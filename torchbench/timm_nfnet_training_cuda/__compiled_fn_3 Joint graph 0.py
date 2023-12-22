from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[16, 3, 3, 3]"; primals_2: "f32[16, 1, 1, 1]"; primals_3: "f32[16]"; primals_4: "f32[32, 16, 3, 3]"; primals_5: "f32[32, 1, 1, 1]"; primals_6: "f32[32]"; primals_7: "f32[64, 32, 3, 3]"; primals_8: "f32[64, 1, 1, 1]"; primals_9: "f32[64]"; primals_10: "f32[128, 64, 3, 3]"; primals_11: "f32[128, 1, 1, 1]"; primals_12: "f32[128]"; primals_13: "f32[256, 128, 1, 1]"; primals_14: "f32[256, 1, 1, 1]"; primals_15: "f32[256]"; primals_16: "f32[128, 128, 1, 1]"; primals_17: "f32[128, 1, 1, 1]"; primals_18: "f32[128]"; primals_19: "f32[128, 128, 3, 3]"; primals_20: "f32[128, 1, 1, 1]"; primals_21: "f32[128]"; primals_22: "f32[128, 128, 3, 3]"; primals_23: "f32[128, 1, 1, 1]"; primals_24: "f32[128]"; primals_25: "f32[256, 128, 1, 1]"; primals_26: "f32[256, 1, 1, 1]"; primals_27: "f32[256]"; primals_28: "f32[]"; primals_29: "f32[512, 256, 1, 1]"; primals_30: "f32[512, 1, 1, 1]"; primals_31: "f32[512]"; primals_32: "f32[256, 256, 1, 1]"; primals_33: "f32[256, 1, 1, 1]"; primals_34: "f32[256]"; primals_35: "f32[256, 128, 3, 3]"; primals_36: "f32[256, 1, 1, 1]"; primals_37: "f32[256]"; primals_38: "f32[256, 128, 3, 3]"; primals_39: "f32[256, 1, 1, 1]"; primals_40: "f32[256]"; primals_41: "f32[512, 256, 1, 1]"; primals_42: "f32[512, 1, 1, 1]"; primals_43: "f32[512]"; primals_44: "f32[]"; primals_45: "f32[256, 512, 1, 1]"; primals_46: "f32[256, 1, 1, 1]"; primals_47: "f32[256]"; primals_48: "f32[256, 128, 3, 3]"; primals_49: "f32[256, 1, 1, 1]"; primals_50: "f32[256]"; primals_51: "f32[256, 128, 3, 3]"; primals_52: "f32[256, 1, 1, 1]"; primals_53: "f32[256]"; primals_54: "f32[512, 256, 1, 1]"; primals_55: "f32[512, 1, 1, 1]"; primals_56: "f32[512]"; primals_57: "f32[]"; primals_58: "f32[1536, 512, 1, 1]"; primals_59: "f32[1536, 1, 1, 1]"; primals_60: "f32[1536]"; primals_61: "f32[768, 512, 1, 1]"; primals_62: "f32[768, 1, 1, 1]"; primals_63: "f32[768]"; primals_64: "f32[768, 128, 3, 3]"; primals_65: "f32[768, 1, 1, 1]"; primals_66: "f32[768]"; primals_67: "f32[768, 128, 3, 3]"; primals_68: "f32[768, 1, 1, 1]"; primals_69: "f32[768]"; primals_70: "f32[1536, 768, 1, 1]"; primals_71: "f32[1536, 1, 1, 1]"; primals_72: "f32[1536]"; primals_73: "f32[]"; primals_74: "f32[768, 1536, 1, 1]"; primals_75: "f32[768, 1, 1, 1]"; primals_76: "f32[768]"; primals_77: "f32[768, 128, 3, 3]"; primals_78: "f32[768, 1, 1, 1]"; primals_79: "f32[768]"; primals_80: "f32[768, 128, 3, 3]"; primals_81: "f32[768, 1, 1, 1]"; primals_82: "f32[768]"; primals_83: "f32[1536, 768, 1, 1]"; primals_84: "f32[1536, 1, 1, 1]"; primals_85: "f32[1536]"; primals_86: "f32[]"; primals_87: "f32[768, 1536, 1, 1]"; primals_88: "f32[768, 1, 1, 1]"; primals_89: "f32[768]"; primals_90: "f32[768, 128, 3, 3]"; primals_91: "f32[768, 1, 1, 1]"; primals_92: "f32[768]"; primals_93: "f32[768, 128, 3, 3]"; primals_94: "f32[768, 1, 1, 1]"; primals_95: "f32[768]"; primals_96: "f32[1536, 768, 1, 1]"; primals_97: "f32[1536, 1, 1, 1]"; primals_98: "f32[1536]"; primals_99: "f32[]"; primals_100: "f32[768, 1536, 1, 1]"; primals_101: "f32[768, 1, 1, 1]"; primals_102: "f32[768]"; primals_103: "f32[768, 128, 3, 3]"; primals_104: "f32[768, 1, 1, 1]"; primals_105: "f32[768]"; primals_106: "f32[768, 128, 3, 3]"; primals_107: "f32[768, 1, 1, 1]"; primals_108: "f32[768]"; primals_109: "f32[1536, 768, 1, 1]"; primals_110: "f32[1536, 1, 1, 1]"; primals_111: "f32[1536]"; primals_112: "f32[]"; primals_113: "f32[768, 1536, 1, 1]"; primals_114: "f32[768, 1, 1, 1]"; primals_115: "f32[768]"; primals_116: "f32[768, 128, 3, 3]"; primals_117: "f32[768, 1, 1, 1]"; primals_118: "f32[768]"; primals_119: "f32[768, 128, 3, 3]"; primals_120: "f32[768, 1, 1, 1]"; primals_121: "f32[768]"; primals_122: "f32[1536, 768, 1, 1]"; primals_123: "f32[1536, 1, 1, 1]"; primals_124: "f32[1536]"; primals_125: "f32[]"; primals_126: "f32[768, 1536, 1, 1]"; primals_127: "f32[768, 1, 1, 1]"; primals_128: "f32[768]"; primals_129: "f32[768, 128, 3, 3]"; primals_130: "f32[768, 1, 1, 1]"; primals_131: "f32[768]"; primals_132: "f32[768, 128, 3, 3]"; primals_133: "f32[768, 1, 1, 1]"; primals_134: "f32[768]"; primals_135: "f32[1536, 768, 1, 1]"; primals_136: "f32[1536, 1, 1, 1]"; primals_137: "f32[1536]"; primals_138: "f32[]"; primals_139: "f32[1536, 1536, 1, 1]"; primals_140: "f32[1536, 1, 1, 1]"; primals_141: "f32[1536]"; primals_142: "f32[768, 1536, 1, 1]"; primals_143: "f32[768, 1, 1, 1]"; primals_144: "f32[768]"; primals_145: "f32[768, 128, 3, 3]"; primals_146: "f32[768, 1, 1, 1]"; primals_147: "f32[768]"; primals_148: "f32[768, 128, 3, 3]"; primals_149: "f32[768, 1, 1, 1]"; primals_150: "f32[768]"; primals_151: "f32[1536, 768, 1, 1]"; primals_152: "f32[1536, 1, 1, 1]"; primals_153: "f32[1536]"; primals_154: "f32[]"; primals_155: "f32[768, 1536, 1, 1]"; primals_156: "f32[768, 1, 1, 1]"; primals_157: "f32[768]"; primals_158: "f32[768, 128, 3, 3]"; primals_159: "f32[768, 1, 1, 1]"; primals_160: "f32[768]"; primals_161: "f32[768, 128, 3, 3]"; primals_162: "f32[768, 1, 1, 1]"; primals_163: "f32[768]"; primals_164: "f32[1536, 768, 1, 1]"; primals_165: "f32[1536, 1, 1, 1]"; primals_166: "f32[1536]"; primals_167: "f32[]"; primals_168: "f32[768, 1536, 1, 1]"; primals_169: "f32[768, 1, 1, 1]"; primals_170: "f32[768]"; primals_171: "f32[768, 128, 3, 3]"; primals_172: "f32[768, 1, 1, 1]"; primals_173: "f32[768]"; primals_174: "f32[768, 128, 3, 3]"; primals_175: "f32[768, 1, 1, 1]"; primals_176: "f32[768]"; primals_177: "f32[1536, 768, 1, 1]"; primals_178: "f32[1536, 1, 1, 1]"; primals_179: "f32[1536]"; primals_180: "f32[]"; primals_181: "f32[3072, 1536, 1, 1]"; primals_182: "f32[3072, 1, 1, 1]"; primals_183: "f32[3072]"; primals_184: "f32[128, 256, 1, 1]"; primals_185: "f32[128]"; primals_186: "f32[256, 128, 1, 1]"; primals_187: "f32[256]"; primals_188: "f32[256, 512, 1, 1]"; primals_189: "f32[256]"; primals_190: "f32[512, 256, 1, 1]"; primals_191: "f32[512]"; primals_192: "f32[256, 512, 1, 1]"; primals_193: "f32[256]"; primals_194: "f32[512, 256, 1, 1]"; primals_195: "f32[512]"; primals_196: "f32[768, 1536, 1, 1]"; primals_197: "f32[768]"; primals_198: "f32[1536, 768, 1, 1]"; primals_199: "f32[1536]"; primals_200: "f32[768, 1536, 1, 1]"; primals_201: "f32[768]"; primals_202: "f32[1536, 768, 1, 1]"; primals_203: "f32[1536]"; primals_204: "f32[768, 1536, 1, 1]"; primals_205: "f32[768]"; primals_206: "f32[1536, 768, 1, 1]"; primals_207: "f32[1536]"; primals_208: "f32[768, 1536, 1, 1]"; primals_209: "f32[768]"; primals_210: "f32[1536, 768, 1, 1]"; primals_211: "f32[1536]"; primals_212: "f32[768, 1536, 1, 1]"; primals_213: "f32[768]"; primals_214: "f32[1536, 768, 1, 1]"; primals_215: "f32[1536]"; primals_216: "f32[768, 1536, 1, 1]"; primals_217: "f32[768]"; primals_218: "f32[1536, 768, 1, 1]"; primals_219: "f32[1536]"; primals_220: "f32[768, 1536, 1, 1]"; primals_221: "f32[768]"; primals_222: "f32[1536, 768, 1, 1]"; primals_223: "f32[1536]"; primals_224: "f32[768, 1536, 1, 1]"; primals_225: "f32[768]"; primals_226: "f32[1536, 768, 1, 1]"; primals_227: "f32[1536]"; primals_228: "f32[768, 1536, 1, 1]"; primals_229: "f32[768]"; primals_230: "f32[1536, 768, 1, 1]"; primals_231: "f32[1536]"; primals_232: "f32[1000, 3072]"; primals_233: "f32[1000]"; primals_234: "f32[4, 3, 192, 192]"; tangents_1: "f32[4, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd: "f32[4, 3, 193, 193]" = torch.ops.aten.constant_pad_nd.default(primals_234, [0, 1, 0, 1], 0.0);  primals_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view: "f32[1, 16, 27]" = torch.ops.aten.view.default(primals_1, [1, 16, -1]);  primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul: "f32[16, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_2, 0.19245008972987526);  primals_2 = None
    view_1: "f32[16]" = torch.ops.aten.view.default(mul, [-1]);  mul = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_2: "f32[16, 3, 3, 3]" = torch.ops.aten.view.default(mul_2, [16, 3, 3, 3]);  mul_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution: "f32[4, 16, 96, 96]" = torch.ops.aten.convolution.default(constant_pad_nd, view_2, primals_3, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_3: "f32[4, 16, 96, 96]" = torch.ops.aten.mul.Tensor(convolution, 0.5)
    mul_4: "f32[4, 16, 96, 96]" = torch.ops.aten.mul.Tensor(convolution, 0.7071067811865476)
    erf: "f32[4, 16, 96, 96]" = torch.ops.aten.erf.default(mul_4);  mul_4 = None
    add_1: "f32[4, 16, 96, 96]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_5: "f32[4, 16, 96, 96]" = torch.ops.aten.mul.Tensor(mul_3, add_1);  mul_3 = add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_6: "f32[4, 16, 96, 96]" = torch.ops.aten.mul.Tensor(mul_5, 1.7015043497085571);  mul_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_3: "f32[1, 32, 144]" = torch.ops.aten.view.default(primals_4, [1, 32, -1]);  primals_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_7: "f32[32, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_5, 0.08333333333333333);  primals_5 = None
    view_4: "f32[32]" = torch.ops.aten.view.default(mul_7, [-1]);  mul_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_1 = torch.ops.aten.var_mean.correction(view_3, [0, 2], correction = 0, keepdim = True)
    getitem_2: "f32[1, 32, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 32, 1]" = var_mean_1[1];  var_mean_1 = None
    add_2: "f32[1, 32, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
    rsqrt_1: "f32[1, 32, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
    sub_1: "f32[1, 32, 144]" = torch.ops.aten.sub.Tensor(view_3, getitem_3)
    mul_8: "f32[1, 32, 144]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_2: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2]);  getitem_3 = None
    squeeze_3: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2]);  rsqrt_1 = None
    unsqueeze_1: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(view_4, -1)
    mul_9: "f32[1, 32, 144]" = torch.ops.aten.mul.Tensor(mul_8, unsqueeze_1);  mul_8 = unsqueeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_5: "f32[32, 16, 3, 3]" = torch.ops.aten.view.default(mul_9, [32, 16, 3, 3]);  mul_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_1: "f32[4, 32, 96, 96]" = torch.ops.aten.convolution.default(mul_6, view_5, primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_10: "f32[4, 32, 96, 96]" = torch.ops.aten.mul.Tensor(convolution_1, 0.5)
    mul_11: "f32[4, 32, 96, 96]" = torch.ops.aten.mul.Tensor(convolution_1, 0.7071067811865476)
    erf_1: "f32[4, 32, 96, 96]" = torch.ops.aten.erf.default(mul_11);  mul_11 = None
    add_3: "f32[4, 32, 96, 96]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_12: "f32[4, 32, 96, 96]" = torch.ops.aten.mul.Tensor(mul_10, add_3);  mul_10 = add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_13: "f32[4, 32, 96, 96]" = torch.ops.aten.mul.Tensor(mul_12, 1.7015043497085571);  mul_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_6: "f32[1, 64, 288]" = torch.ops.aten.view.default(primals_7, [1, 64, -1]);  primals_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_14: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_8, 0.05892556509887896);  primals_8 = None
    view_7: "f32[64]" = torch.ops.aten.view.default(mul_14, [-1]);  mul_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_2 = torch.ops.aten.var_mean.correction(view_6, [0, 2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 64, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 64, 1]" = var_mean_2[1];  var_mean_2 = None
    add_4: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_2: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_2: "f32[1, 64, 288]" = torch.ops.aten.sub.Tensor(view_6, getitem_5)
    mul_15: "f32[1, 64, 288]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_4: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2]);  getitem_5 = None
    squeeze_5: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2]);  rsqrt_2 = None
    unsqueeze_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(view_7, -1)
    mul_16: "f32[1, 64, 288]" = torch.ops.aten.mul.Tensor(mul_15, unsqueeze_2);  mul_15 = unsqueeze_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_8: "f32[64, 32, 3, 3]" = torch.ops.aten.view.default(mul_16, [64, 32, 3, 3]);  mul_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_2: "f32[4, 64, 96, 96]" = torch.ops.aten.convolution.default(mul_13, view_8, primals_9, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_17: "f32[4, 64, 96, 96]" = torch.ops.aten.mul.Tensor(convolution_2, 0.5)
    mul_18: "f32[4, 64, 96, 96]" = torch.ops.aten.mul.Tensor(convolution_2, 0.7071067811865476)
    erf_2: "f32[4, 64, 96, 96]" = torch.ops.aten.erf.default(mul_18);  mul_18 = None
    add_5: "f32[4, 64, 96, 96]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_19: "f32[4, 64, 96, 96]" = torch.ops.aten.mul.Tensor(mul_17, add_5);  mul_17 = add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_20: "f32[4, 64, 96, 96]" = torch.ops.aten.mul.Tensor(mul_19, 1.7015043497085571);  mul_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_1: "f32[4, 64, 97, 97]" = torch.ops.aten.constant_pad_nd.default(mul_20, [0, 1, 0, 1], 0.0);  mul_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_9: "f32[1, 128, 576]" = torch.ops.aten.view.default(primals_10, [1, 128, -1]);  primals_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_21: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_11, 0.041666666666666664);  primals_11 = None
    view_10: "f32[128]" = torch.ops.aten.view.default(mul_21, [-1]);  mul_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_3 = torch.ops.aten.var_mean.correction(view_9, [0, 2], correction = 0, keepdim = True)
    getitem_6: "f32[1, 128, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 128, 1]" = var_mean_3[1];  var_mean_3 = None
    add_6: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
    rsqrt_3: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_3: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_9, getitem_7)
    mul_22: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_6: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2]);  getitem_7 = None
    squeeze_7: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2]);  rsqrt_3 = None
    unsqueeze_3: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_10, -1)
    mul_23: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_3);  mul_22 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_11: "f32[128, 64, 3, 3]" = torch.ops.aten.view.default(mul_23, [128, 64, 3, 3]);  mul_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_3: "f32[4, 128, 48, 48]" = torch.ops.aten.convolution.default(constant_pad_nd_1, view_11, primals_12, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_24: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_3, 0.5)
    mul_25: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_3, 0.7071067811865476)
    erf_3: "f32[4, 128, 48, 48]" = torch.ops.aten.erf.default(mul_25);  mul_25 = None
    add_7: "f32[4, 128, 48, 48]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_26: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_24, add_7);  mul_24 = add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_27: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_26, 1.7015043497085571);  mul_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_28: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_27, 1.0);  mul_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_12: "f32[1, 256, 128]" = torch.ops.aten.view.default(primals_13, [1, 256, -1]);  primals_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_29: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_14, 0.08838834764831845);  primals_14 = None
    view_13: "f32[256]" = torch.ops.aten.view.default(mul_29, [-1]);  mul_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_4 = torch.ops.aten.var_mean.correction(view_12, [0, 2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 256, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 256, 1]" = var_mean_4[1];  var_mean_4 = None
    add_8: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_4: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
    sub_4: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(view_12, getitem_9)
    mul_30: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_8: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2]);  getitem_9 = None
    squeeze_9: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2]);  rsqrt_4 = None
    unsqueeze_4: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(view_13, -1)
    mul_31: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(mul_30, unsqueeze_4);  mul_30 = unsqueeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_14: "f32[256, 128, 1, 1]" = torch.ops.aten.view.default(mul_31, [256, 128, 1, 1]);  mul_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_4: "f32[4, 256, 48, 48]" = torch.ops.aten.convolution.default(mul_28, view_14, primals_15, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_15: "f32[1, 128, 128]" = torch.ops.aten.view.default(primals_16, [1, 128, -1]);  primals_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_32: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_17, 0.08838834764831845);  primals_17 = None
    view_16: "f32[128]" = torch.ops.aten.view.default(mul_32, [-1]);  mul_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_5 = torch.ops.aten.var_mean.correction(view_15, [0, 2], correction = 0, keepdim = True)
    getitem_10: "f32[1, 128, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 128, 1]" = var_mean_5[1];  var_mean_5 = None
    add_9: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
    rsqrt_5: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
    sub_5: "f32[1, 128, 128]" = torch.ops.aten.sub.Tensor(view_15, getitem_11)
    mul_33: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    squeeze_10: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2]);  getitem_11 = None
    squeeze_11: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2]);  rsqrt_5 = None
    unsqueeze_5: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_16, -1)
    mul_34: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(mul_33, unsqueeze_5);  mul_33 = unsqueeze_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_17: "f32[128, 128, 1, 1]" = torch.ops.aten.view.default(mul_34, [128, 128, 1, 1]);  mul_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_5: "f32[4, 128, 48, 48]" = torch.ops.aten.convolution.default(mul_28, view_17, primals_18, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_35: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_5, 0.5)
    mul_36: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_5, 0.7071067811865476)
    erf_4: "f32[4, 128, 48, 48]" = torch.ops.aten.erf.default(mul_36);  mul_36 = None
    add_10: "f32[4, 128, 48, 48]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_37: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_35, add_10);  mul_35 = add_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_38: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_37, 1.7015043497085571);  mul_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_18: "f32[1, 128, 1152]" = torch.ops.aten.view.default(primals_19, [1, 128, -1]);  primals_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_39: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_20, 0.02946278254943948);  primals_20 = None
    view_19: "f32[128]" = torch.ops.aten.view.default(mul_39, [-1]);  mul_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_6 = torch.ops.aten.var_mean.correction(view_18, [0, 2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 128, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 128, 1]" = var_mean_6[1];  var_mean_6 = None
    add_11: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_6: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_6: "f32[1, 128, 1152]" = torch.ops.aten.sub.Tensor(view_18, getitem_13)
    mul_40: "f32[1, 128, 1152]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    squeeze_12: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2]);  getitem_13 = None
    squeeze_13: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2]);  rsqrt_6 = None
    unsqueeze_6: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_19, -1)
    mul_41: "f32[1, 128, 1152]" = torch.ops.aten.mul.Tensor(mul_40, unsqueeze_6);  mul_40 = unsqueeze_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_20: "f32[128, 128, 3, 3]" = torch.ops.aten.view.default(mul_41, [128, 128, 3, 3]);  mul_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_6: "f32[4, 128, 48, 48]" = torch.ops.aten.convolution.default(mul_38, view_20, primals_21, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_42: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_6, 0.5)
    mul_43: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_6, 0.7071067811865476)
    erf_5: "f32[4, 128, 48, 48]" = torch.ops.aten.erf.default(mul_43);  mul_43 = None
    add_12: "f32[4, 128, 48, 48]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_44: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_42, add_12);  mul_42 = add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_45: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_44, 1.7015043497085571);  mul_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_21: "f32[1, 128, 1152]" = torch.ops.aten.view.default(primals_22, [1, 128, -1]);  primals_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_46: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_23, 0.02946278254943948);  primals_23 = None
    view_22: "f32[128]" = torch.ops.aten.view.default(mul_46, [-1]);  mul_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_7 = torch.ops.aten.var_mean.correction(view_21, [0, 2], correction = 0, keepdim = True)
    getitem_14: "f32[1, 128, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 128, 1]" = var_mean_7[1];  var_mean_7 = None
    add_13: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
    rsqrt_7: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
    sub_7: "f32[1, 128, 1152]" = torch.ops.aten.sub.Tensor(view_21, getitem_15)
    mul_47: "f32[1, 128, 1152]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    squeeze_14: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2]);  getitem_15 = None
    squeeze_15: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2]);  rsqrt_7 = None
    unsqueeze_7: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_22, -1)
    mul_48: "f32[1, 128, 1152]" = torch.ops.aten.mul.Tensor(mul_47, unsqueeze_7);  mul_47 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_23: "f32[128, 128, 3, 3]" = torch.ops.aten.view.default(mul_48, [128, 128, 3, 3]);  mul_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_7: "f32[4, 128, 48, 48]" = torch.ops.aten.convolution.default(mul_45, view_23, primals_24, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_49: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_7, 0.5)
    mul_50: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_7, 0.7071067811865476)
    erf_6: "f32[4, 128, 48, 48]" = torch.ops.aten.erf.default(mul_50);  mul_50 = None
    add_14: "f32[4, 128, 48, 48]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_51: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_49, add_14);  mul_49 = add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_52: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_51, 1.7015043497085571);  mul_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_24: "f32[1, 256, 128]" = torch.ops.aten.view.default(primals_25, [1, 256, -1]);  primals_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_53: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_26, 0.08838834764831845);  primals_26 = None
    view_25: "f32[256]" = torch.ops.aten.view.default(mul_53, [-1]);  mul_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_8 = torch.ops.aten.var_mean.correction(view_24, [0, 2], correction = 0, keepdim = True)
    getitem_16: "f32[1, 256, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 256, 1]" = var_mean_8[1];  var_mean_8 = None
    add_15: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_8: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
    sub_8: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(view_24, getitem_17)
    mul_54: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    squeeze_16: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2]);  getitem_17 = None
    squeeze_17: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2]);  rsqrt_8 = None
    unsqueeze_8: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(view_25, -1)
    mul_55: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(mul_54, unsqueeze_8);  mul_54 = unsqueeze_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_26: "f32[256, 128, 1, 1]" = torch.ops.aten.view.default(mul_55, [256, 128, 1, 1]);  mul_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_8: "f32[4, 256, 48, 48]" = torch.ops.aten.convolution.default(mul_52, view_26, primals_27, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean: "f32[4, 256, 1, 1]" = torch.ops.aten.mean.dim(convolution_8, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_9: "f32[4, 128, 1, 1]" = torch.ops.aten.convolution.default(mean, primals_184, primals_185, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu: "f32[4, 128, 1, 1]" = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_10: "f32[4, 256, 1, 1]" = torch.ops.aten.convolution.default(relu, primals_186, primals_187, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid: "f32[4, 256, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_10);  convolution_10 = None
    alias_1: "f32[4, 256, 1, 1]" = torch.ops.aten.alias.default(sigmoid)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_56: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_8, sigmoid)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_57: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_56, 2.0);  mul_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    clone: "f32[4, 256, 48, 48]" = torch.ops.aten.clone.default(mul_57)
    mul_58: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_57, primals_28);  mul_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_59: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_58, 0.2);  mul_58 = None
    add_16: "f32[4, 256, 48, 48]" = torch.ops.aten.add.Tensor(mul_59, convolution_4);  mul_59 = convolution_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_60: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(add_16, 0.5)
    mul_61: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(add_16, 0.7071067811865476)
    erf_7: "f32[4, 256, 48, 48]" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
    add_17: "f32[4, 256, 48, 48]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_62: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_60, add_17);  mul_60 = add_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_63: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_62, 1.7015043497085571);  mul_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_64: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_63, 0.9805806756909201);  mul_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    avg_pool2d: "f32[4, 256, 24, 24]" = torch.ops.aten.avg_pool2d.default(mul_64, [2, 2], [2, 2], [0, 0], True, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_27: "f32[1, 512, 256]" = torch.ops.aten.view.default(primals_29, [1, 512, -1]);  primals_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_65: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_30, 0.0625);  primals_30 = None
    view_28: "f32[512]" = torch.ops.aten.view.default(mul_65, [-1]);  mul_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_9 = torch.ops.aten.var_mean.correction(view_27, [0, 2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 512, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 512, 1]" = var_mean_9[1];  var_mean_9 = None
    add_18: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
    rsqrt_9: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_9: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(view_27, getitem_19)
    mul_66: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    squeeze_18: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2]);  getitem_19 = None
    squeeze_19: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2]);  rsqrt_9 = None
    unsqueeze_9: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(view_28, -1)
    mul_67: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_66, unsqueeze_9);  mul_66 = unsqueeze_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_29: "f32[512, 256, 1, 1]" = torch.ops.aten.view.default(mul_67, [512, 256, 1, 1]);  mul_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_11: "f32[4, 512, 24, 24]" = torch.ops.aten.convolution.default(avg_pool2d, view_29, primals_31, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_30: "f32[1, 256, 256]" = torch.ops.aten.view.default(primals_32, [1, 256, -1]);  primals_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_68: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_33, 0.0625);  primals_33 = None
    view_31: "f32[256]" = torch.ops.aten.view.default(mul_68, [-1]);  mul_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_10 = torch.ops.aten.var_mean.correction(view_30, [0, 2], correction = 0, keepdim = True)
    getitem_20: "f32[1, 256, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 256, 1]" = var_mean_10[1];  var_mean_10 = None
    add_19: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
    rsqrt_10: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
    sub_10: "f32[1, 256, 256]" = torch.ops.aten.sub.Tensor(view_30, getitem_21)
    mul_69: "f32[1, 256, 256]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    squeeze_20: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2]);  getitem_21 = None
    squeeze_21: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2]);  rsqrt_10 = None
    unsqueeze_10: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(view_31, -1)
    mul_70: "f32[1, 256, 256]" = torch.ops.aten.mul.Tensor(mul_69, unsqueeze_10);  mul_69 = unsqueeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_32: "f32[256, 256, 1, 1]" = torch.ops.aten.view.default(mul_70, [256, 256, 1, 1]);  mul_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_12: "f32[4, 256, 48, 48]" = torch.ops.aten.convolution.default(mul_64, view_32, primals_34, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_71: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_12, 0.5)
    mul_72: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_12, 0.7071067811865476)
    erf_8: "f32[4, 256, 48, 48]" = torch.ops.aten.erf.default(mul_72);  mul_72 = None
    add_20: "f32[4, 256, 48, 48]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_73: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_71, add_20);  mul_71 = add_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_74: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_73, 1.7015043497085571);  mul_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_2: "f32[4, 256, 49, 49]" = torch.ops.aten.constant_pad_nd.default(mul_74, [0, 1, 0, 1], 0.0);  mul_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_33: "f32[1, 256, 1152]" = torch.ops.aten.view.default(primals_35, [1, 256, -1]);  primals_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_75: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_36, 0.02946278254943948);  primals_36 = None
    view_34: "f32[256]" = torch.ops.aten.view.default(mul_75, [-1]);  mul_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_11 = torch.ops.aten.var_mean.correction(view_33, [0, 2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 256, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 256, 1]" = var_mean_11[1];  var_mean_11 = None
    add_21: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_11: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_11: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(view_33, getitem_23)
    mul_76: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    squeeze_22: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2]);  getitem_23 = None
    squeeze_23: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2]);  rsqrt_11 = None
    unsqueeze_11: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(view_34, -1)
    mul_77: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(mul_76, unsqueeze_11);  mul_76 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_35: "f32[256, 128, 3, 3]" = torch.ops.aten.view.default(mul_77, [256, 128, 3, 3]);  mul_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_13: "f32[4, 256, 24, 24]" = torch.ops.aten.convolution.default(constant_pad_nd_2, view_35, primals_37, [2, 2], [0, 0], [1, 1], False, [0, 0], 2);  primals_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_78: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_13, 0.5)
    mul_79: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_13, 0.7071067811865476)
    erf_9: "f32[4, 256, 24, 24]" = torch.ops.aten.erf.default(mul_79);  mul_79 = None
    add_22: "f32[4, 256, 24, 24]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_80: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(mul_78, add_22);  mul_78 = add_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_81: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(mul_80, 1.7015043497085571);  mul_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_36: "f32[1, 256, 1152]" = torch.ops.aten.view.default(primals_38, [1, 256, -1]);  primals_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_82: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_39, 0.02946278254943948);  primals_39 = None
    view_37: "f32[256]" = torch.ops.aten.view.default(mul_82, [-1]);  mul_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_12 = torch.ops.aten.var_mean.correction(view_36, [0, 2], correction = 0, keepdim = True)
    getitem_24: "f32[1, 256, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 256, 1]" = var_mean_12[1];  var_mean_12 = None
    add_23: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
    rsqrt_12: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
    sub_12: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(view_36, getitem_25)
    mul_83: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    squeeze_24: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2]);  getitem_25 = None
    squeeze_25: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2]);  rsqrt_12 = None
    unsqueeze_12: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(view_37, -1)
    mul_84: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(mul_83, unsqueeze_12);  mul_83 = unsqueeze_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_38: "f32[256, 128, 3, 3]" = torch.ops.aten.view.default(mul_84, [256, 128, 3, 3]);  mul_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_14: "f32[4, 256, 24, 24]" = torch.ops.aten.convolution.default(mul_81, view_38, primals_40, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  primals_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_85: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_14, 0.5)
    mul_86: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_14, 0.7071067811865476)
    erf_10: "f32[4, 256, 24, 24]" = torch.ops.aten.erf.default(mul_86);  mul_86 = None
    add_24: "f32[4, 256, 24, 24]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_87: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(mul_85, add_24);  mul_85 = add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_88: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(mul_87, 1.7015043497085571);  mul_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_39: "f32[1, 512, 256]" = torch.ops.aten.view.default(primals_41, [1, 512, -1]);  primals_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_89: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_42, 0.0625);  primals_42 = None
    view_40: "f32[512]" = torch.ops.aten.view.default(mul_89, [-1]);  mul_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_13 = torch.ops.aten.var_mean.correction(view_39, [0, 2], correction = 0, keepdim = True)
    getitem_26: "f32[1, 512, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 512, 1]" = var_mean_13[1];  var_mean_13 = None
    add_25: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
    rsqrt_13: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    sub_13: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(view_39, getitem_27)
    mul_90: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    squeeze_26: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2]);  getitem_27 = None
    squeeze_27: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2]);  rsqrt_13 = None
    unsqueeze_13: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(view_40, -1)
    mul_91: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_90, unsqueeze_13);  mul_90 = unsqueeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_41: "f32[512, 256, 1, 1]" = torch.ops.aten.view.default(mul_91, [512, 256, 1, 1]);  mul_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_15: "f32[4, 512, 24, 24]" = torch.ops.aten.convolution.default(mul_88, view_41, primals_43, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_1: "f32[4, 512, 1, 1]" = torch.ops.aten.mean.dim(convolution_15, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_16: "f32[4, 256, 1, 1]" = torch.ops.aten.convolution.default(mean_1, primals_188, primals_189, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_1: "f32[4, 256, 1, 1]" = torch.ops.aten.relu.default(convolution_16);  convolution_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_17: "f32[4, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_1, primals_190, primals_191, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_1: "f32[4, 512, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_17);  convolution_17 = None
    alias_3: "f32[4, 512, 1, 1]" = torch.ops.aten.alias.default(sigmoid_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_92: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_15, sigmoid_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_93: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_92, 2.0);  mul_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    clone_1: "f32[4, 512, 24, 24]" = torch.ops.aten.clone.default(mul_93)
    mul_94: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_93, primals_44);  mul_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_95: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_94, 0.2);  mul_94 = None
    add_26: "f32[4, 512, 24, 24]" = torch.ops.aten.add.Tensor(mul_95, convolution_11);  mul_95 = convolution_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_96: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_26, 0.5)
    mul_97: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_26, 0.7071067811865476)
    erf_11: "f32[4, 512, 24, 24]" = torch.ops.aten.erf.default(mul_97);  mul_97 = None
    add_27: "f32[4, 512, 24, 24]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_98: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_96, add_27);  mul_96 = add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_99: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_98, 1.7015043497085571);  mul_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_100: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_99, 0.9805806756909201);  mul_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_42: "f32[1, 256, 512]" = torch.ops.aten.view.default(primals_45, [1, 256, -1]);  primals_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_101: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_46, 0.04419417382415922);  primals_46 = None
    view_43: "f32[256]" = torch.ops.aten.view.default(mul_101, [-1]);  mul_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_14 = torch.ops.aten.var_mean.correction(view_42, [0, 2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 256, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 256, 1]" = var_mean_14[1];  var_mean_14 = None
    add_28: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_14: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_14: "f32[1, 256, 512]" = torch.ops.aten.sub.Tensor(view_42, getitem_29)
    mul_102: "f32[1, 256, 512]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    squeeze_28: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2]);  getitem_29 = None
    squeeze_29: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2]);  rsqrt_14 = None
    unsqueeze_14: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(view_43, -1)
    mul_103: "f32[1, 256, 512]" = torch.ops.aten.mul.Tensor(mul_102, unsqueeze_14);  mul_102 = unsqueeze_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_44: "f32[256, 512, 1, 1]" = torch.ops.aten.view.default(mul_103, [256, 512, 1, 1]);  mul_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_18: "f32[4, 256, 24, 24]" = torch.ops.aten.convolution.default(mul_100, view_44, primals_47, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_104: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_18, 0.5)
    mul_105: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_18, 0.7071067811865476)
    erf_12: "f32[4, 256, 24, 24]" = torch.ops.aten.erf.default(mul_105);  mul_105 = None
    add_29: "f32[4, 256, 24, 24]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_106: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(mul_104, add_29);  mul_104 = add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_107: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(mul_106, 1.7015043497085571);  mul_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_45: "f32[1, 256, 1152]" = torch.ops.aten.view.default(primals_48, [1, 256, -1]);  primals_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_108: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_49, 0.02946278254943948);  primals_49 = None
    view_46: "f32[256]" = torch.ops.aten.view.default(mul_108, [-1]);  mul_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_15 = torch.ops.aten.var_mean.correction(view_45, [0, 2], correction = 0, keepdim = True)
    getitem_30: "f32[1, 256, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 256, 1]" = var_mean_15[1];  var_mean_15 = None
    add_30: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
    rsqrt_15: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
    sub_15: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(view_45, getitem_31)
    mul_109: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    squeeze_30: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2]);  getitem_31 = None
    squeeze_31: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2]);  rsqrt_15 = None
    unsqueeze_15: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(view_46, -1)
    mul_110: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(mul_109, unsqueeze_15);  mul_109 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_47: "f32[256, 128, 3, 3]" = torch.ops.aten.view.default(mul_110, [256, 128, 3, 3]);  mul_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_19: "f32[4, 256, 24, 24]" = torch.ops.aten.convolution.default(mul_107, view_47, primals_50, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  primals_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_111: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_19, 0.5)
    mul_112: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_19, 0.7071067811865476)
    erf_13: "f32[4, 256, 24, 24]" = torch.ops.aten.erf.default(mul_112);  mul_112 = None
    add_31: "f32[4, 256, 24, 24]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_113: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(mul_111, add_31);  mul_111 = add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_114: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(mul_113, 1.7015043497085571);  mul_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_48: "f32[1, 256, 1152]" = torch.ops.aten.view.default(primals_51, [1, 256, -1]);  primals_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_115: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_52, 0.02946278254943948);  primals_52 = None
    view_49: "f32[256]" = torch.ops.aten.view.default(mul_115, [-1]);  mul_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_16 = torch.ops.aten.var_mean.correction(view_48, [0, 2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 256, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 256, 1]" = var_mean_16[1];  var_mean_16 = None
    add_32: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
    rsqrt_16: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_16: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(view_48, getitem_33)
    mul_116: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    squeeze_32: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2]);  getitem_33 = None
    squeeze_33: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2]);  rsqrt_16 = None
    unsqueeze_16: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(view_49, -1)
    mul_117: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(mul_116, unsqueeze_16);  mul_116 = unsqueeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_50: "f32[256, 128, 3, 3]" = torch.ops.aten.view.default(mul_117, [256, 128, 3, 3]);  mul_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_20: "f32[4, 256, 24, 24]" = torch.ops.aten.convolution.default(mul_114, view_50, primals_53, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  primals_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_118: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_20, 0.5)
    mul_119: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_20, 0.7071067811865476)
    erf_14: "f32[4, 256, 24, 24]" = torch.ops.aten.erf.default(mul_119);  mul_119 = None
    add_33: "f32[4, 256, 24, 24]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_120: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(mul_118, add_33);  mul_118 = add_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_121: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(mul_120, 1.7015043497085571);  mul_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_51: "f32[1, 512, 256]" = torch.ops.aten.view.default(primals_54, [1, 512, -1]);  primals_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_122: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_55, 0.0625);  primals_55 = None
    view_52: "f32[512]" = torch.ops.aten.view.default(mul_122, [-1]);  mul_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_17 = torch.ops.aten.var_mean.correction(view_51, [0, 2], correction = 0, keepdim = True)
    getitem_34: "f32[1, 512, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 512, 1]" = var_mean_17[1];  var_mean_17 = None
    add_34: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
    rsqrt_17: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
    sub_17: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(view_51, getitem_35)
    mul_123: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    squeeze_34: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2]);  getitem_35 = None
    squeeze_35: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2]);  rsqrt_17 = None
    unsqueeze_17: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(view_52, -1)
    mul_124: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_123, unsqueeze_17);  mul_123 = unsqueeze_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_53: "f32[512, 256, 1, 1]" = torch.ops.aten.view.default(mul_124, [512, 256, 1, 1]);  mul_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_21: "f32[4, 512, 24, 24]" = torch.ops.aten.convolution.default(mul_121, view_53, primals_56, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_2: "f32[4, 512, 1, 1]" = torch.ops.aten.mean.dim(convolution_21, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_22: "f32[4, 256, 1, 1]" = torch.ops.aten.convolution.default(mean_2, primals_192, primals_193, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_2: "f32[4, 256, 1, 1]" = torch.ops.aten.relu.default(convolution_22);  convolution_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_23: "f32[4, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_2, primals_194, primals_195, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_2: "f32[4, 512, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_23);  convolution_23 = None
    alias_5: "f32[4, 512, 1, 1]" = torch.ops.aten.alias.default(sigmoid_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_125: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_21, sigmoid_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_126: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_125, 2.0);  mul_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    clone_2: "f32[4, 512, 24, 24]" = torch.ops.aten.clone.default(mul_126)
    mul_127: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_126, primals_57);  mul_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_128: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_127, 0.2);  mul_127 = None
    add_35: "f32[4, 512, 24, 24]" = torch.ops.aten.add.Tensor(mul_128, add_26);  mul_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_129: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_35, 0.5)
    mul_130: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_35, 0.7071067811865476)
    erf_15: "f32[4, 512, 24, 24]" = torch.ops.aten.erf.default(mul_130);  mul_130 = None
    add_36: "f32[4, 512, 24, 24]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_131: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_129, add_36);  mul_129 = add_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_132: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_131, 1.7015043497085571);  mul_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_133: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_132, 0.9622504486493761);  mul_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    avg_pool2d_1: "f32[4, 512, 12, 12]" = torch.ops.aten.avg_pool2d.default(mul_133, [2, 2], [2, 2], [0, 0], True, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_54: "f32[1, 1536, 512]" = torch.ops.aten.view.default(primals_58, [1, 1536, -1]);  primals_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_134: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_59, 0.04419417382415922);  primals_59 = None
    view_55: "f32[1536]" = torch.ops.aten.view.default(mul_134, [-1]);  mul_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_18 = torch.ops.aten.var_mean.correction(view_54, [0, 2], correction = 0, keepdim = True)
    getitem_36: "f32[1, 1536, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 1536, 1]" = var_mean_18[1];  var_mean_18 = None
    add_37: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
    rsqrt_18: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    sub_18: "f32[1, 1536, 512]" = torch.ops.aten.sub.Tensor(view_54, getitem_37)
    mul_135: "f32[1, 1536, 512]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    squeeze_36: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2]);  getitem_37 = None
    squeeze_37: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2]);  rsqrt_18 = None
    unsqueeze_18: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_55, -1)
    mul_136: "f32[1, 1536, 512]" = torch.ops.aten.mul.Tensor(mul_135, unsqueeze_18);  mul_135 = unsqueeze_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_56: "f32[1536, 512, 1, 1]" = torch.ops.aten.view.default(mul_136, [1536, 512, 1, 1]);  mul_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_24: "f32[4, 1536, 12, 12]" = torch.ops.aten.convolution.default(avg_pool2d_1, view_56, primals_60, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_57: "f32[1, 768, 512]" = torch.ops.aten.view.default(primals_61, [1, 768, -1]);  primals_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_137: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_62, 0.04419417382415922);  primals_62 = None
    view_58: "f32[768]" = torch.ops.aten.view.default(mul_137, [-1]);  mul_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_19 = torch.ops.aten.var_mean.correction(view_57, [0, 2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 768, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 768, 1]" = var_mean_19[1];  var_mean_19 = None
    add_38: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
    rsqrt_19: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_19: "f32[1, 768, 512]" = torch.ops.aten.sub.Tensor(view_57, getitem_39)
    mul_138: "f32[1, 768, 512]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    squeeze_38: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2]);  getitem_39 = None
    squeeze_39: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2]);  rsqrt_19 = None
    unsqueeze_19: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_58, -1)
    mul_139: "f32[1, 768, 512]" = torch.ops.aten.mul.Tensor(mul_138, unsqueeze_19);  mul_138 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_59: "f32[768, 512, 1, 1]" = torch.ops.aten.view.default(mul_139, [768, 512, 1, 1]);  mul_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_25: "f32[4, 768, 24, 24]" = torch.ops.aten.convolution.default(mul_133, view_59, primals_63, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_140: "f32[4, 768, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_25, 0.5)
    mul_141: "f32[4, 768, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_25, 0.7071067811865476)
    erf_16: "f32[4, 768, 24, 24]" = torch.ops.aten.erf.default(mul_141);  mul_141 = None
    add_39: "f32[4, 768, 24, 24]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_142: "f32[4, 768, 24, 24]" = torch.ops.aten.mul.Tensor(mul_140, add_39);  mul_140 = add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_143: "f32[4, 768, 24, 24]" = torch.ops.aten.mul.Tensor(mul_142, 1.7015043497085571);  mul_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_3: "f32[4, 768, 25, 25]" = torch.ops.aten.constant_pad_nd.default(mul_143, [0, 1, 0, 1], 0.0);  mul_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_60: "f32[1, 768, 1152]" = torch.ops.aten.view.default(primals_64, [1, 768, -1]);  primals_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_144: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_65, 0.02946278254943948);  primals_65 = None
    view_61: "f32[768]" = torch.ops.aten.view.default(mul_144, [-1]);  mul_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_20 = torch.ops.aten.var_mean.correction(view_60, [0, 2], correction = 0, keepdim = True)
    getitem_40: "f32[1, 768, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 768, 1]" = var_mean_20[1];  var_mean_20 = None
    add_40: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
    rsqrt_20: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
    sub_20: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_60, getitem_41)
    mul_145: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    squeeze_40: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2]);  getitem_41 = None
    squeeze_41: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2]);  rsqrt_20 = None
    unsqueeze_20: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_61, -1)
    mul_146: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_145, unsqueeze_20);  mul_145 = unsqueeze_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_62: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_146, [768, 128, 3, 3]);  mul_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_26: "f32[4, 768, 12, 12]" = torch.ops.aten.convolution.default(constant_pad_nd_3, view_62, primals_66, [2, 2], [0, 0], [1, 1], False, [0, 0], 6);  primals_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_147: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_26, 0.5)
    mul_148: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_26, 0.7071067811865476)
    erf_17: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_148);  mul_148 = None
    add_41: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_149: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_147, add_41);  mul_147 = add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_150: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_149, 1.7015043497085571);  mul_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_63: "f32[1, 768, 1152]" = torch.ops.aten.view.default(primals_67, [1, 768, -1]);  primals_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_151: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_68, 0.02946278254943948);  primals_68 = None
    view_64: "f32[768]" = torch.ops.aten.view.default(mul_151, [-1]);  mul_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_21 = torch.ops.aten.var_mean.correction(view_63, [0, 2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 768, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 768, 1]" = var_mean_21[1];  var_mean_21 = None
    add_42: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
    rsqrt_21: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_21: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_63, getitem_43)
    mul_152: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    squeeze_42: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2]);  getitem_43 = None
    squeeze_43: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2]);  rsqrt_21 = None
    unsqueeze_21: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_64, -1)
    mul_153: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_152, unsqueeze_21);  mul_152 = unsqueeze_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_65: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_153, [768, 128, 3, 3]);  mul_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_27: "f32[4, 768, 12, 12]" = torch.ops.aten.convolution.default(mul_150, view_65, primals_69, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_154: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_27, 0.5)
    mul_155: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_27, 0.7071067811865476)
    erf_18: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_155);  mul_155 = None
    add_43: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_156: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_154, add_43);  mul_154 = add_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_157: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_156, 1.7015043497085571);  mul_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_66: "f32[1, 1536, 768]" = torch.ops.aten.view.default(primals_70, [1, 1536, -1]);  primals_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_158: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_71, 0.03608439182435161);  primals_71 = None
    view_67: "f32[1536]" = torch.ops.aten.view.default(mul_158, [-1]);  mul_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_22 = torch.ops.aten.var_mean.correction(view_66, [0, 2], correction = 0, keepdim = True)
    getitem_44: "f32[1, 1536, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 1536, 1]" = var_mean_22[1];  var_mean_22 = None
    add_44: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
    rsqrt_22: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
    sub_22: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_66, getitem_45)
    mul_159: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    squeeze_44: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2]);  getitem_45 = None
    squeeze_45: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2]);  rsqrt_22 = None
    unsqueeze_22: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_67, -1)
    mul_160: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(mul_159, unsqueeze_22);  mul_159 = unsqueeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_68: "f32[1536, 768, 1, 1]" = torch.ops.aten.view.default(mul_160, [1536, 768, 1, 1]);  mul_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_28: "f32[4, 1536, 12, 12]" = torch.ops.aten.convolution.default(mul_157, view_68, primals_72, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_3: "f32[4, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_28, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_29: "f32[4, 768, 1, 1]" = torch.ops.aten.convolution.default(mean_3, primals_196, primals_197, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_3: "f32[4, 768, 1, 1]" = torch.ops.aten.relu.default(convolution_29);  convolution_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_30: "f32[4, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_3, primals_198, primals_199, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_3: "f32[4, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_30);  convolution_30 = None
    alias_7: "f32[4, 1536, 1, 1]" = torch.ops.aten.alias.default(sigmoid_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_161: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_28, sigmoid_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_162: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_161, 2.0);  mul_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    clone_3: "f32[4, 1536, 12, 12]" = torch.ops.aten.clone.default(mul_162)
    mul_163: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_162, primals_73);  mul_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_164: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_163, 0.2);  mul_163 = None
    add_45: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_164, convolution_24);  mul_164 = convolution_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_165: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_45, 0.5)
    mul_166: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_45, 0.7071067811865476)
    erf_19: "f32[4, 1536, 12, 12]" = torch.ops.aten.erf.default(mul_166);  mul_166 = None
    add_46: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_167: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_165, add_46);  mul_165 = add_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_168: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_167, 1.7015043497085571);  mul_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_169: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_168, 0.9805806756909201);  mul_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_69: "f32[1, 768, 1536]" = torch.ops.aten.view.default(primals_74, [1, 768, -1]);  primals_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_170: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_75, 0.02551551815399144);  primals_75 = None
    view_70: "f32[768]" = torch.ops.aten.view.default(mul_170, [-1]);  mul_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_23 = torch.ops.aten.var_mean.correction(view_69, [0, 2], correction = 0, keepdim = True)
    getitem_46: "f32[1, 768, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 768, 1]" = var_mean_23[1];  var_mean_23 = None
    add_47: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
    rsqrt_23: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    sub_23: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_69, getitem_47)
    mul_171: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    squeeze_46: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2]);  getitem_47 = None
    squeeze_47: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2]);  rsqrt_23 = None
    unsqueeze_23: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_70, -1)
    mul_172: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(mul_171, unsqueeze_23);  mul_171 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_71: "f32[768, 1536, 1, 1]" = torch.ops.aten.view.default(mul_172, [768, 1536, 1, 1]);  mul_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_31: "f32[4, 768, 12, 12]" = torch.ops.aten.convolution.default(mul_169, view_71, primals_76, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_173: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_31, 0.5)
    mul_174: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_31, 0.7071067811865476)
    erf_20: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_174);  mul_174 = None
    add_48: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_175: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_173, add_48);  mul_173 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_176: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_175, 1.7015043497085571);  mul_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_72: "f32[1, 768, 1152]" = torch.ops.aten.view.default(primals_77, [1, 768, -1]);  primals_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_177: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_78, 0.02946278254943948);  primals_78 = None
    view_73: "f32[768]" = torch.ops.aten.view.default(mul_177, [-1]);  mul_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_24 = torch.ops.aten.var_mean.correction(view_72, [0, 2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 768, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 768, 1]" = var_mean_24[1];  var_mean_24 = None
    add_49: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
    rsqrt_24: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    sub_24: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_72, getitem_49)
    mul_178: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    squeeze_48: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2]);  getitem_49 = None
    squeeze_49: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2]);  rsqrt_24 = None
    unsqueeze_24: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_73, -1)
    mul_179: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_178, unsqueeze_24);  mul_178 = unsqueeze_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_74: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_179, [768, 128, 3, 3]);  mul_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_32: "f32[4, 768, 12, 12]" = torch.ops.aten.convolution.default(mul_176, view_74, primals_79, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_180: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_32, 0.5)
    mul_181: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_32, 0.7071067811865476)
    erf_21: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_181);  mul_181 = None
    add_50: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_182: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_180, add_50);  mul_180 = add_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_183: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_182, 1.7015043497085571);  mul_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_75: "f32[1, 768, 1152]" = torch.ops.aten.view.default(primals_80, [1, 768, -1]);  primals_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_184: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_81, 0.02946278254943948);  primals_81 = None
    view_76: "f32[768]" = torch.ops.aten.view.default(mul_184, [-1]);  mul_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_25 = torch.ops.aten.var_mean.correction(view_75, [0, 2], correction = 0, keepdim = True)
    getitem_50: "f32[1, 768, 1]" = var_mean_25[0]
    getitem_51: "f32[1, 768, 1]" = var_mean_25[1];  var_mean_25 = None
    add_51: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
    rsqrt_25: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
    sub_25: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_75, getitem_51)
    mul_185: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    squeeze_50: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2]);  getitem_51 = None
    squeeze_51: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2]);  rsqrt_25 = None
    unsqueeze_25: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_76, -1)
    mul_186: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_185, unsqueeze_25);  mul_185 = unsqueeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_77: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_186, [768, 128, 3, 3]);  mul_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_33: "f32[4, 768, 12, 12]" = torch.ops.aten.convolution.default(mul_183, view_77, primals_82, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_187: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_33, 0.5)
    mul_188: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_33, 0.7071067811865476)
    erf_22: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_188);  mul_188 = None
    add_52: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_189: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_187, add_52);  mul_187 = add_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_190: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_189, 1.7015043497085571);  mul_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_78: "f32[1, 1536, 768]" = torch.ops.aten.view.default(primals_83, [1, 1536, -1]);  primals_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_191: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_84, 0.03608439182435161);  primals_84 = None
    view_79: "f32[1536]" = torch.ops.aten.view.default(mul_191, [-1]);  mul_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_26 = torch.ops.aten.var_mean.correction(view_78, [0, 2], correction = 0, keepdim = True)
    getitem_52: "f32[1, 1536, 1]" = var_mean_26[0]
    getitem_53: "f32[1, 1536, 1]" = var_mean_26[1];  var_mean_26 = None
    add_53: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
    rsqrt_26: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    sub_26: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_78, getitem_53)
    mul_192: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    squeeze_52: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2]);  getitem_53 = None
    squeeze_53: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2]);  rsqrt_26 = None
    unsqueeze_26: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_79, -1)
    mul_193: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(mul_192, unsqueeze_26);  mul_192 = unsqueeze_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_80: "f32[1536, 768, 1, 1]" = torch.ops.aten.view.default(mul_193, [1536, 768, 1, 1]);  mul_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_34: "f32[4, 1536, 12, 12]" = torch.ops.aten.convolution.default(mul_190, view_80, primals_85, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_4: "f32[4, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_34, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_35: "f32[4, 768, 1, 1]" = torch.ops.aten.convolution.default(mean_4, primals_200, primals_201, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_4: "f32[4, 768, 1, 1]" = torch.ops.aten.relu.default(convolution_35);  convolution_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_36: "f32[4, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_4, primals_202, primals_203, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_4: "f32[4, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_36);  convolution_36 = None
    alias_9: "f32[4, 1536, 1, 1]" = torch.ops.aten.alias.default(sigmoid_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_194: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_34, sigmoid_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_195: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_194, 2.0);  mul_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    clone_4: "f32[4, 1536, 12, 12]" = torch.ops.aten.clone.default(mul_195)
    mul_196: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_195, primals_86);  mul_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_197: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_196, 0.2);  mul_196 = None
    add_54: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_197, add_45);  mul_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_198: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_54, 0.5)
    mul_199: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_54, 0.7071067811865476)
    erf_23: "f32[4, 1536, 12, 12]" = torch.ops.aten.erf.default(mul_199);  mul_199 = None
    add_55: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_200: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_198, add_55);  mul_198 = add_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_201: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_200, 1.7015043497085571);  mul_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_202: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_201, 0.9622504486493761);  mul_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_81: "f32[1, 768, 1536]" = torch.ops.aten.view.default(primals_87, [1, 768, -1]);  primals_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_203: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_88, 0.02551551815399144);  primals_88 = None
    view_82: "f32[768]" = torch.ops.aten.view.default(mul_203, [-1]);  mul_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_27 = torch.ops.aten.var_mean.correction(view_81, [0, 2], correction = 0, keepdim = True)
    getitem_54: "f32[1, 768, 1]" = var_mean_27[0]
    getitem_55: "f32[1, 768, 1]" = var_mean_27[1];  var_mean_27 = None
    add_56: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
    rsqrt_27: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_27: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_81, getitem_55)
    mul_204: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
    squeeze_54: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2]);  getitem_55 = None
    squeeze_55: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2]);  rsqrt_27 = None
    unsqueeze_27: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_82, -1)
    mul_205: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(mul_204, unsqueeze_27);  mul_204 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_83: "f32[768, 1536, 1, 1]" = torch.ops.aten.view.default(mul_205, [768, 1536, 1, 1]);  mul_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_37: "f32[4, 768, 12, 12]" = torch.ops.aten.convolution.default(mul_202, view_83, primals_89, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_206: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_37, 0.5)
    mul_207: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_37, 0.7071067811865476)
    erf_24: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_207);  mul_207 = None
    add_57: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
    mul_208: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_206, add_57);  mul_206 = add_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_209: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_208, 1.7015043497085571);  mul_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_84: "f32[1, 768, 1152]" = torch.ops.aten.view.default(primals_90, [1, 768, -1]);  primals_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_210: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_91, 0.02946278254943948);  primals_91 = None
    view_85: "f32[768]" = torch.ops.aten.view.default(mul_210, [-1]);  mul_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_28 = torch.ops.aten.var_mean.correction(view_84, [0, 2], correction = 0, keepdim = True)
    getitem_56: "f32[1, 768, 1]" = var_mean_28[0]
    getitem_57: "f32[1, 768, 1]" = var_mean_28[1];  var_mean_28 = None
    add_58: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
    rsqrt_28: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    sub_28: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_84, getitem_57)
    mul_211: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
    squeeze_56: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2]);  getitem_57 = None
    squeeze_57: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2]);  rsqrt_28 = None
    unsqueeze_28: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_85, -1)
    mul_212: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_211, unsqueeze_28);  mul_211 = unsqueeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_86: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_212, [768, 128, 3, 3]);  mul_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_38: "f32[4, 768, 12, 12]" = torch.ops.aten.convolution.default(mul_209, view_86, primals_92, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_213: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_38, 0.5)
    mul_214: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_38, 0.7071067811865476)
    erf_25: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_214);  mul_214 = None
    add_59: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
    mul_215: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_213, add_59);  mul_213 = add_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_216: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_215, 1.7015043497085571);  mul_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_87: "f32[1, 768, 1152]" = torch.ops.aten.view.default(primals_93, [1, 768, -1]);  primals_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_217: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_94, 0.02946278254943948);  primals_94 = None
    view_88: "f32[768]" = torch.ops.aten.view.default(mul_217, [-1]);  mul_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_29 = torch.ops.aten.var_mean.correction(view_87, [0, 2], correction = 0, keepdim = True)
    getitem_58: "f32[1, 768, 1]" = var_mean_29[0]
    getitem_59: "f32[1, 768, 1]" = var_mean_29[1];  var_mean_29 = None
    add_60: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
    rsqrt_29: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_29: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_87, getitem_59)
    mul_218: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
    squeeze_58: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2]);  getitem_59 = None
    squeeze_59: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2]);  rsqrt_29 = None
    unsqueeze_29: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_88, -1)
    mul_219: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_218, unsqueeze_29);  mul_218 = unsqueeze_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_89: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_219, [768, 128, 3, 3]);  mul_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_39: "f32[4, 768, 12, 12]" = torch.ops.aten.convolution.default(mul_216, view_89, primals_95, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_220: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_39, 0.5)
    mul_221: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_39, 0.7071067811865476)
    erf_26: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_221);  mul_221 = None
    add_61: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
    mul_222: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_220, add_61);  mul_220 = add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_223: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_222, 1.7015043497085571);  mul_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_90: "f32[1, 1536, 768]" = torch.ops.aten.view.default(primals_96, [1, 1536, -1]);  primals_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_224: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_97, 0.03608439182435161);  primals_97 = None
    view_91: "f32[1536]" = torch.ops.aten.view.default(mul_224, [-1]);  mul_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_30 = torch.ops.aten.var_mean.correction(view_90, [0, 2], correction = 0, keepdim = True)
    getitem_60: "f32[1, 1536, 1]" = var_mean_30[0]
    getitem_61: "f32[1, 1536, 1]" = var_mean_30[1];  var_mean_30 = None
    add_62: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
    rsqrt_30: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    sub_30: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_90, getitem_61)
    mul_225: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
    squeeze_60: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2]);  getitem_61 = None
    squeeze_61: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2]);  rsqrt_30 = None
    unsqueeze_30: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_91, -1)
    mul_226: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(mul_225, unsqueeze_30);  mul_225 = unsqueeze_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_92: "f32[1536, 768, 1, 1]" = torch.ops.aten.view.default(mul_226, [1536, 768, 1, 1]);  mul_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_40: "f32[4, 1536, 12, 12]" = torch.ops.aten.convolution.default(mul_223, view_92, primals_98, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_5: "f32[4, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_40, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_41: "f32[4, 768, 1, 1]" = torch.ops.aten.convolution.default(mean_5, primals_204, primals_205, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_5: "f32[4, 768, 1, 1]" = torch.ops.aten.relu.default(convolution_41);  convolution_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_42: "f32[4, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_5, primals_206, primals_207, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_5: "f32[4, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_42);  convolution_42 = None
    alias_11: "f32[4, 1536, 1, 1]" = torch.ops.aten.alias.default(sigmoid_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_227: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_40, sigmoid_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_228: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_227, 2.0);  mul_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    clone_5: "f32[4, 1536, 12, 12]" = torch.ops.aten.clone.default(mul_228)
    mul_229: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_228, primals_99);  mul_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_230: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_229, 0.2);  mul_229 = None
    add_63: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_230, add_54);  mul_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_231: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_63, 0.5)
    mul_232: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_63, 0.7071067811865476)
    erf_27: "f32[4, 1536, 12, 12]" = torch.ops.aten.erf.default(mul_232);  mul_232 = None
    add_64: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
    mul_233: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_231, add_64);  mul_231 = add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_234: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_233, 1.7015043497085571);  mul_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_235: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_234, 0.9449111825230679);  mul_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_93: "f32[1, 768, 1536]" = torch.ops.aten.view.default(primals_100, [1, 768, -1]);  primals_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_236: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_101, 0.02551551815399144);  primals_101 = None
    view_94: "f32[768]" = torch.ops.aten.view.default(mul_236, [-1]);  mul_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_31 = torch.ops.aten.var_mean.correction(view_93, [0, 2], correction = 0, keepdim = True)
    getitem_62: "f32[1, 768, 1]" = var_mean_31[0]
    getitem_63: "f32[1, 768, 1]" = var_mean_31[1];  var_mean_31 = None
    add_65: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
    rsqrt_31: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
    sub_31: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_93, getitem_63)
    mul_237: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
    squeeze_62: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2]);  getitem_63 = None
    squeeze_63: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2]);  rsqrt_31 = None
    unsqueeze_31: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_94, -1)
    mul_238: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(mul_237, unsqueeze_31);  mul_237 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_95: "f32[768, 1536, 1, 1]" = torch.ops.aten.view.default(mul_238, [768, 1536, 1, 1]);  mul_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_43: "f32[4, 768, 12, 12]" = torch.ops.aten.convolution.default(mul_235, view_95, primals_102, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_239: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_43, 0.5)
    mul_240: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_43, 0.7071067811865476)
    erf_28: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_240);  mul_240 = None
    add_66: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
    mul_241: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_239, add_66);  mul_239 = add_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_242: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_241, 1.7015043497085571);  mul_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_96: "f32[1, 768, 1152]" = torch.ops.aten.view.default(primals_103, [1, 768, -1]);  primals_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_243: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_104, 0.02946278254943948);  primals_104 = None
    view_97: "f32[768]" = torch.ops.aten.view.default(mul_243, [-1]);  mul_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_32 = torch.ops.aten.var_mean.correction(view_96, [0, 2], correction = 0, keepdim = True)
    getitem_64: "f32[1, 768, 1]" = var_mean_32[0]
    getitem_65: "f32[1, 768, 1]" = var_mean_32[1];  var_mean_32 = None
    add_67: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
    rsqrt_32: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    sub_32: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_96, getitem_65)
    mul_244: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
    squeeze_64: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2]);  getitem_65 = None
    squeeze_65: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2]);  rsqrt_32 = None
    unsqueeze_32: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_97, -1)
    mul_245: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_244, unsqueeze_32);  mul_244 = unsqueeze_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_98: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_245, [768, 128, 3, 3]);  mul_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_44: "f32[4, 768, 12, 12]" = torch.ops.aten.convolution.default(mul_242, view_98, primals_105, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_246: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_44, 0.5)
    mul_247: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_44, 0.7071067811865476)
    erf_29: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_247);  mul_247 = None
    add_68: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
    mul_248: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_246, add_68);  mul_246 = add_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_249: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_248, 1.7015043497085571);  mul_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_99: "f32[1, 768, 1152]" = torch.ops.aten.view.default(primals_106, [1, 768, -1]);  primals_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_250: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_107, 0.02946278254943948);  primals_107 = None
    view_100: "f32[768]" = torch.ops.aten.view.default(mul_250, [-1]);  mul_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_33 = torch.ops.aten.var_mean.correction(view_99, [0, 2], correction = 0, keepdim = True)
    getitem_66: "f32[1, 768, 1]" = var_mean_33[0]
    getitem_67: "f32[1, 768, 1]" = var_mean_33[1];  var_mean_33 = None
    add_69: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
    rsqrt_33: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
    sub_33: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_99, getitem_67)
    mul_251: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
    squeeze_66: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2]);  getitem_67 = None
    squeeze_67: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2]);  rsqrt_33 = None
    unsqueeze_33: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_100, -1)
    mul_252: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_251, unsqueeze_33);  mul_251 = unsqueeze_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_101: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_252, [768, 128, 3, 3]);  mul_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_45: "f32[4, 768, 12, 12]" = torch.ops.aten.convolution.default(mul_249, view_101, primals_108, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_253: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_45, 0.5)
    mul_254: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_45, 0.7071067811865476)
    erf_30: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_254);  mul_254 = None
    add_70: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
    mul_255: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_253, add_70);  mul_253 = add_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_256: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_255, 1.7015043497085571);  mul_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_102: "f32[1, 1536, 768]" = torch.ops.aten.view.default(primals_109, [1, 1536, -1]);  primals_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_257: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_110, 0.03608439182435161);  primals_110 = None
    view_103: "f32[1536]" = torch.ops.aten.view.default(mul_257, [-1]);  mul_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_34 = torch.ops.aten.var_mean.correction(view_102, [0, 2], correction = 0, keepdim = True)
    getitem_68: "f32[1, 1536, 1]" = var_mean_34[0]
    getitem_69: "f32[1, 1536, 1]" = var_mean_34[1];  var_mean_34 = None
    add_71: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
    rsqrt_34: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
    sub_34: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_102, getitem_69)
    mul_258: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
    squeeze_68: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2]);  getitem_69 = None
    squeeze_69: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2]);  rsqrt_34 = None
    unsqueeze_34: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_103, -1)
    mul_259: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(mul_258, unsqueeze_34);  mul_258 = unsqueeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_104: "f32[1536, 768, 1, 1]" = torch.ops.aten.view.default(mul_259, [1536, 768, 1, 1]);  mul_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_46: "f32[4, 1536, 12, 12]" = torch.ops.aten.convolution.default(mul_256, view_104, primals_111, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_6: "f32[4, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_46, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_47: "f32[4, 768, 1, 1]" = torch.ops.aten.convolution.default(mean_6, primals_208, primals_209, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_6: "f32[4, 768, 1, 1]" = torch.ops.aten.relu.default(convolution_47);  convolution_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_48: "f32[4, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_6, primals_210, primals_211, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_6: "f32[4, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_48);  convolution_48 = None
    alias_13: "f32[4, 1536, 1, 1]" = torch.ops.aten.alias.default(sigmoid_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_260: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_46, sigmoid_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_261: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_260, 2.0);  mul_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    clone_6: "f32[4, 1536, 12, 12]" = torch.ops.aten.clone.default(mul_261)
    mul_262: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_261, primals_112);  mul_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_263: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_262, 0.2);  mul_262 = None
    add_72: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_263, add_63);  mul_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_264: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_72, 0.5)
    mul_265: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_72, 0.7071067811865476)
    erf_31: "f32[4, 1536, 12, 12]" = torch.ops.aten.erf.default(mul_265);  mul_265 = None
    add_73: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
    mul_266: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_264, add_73);  mul_264 = add_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_267: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_266, 1.7015043497085571);  mul_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_268: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_267, 0.9284766908852592);  mul_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_105: "f32[1, 768, 1536]" = torch.ops.aten.view.default(primals_113, [1, 768, -1]);  primals_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_269: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_114, 0.02551551815399144);  primals_114 = None
    view_106: "f32[768]" = torch.ops.aten.view.default(mul_269, [-1]);  mul_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_35 = torch.ops.aten.var_mean.correction(view_105, [0, 2], correction = 0, keepdim = True)
    getitem_70: "f32[1, 768, 1]" = var_mean_35[0]
    getitem_71: "f32[1, 768, 1]" = var_mean_35[1];  var_mean_35 = None
    add_74: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
    rsqrt_35: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_35: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_105, getitem_71)
    mul_270: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
    squeeze_70: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2]);  getitem_71 = None
    squeeze_71: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2]);  rsqrt_35 = None
    unsqueeze_35: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_106, -1)
    mul_271: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(mul_270, unsqueeze_35);  mul_270 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_107: "f32[768, 1536, 1, 1]" = torch.ops.aten.view.default(mul_271, [768, 1536, 1, 1]);  mul_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_49: "f32[4, 768, 12, 12]" = torch.ops.aten.convolution.default(mul_268, view_107, primals_115, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_272: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_49, 0.5)
    mul_273: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_49, 0.7071067811865476)
    erf_32: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_273);  mul_273 = None
    add_75: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
    mul_274: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_272, add_75);  mul_272 = add_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_275: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_274, 1.7015043497085571);  mul_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_108: "f32[1, 768, 1152]" = torch.ops.aten.view.default(primals_116, [1, 768, -1]);  primals_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_276: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_117, 0.02946278254943948);  primals_117 = None
    view_109: "f32[768]" = torch.ops.aten.view.default(mul_276, [-1]);  mul_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_36 = torch.ops.aten.var_mean.correction(view_108, [0, 2], correction = 0, keepdim = True)
    getitem_72: "f32[1, 768, 1]" = var_mean_36[0]
    getitem_73: "f32[1, 768, 1]" = var_mean_36[1];  var_mean_36 = None
    add_76: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
    rsqrt_36: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
    sub_36: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_108, getitem_73)
    mul_277: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
    squeeze_72: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2]);  getitem_73 = None
    squeeze_73: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2]);  rsqrt_36 = None
    unsqueeze_36: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_109, -1)
    mul_278: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_277, unsqueeze_36);  mul_277 = unsqueeze_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_110: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_278, [768, 128, 3, 3]);  mul_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_50: "f32[4, 768, 12, 12]" = torch.ops.aten.convolution.default(mul_275, view_110, primals_118, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_279: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_50, 0.5)
    mul_280: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_50, 0.7071067811865476)
    erf_33: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_280);  mul_280 = None
    add_77: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
    mul_281: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_279, add_77);  mul_279 = add_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_282: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_281, 1.7015043497085571);  mul_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_111: "f32[1, 768, 1152]" = torch.ops.aten.view.default(primals_119, [1, 768, -1]);  primals_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_283: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_120, 0.02946278254943948);  primals_120 = None
    view_112: "f32[768]" = torch.ops.aten.view.default(mul_283, [-1]);  mul_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_37 = torch.ops.aten.var_mean.correction(view_111, [0, 2], correction = 0, keepdim = True)
    getitem_74: "f32[1, 768, 1]" = var_mean_37[0]
    getitem_75: "f32[1, 768, 1]" = var_mean_37[1];  var_mean_37 = None
    add_78: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
    rsqrt_37: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    sub_37: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_111, getitem_75)
    mul_284: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
    squeeze_74: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2]);  getitem_75 = None
    squeeze_75: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2]);  rsqrt_37 = None
    unsqueeze_37: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_112, -1)
    mul_285: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_284, unsqueeze_37);  mul_284 = unsqueeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_113: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_285, [768, 128, 3, 3]);  mul_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_51: "f32[4, 768, 12, 12]" = torch.ops.aten.convolution.default(mul_282, view_113, primals_121, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_286: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_51, 0.5)
    mul_287: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_51, 0.7071067811865476)
    erf_34: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_287);  mul_287 = None
    add_79: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
    mul_288: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_286, add_79);  mul_286 = add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_289: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_288, 1.7015043497085571);  mul_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_114: "f32[1, 1536, 768]" = torch.ops.aten.view.default(primals_122, [1, 1536, -1]);  primals_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_290: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_123, 0.03608439182435161);  primals_123 = None
    view_115: "f32[1536]" = torch.ops.aten.view.default(mul_290, [-1]);  mul_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_38 = torch.ops.aten.var_mean.correction(view_114, [0, 2], correction = 0, keepdim = True)
    getitem_76: "f32[1, 1536, 1]" = var_mean_38[0]
    getitem_77: "f32[1, 1536, 1]" = var_mean_38[1];  var_mean_38 = None
    add_80: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
    rsqrt_38: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
    sub_38: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_114, getitem_77)
    mul_291: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
    squeeze_76: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2]);  getitem_77 = None
    squeeze_77: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2]);  rsqrt_38 = None
    unsqueeze_38: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_115, -1)
    mul_292: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(mul_291, unsqueeze_38);  mul_291 = unsqueeze_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_116: "f32[1536, 768, 1, 1]" = torch.ops.aten.view.default(mul_292, [1536, 768, 1, 1]);  mul_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_52: "f32[4, 1536, 12, 12]" = torch.ops.aten.convolution.default(mul_289, view_116, primals_124, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_7: "f32[4, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_52, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_53: "f32[4, 768, 1, 1]" = torch.ops.aten.convolution.default(mean_7, primals_212, primals_213, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_7: "f32[4, 768, 1, 1]" = torch.ops.aten.relu.default(convolution_53);  convolution_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_54: "f32[4, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_7, primals_214, primals_215, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_7: "f32[4, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_54);  convolution_54 = None
    alias_15: "f32[4, 1536, 1, 1]" = torch.ops.aten.alias.default(sigmoid_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_293: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_52, sigmoid_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_294: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_293, 2.0);  mul_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    clone_7: "f32[4, 1536, 12, 12]" = torch.ops.aten.clone.default(mul_294)
    mul_295: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_294, primals_125);  mul_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_296: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_295, 0.2);  mul_295 = None
    add_81: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_296, add_72);  mul_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_297: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_81, 0.5)
    mul_298: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_81, 0.7071067811865476)
    erf_35: "f32[4, 1536, 12, 12]" = torch.ops.aten.erf.default(mul_298);  mul_298 = None
    add_82: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
    mul_299: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_297, add_82);  mul_297 = add_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_300: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_299, 1.7015043497085571);  mul_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_301: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_300, 0.9128709291752768);  mul_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_117: "f32[1, 768, 1536]" = torch.ops.aten.view.default(primals_126, [1, 768, -1]);  primals_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_302: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_127, 0.02551551815399144);  primals_127 = None
    view_118: "f32[768]" = torch.ops.aten.view.default(mul_302, [-1]);  mul_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_39 = torch.ops.aten.var_mean.correction(view_117, [0, 2], correction = 0, keepdim = True)
    getitem_78: "f32[1, 768, 1]" = var_mean_39[0]
    getitem_79: "f32[1, 768, 1]" = var_mean_39[1];  var_mean_39 = None
    add_83: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
    rsqrt_39: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
    sub_39: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_117, getitem_79)
    mul_303: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
    squeeze_78: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2]);  getitem_79 = None
    squeeze_79: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2]);  rsqrt_39 = None
    unsqueeze_39: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_118, -1)
    mul_304: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(mul_303, unsqueeze_39);  mul_303 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_119: "f32[768, 1536, 1, 1]" = torch.ops.aten.view.default(mul_304, [768, 1536, 1, 1]);  mul_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_55: "f32[4, 768, 12, 12]" = torch.ops.aten.convolution.default(mul_301, view_119, primals_128, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_305: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_55, 0.5)
    mul_306: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_55, 0.7071067811865476)
    erf_36: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_306);  mul_306 = None
    add_84: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
    mul_307: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_305, add_84);  mul_305 = add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_308: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_307, 1.7015043497085571);  mul_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_120: "f32[1, 768, 1152]" = torch.ops.aten.view.default(primals_129, [1, 768, -1]);  primals_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_309: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_130, 0.02946278254943948);  primals_130 = None
    view_121: "f32[768]" = torch.ops.aten.view.default(mul_309, [-1]);  mul_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_40 = torch.ops.aten.var_mean.correction(view_120, [0, 2], correction = 0, keepdim = True)
    getitem_80: "f32[1, 768, 1]" = var_mean_40[0]
    getitem_81: "f32[1, 768, 1]" = var_mean_40[1];  var_mean_40 = None
    add_85: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
    rsqrt_40: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
    sub_40: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_120, getitem_81)
    mul_310: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
    squeeze_80: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2]);  getitem_81 = None
    squeeze_81: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2]);  rsqrt_40 = None
    unsqueeze_40: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_121, -1)
    mul_311: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_310, unsqueeze_40);  mul_310 = unsqueeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_122: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_311, [768, 128, 3, 3]);  mul_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_56: "f32[4, 768, 12, 12]" = torch.ops.aten.convolution.default(mul_308, view_122, primals_131, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_312: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_56, 0.5)
    mul_313: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_56, 0.7071067811865476)
    erf_37: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_313);  mul_313 = None
    add_86: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
    mul_314: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_312, add_86);  mul_312 = add_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_315: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_314, 1.7015043497085571);  mul_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_123: "f32[1, 768, 1152]" = torch.ops.aten.view.default(primals_132, [1, 768, -1]);  primals_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_316: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_133, 0.02946278254943948);  primals_133 = None
    view_124: "f32[768]" = torch.ops.aten.view.default(mul_316, [-1]);  mul_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_41 = torch.ops.aten.var_mean.correction(view_123, [0, 2], correction = 0, keepdim = True)
    getitem_82: "f32[1, 768, 1]" = var_mean_41[0]
    getitem_83: "f32[1, 768, 1]" = var_mean_41[1];  var_mean_41 = None
    add_87: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
    rsqrt_41: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
    sub_41: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_123, getitem_83)
    mul_317: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
    squeeze_82: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2]);  getitem_83 = None
    squeeze_83: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2]);  rsqrt_41 = None
    unsqueeze_41: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_124, -1)
    mul_318: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_317, unsqueeze_41);  mul_317 = unsqueeze_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_125: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_318, [768, 128, 3, 3]);  mul_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_57: "f32[4, 768, 12, 12]" = torch.ops.aten.convolution.default(mul_315, view_125, primals_134, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_319: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_57, 0.5)
    mul_320: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_57, 0.7071067811865476)
    erf_38: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_320);  mul_320 = None
    add_88: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
    mul_321: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_319, add_88);  mul_319 = add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_322: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_321, 1.7015043497085571);  mul_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_126: "f32[1, 1536, 768]" = torch.ops.aten.view.default(primals_135, [1, 1536, -1]);  primals_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_323: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_136, 0.03608439182435161);  primals_136 = None
    view_127: "f32[1536]" = torch.ops.aten.view.default(mul_323, [-1]);  mul_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_42 = torch.ops.aten.var_mean.correction(view_126, [0, 2], correction = 0, keepdim = True)
    getitem_84: "f32[1, 1536, 1]" = var_mean_42[0]
    getitem_85: "f32[1, 1536, 1]" = var_mean_42[1];  var_mean_42 = None
    add_89: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05);  getitem_84 = None
    rsqrt_42: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
    sub_42: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_126, getitem_85)
    mul_324: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
    squeeze_84: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2]);  getitem_85 = None
    squeeze_85: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2]);  rsqrt_42 = None
    unsqueeze_42: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_127, -1)
    mul_325: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(mul_324, unsqueeze_42);  mul_324 = unsqueeze_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_128: "f32[1536, 768, 1, 1]" = torch.ops.aten.view.default(mul_325, [1536, 768, 1, 1]);  mul_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_58: "f32[4, 1536, 12, 12]" = torch.ops.aten.convolution.default(mul_322, view_128, primals_137, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_8: "f32[4, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_58, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_59: "f32[4, 768, 1, 1]" = torch.ops.aten.convolution.default(mean_8, primals_216, primals_217, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_8: "f32[4, 768, 1, 1]" = torch.ops.aten.relu.default(convolution_59);  convolution_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_60: "f32[4, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_8, primals_218, primals_219, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_8: "f32[4, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_60);  convolution_60 = None
    alias_17: "f32[4, 1536, 1, 1]" = torch.ops.aten.alias.default(sigmoid_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_326: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_58, sigmoid_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_327: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_326, 2.0);  mul_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    clone_8: "f32[4, 1536, 12, 12]" = torch.ops.aten.clone.default(mul_327)
    mul_328: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_327, primals_138);  mul_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_329: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_328, 0.2);  mul_328 = None
    add_90: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_329, add_81);  mul_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_330: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_90, 0.5)
    mul_331: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_90, 0.7071067811865476)
    erf_39: "f32[4, 1536, 12, 12]" = torch.ops.aten.erf.default(mul_331);  mul_331 = None
    add_91: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
    mul_332: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_330, add_91);  mul_330 = add_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_333: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_332, 1.7015043497085571);  mul_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_334: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_333, 0.8980265101338745);  mul_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    avg_pool2d_2: "f32[4, 1536, 6, 6]" = torch.ops.aten.avg_pool2d.default(mul_334, [2, 2], [2, 2], [0, 0], True, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_129: "f32[1, 1536, 1536]" = torch.ops.aten.view.default(primals_139, [1, 1536, -1]);  primals_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_335: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_140, 0.02551551815399144);  primals_140 = None
    view_130: "f32[1536]" = torch.ops.aten.view.default(mul_335, [-1]);  mul_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_43 = torch.ops.aten.var_mean.correction(view_129, [0, 2], correction = 0, keepdim = True)
    getitem_86: "f32[1, 1536, 1]" = var_mean_43[0]
    getitem_87: "f32[1, 1536, 1]" = var_mean_43[1];  var_mean_43 = None
    add_92: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
    rsqrt_43: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
    sub_43: "f32[1, 1536, 1536]" = torch.ops.aten.sub.Tensor(view_129, getitem_87)
    mul_336: "f32[1, 1536, 1536]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
    squeeze_86: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2]);  getitem_87 = None
    squeeze_87: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2]);  rsqrt_43 = None
    unsqueeze_43: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_130, -1)
    mul_337: "f32[1, 1536, 1536]" = torch.ops.aten.mul.Tensor(mul_336, unsqueeze_43);  mul_336 = unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_131: "f32[1536, 1536, 1, 1]" = torch.ops.aten.view.default(mul_337, [1536, 1536, 1, 1]);  mul_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_61: "f32[4, 1536, 6, 6]" = torch.ops.aten.convolution.default(avg_pool2d_2, view_131, primals_141, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_132: "f32[1, 768, 1536]" = torch.ops.aten.view.default(primals_142, [1, 768, -1]);  primals_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_338: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_143, 0.02551551815399144);  primals_143 = None
    view_133: "f32[768]" = torch.ops.aten.view.default(mul_338, [-1]);  mul_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_44 = torch.ops.aten.var_mean.correction(view_132, [0, 2], correction = 0, keepdim = True)
    getitem_88: "f32[1, 768, 1]" = var_mean_44[0]
    getitem_89: "f32[1, 768, 1]" = var_mean_44[1];  var_mean_44 = None
    add_93: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
    rsqrt_44: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
    sub_44: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_132, getitem_89)
    mul_339: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
    squeeze_88: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2]);  getitem_89 = None
    squeeze_89: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2]);  rsqrt_44 = None
    unsqueeze_44: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_133, -1)
    mul_340: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(mul_339, unsqueeze_44);  mul_339 = unsqueeze_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_134: "f32[768, 1536, 1, 1]" = torch.ops.aten.view.default(mul_340, [768, 1536, 1, 1]);  mul_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_62: "f32[4, 768, 12, 12]" = torch.ops.aten.convolution.default(mul_334, view_134, primals_144, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_341: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_62, 0.5)
    mul_342: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_62, 0.7071067811865476)
    erf_40: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_342);  mul_342 = None
    add_94: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_40, 1);  erf_40 = None
    mul_343: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_341, add_94);  mul_341 = add_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_344: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_343, 1.7015043497085571);  mul_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_4: "f32[4, 768, 13, 13]" = torch.ops.aten.constant_pad_nd.default(mul_344, [0, 1, 0, 1], 0.0);  mul_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_135: "f32[1, 768, 1152]" = torch.ops.aten.view.default(primals_145, [1, 768, -1]);  primals_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_345: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_146, 0.02946278254943948);  primals_146 = None
    view_136: "f32[768]" = torch.ops.aten.view.default(mul_345, [-1]);  mul_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_45 = torch.ops.aten.var_mean.correction(view_135, [0, 2], correction = 0, keepdim = True)
    getitem_90: "f32[1, 768, 1]" = var_mean_45[0]
    getitem_91: "f32[1, 768, 1]" = var_mean_45[1];  var_mean_45 = None
    add_95: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
    rsqrt_45: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
    sub_45: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_135, getitem_91)
    mul_346: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
    squeeze_90: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2]);  getitem_91 = None
    squeeze_91: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2]);  rsqrt_45 = None
    unsqueeze_45: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_136, -1)
    mul_347: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_346, unsqueeze_45);  mul_346 = unsqueeze_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_137: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_347, [768, 128, 3, 3]);  mul_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_63: "f32[4, 768, 6, 6]" = torch.ops.aten.convolution.default(constant_pad_nd_4, view_137, primals_147, [2, 2], [0, 0], [1, 1], False, [0, 0], 6);  primals_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_348: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_63, 0.5)
    mul_349: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_63, 0.7071067811865476)
    erf_41: "f32[4, 768, 6, 6]" = torch.ops.aten.erf.default(mul_349);  mul_349 = None
    add_96: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(erf_41, 1);  erf_41 = None
    mul_350: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_348, add_96);  mul_348 = add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_351: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_350, 1.7015043497085571);  mul_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_138: "f32[1, 768, 1152]" = torch.ops.aten.view.default(primals_148, [1, 768, -1]);  primals_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_352: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_149, 0.02946278254943948);  primals_149 = None
    view_139: "f32[768]" = torch.ops.aten.view.default(mul_352, [-1]);  mul_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_46 = torch.ops.aten.var_mean.correction(view_138, [0, 2], correction = 0, keepdim = True)
    getitem_92: "f32[1, 768, 1]" = var_mean_46[0]
    getitem_93: "f32[1, 768, 1]" = var_mean_46[1];  var_mean_46 = None
    add_97: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
    rsqrt_46: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
    sub_46: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_138, getitem_93)
    mul_353: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
    squeeze_92: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2]);  getitem_93 = None
    squeeze_93: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2]);  rsqrt_46 = None
    unsqueeze_46: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_139, -1)
    mul_354: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_353, unsqueeze_46);  mul_353 = unsqueeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_140: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_354, [768, 128, 3, 3]);  mul_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_64: "f32[4, 768, 6, 6]" = torch.ops.aten.convolution.default(mul_351, view_140, primals_150, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_355: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_64, 0.5)
    mul_356: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_64, 0.7071067811865476)
    erf_42: "f32[4, 768, 6, 6]" = torch.ops.aten.erf.default(mul_356);  mul_356 = None
    add_98: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(erf_42, 1);  erf_42 = None
    mul_357: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_355, add_98);  mul_355 = add_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_358: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_357, 1.7015043497085571);  mul_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_141: "f32[1, 1536, 768]" = torch.ops.aten.view.default(primals_151, [1, 1536, -1]);  primals_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_359: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_152, 0.03608439182435161);  primals_152 = None
    view_142: "f32[1536]" = torch.ops.aten.view.default(mul_359, [-1]);  mul_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_47 = torch.ops.aten.var_mean.correction(view_141, [0, 2], correction = 0, keepdim = True)
    getitem_94: "f32[1, 1536, 1]" = var_mean_47[0]
    getitem_95: "f32[1, 1536, 1]" = var_mean_47[1];  var_mean_47 = None
    add_99: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
    rsqrt_47: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
    sub_47: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_141, getitem_95)
    mul_360: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
    squeeze_94: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2]);  getitem_95 = None
    squeeze_95: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2]);  rsqrt_47 = None
    unsqueeze_47: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_142, -1)
    mul_361: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(mul_360, unsqueeze_47);  mul_360 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_143: "f32[1536, 768, 1, 1]" = torch.ops.aten.view.default(mul_361, [1536, 768, 1, 1]);  mul_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_65: "f32[4, 1536, 6, 6]" = torch.ops.aten.convolution.default(mul_358, view_143, primals_153, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_9: "f32[4, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_65, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_66: "f32[4, 768, 1, 1]" = torch.ops.aten.convolution.default(mean_9, primals_220, primals_221, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_9: "f32[4, 768, 1, 1]" = torch.ops.aten.relu.default(convolution_66);  convolution_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_67: "f32[4, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_9, primals_222, primals_223, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_9: "f32[4, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_67);  convolution_67 = None
    alias_19: "f32[4, 1536, 1, 1]" = torch.ops.aten.alias.default(sigmoid_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_362: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_65, sigmoid_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_363: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_362, 2.0);  mul_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    clone_9: "f32[4, 1536, 6, 6]" = torch.ops.aten.clone.default(mul_363)
    mul_364: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_363, primals_154);  mul_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_365: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_364, 0.2);  mul_364 = None
    add_100: "f32[4, 1536, 6, 6]" = torch.ops.aten.add.Tensor(mul_365, convolution_61);  mul_365 = convolution_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_366: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(add_100, 0.5)
    mul_367: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(add_100, 0.7071067811865476)
    erf_43: "f32[4, 1536, 6, 6]" = torch.ops.aten.erf.default(mul_367);  mul_367 = None
    add_101: "f32[4, 1536, 6, 6]" = torch.ops.aten.add.Tensor(erf_43, 1);  erf_43 = None
    mul_368: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_366, add_101);  mul_366 = add_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_369: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_368, 1.7015043497085571);  mul_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_370: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_369, 0.9805806756909201);  mul_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_144: "f32[1, 768, 1536]" = torch.ops.aten.view.default(primals_155, [1, 768, -1]);  primals_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_371: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_156, 0.02551551815399144);  primals_156 = None
    view_145: "f32[768]" = torch.ops.aten.view.default(mul_371, [-1]);  mul_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_48 = torch.ops.aten.var_mean.correction(view_144, [0, 2], correction = 0, keepdim = True)
    getitem_96: "f32[1, 768, 1]" = var_mean_48[0]
    getitem_97: "f32[1, 768, 1]" = var_mean_48[1];  var_mean_48 = None
    add_102: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
    rsqrt_48: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
    sub_48: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_144, getitem_97)
    mul_372: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
    squeeze_96: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2]);  getitem_97 = None
    squeeze_97: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2]);  rsqrt_48 = None
    unsqueeze_48: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_145, -1)
    mul_373: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(mul_372, unsqueeze_48);  mul_372 = unsqueeze_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_146: "f32[768, 1536, 1, 1]" = torch.ops.aten.view.default(mul_373, [768, 1536, 1, 1]);  mul_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_68: "f32[4, 768, 6, 6]" = torch.ops.aten.convolution.default(mul_370, view_146, primals_157, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_374: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_68, 0.5)
    mul_375: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_68, 0.7071067811865476)
    erf_44: "f32[4, 768, 6, 6]" = torch.ops.aten.erf.default(mul_375);  mul_375 = None
    add_103: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(erf_44, 1);  erf_44 = None
    mul_376: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_374, add_103);  mul_374 = add_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_377: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_376, 1.7015043497085571);  mul_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_147: "f32[1, 768, 1152]" = torch.ops.aten.view.default(primals_158, [1, 768, -1]);  primals_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_378: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_159, 0.02946278254943948);  primals_159 = None
    view_148: "f32[768]" = torch.ops.aten.view.default(mul_378, [-1]);  mul_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_49 = torch.ops.aten.var_mean.correction(view_147, [0, 2], correction = 0, keepdim = True)
    getitem_98: "f32[1, 768, 1]" = var_mean_49[0]
    getitem_99: "f32[1, 768, 1]" = var_mean_49[1];  var_mean_49 = None
    add_104: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-05);  getitem_98 = None
    rsqrt_49: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
    sub_49: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_147, getitem_99)
    mul_379: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
    squeeze_98: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2]);  getitem_99 = None
    squeeze_99: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2]);  rsqrt_49 = None
    unsqueeze_49: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_148, -1)
    mul_380: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_379, unsqueeze_49);  mul_379 = unsqueeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_149: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_380, [768, 128, 3, 3]);  mul_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_69: "f32[4, 768, 6, 6]" = torch.ops.aten.convolution.default(mul_377, view_149, primals_160, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_381: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_69, 0.5)
    mul_382: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_69, 0.7071067811865476)
    erf_45: "f32[4, 768, 6, 6]" = torch.ops.aten.erf.default(mul_382);  mul_382 = None
    add_105: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(erf_45, 1);  erf_45 = None
    mul_383: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_381, add_105);  mul_381 = add_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_384: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_383, 1.7015043497085571);  mul_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_150: "f32[1, 768, 1152]" = torch.ops.aten.view.default(primals_161, [1, 768, -1]);  primals_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_385: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_162, 0.02946278254943948);  primals_162 = None
    view_151: "f32[768]" = torch.ops.aten.view.default(mul_385, [-1]);  mul_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_50 = torch.ops.aten.var_mean.correction(view_150, [0, 2], correction = 0, keepdim = True)
    getitem_100: "f32[1, 768, 1]" = var_mean_50[0]
    getitem_101: "f32[1, 768, 1]" = var_mean_50[1];  var_mean_50 = None
    add_106: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05);  getitem_100 = None
    rsqrt_50: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
    sub_50: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_150, getitem_101)
    mul_386: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
    squeeze_100: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2]);  getitem_101 = None
    squeeze_101: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2]);  rsqrt_50 = None
    unsqueeze_50: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_151, -1)
    mul_387: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_386, unsqueeze_50);  mul_386 = unsqueeze_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_152: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_387, [768, 128, 3, 3]);  mul_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_70: "f32[4, 768, 6, 6]" = torch.ops.aten.convolution.default(mul_384, view_152, primals_163, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_388: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_70, 0.5)
    mul_389: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_70, 0.7071067811865476)
    erf_46: "f32[4, 768, 6, 6]" = torch.ops.aten.erf.default(mul_389);  mul_389 = None
    add_107: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(erf_46, 1);  erf_46 = None
    mul_390: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_388, add_107);  mul_388 = add_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_391: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_390, 1.7015043497085571);  mul_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_153: "f32[1, 1536, 768]" = torch.ops.aten.view.default(primals_164, [1, 1536, -1]);  primals_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_392: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_165, 0.03608439182435161);  primals_165 = None
    view_154: "f32[1536]" = torch.ops.aten.view.default(mul_392, [-1]);  mul_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_51 = torch.ops.aten.var_mean.correction(view_153, [0, 2], correction = 0, keepdim = True)
    getitem_102: "f32[1, 1536, 1]" = var_mean_51[0]
    getitem_103: "f32[1, 1536, 1]" = var_mean_51[1];  var_mean_51 = None
    add_108: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05);  getitem_102 = None
    rsqrt_51: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    sub_51: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_153, getitem_103)
    mul_393: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
    squeeze_102: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2]);  getitem_103 = None
    squeeze_103: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2]);  rsqrt_51 = None
    unsqueeze_51: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_154, -1)
    mul_394: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(mul_393, unsqueeze_51);  mul_393 = unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_155: "f32[1536, 768, 1, 1]" = torch.ops.aten.view.default(mul_394, [1536, 768, 1, 1]);  mul_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_71: "f32[4, 1536, 6, 6]" = torch.ops.aten.convolution.default(mul_391, view_155, primals_166, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_10: "f32[4, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_71, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_72: "f32[4, 768, 1, 1]" = torch.ops.aten.convolution.default(mean_10, primals_224, primals_225, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_10: "f32[4, 768, 1, 1]" = torch.ops.aten.relu.default(convolution_72);  convolution_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_73: "f32[4, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_10, primals_226, primals_227, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_10: "f32[4, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_73);  convolution_73 = None
    alias_21: "f32[4, 1536, 1, 1]" = torch.ops.aten.alias.default(sigmoid_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_395: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_71, sigmoid_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_396: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_395, 2.0);  mul_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    clone_10: "f32[4, 1536, 6, 6]" = torch.ops.aten.clone.default(mul_396)
    mul_397: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_396, primals_167);  mul_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_398: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_397, 0.2);  mul_397 = None
    add_109: "f32[4, 1536, 6, 6]" = torch.ops.aten.add.Tensor(mul_398, add_100);  mul_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_399: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(add_109, 0.5)
    mul_400: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(add_109, 0.7071067811865476)
    erf_47: "f32[4, 1536, 6, 6]" = torch.ops.aten.erf.default(mul_400);  mul_400 = None
    add_110: "f32[4, 1536, 6, 6]" = torch.ops.aten.add.Tensor(erf_47, 1);  erf_47 = None
    mul_401: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_399, add_110);  mul_399 = add_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_402: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_401, 1.7015043497085571);  mul_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_403: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_402, 0.9622504486493761);  mul_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_156: "f32[1, 768, 1536]" = torch.ops.aten.view.default(primals_168, [1, 768, -1]);  primals_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_404: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_169, 0.02551551815399144);  primals_169 = None
    view_157: "f32[768]" = torch.ops.aten.view.default(mul_404, [-1]);  mul_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_52 = torch.ops.aten.var_mean.correction(view_156, [0, 2], correction = 0, keepdim = True)
    getitem_104: "f32[1, 768, 1]" = var_mean_52[0]
    getitem_105: "f32[1, 768, 1]" = var_mean_52[1];  var_mean_52 = None
    add_111: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
    rsqrt_52: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
    sub_52: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_156, getitem_105)
    mul_405: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
    squeeze_104: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_105, [0, 2]);  getitem_105 = None
    squeeze_105: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2]);  rsqrt_52 = None
    unsqueeze_52: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_157, -1)
    mul_406: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(mul_405, unsqueeze_52);  mul_405 = unsqueeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_158: "f32[768, 1536, 1, 1]" = torch.ops.aten.view.default(mul_406, [768, 1536, 1, 1]);  mul_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_74: "f32[4, 768, 6, 6]" = torch.ops.aten.convolution.default(mul_403, view_158, primals_170, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_407: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_74, 0.5)
    mul_408: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_74, 0.7071067811865476)
    erf_48: "f32[4, 768, 6, 6]" = torch.ops.aten.erf.default(mul_408);  mul_408 = None
    add_112: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(erf_48, 1);  erf_48 = None
    mul_409: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_407, add_112);  mul_407 = add_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_410: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_409, 1.7015043497085571);  mul_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_159: "f32[1, 768, 1152]" = torch.ops.aten.view.default(primals_171, [1, 768, -1]);  primals_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_411: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_172, 0.02946278254943948);  primals_172 = None
    view_160: "f32[768]" = torch.ops.aten.view.default(mul_411, [-1]);  mul_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_53 = torch.ops.aten.var_mean.correction(view_159, [0, 2], correction = 0, keepdim = True)
    getitem_106: "f32[1, 768, 1]" = var_mean_53[0]
    getitem_107: "f32[1, 768, 1]" = var_mean_53[1];  var_mean_53 = None
    add_113: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05);  getitem_106 = None
    rsqrt_53: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
    sub_53: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_159, getitem_107)
    mul_412: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = None
    squeeze_106: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2]);  getitem_107 = None
    squeeze_107: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_53, [0, 2]);  rsqrt_53 = None
    unsqueeze_53: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_160, -1)
    mul_413: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_412, unsqueeze_53);  mul_412 = unsqueeze_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_161: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_413, [768, 128, 3, 3]);  mul_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_75: "f32[4, 768, 6, 6]" = torch.ops.aten.convolution.default(mul_410, view_161, primals_173, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_414: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_75, 0.5)
    mul_415: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_75, 0.7071067811865476)
    erf_49: "f32[4, 768, 6, 6]" = torch.ops.aten.erf.default(mul_415);  mul_415 = None
    add_114: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(erf_49, 1);  erf_49 = None
    mul_416: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_414, add_114);  mul_414 = add_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_417: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_416, 1.7015043497085571);  mul_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_162: "f32[1, 768, 1152]" = torch.ops.aten.view.default(primals_174, [1, 768, -1]);  primals_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_418: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_175, 0.02946278254943948);  primals_175 = None
    view_163: "f32[768]" = torch.ops.aten.view.default(mul_418, [-1]);  mul_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_54 = torch.ops.aten.var_mean.correction(view_162, [0, 2], correction = 0, keepdim = True)
    getitem_108: "f32[1, 768, 1]" = var_mean_54[0]
    getitem_109: "f32[1, 768, 1]" = var_mean_54[1];  var_mean_54 = None
    add_115: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
    rsqrt_54: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
    sub_54: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_162, getitem_109)
    mul_419: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = None
    squeeze_108: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_109, [0, 2]);  getitem_109 = None
    squeeze_109: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2]);  rsqrt_54 = None
    unsqueeze_54: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_163, -1)
    mul_420: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_419, unsqueeze_54);  mul_419 = unsqueeze_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_164: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_420, [768, 128, 3, 3]);  mul_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_76: "f32[4, 768, 6, 6]" = torch.ops.aten.convolution.default(mul_417, view_164, primals_176, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  primals_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_421: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_76, 0.5)
    mul_422: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_76, 0.7071067811865476)
    erf_50: "f32[4, 768, 6, 6]" = torch.ops.aten.erf.default(mul_422);  mul_422 = None
    add_116: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(erf_50, 1);  erf_50 = None
    mul_423: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_421, add_116);  mul_421 = add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_424: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_423, 1.7015043497085571);  mul_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_165: "f32[1, 1536, 768]" = torch.ops.aten.view.default(primals_177, [1, 1536, -1]);  primals_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_425: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_178, 0.03608439182435161);  primals_178 = None
    view_166: "f32[1536]" = torch.ops.aten.view.default(mul_425, [-1]);  mul_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_55 = torch.ops.aten.var_mean.correction(view_165, [0, 2], correction = 0, keepdim = True)
    getitem_110: "f32[1, 1536, 1]" = var_mean_55[0]
    getitem_111: "f32[1, 1536, 1]" = var_mean_55[1];  var_mean_55 = None
    add_117: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
    rsqrt_55: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
    sub_55: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_165, getitem_111)
    mul_426: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = None
    squeeze_110: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_111, [0, 2]);  getitem_111 = None
    squeeze_111: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2]);  rsqrt_55 = None
    unsqueeze_55: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_166, -1)
    mul_427: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(mul_426, unsqueeze_55);  mul_426 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_167: "f32[1536, 768, 1, 1]" = torch.ops.aten.view.default(mul_427, [1536, 768, 1, 1]);  mul_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_77: "f32[4, 1536, 6, 6]" = torch.ops.aten.convolution.default(mul_424, view_167, primals_179, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_11: "f32[4, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_77, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_78: "f32[4, 768, 1, 1]" = torch.ops.aten.convolution.default(mean_11, primals_228, primals_229, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_11: "f32[4, 768, 1, 1]" = torch.ops.aten.relu.default(convolution_78);  convolution_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_79: "f32[4, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_11, primals_230, primals_231, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_11: "f32[4, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_79);  convolution_79 = None
    alias_23: "f32[4, 1536, 1, 1]" = torch.ops.aten.alias.default(sigmoid_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_428: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_77, sigmoid_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_429: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_428, 2.0);  mul_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    clone_11: "f32[4, 1536, 6, 6]" = torch.ops.aten.clone.default(mul_429)
    mul_430: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_429, primals_180);  mul_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_431: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_430, 0.2);  mul_430 = None
    add_118: "f32[4, 1536, 6, 6]" = torch.ops.aten.add.Tensor(mul_431, add_109);  mul_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_168: "f32[1, 3072, 1536]" = torch.ops.aten.view.default(primals_181, [1, 3072, -1]);  primals_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_432: "f32[3072, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_182, 0.02551551815399144);  primals_182 = None
    view_169: "f32[3072]" = torch.ops.aten.view.default(mul_432, [-1]);  mul_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    var_mean_56 = torch.ops.aten.var_mean.correction(view_168, [0, 2], correction = 0, keepdim = True)
    getitem_112: "f32[1, 3072, 1]" = var_mean_56[0]
    getitem_113: "f32[1, 3072, 1]" = var_mean_56[1];  var_mean_56 = None
    add_119: "f32[1, 3072, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
    rsqrt_56: "f32[1, 3072, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
    sub_56: "f32[1, 3072, 1536]" = torch.ops.aten.sub.Tensor(view_168, getitem_113)
    mul_433: "f32[1, 3072, 1536]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = None
    squeeze_112: "f32[3072]" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2]);  getitem_113 = None
    squeeze_113: "f32[3072]" = torch.ops.aten.squeeze.dims(rsqrt_56, [0, 2]);  rsqrt_56 = None
    unsqueeze_56: "f32[3072, 1]" = torch.ops.aten.unsqueeze.default(view_169, -1)
    mul_434: "f32[1, 3072, 1536]" = torch.ops.aten.mul.Tensor(mul_433, unsqueeze_56);  mul_433 = unsqueeze_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_170: "f32[3072, 1536, 1, 1]" = torch.ops.aten.view.default(mul_434, [3072, 1536, 1, 1]);  mul_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    convolution_80: "f32[4, 3072, 6, 6]" = torch.ops.aten.convolution.default(add_118, view_170, primals_183, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_435: "f32[4, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_80, 0.5)
    mul_436: "f32[4, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_80, 0.7071067811865476)
    erf_51: "f32[4, 3072, 6, 6]" = torch.ops.aten.erf.default(mul_436);  mul_436 = None
    add_120: "f32[4, 3072, 6, 6]" = torch.ops.aten.add.Tensor(erf_51, 1);  erf_51 = None
    mul_437: "f32[4, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(mul_435, add_120);  mul_435 = add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_438: "f32[4, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(mul_437, 1.7015043497085571);  mul_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean_12: "f32[4, 3072, 1, 1]" = torch.ops.aten.mean.dim(mul_438, [-1, -2], True);  mul_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_171: "f32[4, 3072]" = torch.ops.aten.view.default(mean_12, [4, 3072]);  mean_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:131, code: x = self.drop(x)
    clone_12: "f32[4, 3072]" = torch.ops.aten.clone.default(view_171);  view_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    permute: "f32[3072, 1000]" = torch.ops.aten.permute.default(primals_232, [1, 0]);  primals_232 = None
    addmm: "f32[4, 1000]" = torch.ops.aten.addmm.default(primals_233, clone_12, permute);  primals_233 = None
    permute_1: "f32[1000, 3072]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    mm: "f32[4, 3072]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 4]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 3072]" = torch.ops.aten.mm.default(permute_2, clone_12);  permute_2 = clone_12 = None
    permute_3: "f32[3072, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_172: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 3072]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_173: "f32[4, 3072, 1, 1]" = torch.ops.aten.view.default(mm, [4, 3072, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[4, 3072, 6, 6]" = torch.ops.aten.expand.default(view_173, [4, 3072, 6, 6]);  view_173 = None
    div: "f32[4, 3072, 6, 6]" = torch.ops.aten.div.Scalar(expand, 36);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_439: "f32[4, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(div, 1.7015043497085571);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_440: "f32[4, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_80, 0.7071067811865476)
    erf_52: "f32[4, 3072, 6, 6]" = torch.ops.aten.erf.default(mul_440);  mul_440 = None
    add_121: "f32[4, 3072, 6, 6]" = torch.ops.aten.add.Tensor(erf_52, 1);  erf_52 = None
    mul_441: "f32[4, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(add_121, 0.5);  add_121 = None
    mul_442: "f32[4, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_80, convolution_80)
    mul_443: "f32[4, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(mul_442, -0.5);  mul_442 = None
    exp: "f32[4, 3072, 6, 6]" = torch.ops.aten.exp.default(mul_443);  mul_443 = None
    mul_444: "f32[4, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
    mul_445: "f32[4, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_80, mul_444);  convolution_80 = mul_444 = None
    add_122: "f32[4, 3072, 6, 6]" = torch.ops.aten.add.Tensor(mul_441, mul_445);  mul_441 = mul_445 = None
    mul_446: "f32[4, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(mul_439, add_122);  mul_439 = add_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_2: "f32[3072]" = torch.ops.aten.sum.dim_IntList(mul_446, [0, 2, 3])
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_446, add_118, view_170, [3072], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_446 = add_118 = view_170 = None
    getitem_114: "f32[4, 1536, 6, 6]" = convolution_backward[0]
    getitem_115: "f32[3072, 1536, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_174: "f32[1, 3072, 1536]" = torch.ops.aten.view.default(getitem_115, [1, 3072, 1536]);  getitem_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_57: "f32[1, 3072]" = torch.ops.aten.unsqueeze.default(squeeze_112, 0);  squeeze_112 = None
    unsqueeze_58: "f32[1, 3072, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_57, 2);  unsqueeze_57 = None
    sum_3: "f32[3072]" = torch.ops.aten.sum.dim_IntList(view_174, [0, 2])
    sub_57: "f32[1, 3072, 1536]" = torch.ops.aten.sub.Tensor(view_168, unsqueeze_58)
    mul_447: "f32[1, 3072, 1536]" = torch.ops.aten.mul.Tensor(view_174, sub_57);  sub_57 = None
    sum_4: "f32[3072]" = torch.ops.aten.sum.dim_IntList(mul_447, [0, 2]);  mul_447 = None
    mul_448: "f32[3072]" = torch.ops.aten.mul.Tensor(sum_3, 0.0006510416666666666);  sum_3 = None
    unsqueeze_59: "f32[1, 3072]" = torch.ops.aten.unsqueeze.default(mul_448, 0);  mul_448 = None
    unsqueeze_60: "f32[1, 3072, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_59, 2);  unsqueeze_59 = None
    mul_449: "f32[3072]" = torch.ops.aten.mul.Tensor(sum_4, 0.0006510416666666666)
    mul_450: "f32[3072]" = torch.ops.aten.mul.Tensor(squeeze_113, squeeze_113)
    mul_451: "f32[3072]" = torch.ops.aten.mul.Tensor(mul_449, mul_450);  mul_449 = mul_450 = None
    unsqueeze_61: "f32[1, 3072]" = torch.ops.aten.unsqueeze.default(mul_451, 0);  mul_451 = None
    unsqueeze_62: "f32[1, 3072, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_61, 2);  unsqueeze_61 = None
    mul_452: "f32[3072]" = torch.ops.aten.mul.Tensor(squeeze_113, view_169);  view_169 = None
    unsqueeze_63: "f32[1, 3072]" = torch.ops.aten.unsqueeze.default(mul_452, 0);  mul_452 = None
    unsqueeze_64: "f32[1, 3072, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_63, 2);  unsqueeze_63 = None
    sub_58: "f32[1, 3072, 1536]" = torch.ops.aten.sub.Tensor(view_168, unsqueeze_58);  view_168 = unsqueeze_58 = None
    mul_453: "f32[1, 3072, 1536]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_62);  sub_58 = unsqueeze_62 = None
    sub_59: "f32[1, 3072, 1536]" = torch.ops.aten.sub.Tensor(view_174, mul_453);  view_174 = mul_453 = None
    sub_60: "f32[1, 3072, 1536]" = torch.ops.aten.sub.Tensor(sub_59, unsqueeze_60);  sub_59 = unsqueeze_60 = None
    mul_454: "f32[1, 3072, 1536]" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_64);  sub_60 = unsqueeze_64 = None
    mul_455: "f32[3072]" = torch.ops.aten.mul.Tensor(sum_4, squeeze_113);  sum_4 = squeeze_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_175: "f32[3072, 1, 1, 1]" = torch.ops.aten.view.default(mul_455, [3072, 1, 1, 1]);  mul_455 = None
    mul_456: "f32[3072, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_175, 0.02551551815399144);  view_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_176: "f32[3072, 1536, 1, 1]" = torch.ops.aten.view.default(mul_454, [3072, 1536, 1, 1]);  mul_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_457: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_114, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    mul_458: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_457, clone_11);  clone_11 = None
    mul_459: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_457, primals_180);  mul_457 = primals_180 = None
    sum_5: "f32[]" = torch.ops.aten.sum.default(mul_458);  mul_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_460: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_459, 2.0);  mul_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_461: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_460, convolution_77);  convolution_77 = None
    mul_462: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_460, sigmoid_11);  mul_460 = sigmoid_11 = None
    sum_6: "f32[4, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_461, [2, 3], True);  mul_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_24: "f32[4, 1536, 1, 1]" = torch.ops.aten.alias.default(alias_23);  alias_23 = None
    sub_61: "f32[4, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_24)
    mul_463: "f32[4, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(alias_24, sub_61);  alias_24 = sub_61 = None
    mul_464: "f32[4, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_6, mul_463);  sum_6 = mul_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_7: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_464, [0, 2, 3])
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_464, relu_11, primals_230, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_464 = primals_230 = None
    getitem_117: "f32[4, 768, 1, 1]" = convolution_backward_1[0]
    getitem_118: "f32[1536, 768, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_26: "f32[4, 768, 1, 1]" = torch.ops.aten.alias.default(relu_11);  relu_11 = None
    alias_27: "f32[4, 768, 1, 1]" = torch.ops.aten.alias.default(alias_26);  alias_26 = None
    le: "b8[4, 768, 1, 1]" = torch.ops.aten.le.Scalar(alias_27, 0);  alias_27 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where: "f32[4, 768, 1, 1]" = torch.ops.aten.where.self(le, scalar_tensor, getitem_117);  le = scalar_tensor = getitem_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_8: "f32[768]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(where, mean_11, primals_228, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where = mean_11 = primals_228 = None
    getitem_120: "f32[4, 1536, 1, 1]" = convolution_backward_2[0]
    getitem_121: "f32[768, 1536, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_1: "f32[4, 1536, 6, 6]" = torch.ops.aten.expand.default(getitem_120, [4, 1536, 6, 6]);  getitem_120 = None
    div_1: "f32[4, 1536, 6, 6]" = torch.ops.aten.div.Scalar(expand_1, 36);  expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_123: "f32[4, 1536, 6, 6]" = torch.ops.aten.add.Tensor(mul_462, div_1);  mul_462 = div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_9: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_123, [0, 2, 3])
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(add_123, mul_424, view_167, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_123 = mul_424 = view_167 = None
    getitem_123: "f32[4, 768, 6, 6]" = convolution_backward_3[0]
    getitem_124: "f32[1536, 768, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_177: "f32[1, 1536, 768]" = torch.ops.aten.view.default(getitem_124, [1, 1536, 768]);  getitem_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_65: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_110, 0);  squeeze_110 = None
    unsqueeze_66: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_65, 2);  unsqueeze_65 = None
    sum_10: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_177, [0, 2])
    sub_62: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_165, unsqueeze_66)
    mul_465: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(view_177, sub_62);  sub_62 = None
    sum_11: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_465, [0, 2]);  mul_465 = None
    mul_466: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_10, 0.0013020833333333333);  sum_10 = None
    unsqueeze_67: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_466, 0);  mul_466 = None
    unsqueeze_68: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_67, 2);  unsqueeze_67 = None
    mul_467: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_11, 0.0013020833333333333)
    mul_468: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_111, squeeze_111)
    mul_469: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_467, mul_468);  mul_467 = mul_468 = None
    unsqueeze_69: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_469, 0);  mul_469 = None
    unsqueeze_70: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_69, 2);  unsqueeze_69 = None
    mul_470: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_111, view_166);  view_166 = None
    unsqueeze_71: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_470, 0);  mul_470 = None
    unsqueeze_72: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_71, 2);  unsqueeze_71 = None
    sub_63: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_165, unsqueeze_66);  view_165 = unsqueeze_66 = None
    mul_471: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_70);  sub_63 = unsqueeze_70 = None
    sub_64: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_177, mul_471);  view_177 = mul_471 = None
    sub_65: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(sub_64, unsqueeze_68);  sub_64 = unsqueeze_68 = None
    mul_472: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_72);  sub_65 = unsqueeze_72 = None
    mul_473: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_111);  sum_11 = squeeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_178: "f32[1536, 1, 1, 1]" = torch.ops.aten.view.default(mul_473, [1536, 1, 1, 1]);  mul_473 = None
    mul_474: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_178, 0.03608439182435161);  view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_179: "f32[1536, 768, 1, 1]" = torch.ops.aten.view.default(mul_472, [1536, 768, 1, 1]);  mul_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_475: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_123, 1.7015043497085571);  getitem_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_476: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_76, 0.7071067811865476)
    erf_53: "f32[4, 768, 6, 6]" = torch.ops.aten.erf.default(mul_476);  mul_476 = None
    add_124: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(erf_53, 1);  erf_53 = None
    mul_477: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(add_124, 0.5);  add_124 = None
    mul_478: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_76, convolution_76)
    mul_479: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_478, -0.5);  mul_478 = None
    exp_1: "f32[4, 768, 6, 6]" = torch.ops.aten.exp.default(mul_479);  mul_479 = None
    mul_480: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
    mul_481: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_76, mul_480);  convolution_76 = mul_480 = None
    add_125: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(mul_477, mul_481);  mul_477 = mul_481 = None
    mul_482: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_475, add_125);  mul_475 = add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_12: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_482, [0, 2, 3])
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_482, mul_417, view_164, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_482 = mul_417 = view_164 = None
    getitem_126: "f32[4, 768, 6, 6]" = convolution_backward_4[0]
    getitem_127: "f32[768, 128, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_180: "f32[1, 768, 1152]" = torch.ops.aten.view.default(getitem_127, [1, 768, 1152]);  getitem_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_73: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
    unsqueeze_74: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_73, 2);  unsqueeze_73 = None
    sum_13: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_180, [0, 2])
    sub_66: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_162, unsqueeze_74)
    mul_483: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(view_180, sub_66);  sub_66 = None
    sum_14: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_483, [0, 2]);  mul_483 = None
    mul_484: "f32[768]" = torch.ops.aten.mul.Tensor(sum_13, 0.0008680555555555555);  sum_13 = None
    unsqueeze_75: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_484, 0);  mul_484 = None
    unsqueeze_76: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_75, 2);  unsqueeze_75 = None
    mul_485: "f32[768]" = torch.ops.aten.mul.Tensor(sum_14, 0.0008680555555555555)
    mul_486: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_487: "f32[768]" = torch.ops.aten.mul.Tensor(mul_485, mul_486);  mul_485 = mul_486 = None
    unsqueeze_77: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_487, 0);  mul_487 = None
    unsqueeze_78: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_77, 2);  unsqueeze_77 = None
    mul_488: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_109, view_163);  view_163 = None
    unsqueeze_79: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_488, 0);  mul_488 = None
    unsqueeze_80: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_79, 2);  unsqueeze_79 = None
    sub_67: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_162, unsqueeze_74);  view_162 = unsqueeze_74 = None
    mul_489: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_78);  sub_67 = unsqueeze_78 = None
    sub_68: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_180, mul_489);  view_180 = mul_489 = None
    sub_69: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_68, unsqueeze_76);  sub_68 = unsqueeze_76 = None
    mul_490: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_80);  sub_69 = unsqueeze_80 = None
    mul_491: "f32[768]" = torch.ops.aten.mul.Tensor(sum_14, squeeze_109);  sum_14 = squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_181: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_491, [768, 1, 1, 1]);  mul_491 = None
    mul_492: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_181, 0.02946278254943948);  view_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_182: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_490, [768, 128, 3, 3]);  mul_490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_493: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_126, 1.7015043497085571);  getitem_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_494: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_75, 0.7071067811865476)
    erf_54: "f32[4, 768, 6, 6]" = torch.ops.aten.erf.default(mul_494);  mul_494 = None
    add_126: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(erf_54, 1);  erf_54 = None
    mul_495: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(add_126, 0.5);  add_126 = None
    mul_496: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_75, convolution_75)
    mul_497: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_496, -0.5);  mul_496 = None
    exp_2: "f32[4, 768, 6, 6]" = torch.ops.aten.exp.default(mul_497);  mul_497 = None
    mul_498: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
    mul_499: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_75, mul_498);  convolution_75 = mul_498 = None
    add_127: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(mul_495, mul_499);  mul_495 = mul_499 = None
    mul_500: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_493, add_127);  mul_493 = add_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_15: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_500, [0, 2, 3])
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_500, mul_410, view_161, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_500 = mul_410 = view_161 = None
    getitem_129: "f32[4, 768, 6, 6]" = convolution_backward_5[0]
    getitem_130: "f32[768, 128, 3, 3]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_183: "f32[1, 768, 1152]" = torch.ops.aten.view.default(getitem_130, [1, 768, 1152]);  getitem_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_81: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_106, 0);  squeeze_106 = None
    unsqueeze_82: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_81, 2);  unsqueeze_81 = None
    sum_16: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_183, [0, 2])
    sub_70: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_159, unsqueeze_82)
    mul_501: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(view_183, sub_70);  sub_70 = None
    sum_17: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_501, [0, 2]);  mul_501 = None
    mul_502: "f32[768]" = torch.ops.aten.mul.Tensor(sum_16, 0.0008680555555555555);  sum_16 = None
    unsqueeze_83: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_502, 0);  mul_502 = None
    unsqueeze_84: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_83, 2);  unsqueeze_83 = None
    mul_503: "f32[768]" = torch.ops.aten.mul.Tensor(sum_17, 0.0008680555555555555)
    mul_504: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_107, squeeze_107)
    mul_505: "f32[768]" = torch.ops.aten.mul.Tensor(mul_503, mul_504);  mul_503 = mul_504 = None
    unsqueeze_85: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_505, 0);  mul_505 = None
    unsqueeze_86: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_85, 2);  unsqueeze_85 = None
    mul_506: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_107, view_160);  view_160 = None
    unsqueeze_87: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_506, 0);  mul_506 = None
    unsqueeze_88: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_87, 2);  unsqueeze_87 = None
    sub_71: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_159, unsqueeze_82);  view_159 = unsqueeze_82 = None
    mul_507: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_86);  sub_71 = unsqueeze_86 = None
    sub_72: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_183, mul_507);  view_183 = mul_507 = None
    sub_73: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_72, unsqueeze_84);  sub_72 = unsqueeze_84 = None
    mul_508: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_88);  sub_73 = unsqueeze_88 = None
    mul_509: "f32[768]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_107);  sum_17 = squeeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_184: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_509, [768, 1, 1, 1]);  mul_509 = None
    mul_510: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_184, 0.02946278254943948);  view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_185: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_508, [768, 128, 3, 3]);  mul_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_511: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_129, 1.7015043497085571);  getitem_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_512: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_74, 0.7071067811865476)
    erf_55: "f32[4, 768, 6, 6]" = torch.ops.aten.erf.default(mul_512);  mul_512 = None
    add_128: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(erf_55, 1);  erf_55 = None
    mul_513: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(add_128, 0.5);  add_128 = None
    mul_514: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_74, convolution_74)
    mul_515: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_514, -0.5);  mul_514 = None
    exp_3: "f32[4, 768, 6, 6]" = torch.ops.aten.exp.default(mul_515);  mul_515 = None
    mul_516: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(exp_3, 0.3989422804014327);  exp_3 = None
    mul_517: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_74, mul_516);  convolution_74 = mul_516 = None
    add_129: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(mul_513, mul_517);  mul_513 = mul_517 = None
    mul_518: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_511, add_129);  mul_511 = add_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_18: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_518, [0, 2, 3])
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_518, mul_403, view_158, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_518 = mul_403 = view_158 = None
    getitem_132: "f32[4, 1536, 6, 6]" = convolution_backward_6[0]
    getitem_133: "f32[768, 1536, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_186: "f32[1, 768, 1536]" = torch.ops.aten.view.default(getitem_133, [1, 768, 1536]);  getitem_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_89: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_104, 0);  squeeze_104 = None
    unsqueeze_90: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_89, 2);  unsqueeze_89 = None
    sum_19: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_186, [0, 2])
    sub_74: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_156, unsqueeze_90)
    mul_519: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(view_186, sub_74);  sub_74 = None
    sum_20: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_519, [0, 2]);  mul_519 = None
    mul_520: "f32[768]" = torch.ops.aten.mul.Tensor(sum_19, 0.0006510416666666666);  sum_19 = None
    unsqueeze_91: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_520, 0);  mul_520 = None
    unsqueeze_92: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_91, 2);  unsqueeze_91 = None
    mul_521: "f32[768]" = torch.ops.aten.mul.Tensor(sum_20, 0.0006510416666666666)
    mul_522: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_105, squeeze_105)
    mul_523: "f32[768]" = torch.ops.aten.mul.Tensor(mul_521, mul_522);  mul_521 = mul_522 = None
    unsqueeze_93: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_523, 0);  mul_523 = None
    unsqueeze_94: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_93, 2);  unsqueeze_93 = None
    mul_524: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_105, view_157);  view_157 = None
    unsqueeze_95: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_524, 0);  mul_524 = None
    unsqueeze_96: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_95, 2);  unsqueeze_95 = None
    sub_75: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_156, unsqueeze_90);  view_156 = unsqueeze_90 = None
    mul_525: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_94);  sub_75 = unsqueeze_94 = None
    sub_76: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_186, mul_525);  view_186 = mul_525 = None
    sub_77: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(sub_76, unsqueeze_92);  sub_76 = unsqueeze_92 = None
    mul_526: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_96);  sub_77 = unsqueeze_96 = None
    mul_527: "f32[768]" = torch.ops.aten.mul.Tensor(sum_20, squeeze_105);  sum_20 = squeeze_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_187: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_527, [768, 1, 1, 1]);  mul_527 = None
    mul_528: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_187, 0.02551551815399144);  view_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_188: "f32[768, 1536, 1, 1]" = torch.ops.aten.view.default(mul_526, [768, 1536, 1, 1]);  mul_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_529: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_132, 0.9622504486493761);  getitem_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_530: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_529, 1.7015043497085571);  mul_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_531: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(add_109, 0.7071067811865476)
    erf_56: "f32[4, 1536, 6, 6]" = torch.ops.aten.erf.default(mul_531);  mul_531 = None
    add_130: "f32[4, 1536, 6, 6]" = torch.ops.aten.add.Tensor(erf_56, 1);  erf_56 = None
    mul_532: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(add_130, 0.5);  add_130 = None
    mul_533: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(add_109, add_109)
    mul_534: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_533, -0.5);  mul_533 = None
    exp_4: "f32[4, 1536, 6, 6]" = torch.ops.aten.exp.default(mul_534);  mul_534 = None
    mul_535: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(exp_4, 0.3989422804014327);  exp_4 = None
    mul_536: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(add_109, mul_535);  add_109 = mul_535 = None
    add_131: "f32[4, 1536, 6, 6]" = torch.ops.aten.add.Tensor(mul_532, mul_536);  mul_532 = mul_536 = None
    mul_537: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_530, add_131);  mul_530 = add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    add_132: "f32[4, 1536, 6, 6]" = torch.ops.aten.add.Tensor(getitem_114, mul_537);  getitem_114 = mul_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_538: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(add_132, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    mul_539: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_538, clone_10);  clone_10 = None
    mul_540: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_538, primals_167);  mul_538 = primals_167 = None
    sum_21: "f32[]" = torch.ops.aten.sum.default(mul_539);  mul_539 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_541: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_540, 2.0);  mul_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_542: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_541, convolution_71);  convolution_71 = None
    mul_543: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_541, sigmoid_10);  mul_541 = sigmoid_10 = None
    sum_22: "f32[4, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_542, [2, 3], True);  mul_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_28: "f32[4, 1536, 1, 1]" = torch.ops.aten.alias.default(alias_21);  alias_21 = None
    sub_78: "f32[4, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_28)
    mul_544: "f32[4, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(alias_28, sub_78);  alias_28 = sub_78 = None
    mul_545: "f32[4, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_22, mul_544);  sum_22 = mul_544 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_23: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_545, [0, 2, 3])
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_545, relu_10, primals_226, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_545 = primals_226 = None
    getitem_135: "f32[4, 768, 1, 1]" = convolution_backward_7[0]
    getitem_136: "f32[1536, 768, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_30: "f32[4, 768, 1, 1]" = torch.ops.aten.alias.default(relu_10);  relu_10 = None
    alias_31: "f32[4, 768, 1, 1]" = torch.ops.aten.alias.default(alias_30);  alias_30 = None
    le_1: "b8[4, 768, 1, 1]" = torch.ops.aten.le.Scalar(alias_31, 0);  alias_31 = None
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_1: "f32[4, 768, 1, 1]" = torch.ops.aten.where.self(le_1, scalar_tensor_1, getitem_135);  le_1 = scalar_tensor_1 = getitem_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_24: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(where_1, mean_10, primals_224, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_1 = mean_10 = primals_224 = None
    getitem_138: "f32[4, 1536, 1, 1]" = convolution_backward_8[0]
    getitem_139: "f32[768, 1536, 1, 1]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_2: "f32[4, 1536, 6, 6]" = torch.ops.aten.expand.default(getitem_138, [4, 1536, 6, 6]);  getitem_138 = None
    div_2: "f32[4, 1536, 6, 6]" = torch.ops.aten.div.Scalar(expand_2, 36);  expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_133: "f32[4, 1536, 6, 6]" = torch.ops.aten.add.Tensor(mul_543, div_2);  mul_543 = div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_25: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_133, [0, 2, 3])
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(add_133, mul_391, view_155, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_133 = mul_391 = view_155 = None
    getitem_141: "f32[4, 768, 6, 6]" = convolution_backward_9[0]
    getitem_142: "f32[1536, 768, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_189: "f32[1, 1536, 768]" = torch.ops.aten.view.default(getitem_142, [1, 1536, 768]);  getitem_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_97: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
    unsqueeze_98: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_97, 2);  unsqueeze_97 = None
    sum_26: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_189, [0, 2])
    sub_79: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_153, unsqueeze_98)
    mul_546: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(view_189, sub_79);  sub_79 = None
    sum_27: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_546, [0, 2]);  mul_546 = None
    mul_547: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_26, 0.0013020833333333333);  sum_26 = None
    unsqueeze_99: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_547, 0);  mul_547 = None
    unsqueeze_100: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_99, 2);  unsqueeze_99 = None
    mul_548: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_27, 0.0013020833333333333)
    mul_549: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_550: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_548, mul_549);  mul_548 = mul_549 = None
    unsqueeze_101: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_550, 0);  mul_550 = None
    unsqueeze_102: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_101, 2);  unsqueeze_101 = None
    mul_551: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_103, view_154);  view_154 = None
    unsqueeze_103: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_551, 0);  mul_551 = None
    unsqueeze_104: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_103, 2);  unsqueeze_103 = None
    sub_80: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_153, unsqueeze_98);  view_153 = unsqueeze_98 = None
    mul_552: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_102);  sub_80 = unsqueeze_102 = None
    sub_81: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_189, mul_552);  view_189 = mul_552 = None
    sub_82: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(sub_81, unsqueeze_100);  sub_81 = unsqueeze_100 = None
    mul_553: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_104);  sub_82 = unsqueeze_104 = None
    mul_554: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_103);  sum_27 = squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_190: "f32[1536, 1, 1, 1]" = torch.ops.aten.view.default(mul_554, [1536, 1, 1, 1]);  mul_554 = None
    mul_555: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_190, 0.03608439182435161);  view_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_191: "f32[1536, 768, 1, 1]" = torch.ops.aten.view.default(mul_553, [1536, 768, 1, 1]);  mul_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_556: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_141, 1.7015043497085571);  getitem_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_557: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_70, 0.7071067811865476)
    erf_57: "f32[4, 768, 6, 6]" = torch.ops.aten.erf.default(mul_557);  mul_557 = None
    add_134: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(erf_57, 1);  erf_57 = None
    mul_558: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(add_134, 0.5);  add_134 = None
    mul_559: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_70, convolution_70)
    mul_560: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_559, -0.5);  mul_559 = None
    exp_5: "f32[4, 768, 6, 6]" = torch.ops.aten.exp.default(mul_560);  mul_560 = None
    mul_561: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(exp_5, 0.3989422804014327);  exp_5 = None
    mul_562: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_70, mul_561);  convolution_70 = mul_561 = None
    add_135: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(mul_558, mul_562);  mul_558 = mul_562 = None
    mul_563: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_556, add_135);  mul_556 = add_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_28: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_563, [0, 2, 3])
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_563, mul_384, view_152, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_563 = mul_384 = view_152 = None
    getitem_144: "f32[4, 768, 6, 6]" = convolution_backward_10[0]
    getitem_145: "f32[768, 128, 3, 3]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_192: "f32[1, 768, 1152]" = torch.ops.aten.view.default(getitem_145, [1, 768, 1152]);  getitem_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_105: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_100, 0);  squeeze_100 = None
    unsqueeze_106: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_105, 2);  unsqueeze_105 = None
    sum_29: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_192, [0, 2])
    sub_83: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_150, unsqueeze_106)
    mul_564: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(view_192, sub_83);  sub_83 = None
    sum_30: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_564, [0, 2]);  mul_564 = None
    mul_565: "f32[768]" = torch.ops.aten.mul.Tensor(sum_29, 0.0008680555555555555);  sum_29 = None
    unsqueeze_107: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_565, 0);  mul_565 = None
    unsqueeze_108: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_107, 2);  unsqueeze_107 = None
    mul_566: "f32[768]" = torch.ops.aten.mul.Tensor(sum_30, 0.0008680555555555555)
    mul_567: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_101, squeeze_101)
    mul_568: "f32[768]" = torch.ops.aten.mul.Tensor(mul_566, mul_567);  mul_566 = mul_567 = None
    unsqueeze_109: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_568, 0);  mul_568 = None
    unsqueeze_110: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_109, 2);  unsqueeze_109 = None
    mul_569: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_101, view_151);  view_151 = None
    unsqueeze_111: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_569, 0);  mul_569 = None
    unsqueeze_112: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_111, 2);  unsqueeze_111 = None
    sub_84: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_150, unsqueeze_106);  view_150 = unsqueeze_106 = None
    mul_570: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_110);  sub_84 = unsqueeze_110 = None
    sub_85: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_192, mul_570);  view_192 = mul_570 = None
    sub_86: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_85, unsqueeze_108);  sub_85 = unsqueeze_108 = None
    mul_571: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_112);  sub_86 = unsqueeze_112 = None
    mul_572: "f32[768]" = torch.ops.aten.mul.Tensor(sum_30, squeeze_101);  sum_30 = squeeze_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_193: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_572, [768, 1, 1, 1]);  mul_572 = None
    mul_573: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_193, 0.02946278254943948);  view_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_194: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_571, [768, 128, 3, 3]);  mul_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_574: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_144, 1.7015043497085571);  getitem_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_575: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_69, 0.7071067811865476)
    erf_58: "f32[4, 768, 6, 6]" = torch.ops.aten.erf.default(mul_575);  mul_575 = None
    add_136: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(erf_58, 1);  erf_58 = None
    mul_576: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(add_136, 0.5);  add_136 = None
    mul_577: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_69, convolution_69)
    mul_578: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_577, -0.5);  mul_577 = None
    exp_6: "f32[4, 768, 6, 6]" = torch.ops.aten.exp.default(mul_578);  mul_578 = None
    mul_579: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
    mul_580: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_69, mul_579);  convolution_69 = mul_579 = None
    add_137: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(mul_576, mul_580);  mul_576 = mul_580 = None
    mul_581: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_574, add_137);  mul_574 = add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_31: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_581, [0, 2, 3])
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_581, mul_377, view_149, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_581 = mul_377 = view_149 = None
    getitem_147: "f32[4, 768, 6, 6]" = convolution_backward_11[0]
    getitem_148: "f32[768, 128, 3, 3]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_195: "f32[1, 768, 1152]" = torch.ops.aten.view.default(getitem_148, [1, 768, 1152]);  getitem_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_113: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_98, 0);  squeeze_98 = None
    unsqueeze_114: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_113, 2);  unsqueeze_113 = None
    sum_32: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_195, [0, 2])
    sub_87: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_147, unsqueeze_114)
    mul_582: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(view_195, sub_87);  sub_87 = None
    sum_33: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_582, [0, 2]);  mul_582 = None
    mul_583: "f32[768]" = torch.ops.aten.mul.Tensor(sum_32, 0.0008680555555555555);  sum_32 = None
    unsqueeze_115: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_583, 0);  mul_583 = None
    unsqueeze_116: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_115, 2);  unsqueeze_115 = None
    mul_584: "f32[768]" = torch.ops.aten.mul.Tensor(sum_33, 0.0008680555555555555)
    mul_585: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_99, squeeze_99)
    mul_586: "f32[768]" = torch.ops.aten.mul.Tensor(mul_584, mul_585);  mul_584 = mul_585 = None
    unsqueeze_117: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_586, 0);  mul_586 = None
    unsqueeze_118: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_117, 2);  unsqueeze_117 = None
    mul_587: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_99, view_148);  view_148 = None
    unsqueeze_119: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_587, 0);  mul_587 = None
    unsqueeze_120: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_119, 2);  unsqueeze_119 = None
    sub_88: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_147, unsqueeze_114);  view_147 = unsqueeze_114 = None
    mul_588: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_118);  sub_88 = unsqueeze_118 = None
    sub_89: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_195, mul_588);  view_195 = mul_588 = None
    sub_90: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_89, unsqueeze_116);  sub_89 = unsqueeze_116 = None
    mul_589: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_120);  sub_90 = unsqueeze_120 = None
    mul_590: "f32[768]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_99);  sum_33 = squeeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_196: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_590, [768, 1, 1, 1]);  mul_590 = None
    mul_591: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_196, 0.02946278254943948);  view_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_197: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_589, [768, 128, 3, 3]);  mul_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_592: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_147, 1.7015043497085571);  getitem_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_593: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_68, 0.7071067811865476)
    erf_59: "f32[4, 768, 6, 6]" = torch.ops.aten.erf.default(mul_593);  mul_593 = None
    add_138: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(erf_59, 1);  erf_59 = None
    mul_594: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(add_138, 0.5);  add_138 = None
    mul_595: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_68, convolution_68)
    mul_596: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_595, -0.5);  mul_595 = None
    exp_7: "f32[4, 768, 6, 6]" = torch.ops.aten.exp.default(mul_596);  mul_596 = None
    mul_597: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(exp_7, 0.3989422804014327);  exp_7 = None
    mul_598: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_68, mul_597);  convolution_68 = mul_597 = None
    add_139: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(mul_594, mul_598);  mul_594 = mul_598 = None
    mul_599: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_592, add_139);  mul_592 = add_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_34: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_599, [0, 2, 3])
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_599, mul_370, view_146, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_599 = mul_370 = view_146 = None
    getitem_150: "f32[4, 1536, 6, 6]" = convolution_backward_12[0]
    getitem_151: "f32[768, 1536, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_198: "f32[1, 768, 1536]" = torch.ops.aten.view.default(getitem_151, [1, 768, 1536]);  getitem_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_121: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
    unsqueeze_122: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_121, 2);  unsqueeze_121 = None
    sum_35: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_198, [0, 2])
    sub_91: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_144, unsqueeze_122)
    mul_600: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(view_198, sub_91);  sub_91 = None
    sum_36: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_600, [0, 2]);  mul_600 = None
    mul_601: "f32[768]" = torch.ops.aten.mul.Tensor(sum_35, 0.0006510416666666666);  sum_35 = None
    unsqueeze_123: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_601, 0);  mul_601 = None
    unsqueeze_124: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_123, 2);  unsqueeze_123 = None
    mul_602: "f32[768]" = torch.ops.aten.mul.Tensor(sum_36, 0.0006510416666666666)
    mul_603: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_604: "f32[768]" = torch.ops.aten.mul.Tensor(mul_602, mul_603);  mul_602 = mul_603 = None
    unsqueeze_125: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_604, 0);  mul_604 = None
    unsqueeze_126: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_125, 2);  unsqueeze_125 = None
    mul_605: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_97, view_145);  view_145 = None
    unsqueeze_127: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_605, 0);  mul_605 = None
    unsqueeze_128: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_127, 2);  unsqueeze_127 = None
    sub_92: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_144, unsqueeze_122);  view_144 = unsqueeze_122 = None
    mul_606: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_126);  sub_92 = unsqueeze_126 = None
    sub_93: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_198, mul_606);  view_198 = mul_606 = None
    sub_94: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(sub_93, unsqueeze_124);  sub_93 = unsqueeze_124 = None
    mul_607: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_128);  sub_94 = unsqueeze_128 = None
    mul_608: "f32[768]" = torch.ops.aten.mul.Tensor(sum_36, squeeze_97);  sum_36 = squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_199: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_608, [768, 1, 1, 1]);  mul_608 = None
    mul_609: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_199, 0.02551551815399144);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_200: "f32[768, 1536, 1, 1]" = torch.ops.aten.view.default(mul_607, [768, 1536, 1, 1]);  mul_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_610: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_150, 0.9805806756909201);  getitem_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_611: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_610, 1.7015043497085571);  mul_610 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_612: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(add_100, 0.7071067811865476)
    erf_60: "f32[4, 1536, 6, 6]" = torch.ops.aten.erf.default(mul_612);  mul_612 = None
    add_140: "f32[4, 1536, 6, 6]" = torch.ops.aten.add.Tensor(erf_60, 1);  erf_60 = None
    mul_613: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(add_140, 0.5);  add_140 = None
    mul_614: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(add_100, add_100)
    mul_615: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_614, -0.5);  mul_614 = None
    exp_8: "f32[4, 1536, 6, 6]" = torch.ops.aten.exp.default(mul_615);  mul_615 = None
    mul_616: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
    mul_617: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(add_100, mul_616);  add_100 = mul_616 = None
    add_141: "f32[4, 1536, 6, 6]" = torch.ops.aten.add.Tensor(mul_613, mul_617);  mul_613 = mul_617 = None
    mul_618: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_611, add_141);  mul_611 = add_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    add_142: "f32[4, 1536, 6, 6]" = torch.ops.aten.add.Tensor(add_132, mul_618);  add_132 = mul_618 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_619: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(add_142, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    mul_620: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_619, clone_9);  clone_9 = None
    mul_621: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_619, primals_154);  mul_619 = primals_154 = None
    sum_37: "f32[]" = torch.ops.aten.sum.default(mul_620);  mul_620 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_622: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_621, 2.0);  mul_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_623: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_622, convolution_65);  convolution_65 = None
    mul_624: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_622, sigmoid_9);  mul_622 = sigmoid_9 = None
    sum_38: "f32[4, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_623, [2, 3], True);  mul_623 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_32: "f32[4, 1536, 1, 1]" = torch.ops.aten.alias.default(alias_19);  alias_19 = None
    sub_95: "f32[4, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_32)
    mul_625: "f32[4, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(alias_32, sub_95);  alias_32 = sub_95 = None
    mul_626: "f32[4, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_38, mul_625);  sum_38 = mul_625 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_39: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_626, [0, 2, 3])
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_626, relu_9, primals_222, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_626 = primals_222 = None
    getitem_153: "f32[4, 768, 1, 1]" = convolution_backward_13[0]
    getitem_154: "f32[1536, 768, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_34: "f32[4, 768, 1, 1]" = torch.ops.aten.alias.default(relu_9);  relu_9 = None
    alias_35: "f32[4, 768, 1, 1]" = torch.ops.aten.alias.default(alias_34);  alias_34 = None
    le_2: "b8[4, 768, 1, 1]" = torch.ops.aten.le.Scalar(alias_35, 0);  alias_35 = None
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_2: "f32[4, 768, 1, 1]" = torch.ops.aten.where.self(le_2, scalar_tensor_2, getitem_153);  le_2 = scalar_tensor_2 = getitem_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(where_2, mean_9, primals_220, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_2 = mean_9 = primals_220 = None
    getitem_156: "f32[4, 1536, 1, 1]" = convolution_backward_14[0]
    getitem_157: "f32[768, 1536, 1, 1]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_3: "f32[4, 1536, 6, 6]" = torch.ops.aten.expand.default(getitem_156, [4, 1536, 6, 6]);  getitem_156 = None
    div_3: "f32[4, 1536, 6, 6]" = torch.ops.aten.div.Scalar(expand_3, 36);  expand_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_143: "f32[4, 1536, 6, 6]" = torch.ops.aten.add.Tensor(mul_624, div_3);  mul_624 = div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_41: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_143, [0, 2, 3])
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(add_143, mul_358, view_143, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_143 = mul_358 = view_143 = None
    getitem_159: "f32[4, 768, 6, 6]" = convolution_backward_15[0]
    getitem_160: "f32[1536, 768, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_201: "f32[1, 1536, 768]" = torch.ops.aten.view.default(getitem_160, [1, 1536, 768]);  getitem_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_129: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_94, 0);  squeeze_94 = None
    unsqueeze_130: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_129, 2);  unsqueeze_129 = None
    sum_42: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_201, [0, 2])
    sub_96: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_141, unsqueeze_130)
    mul_627: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(view_201, sub_96);  sub_96 = None
    sum_43: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_627, [0, 2]);  mul_627 = None
    mul_628: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_42, 0.0013020833333333333);  sum_42 = None
    unsqueeze_131: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_628, 0);  mul_628 = None
    unsqueeze_132: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_131, 2);  unsqueeze_131 = None
    mul_629: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_43, 0.0013020833333333333)
    mul_630: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_95, squeeze_95)
    mul_631: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_629, mul_630);  mul_629 = mul_630 = None
    unsqueeze_133: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_631, 0);  mul_631 = None
    unsqueeze_134: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_133, 2);  unsqueeze_133 = None
    mul_632: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_95, view_142);  view_142 = None
    unsqueeze_135: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_632, 0);  mul_632 = None
    unsqueeze_136: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_135, 2);  unsqueeze_135 = None
    sub_97: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_141, unsqueeze_130);  view_141 = unsqueeze_130 = None
    mul_633: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_134);  sub_97 = unsqueeze_134 = None
    sub_98: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_201, mul_633);  view_201 = mul_633 = None
    sub_99: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(sub_98, unsqueeze_132);  sub_98 = unsqueeze_132 = None
    mul_634: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_136);  sub_99 = unsqueeze_136 = None
    mul_635: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_95);  sum_43 = squeeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_202: "f32[1536, 1, 1, 1]" = torch.ops.aten.view.default(mul_635, [1536, 1, 1, 1]);  mul_635 = None
    mul_636: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_202, 0.03608439182435161);  view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_203: "f32[1536, 768, 1, 1]" = torch.ops.aten.view.default(mul_634, [1536, 768, 1, 1]);  mul_634 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_637: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_159, 1.7015043497085571);  getitem_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_638: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_64, 0.7071067811865476)
    erf_61: "f32[4, 768, 6, 6]" = torch.ops.aten.erf.default(mul_638);  mul_638 = None
    add_144: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(erf_61, 1);  erf_61 = None
    mul_639: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(add_144, 0.5);  add_144 = None
    mul_640: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_64, convolution_64)
    mul_641: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_640, -0.5);  mul_640 = None
    exp_9: "f32[4, 768, 6, 6]" = torch.ops.aten.exp.default(mul_641);  mul_641 = None
    mul_642: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
    mul_643: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_64, mul_642);  convolution_64 = mul_642 = None
    add_145: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(mul_639, mul_643);  mul_639 = mul_643 = None
    mul_644: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_637, add_145);  mul_637 = add_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_44: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_644, [0, 2, 3])
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_644, mul_351, view_140, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_644 = mul_351 = view_140 = None
    getitem_162: "f32[4, 768, 6, 6]" = convolution_backward_16[0]
    getitem_163: "f32[768, 128, 3, 3]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_204: "f32[1, 768, 1152]" = torch.ops.aten.view.default(getitem_163, [1, 768, 1152]);  getitem_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_137: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_92, 0);  squeeze_92 = None
    unsqueeze_138: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_137, 2);  unsqueeze_137 = None
    sum_45: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_204, [0, 2])
    sub_100: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_138, unsqueeze_138)
    mul_645: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(view_204, sub_100);  sub_100 = None
    sum_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_645, [0, 2]);  mul_645 = None
    mul_646: "f32[768]" = torch.ops.aten.mul.Tensor(sum_45, 0.0008680555555555555);  sum_45 = None
    unsqueeze_139: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_646, 0);  mul_646 = None
    unsqueeze_140: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_139, 2);  unsqueeze_139 = None
    mul_647: "f32[768]" = torch.ops.aten.mul.Tensor(sum_46, 0.0008680555555555555)
    mul_648: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_93, squeeze_93)
    mul_649: "f32[768]" = torch.ops.aten.mul.Tensor(mul_647, mul_648);  mul_647 = mul_648 = None
    unsqueeze_141: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_649, 0);  mul_649 = None
    unsqueeze_142: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_141, 2);  unsqueeze_141 = None
    mul_650: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_93, view_139);  view_139 = None
    unsqueeze_143: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_650, 0);  mul_650 = None
    unsqueeze_144: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_143, 2);  unsqueeze_143 = None
    sub_101: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_138, unsqueeze_138);  view_138 = unsqueeze_138 = None
    mul_651: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_142);  sub_101 = unsqueeze_142 = None
    sub_102: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_204, mul_651);  view_204 = mul_651 = None
    sub_103: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_102, unsqueeze_140);  sub_102 = unsqueeze_140 = None
    mul_652: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_144);  sub_103 = unsqueeze_144 = None
    mul_653: "f32[768]" = torch.ops.aten.mul.Tensor(sum_46, squeeze_93);  sum_46 = squeeze_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_205: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_653, [768, 1, 1, 1]);  mul_653 = None
    mul_654: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_205, 0.02946278254943948);  view_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_206: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_652, [768, 128, 3, 3]);  mul_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_655: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_162, 1.7015043497085571);  getitem_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_656: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_63, 0.7071067811865476)
    erf_62: "f32[4, 768, 6, 6]" = torch.ops.aten.erf.default(mul_656);  mul_656 = None
    add_146: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(erf_62, 1);  erf_62 = None
    mul_657: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(add_146, 0.5);  add_146 = None
    mul_658: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_63, convolution_63)
    mul_659: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_658, -0.5);  mul_658 = None
    exp_10: "f32[4, 768, 6, 6]" = torch.ops.aten.exp.default(mul_659);  mul_659 = None
    mul_660: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
    mul_661: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_63, mul_660);  convolution_63 = mul_660 = None
    add_147: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(mul_657, mul_661);  mul_657 = mul_661 = None
    mul_662: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_655, add_147);  mul_655 = add_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_47: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_662, [0, 2, 3])
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_662, constant_pad_nd_4, view_137, [768], [2, 2], [0, 0], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_662 = constant_pad_nd_4 = view_137 = None
    getitem_165: "f32[4, 768, 13, 13]" = convolution_backward_17[0]
    getitem_166: "f32[768, 128, 3, 3]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_207: "f32[1, 768, 1152]" = torch.ops.aten.view.default(getitem_166, [1, 768, 1152]);  getitem_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_145: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    unsqueeze_146: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_145, 2);  unsqueeze_145 = None
    sum_48: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_207, [0, 2])
    sub_104: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_135, unsqueeze_146)
    mul_663: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(view_207, sub_104);  sub_104 = None
    sum_49: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_663, [0, 2]);  mul_663 = None
    mul_664: "f32[768]" = torch.ops.aten.mul.Tensor(sum_48, 0.0008680555555555555);  sum_48 = None
    unsqueeze_147: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_664, 0);  mul_664 = None
    unsqueeze_148: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_147, 2);  unsqueeze_147 = None
    mul_665: "f32[768]" = torch.ops.aten.mul.Tensor(sum_49, 0.0008680555555555555)
    mul_666: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_667: "f32[768]" = torch.ops.aten.mul.Tensor(mul_665, mul_666);  mul_665 = mul_666 = None
    unsqueeze_149: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_667, 0);  mul_667 = None
    unsqueeze_150: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_149, 2);  unsqueeze_149 = None
    mul_668: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_91, view_136);  view_136 = None
    unsqueeze_151: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_668, 0);  mul_668 = None
    unsqueeze_152: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_151, 2);  unsqueeze_151 = None
    sub_105: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_135, unsqueeze_146);  view_135 = unsqueeze_146 = None
    mul_669: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_150);  sub_105 = unsqueeze_150 = None
    sub_106: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_207, mul_669);  view_207 = mul_669 = None
    sub_107: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_106, unsqueeze_148);  sub_106 = unsqueeze_148 = None
    mul_670: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_152);  sub_107 = unsqueeze_152 = None
    mul_671: "f32[768]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_91);  sum_49 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_208: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_671, [768, 1, 1, 1]);  mul_671 = None
    mul_672: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_208, 0.02946278254943948);  view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_209: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_670, [768, 128, 3, 3]);  mul_670 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_5: "f32[4, 768, 12, 12]" = torch.ops.aten.constant_pad_nd.default(getitem_165, [0, -1, 0, -1]);  getitem_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_673: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(constant_pad_nd_5, 1.7015043497085571);  constant_pad_nd_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_674: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_62, 0.7071067811865476)
    erf_63: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_674);  mul_674 = None
    add_148: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_63, 1);  erf_63 = None
    mul_675: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(add_148, 0.5);  add_148 = None
    mul_676: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_62, convolution_62)
    mul_677: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_676, -0.5);  mul_676 = None
    exp_11: "f32[4, 768, 12, 12]" = torch.ops.aten.exp.default(mul_677);  mul_677 = None
    mul_678: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
    mul_679: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_62, mul_678);  convolution_62 = mul_678 = None
    add_149: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(mul_675, mul_679);  mul_675 = mul_679 = None
    mul_680: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_673, add_149);  mul_673 = add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_50: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_680, [0, 2, 3])
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_680, mul_334, view_134, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_680 = view_134 = None
    getitem_168: "f32[4, 1536, 12, 12]" = convolution_backward_18[0]
    getitem_169: "f32[768, 1536, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_210: "f32[1, 768, 1536]" = torch.ops.aten.view.default(getitem_169, [1, 768, 1536]);  getitem_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_153: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_88, 0);  squeeze_88 = None
    unsqueeze_154: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_153, 2);  unsqueeze_153 = None
    sum_51: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_210, [0, 2])
    sub_108: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_132, unsqueeze_154)
    mul_681: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(view_210, sub_108);  sub_108 = None
    sum_52: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_681, [0, 2]);  mul_681 = None
    mul_682: "f32[768]" = torch.ops.aten.mul.Tensor(sum_51, 0.0006510416666666666);  sum_51 = None
    unsqueeze_155: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_682, 0);  mul_682 = None
    unsqueeze_156: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_155, 2);  unsqueeze_155 = None
    mul_683: "f32[768]" = torch.ops.aten.mul.Tensor(sum_52, 0.0006510416666666666)
    mul_684: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_89, squeeze_89)
    mul_685: "f32[768]" = torch.ops.aten.mul.Tensor(mul_683, mul_684);  mul_683 = mul_684 = None
    unsqueeze_157: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_685, 0);  mul_685 = None
    unsqueeze_158: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_157, 2);  unsqueeze_157 = None
    mul_686: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_89, view_133);  view_133 = None
    unsqueeze_159: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_686, 0);  mul_686 = None
    unsqueeze_160: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_159, 2);  unsqueeze_159 = None
    sub_109: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_132, unsqueeze_154);  view_132 = unsqueeze_154 = None
    mul_687: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_158);  sub_109 = unsqueeze_158 = None
    sub_110: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_210, mul_687);  view_210 = mul_687 = None
    sub_111: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(sub_110, unsqueeze_156);  sub_110 = unsqueeze_156 = None
    mul_688: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_160);  sub_111 = unsqueeze_160 = None
    mul_689: "f32[768]" = torch.ops.aten.mul.Tensor(sum_52, squeeze_89);  sum_52 = squeeze_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_211: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_689, [768, 1, 1, 1]);  mul_689 = None
    mul_690: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_211, 0.02551551815399144);  view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_212: "f32[768, 1536, 1, 1]" = torch.ops.aten.view.default(mul_688, [768, 1536, 1, 1]);  mul_688 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_53: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_142, [0, 2, 3])
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(add_142, avg_pool2d_2, view_131, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_142 = avg_pool2d_2 = view_131 = None
    getitem_171: "f32[4, 1536, 6, 6]" = convolution_backward_19[0]
    getitem_172: "f32[1536, 1536, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_213: "f32[1, 1536, 1536]" = torch.ops.aten.view.default(getitem_172, [1, 1536, 1536]);  getitem_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_161: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_86, 0);  squeeze_86 = None
    unsqueeze_162: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_161, 2);  unsqueeze_161 = None
    sum_54: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_213, [0, 2])
    sub_112: "f32[1, 1536, 1536]" = torch.ops.aten.sub.Tensor(view_129, unsqueeze_162)
    mul_691: "f32[1, 1536, 1536]" = torch.ops.aten.mul.Tensor(view_213, sub_112);  sub_112 = None
    sum_55: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_691, [0, 2]);  mul_691 = None
    mul_692: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_54, 0.0006510416666666666);  sum_54 = None
    unsqueeze_163: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_692, 0);  mul_692 = None
    unsqueeze_164: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_163, 2);  unsqueeze_163 = None
    mul_693: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_55, 0.0006510416666666666)
    mul_694: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_87, squeeze_87)
    mul_695: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_693, mul_694);  mul_693 = mul_694 = None
    unsqueeze_165: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_695, 0);  mul_695 = None
    unsqueeze_166: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_165, 2);  unsqueeze_165 = None
    mul_696: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_87, view_130);  view_130 = None
    unsqueeze_167: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_696, 0);  mul_696 = None
    unsqueeze_168: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 2);  unsqueeze_167 = None
    sub_113: "f32[1, 1536, 1536]" = torch.ops.aten.sub.Tensor(view_129, unsqueeze_162);  view_129 = unsqueeze_162 = None
    mul_697: "f32[1, 1536, 1536]" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_166);  sub_113 = unsqueeze_166 = None
    sub_114: "f32[1, 1536, 1536]" = torch.ops.aten.sub.Tensor(view_213, mul_697);  view_213 = mul_697 = None
    sub_115: "f32[1, 1536, 1536]" = torch.ops.aten.sub.Tensor(sub_114, unsqueeze_164);  sub_114 = unsqueeze_164 = None
    mul_698: "f32[1, 1536, 1536]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_168);  sub_115 = unsqueeze_168 = None
    mul_699: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_87);  sum_55 = squeeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_214: "f32[1536, 1, 1, 1]" = torch.ops.aten.view.default(mul_699, [1536, 1, 1, 1]);  mul_699 = None
    mul_700: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_214, 0.02551551815399144);  view_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_215: "f32[1536, 1536, 1, 1]" = torch.ops.aten.view.default(mul_698, [1536, 1536, 1, 1]);  mul_698 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    avg_pool2d_backward: "f32[4, 1536, 12, 12]" = torch.ops.aten.avg_pool2d_backward.default(getitem_171, mul_334, [2, 2], [2, 2], [0, 0], True, False, None);  getitem_171 = mul_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    add_150: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(getitem_168, avg_pool2d_backward);  getitem_168 = avg_pool2d_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_701: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_150, 0.8980265101338745);  add_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_702: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_701, 1.7015043497085571);  mul_701 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_703: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_90, 0.7071067811865476)
    erf_64: "f32[4, 1536, 12, 12]" = torch.ops.aten.erf.default(mul_703);  mul_703 = None
    add_151: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(erf_64, 1);  erf_64 = None
    mul_704: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_151, 0.5);  add_151 = None
    mul_705: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_90, add_90)
    mul_706: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_705, -0.5);  mul_705 = None
    exp_12: "f32[4, 1536, 12, 12]" = torch.ops.aten.exp.default(mul_706);  mul_706 = None
    mul_707: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
    mul_708: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_90, mul_707);  add_90 = mul_707 = None
    add_152: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_704, mul_708);  mul_704 = mul_708 = None
    mul_709: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_702, add_152);  mul_702 = add_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_710: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_709, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    mul_711: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_710, clone_8);  clone_8 = None
    mul_712: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_710, primals_138);  mul_710 = primals_138 = None
    sum_56: "f32[]" = torch.ops.aten.sum.default(mul_711);  mul_711 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_713: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_712, 2.0);  mul_712 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_714: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_713, convolution_58);  convolution_58 = None
    mul_715: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_713, sigmoid_8);  mul_713 = sigmoid_8 = None
    sum_57: "f32[4, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_714, [2, 3], True);  mul_714 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_36: "f32[4, 1536, 1, 1]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    sub_116: "f32[4, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_36)
    mul_716: "f32[4, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(alias_36, sub_116);  alias_36 = sub_116 = None
    mul_717: "f32[4, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_57, mul_716);  sum_57 = mul_716 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_58: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_717, [0, 2, 3])
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_717, relu_8, primals_218, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_717 = primals_218 = None
    getitem_174: "f32[4, 768, 1, 1]" = convolution_backward_20[0]
    getitem_175: "f32[1536, 768, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_38: "f32[4, 768, 1, 1]" = torch.ops.aten.alias.default(relu_8);  relu_8 = None
    alias_39: "f32[4, 768, 1, 1]" = torch.ops.aten.alias.default(alias_38);  alias_38 = None
    le_3: "b8[4, 768, 1, 1]" = torch.ops.aten.le.Scalar(alias_39, 0);  alias_39 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_3: "f32[4, 768, 1, 1]" = torch.ops.aten.where.self(le_3, scalar_tensor_3, getitem_174);  le_3 = scalar_tensor_3 = getitem_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_59: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(where_3, mean_8, primals_216, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_3 = mean_8 = primals_216 = None
    getitem_177: "f32[4, 1536, 1, 1]" = convolution_backward_21[0]
    getitem_178: "f32[768, 1536, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_4: "f32[4, 1536, 12, 12]" = torch.ops.aten.expand.default(getitem_177, [4, 1536, 12, 12]);  getitem_177 = None
    div_4: "f32[4, 1536, 12, 12]" = torch.ops.aten.div.Scalar(expand_4, 144);  expand_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_153: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_715, div_4);  mul_715 = div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_60: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_153, [0, 2, 3])
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(add_153, mul_322, view_128, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_153 = mul_322 = view_128 = None
    getitem_180: "f32[4, 768, 12, 12]" = convolution_backward_22[0]
    getitem_181: "f32[1536, 768, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_216: "f32[1, 1536, 768]" = torch.ops.aten.view.default(getitem_181, [1, 1536, 768]);  getitem_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_169: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    unsqueeze_170: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_169, 2);  unsqueeze_169 = None
    sum_61: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_216, [0, 2])
    sub_117: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_126, unsqueeze_170)
    mul_718: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(view_216, sub_117);  sub_117 = None
    sum_62: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_718, [0, 2]);  mul_718 = None
    mul_719: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_61, 0.0013020833333333333);  sum_61 = None
    unsqueeze_171: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_719, 0);  mul_719 = None
    unsqueeze_172: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_171, 2);  unsqueeze_171 = None
    mul_720: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_62, 0.0013020833333333333)
    mul_721: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_722: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_720, mul_721);  mul_720 = mul_721 = None
    unsqueeze_173: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_722, 0);  mul_722 = None
    unsqueeze_174: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_173, 2);  unsqueeze_173 = None
    mul_723: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_85, view_127);  view_127 = None
    unsqueeze_175: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_723, 0);  mul_723 = None
    unsqueeze_176: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_175, 2);  unsqueeze_175 = None
    sub_118: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_126, unsqueeze_170);  view_126 = unsqueeze_170 = None
    mul_724: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_174);  sub_118 = unsqueeze_174 = None
    sub_119: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_216, mul_724);  view_216 = mul_724 = None
    sub_120: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(sub_119, unsqueeze_172);  sub_119 = unsqueeze_172 = None
    mul_725: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_176);  sub_120 = unsqueeze_176 = None
    mul_726: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_62, squeeze_85);  sum_62 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_217: "f32[1536, 1, 1, 1]" = torch.ops.aten.view.default(mul_726, [1536, 1, 1, 1]);  mul_726 = None
    mul_727: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_217, 0.03608439182435161);  view_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_218: "f32[1536, 768, 1, 1]" = torch.ops.aten.view.default(mul_725, [1536, 768, 1, 1]);  mul_725 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_728: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_180, 1.7015043497085571);  getitem_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_729: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_57, 0.7071067811865476)
    erf_65: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_729);  mul_729 = None
    add_154: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_65, 1);  erf_65 = None
    mul_730: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(add_154, 0.5);  add_154 = None
    mul_731: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_57, convolution_57)
    mul_732: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_731, -0.5);  mul_731 = None
    exp_13: "f32[4, 768, 12, 12]" = torch.ops.aten.exp.default(mul_732);  mul_732 = None
    mul_733: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
    mul_734: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_57, mul_733);  convolution_57 = mul_733 = None
    add_155: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(mul_730, mul_734);  mul_730 = mul_734 = None
    mul_735: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_728, add_155);  mul_728 = add_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_63: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_735, [0, 2, 3])
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_735, mul_315, view_125, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_735 = mul_315 = view_125 = None
    getitem_183: "f32[4, 768, 12, 12]" = convolution_backward_23[0]
    getitem_184: "f32[768, 128, 3, 3]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_219: "f32[1, 768, 1152]" = torch.ops.aten.view.default(getitem_184, [1, 768, 1152]);  getitem_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_177: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_82, 0);  squeeze_82 = None
    unsqueeze_178: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_177, 2);  unsqueeze_177 = None
    sum_64: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_219, [0, 2])
    sub_121: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_123, unsqueeze_178)
    mul_736: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(view_219, sub_121);  sub_121 = None
    sum_65: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_736, [0, 2]);  mul_736 = None
    mul_737: "f32[768]" = torch.ops.aten.mul.Tensor(sum_64, 0.0008680555555555555);  sum_64 = None
    unsqueeze_179: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_737, 0);  mul_737 = None
    unsqueeze_180: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_179, 2);  unsqueeze_179 = None
    mul_738: "f32[768]" = torch.ops.aten.mul.Tensor(sum_65, 0.0008680555555555555)
    mul_739: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_83, squeeze_83)
    mul_740: "f32[768]" = torch.ops.aten.mul.Tensor(mul_738, mul_739);  mul_738 = mul_739 = None
    unsqueeze_181: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_740, 0);  mul_740 = None
    unsqueeze_182: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_181, 2);  unsqueeze_181 = None
    mul_741: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_83, view_124);  view_124 = None
    unsqueeze_183: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_741, 0);  mul_741 = None
    unsqueeze_184: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_183, 2);  unsqueeze_183 = None
    sub_122: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_123, unsqueeze_178);  view_123 = unsqueeze_178 = None
    mul_742: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_122, unsqueeze_182);  sub_122 = unsqueeze_182 = None
    sub_123: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_219, mul_742);  view_219 = mul_742 = None
    sub_124: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_123, unsqueeze_180);  sub_123 = unsqueeze_180 = None
    mul_743: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_184);  sub_124 = unsqueeze_184 = None
    mul_744: "f32[768]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_83);  sum_65 = squeeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_220: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_744, [768, 1, 1, 1]);  mul_744 = None
    mul_745: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_220, 0.02946278254943948);  view_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_221: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_743, [768, 128, 3, 3]);  mul_743 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_746: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_183, 1.7015043497085571);  getitem_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_747: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_56, 0.7071067811865476)
    erf_66: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_747);  mul_747 = None
    add_156: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_66, 1);  erf_66 = None
    mul_748: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(add_156, 0.5);  add_156 = None
    mul_749: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_56, convolution_56)
    mul_750: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_749, -0.5);  mul_749 = None
    exp_14: "f32[4, 768, 12, 12]" = torch.ops.aten.exp.default(mul_750);  mul_750 = None
    mul_751: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
    mul_752: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_56, mul_751);  convolution_56 = mul_751 = None
    add_157: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(mul_748, mul_752);  mul_748 = mul_752 = None
    mul_753: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_746, add_157);  mul_746 = add_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_66: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_753, [0, 2, 3])
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_753, mul_308, view_122, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_753 = mul_308 = view_122 = None
    getitem_186: "f32[4, 768, 12, 12]" = convolution_backward_24[0]
    getitem_187: "f32[768, 128, 3, 3]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_222: "f32[1, 768, 1152]" = torch.ops.aten.view.default(getitem_187, [1, 768, 1152]);  getitem_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_185: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_80, 0);  squeeze_80 = None
    unsqueeze_186: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_185, 2);  unsqueeze_185 = None
    sum_67: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_222, [0, 2])
    sub_125: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_120, unsqueeze_186)
    mul_754: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(view_222, sub_125);  sub_125 = None
    sum_68: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_754, [0, 2]);  mul_754 = None
    mul_755: "f32[768]" = torch.ops.aten.mul.Tensor(sum_67, 0.0008680555555555555);  sum_67 = None
    unsqueeze_187: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_755, 0);  mul_755 = None
    unsqueeze_188: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_187, 2);  unsqueeze_187 = None
    mul_756: "f32[768]" = torch.ops.aten.mul.Tensor(sum_68, 0.0008680555555555555)
    mul_757: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_81, squeeze_81)
    mul_758: "f32[768]" = torch.ops.aten.mul.Tensor(mul_756, mul_757);  mul_756 = mul_757 = None
    unsqueeze_189: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_758, 0);  mul_758 = None
    unsqueeze_190: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_189, 2);  unsqueeze_189 = None
    mul_759: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_81, view_121);  view_121 = None
    unsqueeze_191: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_759, 0);  mul_759 = None
    unsqueeze_192: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_191, 2);  unsqueeze_191 = None
    sub_126: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_120, unsqueeze_186);  view_120 = unsqueeze_186 = None
    mul_760: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_190);  sub_126 = unsqueeze_190 = None
    sub_127: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_222, mul_760);  view_222 = mul_760 = None
    sub_128: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_127, unsqueeze_188);  sub_127 = unsqueeze_188 = None
    mul_761: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_192);  sub_128 = unsqueeze_192 = None
    mul_762: "f32[768]" = torch.ops.aten.mul.Tensor(sum_68, squeeze_81);  sum_68 = squeeze_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_223: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_762, [768, 1, 1, 1]);  mul_762 = None
    mul_763: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_223, 0.02946278254943948);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_224: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_761, [768, 128, 3, 3]);  mul_761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_764: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_186, 1.7015043497085571);  getitem_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_765: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_55, 0.7071067811865476)
    erf_67: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_765);  mul_765 = None
    add_158: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_67, 1);  erf_67 = None
    mul_766: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(add_158, 0.5);  add_158 = None
    mul_767: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_55, convolution_55)
    mul_768: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_767, -0.5);  mul_767 = None
    exp_15: "f32[4, 768, 12, 12]" = torch.ops.aten.exp.default(mul_768);  mul_768 = None
    mul_769: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
    mul_770: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_55, mul_769);  convolution_55 = mul_769 = None
    add_159: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(mul_766, mul_770);  mul_766 = mul_770 = None
    mul_771: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_764, add_159);  mul_764 = add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_69: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_771, [0, 2, 3])
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_771, mul_301, view_119, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_771 = mul_301 = view_119 = None
    getitem_189: "f32[4, 1536, 12, 12]" = convolution_backward_25[0]
    getitem_190: "f32[768, 1536, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_225: "f32[1, 768, 1536]" = torch.ops.aten.view.default(getitem_190, [1, 768, 1536]);  getitem_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_193: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_194: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_193, 2);  unsqueeze_193 = None
    sum_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_225, [0, 2])
    sub_129: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_117, unsqueeze_194)
    mul_772: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(view_225, sub_129);  sub_129 = None
    sum_71: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_772, [0, 2]);  mul_772 = None
    mul_773: "f32[768]" = torch.ops.aten.mul.Tensor(sum_70, 0.0006510416666666666);  sum_70 = None
    unsqueeze_195: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_773, 0);  mul_773 = None
    unsqueeze_196: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_195, 2);  unsqueeze_195 = None
    mul_774: "f32[768]" = torch.ops.aten.mul.Tensor(sum_71, 0.0006510416666666666)
    mul_775: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_776: "f32[768]" = torch.ops.aten.mul.Tensor(mul_774, mul_775);  mul_774 = mul_775 = None
    unsqueeze_197: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_776, 0);  mul_776 = None
    unsqueeze_198: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 2);  unsqueeze_197 = None
    mul_777: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_79, view_118);  view_118 = None
    unsqueeze_199: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_777, 0);  mul_777 = None
    unsqueeze_200: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_199, 2);  unsqueeze_199 = None
    sub_130: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_117, unsqueeze_194);  view_117 = unsqueeze_194 = None
    mul_778: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_198);  sub_130 = unsqueeze_198 = None
    sub_131: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_225, mul_778);  view_225 = mul_778 = None
    sub_132: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(sub_131, unsqueeze_196);  sub_131 = unsqueeze_196 = None
    mul_779: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_200);  sub_132 = unsqueeze_200 = None
    mul_780: "f32[768]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_79);  sum_71 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_226: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_780, [768, 1, 1, 1]);  mul_780 = None
    mul_781: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_226, 0.02551551815399144);  view_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_227: "f32[768, 1536, 1, 1]" = torch.ops.aten.view.default(mul_779, [768, 1536, 1, 1]);  mul_779 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_782: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_189, 0.9128709291752768);  getitem_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_783: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_782, 1.7015043497085571);  mul_782 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_784: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_81, 0.7071067811865476)
    erf_68: "f32[4, 1536, 12, 12]" = torch.ops.aten.erf.default(mul_784);  mul_784 = None
    add_160: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(erf_68, 1);  erf_68 = None
    mul_785: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_160, 0.5);  add_160 = None
    mul_786: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_81, add_81)
    mul_787: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_786, -0.5);  mul_786 = None
    exp_16: "f32[4, 1536, 12, 12]" = torch.ops.aten.exp.default(mul_787);  mul_787 = None
    mul_788: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
    mul_789: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_81, mul_788);  add_81 = mul_788 = None
    add_161: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_785, mul_789);  mul_785 = mul_789 = None
    mul_790: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_783, add_161);  mul_783 = add_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    add_162: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_709, mul_790);  mul_709 = mul_790 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_791: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_162, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    mul_792: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_791, clone_7);  clone_7 = None
    mul_793: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_791, primals_125);  mul_791 = primals_125 = None
    sum_72: "f32[]" = torch.ops.aten.sum.default(mul_792);  mul_792 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_794: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_793, 2.0);  mul_793 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_795: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_794, convolution_52);  convolution_52 = None
    mul_796: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_794, sigmoid_7);  mul_794 = sigmoid_7 = None
    sum_73: "f32[4, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_795, [2, 3], True);  mul_795 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_40: "f32[4, 1536, 1, 1]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    sub_133: "f32[4, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_40)
    mul_797: "f32[4, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(alias_40, sub_133);  alias_40 = sub_133 = None
    mul_798: "f32[4, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_73, mul_797);  sum_73 = mul_797 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_74: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_798, [0, 2, 3])
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_798, relu_7, primals_214, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_798 = primals_214 = None
    getitem_192: "f32[4, 768, 1, 1]" = convolution_backward_26[0]
    getitem_193: "f32[1536, 768, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_42: "f32[4, 768, 1, 1]" = torch.ops.aten.alias.default(relu_7);  relu_7 = None
    alias_43: "f32[4, 768, 1, 1]" = torch.ops.aten.alias.default(alias_42);  alias_42 = None
    le_4: "b8[4, 768, 1, 1]" = torch.ops.aten.le.Scalar(alias_43, 0);  alias_43 = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_4: "f32[4, 768, 1, 1]" = torch.ops.aten.where.self(le_4, scalar_tensor_4, getitem_192);  le_4 = scalar_tensor_4 = getitem_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_75: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(where_4, mean_7, primals_212, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_4 = mean_7 = primals_212 = None
    getitem_195: "f32[4, 1536, 1, 1]" = convolution_backward_27[0]
    getitem_196: "f32[768, 1536, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_5: "f32[4, 1536, 12, 12]" = torch.ops.aten.expand.default(getitem_195, [4, 1536, 12, 12]);  getitem_195 = None
    div_5: "f32[4, 1536, 12, 12]" = torch.ops.aten.div.Scalar(expand_5, 144);  expand_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_163: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_796, div_5);  mul_796 = div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_76: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_163, [0, 2, 3])
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(add_163, mul_289, view_116, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_163 = mul_289 = view_116 = None
    getitem_198: "f32[4, 768, 12, 12]" = convolution_backward_28[0]
    getitem_199: "f32[1536, 768, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_228: "f32[1, 1536, 768]" = torch.ops.aten.view.default(getitem_199, [1, 1536, 768]);  getitem_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_201: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_76, 0);  squeeze_76 = None
    unsqueeze_202: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_201, 2);  unsqueeze_201 = None
    sum_77: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_228, [0, 2])
    sub_134: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_114, unsqueeze_202)
    mul_799: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(view_228, sub_134);  sub_134 = None
    sum_78: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_799, [0, 2]);  mul_799 = None
    mul_800: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_77, 0.0013020833333333333);  sum_77 = None
    unsqueeze_203: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_800, 0);  mul_800 = None
    unsqueeze_204: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 2);  unsqueeze_203 = None
    mul_801: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_78, 0.0013020833333333333)
    mul_802: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_77, squeeze_77)
    mul_803: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_801, mul_802);  mul_801 = mul_802 = None
    unsqueeze_205: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_803, 0);  mul_803 = None
    unsqueeze_206: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 2);  unsqueeze_205 = None
    mul_804: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_77, view_115);  view_115 = None
    unsqueeze_207: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_804, 0);  mul_804 = None
    unsqueeze_208: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_207, 2);  unsqueeze_207 = None
    sub_135: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_114, unsqueeze_202);  view_114 = unsqueeze_202 = None
    mul_805: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_206);  sub_135 = unsqueeze_206 = None
    sub_136: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_228, mul_805);  view_228 = mul_805 = None
    sub_137: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(sub_136, unsqueeze_204);  sub_136 = unsqueeze_204 = None
    mul_806: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_208);  sub_137 = unsqueeze_208 = None
    mul_807: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_78, squeeze_77);  sum_78 = squeeze_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_229: "f32[1536, 1, 1, 1]" = torch.ops.aten.view.default(mul_807, [1536, 1, 1, 1]);  mul_807 = None
    mul_808: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_229, 0.03608439182435161);  view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_230: "f32[1536, 768, 1, 1]" = torch.ops.aten.view.default(mul_806, [1536, 768, 1, 1]);  mul_806 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_809: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_198, 1.7015043497085571);  getitem_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_810: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_51, 0.7071067811865476)
    erf_69: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_810);  mul_810 = None
    add_164: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_69, 1);  erf_69 = None
    mul_811: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(add_164, 0.5);  add_164 = None
    mul_812: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_51, convolution_51)
    mul_813: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_812, -0.5);  mul_812 = None
    exp_17: "f32[4, 768, 12, 12]" = torch.ops.aten.exp.default(mul_813);  mul_813 = None
    mul_814: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
    mul_815: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_51, mul_814);  convolution_51 = mul_814 = None
    add_165: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(mul_811, mul_815);  mul_811 = mul_815 = None
    mul_816: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_809, add_165);  mul_809 = add_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_79: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_816, [0, 2, 3])
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_816, mul_282, view_113, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_816 = mul_282 = view_113 = None
    getitem_201: "f32[4, 768, 12, 12]" = convolution_backward_29[0]
    getitem_202: "f32[768, 128, 3, 3]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_231: "f32[1, 768, 1152]" = torch.ops.aten.view.default(getitem_202, [1, 768, 1152]);  getitem_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_209: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_74, 0);  squeeze_74 = None
    unsqueeze_210: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 2);  unsqueeze_209 = None
    sum_80: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_231, [0, 2])
    sub_138: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_111, unsqueeze_210)
    mul_817: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(view_231, sub_138);  sub_138 = None
    sum_81: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_817, [0, 2]);  mul_817 = None
    mul_818: "f32[768]" = torch.ops.aten.mul.Tensor(sum_80, 0.0008680555555555555);  sum_80 = None
    unsqueeze_211: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_818, 0);  mul_818 = None
    unsqueeze_212: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 2);  unsqueeze_211 = None
    mul_819: "f32[768]" = torch.ops.aten.mul.Tensor(sum_81, 0.0008680555555555555)
    mul_820: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_75, squeeze_75)
    mul_821: "f32[768]" = torch.ops.aten.mul.Tensor(mul_819, mul_820);  mul_819 = mul_820 = None
    unsqueeze_213: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_821, 0);  mul_821 = None
    unsqueeze_214: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 2);  unsqueeze_213 = None
    mul_822: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_75, view_112);  view_112 = None
    unsqueeze_215: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_822, 0);  mul_822 = None
    unsqueeze_216: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 2);  unsqueeze_215 = None
    sub_139: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_111, unsqueeze_210);  view_111 = unsqueeze_210 = None
    mul_823: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_214);  sub_139 = unsqueeze_214 = None
    sub_140: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_231, mul_823);  view_231 = mul_823 = None
    sub_141: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_140, unsqueeze_212);  sub_140 = unsqueeze_212 = None
    mul_824: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_216);  sub_141 = unsqueeze_216 = None
    mul_825: "f32[768]" = torch.ops.aten.mul.Tensor(sum_81, squeeze_75);  sum_81 = squeeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_232: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_825, [768, 1, 1, 1]);  mul_825 = None
    mul_826: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_232, 0.02946278254943948);  view_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_233: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_824, [768, 128, 3, 3]);  mul_824 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_827: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_201, 1.7015043497085571);  getitem_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_828: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_50, 0.7071067811865476)
    erf_70: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_828);  mul_828 = None
    add_166: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_70, 1);  erf_70 = None
    mul_829: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(add_166, 0.5);  add_166 = None
    mul_830: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_50, convolution_50)
    mul_831: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_830, -0.5);  mul_830 = None
    exp_18: "f32[4, 768, 12, 12]" = torch.ops.aten.exp.default(mul_831);  mul_831 = None
    mul_832: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
    mul_833: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_50, mul_832);  convolution_50 = mul_832 = None
    add_167: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(mul_829, mul_833);  mul_829 = mul_833 = None
    mul_834: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_827, add_167);  mul_827 = add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_82: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_834, [0, 2, 3])
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_834, mul_275, view_110, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_834 = mul_275 = view_110 = None
    getitem_204: "f32[4, 768, 12, 12]" = convolution_backward_30[0]
    getitem_205: "f32[768, 128, 3, 3]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_234: "f32[1, 768, 1152]" = torch.ops.aten.view.default(getitem_205, [1, 768, 1152]);  getitem_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_217: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_218: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 2);  unsqueeze_217 = None
    sum_83: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_234, [0, 2])
    sub_142: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_108, unsqueeze_218)
    mul_835: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(view_234, sub_142);  sub_142 = None
    sum_84: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_835, [0, 2]);  mul_835 = None
    mul_836: "f32[768]" = torch.ops.aten.mul.Tensor(sum_83, 0.0008680555555555555);  sum_83 = None
    unsqueeze_219: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_836, 0);  mul_836 = None
    unsqueeze_220: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_219, 2);  unsqueeze_219 = None
    mul_837: "f32[768]" = torch.ops.aten.mul.Tensor(sum_84, 0.0008680555555555555)
    mul_838: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_839: "f32[768]" = torch.ops.aten.mul.Tensor(mul_837, mul_838);  mul_837 = mul_838 = None
    unsqueeze_221: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_839, 0);  mul_839 = None
    unsqueeze_222: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 2);  unsqueeze_221 = None
    mul_840: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_73, view_109);  view_109 = None
    unsqueeze_223: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_840, 0);  mul_840 = None
    unsqueeze_224: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 2);  unsqueeze_223 = None
    sub_143: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_108, unsqueeze_218);  view_108 = unsqueeze_218 = None
    mul_841: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_222);  sub_143 = unsqueeze_222 = None
    sub_144: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_234, mul_841);  view_234 = mul_841 = None
    sub_145: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_144, unsqueeze_220);  sub_144 = unsqueeze_220 = None
    mul_842: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_224);  sub_145 = unsqueeze_224 = None
    mul_843: "f32[768]" = torch.ops.aten.mul.Tensor(sum_84, squeeze_73);  sum_84 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_235: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_843, [768, 1, 1, 1]);  mul_843 = None
    mul_844: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_235, 0.02946278254943948);  view_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_236: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_842, [768, 128, 3, 3]);  mul_842 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_845: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_204, 1.7015043497085571);  getitem_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_846: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_49, 0.7071067811865476)
    erf_71: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_846);  mul_846 = None
    add_168: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_71, 1);  erf_71 = None
    mul_847: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(add_168, 0.5);  add_168 = None
    mul_848: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_49, convolution_49)
    mul_849: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_848, -0.5);  mul_848 = None
    exp_19: "f32[4, 768, 12, 12]" = torch.ops.aten.exp.default(mul_849);  mul_849 = None
    mul_850: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
    mul_851: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_49, mul_850);  convolution_49 = mul_850 = None
    add_169: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(mul_847, mul_851);  mul_847 = mul_851 = None
    mul_852: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_845, add_169);  mul_845 = add_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_85: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_852, [0, 2, 3])
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_852, mul_268, view_107, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_852 = mul_268 = view_107 = None
    getitem_207: "f32[4, 1536, 12, 12]" = convolution_backward_31[0]
    getitem_208: "f32[768, 1536, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_237: "f32[1, 768, 1536]" = torch.ops.aten.view.default(getitem_208, [1, 768, 1536]);  getitem_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_225: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_70, 0);  squeeze_70 = None
    unsqueeze_226: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_225, 2);  unsqueeze_225 = None
    sum_86: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_237, [0, 2])
    sub_146: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_105, unsqueeze_226)
    mul_853: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(view_237, sub_146);  sub_146 = None
    sum_87: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_853, [0, 2]);  mul_853 = None
    mul_854: "f32[768]" = torch.ops.aten.mul.Tensor(sum_86, 0.0006510416666666666);  sum_86 = None
    unsqueeze_227: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_854, 0);  mul_854 = None
    unsqueeze_228: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 2);  unsqueeze_227 = None
    mul_855: "f32[768]" = torch.ops.aten.mul.Tensor(sum_87, 0.0006510416666666666)
    mul_856: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_71, squeeze_71)
    mul_857: "f32[768]" = torch.ops.aten.mul.Tensor(mul_855, mul_856);  mul_855 = mul_856 = None
    unsqueeze_229: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_857, 0);  mul_857 = None
    unsqueeze_230: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 2);  unsqueeze_229 = None
    mul_858: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_71, view_106);  view_106 = None
    unsqueeze_231: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_858, 0);  mul_858 = None
    unsqueeze_232: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_231, 2);  unsqueeze_231 = None
    sub_147: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_105, unsqueeze_226);  view_105 = unsqueeze_226 = None
    mul_859: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_230);  sub_147 = unsqueeze_230 = None
    sub_148: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_237, mul_859);  view_237 = mul_859 = None
    sub_149: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(sub_148, unsqueeze_228);  sub_148 = unsqueeze_228 = None
    mul_860: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_232);  sub_149 = unsqueeze_232 = None
    mul_861: "f32[768]" = torch.ops.aten.mul.Tensor(sum_87, squeeze_71);  sum_87 = squeeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_238: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_861, [768, 1, 1, 1]);  mul_861 = None
    mul_862: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_238, 0.02551551815399144);  view_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_239: "f32[768, 1536, 1, 1]" = torch.ops.aten.view.default(mul_860, [768, 1536, 1, 1]);  mul_860 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_863: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_207, 0.9284766908852592);  getitem_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_864: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_863, 1.7015043497085571);  mul_863 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_865: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_72, 0.7071067811865476)
    erf_72: "f32[4, 1536, 12, 12]" = torch.ops.aten.erf.default(mul_865);  mul_865 = None
    add_170: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(erf_72, 1);  erf_72 = None
    mul_866: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_170, 0.5);  add_170 = None
    mul_867: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_72, add_72)
    mul_868: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_867, -0.5);  mul_867 = None
    exp_20: "f32[4, 1536, 12, 12]" = torch.ops.aten.exp.default(mul_868);  mul_868 = None
    mul_869: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
    mul_870: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_72, mul_869);  add_72 = mul_869 = None
    add_171: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_866, mul_870);  mul_866 = mul_870 = None
    mul_871: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_864, add_171);  mul_864 = add_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    add_172: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(add_162, mul_871);  add_162 = mul_871 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_872: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_172, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    mul_873: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_872, clone_6);  clone_6 = None
    mul_874: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_872, primals_112);  mul_872 = primals_112 = None
    sum_88: "f32[]" = torch.ops.aten.sum.default(mul_873);  mul_873 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_875: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_874, 2.0);  mul_874 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_876: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_875, convolution_46);  convolution_46 = None
    mul_877: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_875, sigmoid_6);  mul_875 = sigmoid_6 = None
    sum_89: "f32[4, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_876, [2, 3], True);  mul_876 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_44: "f32[4, 1536, 1, 1]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    sub_150: "f32[4, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_44)
    mul_878: "f32[4, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(alias_44, sub_150);  alias_44 = sub_150 = None
    mul_879: "f32[4, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_89, mul_878);  sum_89 = mul_878 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_90: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_879, [0, 2, 3])
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_879, relu_6, primals_210, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_879 = primals_210 = None
    getitem_210: "f32[4, 768, 1, 1]" = convolution_backward_32[0]
    getitem_211: "f32[1536, 768, 1, 1]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_46: "f32[4, 768, 1, 1]" = torch.ops.aten.alias.default(relu_6);  relu_6 = None
    alias_47: "f32[4, 768, 1, 1]" = torch.ops.aten.alias.default(alias_46);  alias_46 = None
    le_5: "b8[4, 768, 1, 1]" = torch.ops.aten.le.Scalar(alias_47, 0);  alias_47 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_5: "f32[4, 768, 1, 1]" = torch.ops.aten.where.self(le_5, scalar_tensor_5, getitem_210);  le_5 = scalar_tensor_5 = getitem_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_91: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(where_5, mean_6, primals_208, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_5 = mean_6 = primals_208 = None
    getitem_213: "f32[4, 1536, 1, 1]" = convolution_backward_33[0]
    getitem_214: "f32[768, 1536, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_6: "f32[4, 1536, 12, 12]" = torch.ops.aten.expand.default(getitem_213, [4, 1536, 12, 12]);  getitem_213 = None
    div_6: "f32[4, 1536, 12, 12]" = torch.ops.aten.div.Scalar(expand_6, 144);  expand_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_173: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_877, div_6);  mul_877 = div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_92: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_173, [0, 2, 3])
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(add_173, mul_256, view_104, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_173 = mul_256 = view_104 = None
    getitem_216: "f32[4, 768, 12, 12]" = convolution_backward_34[0]
    getitem_217: "f32[1536, 768, 1, 1]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_240: "f32[1, 1536, 768]" = torch.ops.aten.view.default(getitem_217, [1, 1536, 768]);  getitem_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_233: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_68, 0);  squeeze_68 = None
    unsqueeze_234: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 2);  unsqueeze_233 = None
    sum_93: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_240, [0, 2])
    sub_151: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_102, unsqueeze_234)
    mul_880: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(view_240, sub_151);  sub_151 = None
    sum_94: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_880, [0, 2]);  mul_880 = None
    mul_881: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_93, 0.0013020833333333333);  sum_93 = None
    unsqueeze_235: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_881, 0);  mul_881 = None
    unsqueeze_236: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 2);  unsqueeze_235 = None
    mul_882: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_94, 0.0013020833333333333)
    mul_883: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_69, squeeze_69)
    mul_884: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_882, mul_883);  mul_882 = mul_883 = None
    unsqueeze_237: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_884, 0);  mul_884 = None
    unsqueeze_238: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_237, 2);  unsqueeze_237 = None
    mul_885: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_69, view_103);  view_103 = None
    unsqueeze_239: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_885, 0);  mul_885 = None
    unsqueeze_240: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 2);  unsqueeze_239 = None
    sub_152: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_102, unsqueeze_234);  view_102 = unsqueeze_234 = None
    mul_886: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_238);  sub_152 = unsqueeze_238 = None
    sub_153: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_240, mul_886);  view_240 = mul_886 = None
    sub_154: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(sub_153, unsqueeze_236);  sub_153 = unsqueeze_236 = None
    mul_887: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_240);  sub_154 = unsqueeze_240 = None
    mul_888: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_94, squeeze_69);  sum_94 = squeeze_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_241: "f32[1536, 1, 1, 1]" = torch.ops.aten.view.default(mul_888, [1536, 1, 1, 1]);  mul_888 = None
    mul_889: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_241, 0.03608439182435161);  view_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_242: "f32[1536, 768, 1, 1]" = torch.ops.aten.view.default(mul_887, [1536, 768, 1, 1]);  mul_887 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_890: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_216, 1.7015043497085571);  getitem_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_891: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_45, 0.7071067811865476)
    erf_73: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_891);  mul_891 = None
    add_174: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_73, 1);  erf_73 = None
    mul_892: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(add_174, 0.5);  add_174 = None
    mul_893: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_45, convolution_45)
    mul_894: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_893, -0.5);  mul_893 = None
    exp_21: "f32[4, 768, 12, 12]" = torch.ops.aten.exp.default(mul_894);  mul_894 = None
    mul_895: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
    mul_896: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_45, mul_895);  convolution_45 = mul_895 = None
    add_175: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(mul_892, mul_896);  mul_892 = mul_896 = None
    mul_897: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_890, add_175);  mul_890 = add_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_95: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_897, [0, 2, 3])
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_897, mul_249, view_101, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_897 = mul_249 = view_101 = None
    getitem_219: "f32[4, 768, 12, 12]" = convolution_backward_35[0]
    getitem_220: "f32[768, 128, 3, 3]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_243: "f32[1, 768, 1152]" = torch.ops.aten.view.default(getitem_220, [1, 768, 1152]);  getitem_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_241: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_242: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 2);  unsqueeze_241 = None
    sum_96: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_243, [0, 2])
    sub_155: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_99, unsqueeze_242)
    mul_898: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(view_243, sub_155);  sub_155 = None
    sum_97: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_898, [0, 2]);  mul_898 = None
    mul_899: "f32[768]" = torch.ops.aten.mul.Tensor(sum_96, 0.0008680555555555555);  sum_96 = None
    unsqueeze_243: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_899, 0);  mul_899 = None
    unsqueeze_244: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_243, 2);  unsqueeze_243 = None
    mul_900: "f32[768]" = torch.ops.aten.mul.Tensor(sum_97, 0.0008680555555555555)
    mul_901: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_902: "f32[768]" = torch.ops.aten.mul.Tensor(mul_900, mul_901);  mul_900 = mul_901 = None
    unsqueeze_245: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_902, 0);  mul_902 = None
    unsqueeze_246: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 2);  unsqueeze_245 = None
    mul_903: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_67, view_100);  view_100 = None
    unsqueeze_247: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_903, 0);  mul_903 = None
    unsqueeze_248: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 2);  unsqueeze_247 = None
    sub_156: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_99, unsqueeze_242);  view_99 = unsqueeze_242 = None
    mul_904: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_246);  sub_156 = unsqueeze_246 = None
    sub_157: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_243, mul_904);  view_243 = mul_904 = None
    sub_158: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_157, unsqueeze_244);  sub_157 = unsqueeze_244 = None
    mul_905: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_158, unsqueeze_248);  sub_158 = unsqueeze_248 = None
    mul_906: "f32[768]" = torch.ops.aten.mul.Tensor(sum_97, squeeze_67);  sum_97 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_244: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_906, [768, 1, 1, 1]);  mul_906 = None
    mul_907: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_244, 0.02946278254943948);  view_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_245: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_905, [768, 128, 3, 3]);  mul_905 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_908: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_219, 1.7015043497085571);  getitem_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_909: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_44, 0.7071067811865476)
    erf_74: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_909);  mul_909 = None
    add_176: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_74, 1);  erf_74 = None
    mul_910: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(add_176, 0.5);  add_176 = None
    mul_911: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_44, convolution_44)
    mul_912: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_911, -0.5);  mul_911 = None
    exp_22: "f32[4, 768, 12, 12]" = torch.ops.aten.exp.default(mul_912);  mul_912 = None
    mul_913: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_914: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_44, mul_913);  convolution_44 = mul_913 = None
    add_177: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(mul_910, mul_914);  mul_910 = mul_914 = None
    mul_915: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_908, add_177);  mul_908 = add_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_98: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_915, [0, 2, 3])
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_915, mul_242, view_98, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_915 = mul_242 = view_98 = None
    getitem_222: "f32[4, 768, 12, 12]" = convolution_backward_36[0]
    getitem_223: "f32[768, 128, 3, 3]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_246: "f32[1, 768, 1152]" = torch.ops.aten.view.default(getitem_223, [1, 768, 1152]);  getitem_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_249: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_64, 0);  squeeze_64 = None
    unsqueeze_250: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_249, 2);  unsqueeze_249 = None
    sum_99: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_246, [0, 2])
    sub_159: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_96, unsqueeze_250)
    mul_916: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(view_246, sub_159);  sub_159 = None
    sum_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_916, [0, 2]);  mul_916 = None
    mul_917: "f32[768]" = torch.ops.aten.mul.Tensor(sum_99, 0.0008680555555555555);  sum_99 = None
    unsqueeze_251: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_917, 0);  mul_917 = None
    unsqueeze_252: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 2);  unsqueeze_251 = None
    mul_918: "f32[768]" = torch.ops.aten.mul.Tensor(sum_100, 0.0008680555555555555)
    mul_919: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_65, squeeze_65)
    mul_920: "f32[768]" = torch.ops.aten.mul.Tensor(mul_918, mul_919);  mul_918 = mul_919 = None
    unsqueeze_253: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_920, 0);  mul_920 = None
    unsqueeze_254: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 2);  unsqueeze_253 = None
    mul_921: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_65, view_97);  view_97 = None
    unsqueeze_255: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_921, 0);  mul_921 = None
    unsqueeze_256: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 2);  unsqueeze_255 = None
    sub_160: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_96, unsqueeze_250);  view_96 = unsqueeze_250 = None
    mul_922: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_254);  sub_160 = unsqueeze_254 = None
    sub_161: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_246, mul_922);  view_246 = mul_922 = None
    sub_162: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_161, unsqueeze_252);  sub_161 = unsqueeze_252 = None
    mul_923: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_256);  sub_162 = unsqueeze_256 = None
    mul_924: "f32[768]" = torch.ops.aten.mul.Tensor(sum_100, squeeze_65);  sum_100 = squeeze_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_247: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_924, [768, 1, 1, 1]);  mul_924 = None
    mul_925: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_247, 0.02946278254943948);  view_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_248: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_923, [768, 128, 3, 3]);  mul_923 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_926: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_222, 1.7015043497085571);  getitem_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_927: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_43, 0.7071067811865476)
    erf_75: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_927);  mul_927 = None
    add_178: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_75, 1);  erf_75 = None
    mul_928: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(add_178, 0.5);  add_178 = None
    mul_929: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_43, convolution_43)
    mul_930: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_929, -0.5);  mul_929 = None
    exp_23: "f32[4, 768, 12, 12]" = torch.ops.aten.exp.default(mul_930);  mul_930 = None
    mul_931: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_932: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_43, mul_931);  convolution_43 = mul_931 = None
    add_179: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(mul_928, mul_932);  mul_928 = mul_932 = None
    mul_933: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_926, add_179);  mul_926 = add_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_101: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_933, [0, 2, 3])
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_933, mul_235, view_95, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_933 = mul_235 = view_95 = None
    getitem_225: "f32[4, 1536, 12, 12]" = convolution_backward_37[0]
    getitem_226: "f32[768, 1536, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_249: "f32[1, 768, 1536]" = torch.ops.aten.view.default(getitem_226, [1, 768, 1536]);  getitem_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_257: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_62, 0);  squeeze_62 = None
    unsqueeze_258: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 2);  unsqueeze_257 = None
    sum_102: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_249, [0, 2])
    sub_163: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_93, unsqueeze_258)
    mul_934: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(view_249, sub_163);  sub_163 = None
    sum_103: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_934, [0, 2]);  mul_934 = None
    mul_935: "f32[768]" = torch.ops.aten.mul.Tensor(sum_102, 0.0006510416666666666);  sum_102 = None
    unsqueeze_259: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_935, 0);  mul_935 = None
    unsqueeze_260: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 2);  unsqueeze_259 = None
    mul_936: "f32[768]" = torch.ops.aten.mul.Tensor(sum_103, 0.0006510416666666666)
    mul_937: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_63, squeeze_63)
    mul_938: "f32[768]" = torch.ops.aten.mul.Tensor(mul_936, mul_937);  mul_936 = mul_937 = None
    unsqueeze_261: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_938, 0);  mul_938 = None
    unsqueeze_262: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 2);  unsqueeze_261 = None
    mul_939: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_63, view_94);  view_94 = None
    unsqueeze_263: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_939, 0);  mul_939 = None
    unsqueeze_264: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 2);  unsqueeze_263 = None
    sub_164: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_93, unsqueeze_258);  view_93 = unsqueeze_258 = None
    mul_940: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_262);  sub_164 = unsqueeze_262 = None
    sub_165: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_249, mul_940);  view_249 = mul_940 = None
    sub_166: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(sub_165, unsqueeze_260);  sub_165 = unsqueeze_260 = None
    mul_941: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_264);  sub_166 = unsqueeze_264 = None
    mul_942: "f32[768]" = torch.ops.aten.mul.Tensor(sum_103, squeeze_63);  sum_103 = squeeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_250: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_942, [768, 1, 1, 1]);  mul_942 = None
    mul_943: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_250, 0.02551551815399144);  view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_251: "f32[768, 1536, 1, 1]" = torch.ops.aten.view.default(mul_941, [768, 1536, 1, 1]);  mul_941 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_944: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_225, 0.9449111825230679);  getitem_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_945: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_944, 1.7015043497085571);  mul_944 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_946: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_63, 0.7071067811865476)
    erf_76: "f32[4, 1536, 12, 12]" = torch.ops.aten.erf.default(mul_946);  mul_946 = None
    add_180: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(erf_76, 1);  erf_76 = None
    mul_947: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_180, 0.5);  add_180 = None
    mul_948: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_63, add_63)
    mul_949: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_948, -0.5);  mul_948 = None
    exp_24: "f32[4, 1536, 12, 12]" = torch.ops.aten.exp.default(mul_949);  mul_949 = None
    mul_950: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_951: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_63, mul_950);  add_63 = mul_950 = None
    add_181: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_947, mul_951);  mul_947 = mul_951 = None
    mul_952: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_945, add_181);  mul_945 = add_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    add_182: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(add_172, mul_952);  add_172 = mul_952 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_953: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_182, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    mul_954: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_953, clone_5);  clone_5 = None
    mul_955: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_953, primals_99);  mul_953 = primals_99 = None
    sum_104: "f32[]" = torch.ops.aten.sum.default(mul_954);  mul_954 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_956: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_955, 2.0);  mul_955 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_957: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_956, convolution_40);  convolution_40 = None
    mul_958: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_956, sigmoid_5);  mul_956 = sigmoid_5 = None
    sum_105: "f32[4, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_957, [2, 3], True);  mul_957 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_48: "f32[4, 1536, 1, 1]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    sub_167: "f32[4, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_48)
    mul_959: "f32[4, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(alias_48, sub_167);  alias_48 = sub_167 = None
    mul_960: "f32[4, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_105, mul_959);  sum_105 = mul_959 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_106: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_960, [0, 2, 3])
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_960, relu_5, primals_206, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_960 = primals_206 = None
    getitem_228: "f32[4, 768, 1, 1]" = convolution_backward_38[0]
    getitem_229: "f32[1536, 768, 1, 1]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_50: "f32[4, 768, 1, 1]" = torch.ops.aten.alias.default(relu_5);  relu_5 = None
    alias_51: "f32[4, 768, 1, 1]" = torch.ops.aten.alias.default(alias_50);  alias_50 = None
    le_6: "b8[4, 768, 1, 1]" = torch.ops.aten.le.Scalar(alias_51, 0);  alias_51 = None
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_6: "f32[4, 768, 1, 1]" = torch.ops.aten.where.self(le_6, scalar_tensor_6, getitem_228);  le_6 = scalar_tensor_6 = getitem_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_107: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(where_6, mean_5, primals_204, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_6 = mean_5 = primals_204 = None
    getitem_231: "f32[4, 1536, 1, 1]" = convolution_backward_39[0]
    getitem_232: "f32[768, 1536, 1, 1]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_7: "f32[4, 1536, 12, 12]" = torch.ops.aten.expand.default(getitem_231, [4, 1536, 12, 12]);  getitem_231 = None
    div_7: "f32[4, 1536, 12, 12]" = torch.ops.aten.div.Scalar(expand_7, 144);  expand_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_183: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_958, div_7);  mul_958 = div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_108: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_183, [0, 2, 3])
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(add_183, mul_223, view_92, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_183 = mul_223 = view_92 = None
    getitem_234: "f32[4, 768, 12, 12]" = convolution_backward_40[0]
    getitem_235: "f32[1536, 768, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_252: "f32[1, 1536, 768]" = torch.ops.aten.view.default(getitem_235, [1, 1536, 768]);  getitem_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_265: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_266: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 2);  unsqueeze_265 = None
    sum_109: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_252, [0, 2])
    sub_168: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_90, unsqueeze_266)
    mul_961: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(view_252, sub_168);  sub_168 = None
    sum_110: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_961, [0, 2]);  mul_961 = None
    mul_962: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_109, 0.0013020833333333333);  sum_109 = None
    unsqueeze_267: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_962, 0);  mul_962 = None
    unsqueeze_268: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 2);  unsqueeze_267 = None
    mul_963: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_110, 0.0013020833333333333)
    mul_964: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_965: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_963, mul_964);  mul_963 = mul_964 = None
    unsqueeze_269: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_965, 0);  mul_965 = None
    unsqueeze_270: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 2);  unsqueeze_269 = None
    mul_966: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_61, view_91);  view_91 = None
    unsqueeze_271: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_966, 0);  mul_966 = None
    unsqueeze_272: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 2);  unsqueeze_271 = None
    sub_169: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_90, unsqueeze_266);  view_90 = unsqueeze_266 = None
    mul_967: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_270);  sub_169 = unsqueeze_270 = None
    sub_170: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_252, mul_967);  view_252 = mul_967 = None
    sub_171: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(sub_170, unsqueeze_268);  sub_170 = unsqueeze_268 = None
    mul_968: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_272);  sub_171 = unsqueeze_272 = None
    mul_969: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_110, squeeze_61);  sum_110 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_253: "f32[1536, 1, 1, 1]" = torch.ops.aten.view.default(mul_969, [1536, 1, 1, 1]);  mul_969 = None
    mul_970: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_253, 0.03608439182435161);  view_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_254: "f32[1536, 768, 1, 1]" = torch.ops.aten.view.default(mul_968, [1536, 768, 1, 1]);  mul_968 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_971: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_234, 1.7015043497085571);  getitem_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_972: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_39, 0.7071067811865476)
    erf_77: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_972);  mul_972 = None
    add_184: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_77, 1);  erf_77 = None
    mul_973: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(add_184, 0.5);  add_184 = None
    mul_974: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_39, convolution_39)
    mul_975: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_974, -0.5);  mul_974 = None
    exp_25: "f32[4, 768, 12, 12]" = torch.ops.aten.exp.default(mul_975);  mul_975 = None
    mul_976: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
    mul_977: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_39, mul_976);  convolution_39 = mul_976 = None
    add_185: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(mul_973, mul_977);  mul_973 = mul_977 = None
    mul_978: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_971, add_185);  mul_971 = add_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_111: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_978, [0, 2, 3])
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_978, mul_216, view_89, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_978 = mul_216 = view_89 = None
    getitem_237: "f32[4, 768, 12, 12]" = convolution_backward_41[0]
    getitem_238: "f32[768, 128, 3, 3]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_255: "f32[1, 768, 1152]" = torch.ops.aten.view.default(getitem_238, [1, 768, 1152]);  getitem_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_273: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_58, 0);  squeeze_58 = None
    unsqueeze_274: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 2);  unsqueeze_273 = None
    sum_112: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_255, [0, 2])
    sub_172: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_87, unsqueeze_274)
    mul_979: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(view_255, sub_172);  sub_172 = None
    sum_113: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_979, [0, 2]);  mul_979 = None
    mul_980: "f32[768]" = torch.ops.aten.mul.Tensor(sum_112, 0.0008680555555555555);  sum_112 = None
    unsqueeze_275: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_980, 0);  mul_980 = None
    unsqueeze_276: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 2);  unsqueeze_275 = None
    mul_981: "f32[768]" = torch.ops.aten.mul.Tensor(sum_113, 0.0008680555555555555)
    mul_982: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_59, squeeze_59)
    mul_983: "f32[768]" = torch.ops.aten.mul.Tensor(mul_981, mul_982);  mul_981 = mul_982 = None
    unsqueeze_277: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_983, 0);  mul_983 = None
    unsqueeze_278: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 2);  unsqueeze_277 = None
    mul_984: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_59, view_88);  view_88 = None
    unsqueeze_279: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_984, 0);  mul_984 = None
    unsqueeze_280: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 2);  unsqueeze_279 = None
    sub_173: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_87, unsqueeze_274);  view_87 = unsqueeze_274 = None
    mul_985: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_278);  sub_173 = unsqueeze_278 = None
    sub_174: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_255, mul_985);  view_255 = mul_985 = None
    sub_175: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_174, unsqueeze_276);  sub_174 = unsqueeze_276 = None
    mul_986: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_280);  sub_175 = unsqueeze_280 = None
    mul_987: "f32[768]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_59);  sum_113 = squeeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_256: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_987, [768, 1, 1, 1]);  mul_987 = None
    mul_988: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_256, 0.02946278254943948);  view_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_257: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_986, [768, 128, 3, 3]);  mul_986 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_989: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_237, 1.7015043497085571);  getitem_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_990: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_38, 0.7071067811865476)
    erf_78: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_990);  mul_990 = None
    add_186: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_78, 1);  erf_78 = None
    mul_991: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(add_186, 0.5);  add_186 = None
    mul_992: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_38, convolution_38)
    mul_993: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_992, -0.5);  mul_992 = None
    exp_26: "f32[4, 768, 12, 12]" = torch.ops.aten.exp.default(mul_993);  mul_993 = None
    mul_994: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_995: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_38, mul_994);  convolution_38 = mul_994 = None
    add_187: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(mul_991, mul_995);  mul_991 = mul_995 = None
    mul_996: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_989, add_187);  mul_989 = add_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_114: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_996, [0, 2, 3])
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_996, mul_209, view_86, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_996 = mul_209 = view_86 = None
    getitem_240: "f32[4, 768, 12, 12]" = convolution_backward_42[0]
    getitem_241: "f32[768, 128, 3, 3]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_258: "f32[1, 768, 1152]" = torch.ops.aten.view.default(getitem_241, [1, 768, 1152]);  getitem_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_281: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_56, 0);  squeeze_56 = None
    unsqueeze_282: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 2);  unsqueeze_281 = None
    sum_115: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_258, [0, 2])
    sub_176: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_84, unsqueeze_282)
    mul_997: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(view_258, sub_176);  sub_176 = None
    sum_116: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_997, [0, 2]);  mul_997 = None
    mul_998: "f32[768]" = torch.ops.aten.mul.Tensor(sum_115, 0.0008680555555555555);  sum_115 = None
    unsqueeze_283: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_998, 0);  mul_998 = None
    unsqueeze_284: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 2);  unsqueeze_283 = None
    mul_999: "f32[768]" = torch.ops.aten.mul.Tensor(sum_116, 0.0008680555555555555)
    mul_1000: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_57, squeeze_57)
    mul_1001: "f32[768]" = torch.ops.aten.mul.Tensor(mul_999, mul_1000);  mul_999 = mul_1000 = None
    unsqueeze_285: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1001, 0);  mul_1001 = None
    unsqueeze_286: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 2);  unsqueeze_285 = None
    mul_1002: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_57, view_85);  view_85 = None
    unsqueeze_287: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1002, 0);  mul_1002 = None
    unsqueeze_288: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 2);  unsqueeze_287 = None
    sub_177: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_84, unsqueeze_282);  view_84 = unsqueeze_282 = None
    mul_1003: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_286);  sub_177 = unsqueeze_286 = None
    sub_178: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_258, mul_1003);  view_258 = mul_1003 = None
    sub_179: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_178, unsqueeze_284);  sub_178 = unsqueeze_284 = None
    mul_1004: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_288);  sub_179 = unsqueeze_288 = None
    mul_1005: "f32[768]" = torch.ops.aten.mul.Tensor(sum_116, squeeze_57);  sum_116 = squeeze_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_259: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_1005, [768, 1, 1, 1]);  mul_1005 = None
    mul_1006: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_259, 0.02946278254943948);  view_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_260: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_1004, [768, 128, 3, 3]);  mul_1004 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1007: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_240, 1.7015043497085571);  getitem_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1008: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_37, 0.7071067811865476)
    erf_79: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_1008);  mul_1008 = None
    add_188: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_79, 1);  erf_79 = None
    mul_1009: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(add_188, 0.5);  add_188 = None
    mul_1010: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_37, convolution_37)
    mul_1011: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1010, -0.5);  mul_1010 = None
    exp_27: "f32[4, 768, 12, 12]" = torch.ops.aten.exp.default(mul_1011);  mul_1011 = None
    mul_1012: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
    mul_1013: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_37, mul_1012);  convolution_37 = mul_1012 = None
    add_189: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(mul_1009, mul_1013);  mul_1009 = mul_1013 = None
    mul_1014: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1007, add_189);  mul_1007 = add_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_117: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1014, [0, 2, 3])
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_1014, mul_202, view_83, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1014 = mul_202 = view_83 = None
    getitem_243: "f32[4, 1536, 12, 12]" = convolution_backward_43[0]
    getitem_244: "f32[768, 1536, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_261: "f32[1, 768, 1536]" = torch.ops.aten.view.default(getitem_244, [1, 768, 1536]);  getitem_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_289: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_290: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 2);  unsqueeze_289 = None
    sum_118: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_261, [0, 2])
    sub_180: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_81, unsqueeze_290)
    mul_1015: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(view_261, sub_180);  sub_180 = None
    sum_119: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1015, [0, 2]);  mul_1015 = None
    mul_1016: "f32[768]" = torch.ops.aten.mul.Tensor(sum_118, 0.0006510416666666666);  sum_118 = None
    unsqueeze_291: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1016, 0);  mul_1016 = None
    unsqueeze_292: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 2);  unsqueeze_291 = None
    mul_1017: "f32[768]" = torch.ops.aten.mul.Tensor(sum_119, 0.0006510416666666666)
    mul_1018: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_1019: "f32[768]" = torch.ops.aten.mul.Tensor(mul_1017, mul_1018);  mul_1017 = mul_1018 = None
    unsqueeze_293: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1019, 0);  mul_1019 = None
    unsqueeze_294: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 2);  unsqueeze_293 = None
    mul_1020: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_55, view_82);  view_82 = None
    unsqueeze_295: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1020, 0);  mul_1020 = None
    unsqueeze_296: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 2);  unsqueeze_295 = None
    sub_181: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_81, unsqueeze_290);  view_81 = unsqueeze_290 = None
    mul_1021: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_294);  sub_181 = unsqueeze_294 = None
    sub_182: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_261, mul_1021);  view_261 = mul_1021 = None
    sub_183: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(sub_182, unsqueeze_292);  sub_182 = unsqueeze_292 = None
    mul_1022: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_183, unsqueeze_296);  sub_183 = unsqueeze_296 = None
    mul_1023: "f32[768]" = torch.ops.aten.mul.Tensor(sum_119, squeeze_55);  sum_119 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_262: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_1023, [768, 1, 1, 1]);  mul_1023 = None
    mul_1024: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_262, 0.02551551815399144);  view_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_263: "f32[768, 1536, 1, 1]" = torch.ops.aten.view.default(mul_1022, [768, 1536, 1, 1]);  mul_1022 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_1025: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_243, 0.9622504486493761);  getitem_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1026: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1025, 1.7015043497085571);  mul_1025 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1027: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_54, 0.7071067811865476)
    erf_80: "f32[4, 1536, 12, 12]" = torch.ops.aten.erf.default(mul_1027);  mul_1027 = None
    add_190: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(erf_80, 1);  erf_80 = None
    mul_1028: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_190, 0.5);  add_190 = None
    mul_1029: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_54, add_54)
    mul_1030: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1029, -0.5);  mul_1029 = None
    exp_28: "f32[4, 1536, 12, 12]" = torch.ops.aten.exp.default(mul_1030);  mul_1030 = None
    mul_1031: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
    mul_1032: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_54, mul_1031);  add_54 = mul_1031 = None
    add_191: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_1028, mul_1032);  mul_1028 = mul_1032 = None
    mul_1033: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1026, add_191);  mul_1026 = add_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    add_192: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(add_182, mul_1033);  add_182 = mul_1033 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_1034: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_192, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    mul_1035: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1034, clone_4);  clone_4 = None
    mul_1036: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1034, primals_86);  mul_1034 = primals_86 = None
    sum_120: "f32[]" = torch.ops.aten.sum.default(mul_1035);  mul_1035 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_1037: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1036, 2.0);  mul_1036 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_1038: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1037, convolution_34);  convolution_34 = None
    mul_1039: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1037, sigmoid_4);  mul_1037 = sigmoid_4 = None
    sum_121: "f32[4, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1038, [2, 3], True);  mul_1038 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_52: "f32[4, 1536, 1, 1]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    sub_184: "f32[4, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_52)
    mul_1040: "f32[4, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(alias_52, sub_184);  alias_52 = sub_184 = None
    mul_1041: "f32[4, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_121, mul_1040);  sum_121 = mul_1040 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_122: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_1041, [0, 2, 3])
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_1041, relu_4, primals_202, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1041 = primals_202 = None
    getitem_246: "f32[4, 768, 1, 1]" = convolution_backward_44[0]
    getitem_247: "f32[1536, 768, 1, 1]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_54: "f32[4, 768, 1, 1]" = torch.ops.aten.alias.default(relu_4);  relu_4 = None
    alias_55: "f32[4, 768, 1, 1]" = torch.ops.aten.alias.default(alias_54);  alias_54 = None
    le_7: "b8[4, 768, 1, 1]" = torch.ops.aten.le.Scalar(alias_55, 0);  alias_55 = None
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_7: "f32[4, 768, 1, 1]" = torch.ops.aten.where.self(le_7, scalar_tensor_7, getitem_246);  le_7 = scalar_tensor_7 = getitem_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_123: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(where_7, mean_4, primals_200, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_7 = mean_4 = primals_200 = None
    getitem_249: "f32[4, 1536, 1, 1]" = convolution_backward_45[0]
    getitem_250: "f32[768, 1536, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_8: "f32[4, 1536, 12, 12]" = torch.ops.aten.expand.default(getitem_249, [4, 1536, 12, 12]);  getitem_249 = None
    div_8: "f32[4, 1536, 12, 12]" = torch.ops.aten.div.Scalar(expand_8, 144);  expand_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_193: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_1039, div_8);  mul_1039 = div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_124: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_193, [0, 2, 3])
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(add_193, mul_190, view_80, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_193 = mul_190 = view_80 = None
    getitem_252: "f32[4, 768, 12, 12]" = convolution_backward_46[0]
    getitem_253: "f32[1536, 768, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_264: "f32[1, 1536, 768]" = torch.ops.aten.view.default(getitem_253, [1, 1536, 768]);  getitem_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_297: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_52, 0);  squeeze_52 = None
    unsqueeze_298: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 2);  unsqueeze_297 = None
    sum_125: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_264, [0, 2])
    sub_185: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_78, unsqueeze_298)
    mul_1042: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(view_264, sub_185);  sub_185 = None
    sum_126: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_1042, [0, 2]);  mul_1042 = None
    mul_1043: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_125, 0.0013020833333333333);  sum_125 = None
    unsqueeze_299: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_1043, 0);  mul_1043 = None
    unsqueeze_300: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 2);  unsqueeze_299 = None
    mul_1044: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_126, 0.0013020833333333333)
    mul_1045: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_53, squeeze_53)
    mul_1046: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_1044, mul_1045);  mul_1044 = mul_1045 = None
    unsqueeze_301: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_1046, 0);  mul_1046 = None
    unsqueeze_302: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 2);  unsqueeze_301 = None
    mul_1047: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_53, view_79);  view_79 = None
    unsqueeze_303: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_1047, 0);  mul_1047 = None
    unsqueeze_304: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 2);  unsqueeze_303 = None
    sub_186: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_78, unsqueeze_298);  view_78 = unsqueeze_298 = None
    mul_1048: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_186, unsqueeze_302);  sub_186 = unsqueeze_302 = None
    sub_187: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_264, mul_1048);  view_264 = mul_1048 = None
    sub_188: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(sub_187, unsqueeze_300);  sub_187 = unsqueeze_300 = None
    mul_1049: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_304);  sub_188 = unsqueeze_304 = None
    mul_1050: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_126, squeeze_53);  sum_126 = squeeze_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_265: "f32[1536, 1, 1, 1]" = torch.ops.aten.view.default(mul_1050, [1536, 1, 1, 1]);  mul_1050 = None
    mul_1051: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_265, 0.03608439182435161);  view_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_266: "f32[1536, 768, 1, 1]" = torch.ops.aten.view.default(mul_1049, [1536, 768, 1, 1]);  mul_1049 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1052: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_252, 1.7015043497085571);  getitem_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1053: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_33, 0.7071067811865476)
    erf_81: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_1053);  mul_1053 = None
    add_194: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_81, 1);  erf_81 = None
    mul_1054: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(add_194, 0.5);  add_194 = None
    mul_1055: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_33, convolution_33)
    mul_1056: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1055, -0.5);  mul_1055 = None
    exp_29: "f32[4, 768, 12, 12]" = torch.ops.aten.exp.default(mul_1056);  mul_1056 = None
    mul_1057: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
    mul_1058: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_33, mul_1057);  convolution_33 = mul_1057 = None
    add_195: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(mul_1054, mul_1058);  mul_1054 = mul_1058 = None
    mul_1059: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1052, add_195);  mul_1052 = add_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_127: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1059, [0, 2, 3])
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_1059, mul_183, view_77, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_1059 = mul_183 = view_77 = None
    getitem_255: "f32[4, 768, 12, 12]" = convolution_backward_47[0]
    getitem_256: "f32[768, 128, 3, 3]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_267: "f32[1, 768, 1152]" = torch.ops.aten.view.default(getitem_256, [1, 768, 1152]);  getitem_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_305: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_50, 0);  squeeze_50 = None
    unsqueeze_306: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 2);  unsqueeze_305 = None
    sum_128: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_267, [0, 2])
    sub_189: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_75, unsqueeze_306)
    mul_1060: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(view_267, sub_189);  sub_189 = None
    sum_129: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1060, [0, 2]);  mul_1060 = None
    mul_1061: "f32[768]" = torch.ops.aten.mul.Tensor(sum_128, 0.0008680555555555555);  sum_128 = None
    unsqueeze_307: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1061, 0);  mul_1061 = None
    unsqueeze_308: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 2);  unsqueeze_307 = None
    mul_1062: "f32[768]" = torch.ops.aten.mul.Tensor(sum_129, 0.0008680555555555555)
    mul_1063: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_51, squeeze_51)
    mul_1064: "f32[768]" = torch.ops.aten.mul.Tensor(mul_1062, mul_1063);  mul_1062 = mul_1063 = None
    unsqueeze_309: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1064, 0);  mul_1064 = None
    unsqueeze_310: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 2);  unsqueeze_309 = None
    mul_1065: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_51, view_76);  view_76 = None
    unsqueeze_311: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1065, 0);  mul_1065 = None
    unsqueeze_312: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 2);  unsqueeze_311 = None
    sub_190: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_75, unsqueeze_306);  view_75 = unsqueeze_306 = None
    mul_1066: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_190, unsqueeze_310);  sub_190 = unsqueeze_310 = None
    sub_191: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_267, mul_1066);  view_267 = mul_1066 = None
    sub_192: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_191, unsqueeze_308);  sub_191 = unsqueeze_308 = None
    mul_1067: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_312);  sub_192 = unsqueeze_312 = None
    mul_1068: "f32[768]" = torch.ops.aten.mul.Tensor(sum_129, squeeze_51);  sum_129 = squeeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_268: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_1068, [768, 1, 1, 1]);  mul_1068 = None
    mul_1069: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_268, 0.02946278254943948);  view_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_269: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_1067, [768, 128, 3, 3]);  mul_1067 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1070: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_255, 1.7015043497085571);  getitem_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1071: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_32, 0.7071067811865476)
    erf_82: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_1071);  mul_1071 = None
    add_196: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_82, 1);  erf_82 = None
    mul_1072: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(add_196, 0.5);  add_196 = None
    mul_1073: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_32, convolution_32)
    mul_1074: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1073, -0.5);  mul_1073 = None
    exp_30: "f32[4, 768, 12, 12]" = torch.ops.aten.exp.default(mul_1074);  mul_1074 = None
    mul_1075: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(exp_30, 0.3989422804014327);  exp_30 = None
    mul_1076: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_32, mul_1075);  convolution_32 = mul_1075 = None
    add_197: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(mul_1072, mul_1076);  mul_1072 = mul_1076 = None
    mul_1077: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1070, add_197);  mul_1070 = add_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_130: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1077, [0, 2, 3])
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_1077, mul_176, view_74, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_1077 = mul_176 = view_74 = None
    getitem_258: "f32[4, 768, 12, 12]" = convolution_backward_48[0]
    getitem_259: "f32[768, 128, 3, 3]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_270: "f32[1, 768, 1152]" = torch.ops.aten.view.default(getitem_259, [1, 768, 1152]);  getitem_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_313: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_314: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 2);  unsqueeze_313 = None
    sum_131: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_270, [0, 2])
    sub_193: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_72, unsqueeze_314)
    mul_1078: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(view_270, sub_193);  sub_193 = None
    sum_132: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1078, [0, 2]);  mul_1078 = None
    mul_1079: "f32[768]" = torch.ops.aten.mul.Tensor(sum_131, 0.0008680555555555555);  sum_131 = None
    unsqueeze_315: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1079, 0);  mul_1079 = None
    unsqueeze_316: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 2);  unsqueeze_315 = None
    mul_1080: "f32[768]" = torch.ops.aten.mul.Tensor(sum_132, 0.0008680555555555555)
    mul_1081: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_1082: "f32[768]" = torch.ops.aten.mul.Tensor(mul_1080, mul_1081);  mul_1080 = mul_1081 = None
    unsqueeze_317: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1082, 0);  mul_1082 = None
    unsqueeze_318: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 2);  unsqueeze_317 = None
    mul_1083: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_49, view_73);  view_73 = None
    unsqueeze_319: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1083, 0);  mul_1083 = None
    unsqueeze_320: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 2);  unsqueeze_319 = None
    sub_194: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_72, unsqueeze_314);  view_72 = unsqueeze_314 = None
    mul_1084: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_194, unsqueeze_318);  sub_194 = unsqueeze_318 = None
    sub_195: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_270, mul_1084);  view_270 = mul_1084 = None
    sub_196: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_195, unsqueeze_316);  sub_195 = unsqueeze_316 = None
    mul_1085: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_320);  sub_196 = unsqueeze_320 = None
    mul_1086: "f32[768]" = torch.ops.aten.mul.Tensor(sum_132, squeeze_49);  sum_132 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_271: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_1086, [768, 1, 1, 1]);  mul_1086 = None
    mul_1087: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_271, 0.02946278254943948);  view_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_272: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_1085, [768, 128, 3, 3]);  mul_1085 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1088: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_258, 1.7015043497085571);  getitem_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1089: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_31, 0.7071067811865476)
    erf_83: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_1089);  mul_1089 = None
    add_198: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_83, 1);  erf_83 = None
    mul_1090: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(add_198, 0.5);  add_198 = None
    mul_1091: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_31, convolution_31)
    mul_1092: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1091, -0.5);  mul_1091 = None
    exp_31: "f32[4, 768, 12, 12]" = torch.ops.aten.exp.default(mul_1092);  mul_1092 = None
    mul_1093: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(exp_31, 0.3989422804014327);  exp_31 = None
    mul_1094: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_31, mul_1093);  convolution_31 = mul_1093 = None
    add_199: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(mul_1090, mul_1094);  mul_1090 = mul_1094 = None
    mul_1095: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1088, add_199);  mul_1088 = add_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_133: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1095, [0, 2, 3])
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_1095, mul_169, view_71, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1095 = mul_169 = view_71 = None
    getitem_261: "f32[4, 1536, 12, 12]" = convolution_backward_49[0]
    getitem_262: "f32[768, 1536, 1, 1]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_273: "f32[1, 768, 1536]" = torch.ops.aten.view.default(getitem_262, [1, 768, 1536]);  getitem_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_321: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_46, 0);  squeeze_46 = None
    unsqueeze_322: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 2);  unsqueeze_321 = None
    sum_134: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_273, [0, 2])
    sub_197: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_69, unsqueeze_322)
    mul_1096: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(view_273, sub_197);  sub_197 = None
    sum_135: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1096, [0, 2]);  mul_1096 = None
    mul_1097: "f32[768]" = torch.ops.aten.mul.Tensor(sum_134, 0.0006510416666666666);  sum_134 = None
    unsqueeze_323: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1097, 0);  mul_1097 = None
    unsqueeze_324: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 2);  unsqueeze_323 = None
    mul_1098: "f32[768]" = torch.ops.aten.mul.Tensor(sum_135, 0.0006510416666666666)
    mul_1099: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_47, squeeze_47)
    mul_1100: "f32[768]" = torch.ops.aten.mul.Tensor(mul_1098, mul_1099);  mul_1098 = mul_1099 = None
    unsqueeze_325: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1100, 0);  mul_1100 = None
    unsqueeze_326: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 2);  unsqueeze_325 = None
    mul_1101: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_47, view_70);  view_70 = None
    unsqueeze_327: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1101, 0);  mul_1101 = None
    unsqueeze_328: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 2);  unsqueeze_327 = None
    sub_198: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_69, unsqueeze_322);  view_69 = unsqueeze_322 = None
    mul_1102: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_198, unsqueeze_326);  sub_198 = unsqueeze_326 = None
    sub_199: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_273, mul_1102);  view_273 = mul_1102 = None
    sub_200: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(sub_199, unsqueeze_324);  sub_199 = unsqueeze_324 = None
    mul_1103: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_328);  sub_200 = unsqueeze_328 = None
    mul_1104: "f32[768]" = torch.ops.aten.mul.Tensor(sum_135, squeeze_47);  sum_135 = squeeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_274: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_1104, [768, 1, 1, 1]);  mul_1104 = None
    mul_1105: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_274, 0.02551551815399144);  view_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_275: "f32[768, 1536, 1, 1]" = torch.ops.aten.view.default(mul_1103, [768, 1536, 1, 1]);  mul_1103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_1106: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_261, 0.9805806756909201);  getitem_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1107: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1106, 1.7015043497085571);  mul_1106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1108: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_45, 0.7071067811865476)
    erf_84: "f32[4, 1536, 12, 12]" = torch.ops.aten.erf.default(mul_1108);  mul_1108 = None
    add_200: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(erf_84, 1);  erf_84 = None
    mul_1109: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_200, 0.5);  add_200 = None
    mul_1110: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_45, add_45)
    mul_1111: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1110, -0.5);  mul_1110 = None
    exp_32: "f32[4, 1536, 12, 12]" = torch.ops.aten.exp.default(mul_1111);  mul_1111 = None
    mul_1112: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(exp_32, 0.3989422804014327);  exp_32 = None
    mul_1113: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_45, mul_1112);  add_45 = mul_1112 = None
    add_201: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_1109, mul_1113);  mul_1109 = mul_1113 = None
    mul_1114: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1107, add_201);  mul_1107 = add_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    add_202: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(add_192, mul_1114);  add_192 = mul_1114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_1115: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_202, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    mul_1116: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1115, clone_3);  clone_3 = None
    mul_1117: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1115, primals_73);  mul_1115 = primals_73 = None
    sum_136: "f32[]" = torch.ops.aten.sum.default(mul_1116);  mul_1116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_1118: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1117, 2.0);  mul_1117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_1119: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1118, convolution_28);  convolution_28 = None
    mul_1120: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1118, sigmoid_3);  mul_1118 = sigmoid_3 = None
    sum_137: "f32[4, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1119, [2, 3], True);  mul_1119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_56: "f32[4, 1536, 1, 1]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    sub_201: "f32[4, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_56)
    mul_1121: "f32[4, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(alias_56, sub_201);  alias_56 = sub_201 = None
    mul_1122: "f32[4, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_137, mul_1121);  sum_137 = mul_1121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_138: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_1122, [0, 2, 3])
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_1122, relu_3, primals_198, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1122 = primals_198 = None
    getitem_264: "f32[4, 768, 1, 1]" = convolution_backward_50[0]
    getitem_265: "f32[1536, 768, 1, 1]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_58: "f32[4, 768, 1, 1]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
    alias_59: "f32[4, 768, 1, 1]" = torch.ops.aten.alias.default(alias_58);  alias_58 = None
    le_8: "b8[4, 768, 1, 1]" = torch.ops.aten.le.Scalar(alias_59, 0);  alias_59 = None
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_8: "f32[4, 768, 1, 1]" = torch.ops.aten.where.self(le_8, scalar_tensor_8, getitem_264);  le_8 = scalar_tensor_8 = getitem_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_139: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(where_8, mean_3, primals_196, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_8 = mean_3 = primals_196 = None
    getitem_267: "f32[4, 1536, 1, 1]" = convolution_backward_51[0]
    getitem_268: "f32[768, 1536, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_9: "f32[4, 1536, 12, 12]" = torch.ops.aten.expand.default(getitem_267, [4, 1536, 12, 12]);  getitem_267 = None
    div_9: "f32[4, 1536, 12, 12]" = torch.ops.aten.div.Scalar(expand_9, 144);  expand_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_203: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_1120, div_9);  mul_1120 = div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_140: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_203, [0, 2, 3])
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(add_203, mul_157, view_68, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_203 = mul_157 = view_68 = None
    getitem_270: "f32[4, 768, 12, 12]" = convolution_backward_52[0]
    getitem_271: "f32[1536, 768, 1, 1]" = convolution_backward_52[1];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_276: "f32[1, 1536, 768]" = torch.ops.aten.view.default(getitem_271, [1, 1536, 768]);  getitem_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_329: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_44, 0);  squeeze_44 = None
    unsqueeze_330: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 2);  unsqueeze_329 = None
    sum_141: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_276, [0, 2])
    sub_202: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_66, unsqueeze_330)
    mul_1123: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(view_276, sub_202);  sub_202 = None
    sum_142: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_1123, [0, 2]);  mul_1123 = None
    mul_1124: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_141, 0.0013020833333333333);  sum_141 = None
    unsqueeze_331: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_1124, 0);  mul_1124 = None
    unsqueeze_332: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 2);  unsqueeze_331 = None
    mul_1125: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_142, 0.0013020833333333333)
    mul_1126: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_45, squeeze_45)
    mul_1127: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_1125, mul_1126);  mul_1125 = mul_1126 = None
    unsqueeze_333: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_1127, 0);  mul_1127 = None
    unsqueeze_334: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 2);  unsqueeze_333 = None
    mul_1128: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_45, view_67);  view_67 = None
    unsqueeze_335: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_1128, 0);  mul_1128 = None
    unsqueeze_336: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 2);  unsqueeze_335 = None
    sub_203: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_66, unsqueeze_330);  view_66 = unsqueeze_330 = None
    mul_1129: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_334);  sub_203 = unsqueeze_334 = None
    sub_204: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_276, mul_1129);  view_276 = mul_1129 = None
    sub_205: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(sub_204, unsqueeze_332);  sub_204 = unsqueeze_332 = None
    mul_1130: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_336);  sub_205 = unsqueeze_336 = None
    mul_1131: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_142, squeeze_45);  sum_142 = squeeze_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_277: "f32[1536, 1, 1, 1]" = torch.ops.aten.view.default(mul_1131, [1536, 1, 1, 1]);  mul_1131 = None
    mul_1132: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_277, 0.03608439182435161);  view_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_278: "f32[1536, 768, 1, 1]" = torch.ops.aten.view.default(mul_1130, [1536, 768, 1, 1]);  mul_1130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1133: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_270, 1.7015043497085571);  getitem_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1134: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_27, 0.7071067811865476)
    erf_85: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_1134);  mul_1134 = None
    add_204: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_85, 1);  erf_85 = None
    mul_1135: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(add_204, 0.5);  add_204 = None
    mul_1136: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_27, convolution_27)
    mul_1137: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1136, -0.5);  mul_1136 = None
    exp_33: "f32[4, 768, 12, 12]" = torch.ops.aten.exp.default(mul_1137);  mul_1137 = None
    mul_1138: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(exp_33, 0.3989422804014327);  exp_33 = None
    mul_1139: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_27, mul_1138);  convolution_27 = mul_1138 = None
    add_205: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(mul_1135, mul_1139);  mul_1135 = mul_1139 = None
    mul_1140: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1133, add_205);  mul_1133 = add_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_143: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1140, [0, 2, 3])
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_1140, mul_150, view_65, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_1140 = mul_150 = view_65 = None
    getitem_273: "f32[4, 768, 12, 12]" = convolution_backward_53[0]
    getitem_274: "f32[768, 128, 3, 3]" = convolution_backward_53[1];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_279: "f32[1, 768, 1152]" = torch.ops.aten.view.default(getitem_274, [1, 768, 1152]);  getitem_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_337: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_338: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 2);  unsqueeze_337 = None
    sum_144: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_279, [0, 2])
    sub_206: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_63, unsqueeze_338)
    mul_1141: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(view_279, sub_206);  sub_206 = None
    sum_145: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1141, [0, 2]);  mul_1141 = None
    mul_1142: "f32[768]" = torch.ops.aten.mul.Tensor(sum_144, 0.0008680555555555555);  sum_144 = None
    unsqueeze_339: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1142, 0);  mul_1142 = None
    unsqueeze_340: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 2);  unsqueeze_339 = None
    mul_1143: "f32[768]" = torch.ops.aten.mul.Tensor(sum_145, 0.0008680555555555555)
    mul_1144: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_1145: "f32[768]" = torch.ops.aten.mul.Tensor(mul_1143, mul_1144);  mul_1143 = mul_1144 = None
    unsqueeze_341: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1145, 0);  mul_1145 = None
    unsqueeze_342: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 2);  unsqueeze_341 = None
    mul_1146: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_43, view_64);  view_64 = None
    unsqueeze_343: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1146, 0);  mul_1146 = None
    unsqueeze_344: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 2);  unsqueeze_343 = None
    sub_207: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_63, unsqueeze_338);  view_63 = unsqueeze_338 = None
    mul_1147: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_207, unsqueeze_342);  sub_207 = unsqueeze_342 = None
    sub_208: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_279, mul_1147);  view_279 = mul_1147 = None
    sub_209: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_208, unsqueeze_340);  sub_208 = unsqueeze_340 = None
    mul_1148: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_344);  sub_209 = unsqueeze_344 = None
    mul_1149: "f32[768]" = torch.ops.aten.mul.Tensor(sum_145, squeeze_43);  sum_145 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_280: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_1149, [768, 1, 1, 1]);  mul_1149 = None
    mul_1150: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_280, 0.02946278254943948);  view_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_281: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_1148, [768, 128, 3, 3]);  mul_1148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1151: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_273, 1.7015043497085571);  getitem_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1152: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_26, 0.7071067811865476)
    erf_86: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_1152);  mul_1152 = None
    add_206: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_86, 1);  erf_86 = None
    mul_1153: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(add_206, 0.5);  add_206 = None
    mul_1154: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_26, convolution_26)
    mul_1155: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1154, -0.5);  mul_1154 = None
    exp_34: "f32[4, 768, 12, 12]" = torch.ops.aten.exp.default(mul_1155);  mul_1155 = None
    mul_1156: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(exp_34, 0.3989422804014327);  exp_34 = None
    mul_1157: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_26, mul_1156);  convolution_26 = mul_1156 = None
    add_207: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(mul_1153, mul_1157);  mul_1153 = mul_1157 = None
    mul_1158: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1151, add_207);  mul_1151 = add_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_146: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1158, [0, 2, 3])
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_1158, constant_pad_nd_3, view_62, [768], [2, 2], [0, 0], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_1158 = constant_pad_nd_3 = view_62 = None
    getitem_276: "f32[4, 768, 25, 25]" = convolution_backward_54[0]
    getitem_277: "f32[768, 128, 3, 3]" = convolution_backward_54[1];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_282: "f32[1, 768, 1152]" = torch.ops.aten.view.default(getitem_277, [1, 768, 1152]);  getitem_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_345: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_40, 0);  squeeze_40 = None
    unsqueeze_346: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_345, 2);  unsqueeze_345 = None
    sum_147: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_282, [0, 2])
    sub_210: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_60, unsqueeze_346)
    mul_1159: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(view_282, sub_210);  sub_210 = None
    sum_148: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1159, [0, 2]);  mul_1159 = None
    mul_1160: "f32[768]" = torch.ops.aten.mul.Tensor(sum_147, 0.0008680555555555555);  sum_147 = None
    unsqueeze_347: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1160, 0);  mul_1160 = None
    unsqueeze_348: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 2);  unsqueeze_347 = None
    mul_1161: "f32[768]" = torch.ops.aten.mul.Tensor(sum_148, 0.0008680555555555555)
    mul_1162: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_41, squeeze_41)
    mul_1163: "f32[768]" = torch.ops.aten.mul.Tensor(mul_1161, mul_1162);  mul_1161 = mul_1162 = None
    unsqueeze_349: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1163, 0);  mul_1163 = None
    unsqueeze_350: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 2);  unsqueeze_349 = None
    mul_1164: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_41, view_61);  view_61 = None
    unsqueeze_351: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1164, 0);  mul_1164 = None
    unsqueeze_352: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 2);  unsqueeze_351 = None
    sub_211: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_60, unsqueeze_346);  view_60 = unsqueeze_346 = None
    mul_1165: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_350);  sub_211 = unsqueeze_350 = None
    sub_212: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_282, mul_1165);  view_282 = mul_1165 = None
    sub_213: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_212, unsqueeze_348);  sub_212 = unsqueeze_348 = None
    mul_1166: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_352);  sub_213 = unsqueeze_352 = None
    mul_1167: "f32[768]" = torch.ops.aten.mul.Tensor(sum_148, squeeze_41);  sum_148 = squeeze_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_283: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_1167, [768, 1, 1, 1]);  mul_1167 = None
    mul_1168: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_283, 0.02946278254943948);  view_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_284: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_1166, [768, 128, 3, 3]);  mul_1166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_6: "f32[4, 768, 24, 24]" = torch.ops.aten.constant_pad_nd.default(getitem_276, [0, -1, 0, -1]);  getitem_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1169: "f32[4, 768, 24, 24]" = torch.ops.aten.mul.Tensor(constant_pad_nd_6, 1.7015043497085571);  constant_pad_nd_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1170: "f32[4, 768, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_25, 0.7071067811865476)
    erf_87: "f32[4, 768, 24, 24]" = torch.ops.aten.erf.default(mul_1170);  mul_1170 = None
    add_208: "f32[4, 768, 24, 24]" = torch.ops.aten.add.Tensor(erf_87, 1);  erf_87 = None
    mul_1171: "f32[4, 768, 24, 24]" = torch.ops.aten.mul.Tensor(add_208, 0.5);  add_208 = None
    mul_1172: "f32[4, 768, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_25, convolution_25)
    mul_1173: "f32[4, 768, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1172, -0.5);  mul_1172 = None
    exp_35: "f32[4, 768, 24, 24]" = torch.ops.aten.exp.default(mul_1173);  mul_1173 = None
    mul_1174: "f32[4, 768, 24, 24]" = torch.ops.aten.mul.Tensor(exp_35, 0.3989422804014327);  exp_35 = None
    mul_1175: "f32[4, 768, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_25, mul_1174);  convolution_25 = mul_1174 = None
    add_209: "f32[4, 768, 24, 24]" = torch.ops.aten.add.Tensor(mul_1171, mul_1175);  mul_1171 = mul_1175 = None
    mul_1176: "f32[4, 768, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1169, add_209);  mul_1169 = add_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_149: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1176, [0, 2, 3])
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_1176, mul_133, view_59, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1176 = view_59 = None
    getitem_279: "f32[4, 512, 24, 24]" = convolution_backward_55[0]
    getitem_280: "f32[768, 512, 1, 1]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_285: "f32[1, 768, 512]" = torch.ops.aten.view.default(getitem_280, [1, 768, 512]);  getitem_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_353: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_38, 0);  squeeze_38 = None
    unsqueeze_354: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 2);  unsqueeze_353 = None
    sum_150: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_285, [0, 2])
    sub_214: "f32[1, 768, 512]" = torch.ops.aten.sub.Tensor(view_57, unsqueeze_354)
    mul_1177: "f32[1, 768, 512]" = torch.ops.aten.mul.Tensor(view_285, sub_214);  sub_214 = None
    sum_151: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1177, [0, 2]);  mul_1177 = None
    mul_1178: "f32[768]" = torch.ops.aten.mul.Tensor(sum_150, 0.001953125);  sum_150 = None
    unsqueeze_355: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1178, 0);  mul_1178 = None
    unsqueeze_356: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 2);  unsqueeze_355 = None
    mul_1179: "f32[768]" = torch.ops.aten.mul.Tensor(sum_151, 0.001953125)
    mul_1180: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_39, squeeze_39)
    mul_1181: "f32[768]" = torch.ops.aten.mul.Tensor(mul_1179, mul_1180);  mul_1179 = mul_1180 = None
    unsqueeze_357: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1181, 0);  mul_1181 = None
    unsqueeze_358: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 2);  unsqueeze_357 = None
    mul_1182: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_39, view_58);  view_58 = None
    unsqueeze_359: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1182, 0);  mul_1182 = None
    unsqueeze_360: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 2);  unsqueeze_359 = None
    sub_215: "f32[1, 768, 512]" = torch.ops.aten.sub.Tensor(view_57, unsqueeze_354);  view_57 = unsqueeze_354 = None
    mul_1183: "f32[1, 768, 512]" = torch.ops.aten.mul.Tensor(sub_215, unsqueeze_358);  sub_215 = unsqueeze_358 = None
    sub_216: "f32[1, 768, 512]" = torch.ops.aten.sub.Tensor(view_285, mul_1183);  view_285 = mul_1183 = None
    sub_217: "f32[1, 768, 512]" = torch.ops.aten.sub.Tensor(sub_216, unsqueeze_356);  sub_216 = unsqueeze_356 = None
    mul_1184: "f32[1, 768, 512]" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_360);  sub_217 = unsqueeze_360 = None
    mul_1185: "f32[768]" = torch.ops.aten.mul.Tensor(sum_151, squeeze_39);  sum_151 = squeeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_286: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_1185, [768, 1, 1, 1]);  mul_1185 = None
    mul_1186: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_286, 0.04419417382415922);  view_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_287: "f32[768, 512, 1, 1]" = torch.ops.aten.view.default(mul_1184, [768, 512, 1, 1]);  mul_1184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_152: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_202, [0, 2, 3])
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(add_202, avg_pool2d_1, view_56, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_202 = avg_pool2d_1 = view_56 = None
    getitem_282: "f32[4, 512, 12, 12]" = convolution_backward_56[0]
    getitem_283: "f32[1536, 512, 1, 1]" = convolution_backward_56[1];  convolution_backward_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_288: "f32[1, 1536, 512]" = torch.ops.aten.view.default(getitem_283, [1, 1536, 512]);  getitem_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_361: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_362: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
    sum_153: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_288, [0, 2])
    sub_218: "f32[1, 1536, 512]" = torch.ops.aten.sub.Tensor(view_54, unsqueeze_362)
    mul_1187: "f32[1, 1536, 512]" = torch.ops.aten.mul.Tensor(view_288, sub_218);  sub_218 = None
    sum_154: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_1187, [0, 2]);  mul_1187 = None
    mul_1188: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_153, 0.001953125);  sum_153 = None
    unsqueeze_363: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_1188, 0);  mul_1188 = None
    unsqueeze_364: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 2);  unsqueeze_363 = None
    mul_1189: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_154, 0.001953125)
    mul_1190: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_1191: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_1189, mul_1190);  mul_1189 = mul_1190 = None
    unsqueeze_365: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_1191, 0);  mul_1191 = None
    unsqueeze_366: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 2);  unsqueeze_365 = None
    mul_1192: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_37, view_55);  view_55 = None
    unsqueeze_367: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_1192, 0);  mul_1192 = None
    unsqueeze_368: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 2);  unsqueeze_367 = None
    sub_219: "f32[1, 1536, 512]" = torch.ops.aten.sub.Tensor(view_54, unsqueeze_362);  view_54 = unsqueeze_362 = None
    mul_1193: "f32[1, 1536, 512]" = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_366);  sub_219 = unsqueeze_366 = None
    sub_220: "f32[1, 1536, 512]" = torch.ops.aten.sub.Tensor(view_288, mul_1193);  view_288 = mul_1193 = None
    sub_221: "f32[1, 1536, 512]" = torch.ops.aten.sub.Tensor(sub_220, unsqueeze_364);  sub_220 = unsqueeze_364 = None
    mul_1194: "f32[1, 1536, 512]" = torch.ops.aten.mul.Tensor(sub_221, unsqueeze_368);  sub_221 = unsqueeze_368 = None
    mul_1195: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_154, squeeze_37);  sum_154 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_289: "f32[1536, 1, 1, 1]" = torch.ops.aten.view.default(mul_1195, [1536, 1, 1, 1]);  mul_1195 = None
    mul_1196: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_289, 0.04419417382415922);  view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_290: "f32[1536, 512, 1, 1]" = torch.ops.aten.view.default(mul_1194, [1536, 512, 1, 1]);  mul_1194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    avg_pool2d_backward_1: "f32[4, 512, 24, 24]" = torch.ops.aten.avg_pool2d_backward.default(getitem_282, mul_133, [2, 2], [2, 2], [0, 0], True, False, None);  getitem_282 = mul_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    add_210: "f32[4, 512, 24, 24]" = torch.ops.aten.add.Tensor(getitem_279, avg_pool2d_backward_1);  getitem_279 = avg_pool2d_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_1197: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_210, 0.9622504486493761);  add_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1198: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1197, 1.7015043497085571);  mul_1197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1199: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_35, 0.7071067811865476)
    erf_88: "f32[4, 512, 24, 24]" = torch.ops.aten.erf.default(mul_1199);  mul_1199 = None
    add_211: "f32[4, 512, 24, 24]" = torch.ops.aten.add.Tensor(erf_88, 1);  erf_88 = None
    mul_1200: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_211, 0.5);  add_211 = None
    mul_1201: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_35, add_35)
    mul_1202: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1201, -0.5);  mul_1201 = None
    exp_36: "f32[4, 512, 24, 24]" = torch.ops.aten.exp.default(mul_1202);  mul_1202 = None
    mul_1203: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(exp_36, 0.3989422804014327);  exp_36 = None
    mul_1204: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_35, mul_1203);  add_35 = mul_1203 = None
    add_212: "f32[4, 512, 24, 24]" = torch.ops.aten.add.Tensor(mul_1200, mul_1204);  mul_1200 = mul_1204 = None
    mul_1205: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1198, add_212);  mul_1198 = add_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_1206: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1205, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    mul_1207: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1206, clone_2);  clone_2 = None
    mul_1208: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1206, primals_57);  mul_1206 = primals_57 = None
    sum_155: "f32[]" = torch.ops.aten.sum.default(mul_1207);  mul_1207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_1209: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1208, 2.0);  mul_1208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_1210: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1209, convolution_21);  convolution_21 = None
    mul_1211: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1209, sigmoid_2);  mul_1209 = sigmoid_2 = None
    sum_156: "f32[4, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1210, [2, 3], True);  mul_1210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_60: "f32[4, 512, 1, 1]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    sub_222: "f32[4, 512, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_60)
    mul_1212: "f32[4, 512, 1, 1]" = torch.ops.aten.mul.Tensor(alias_60, sub_222);  alias_60 = sub_222 = None
    mul_1213: "f32[4, 512, 1, 1]" = torch.ops.aten.mul.Tensor(sum_156, mul_1212);  sum_156 = mul_1212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_157: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1213, [0, 2, 3])
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_1213, relu_2, primals_194, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1213 = primals_194 = None
    getitem_285: "f32[4, 256, 1, 1]" = convolution_backward_57[0]
    getitem_286: "f32[512, 256, 1, 1]" = convolution_backward_57[1];  convolution_backward_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_62: "f32[4, 256, 1, 1]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_63: "f32[4, 256, 1, 1]" = torch.ops.aten.alias.default(alias_62);  alias_62 = None
    le_9: "b8[4, 256, 1, 1]" = torch.ops.aten.le.Scalar(alias_63, 0);  alias_63 = None
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_9: "f32[4, 256, 1, 1]" = torch.ops.aten.where.self(le_9, scalar_tensor_9, getitem_285);  le_9 = scalar_tensor_9 = getitem_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_158: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(where_9, mean_2, primals_192, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_9 = mean_2 = primals_192 = None
    getitem_288: "f32[4, 512, 1, 1]" = convolution_backward_58[0]
    getitem_289: "f32[256, 512, 1, 1]" = convolution_backward_58[1];  convolution_backward_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_10: "f32[4, 512, 24, 24]" = torch.ops.aten.expand.default(getitem_288, [4, 512, 24, 24]);  getitem_288 = None
    div_10: "f32[4, 512, 24, 24]" = torch.ops.aten.div.Scalar(expand_10, 576);  expand_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_213: "f32[4, 512, 24, 24]" = torch.ops.aten.add.Tensor(mul_1211, div_10);  mul_1211 = div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_159: "f32[512]" = torch.ops.aten.sum.dim_IntList(add_213, [0, 2, 3])
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(add_213, mul_121, view_53, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_213 = mul_121 = view_53 = None
    getitem_291: "f32[4, 256, 24, 24]" = convolution_backward_59[0]
    getitem_292: "f32[512, 256, 1, 1]" = convolution_backward_59[1];  convolution_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_291: "f32[1, 512, 256]" = torch.ops.aten.view.default(getitem_292, [1, 512, 256]);  getitem_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_369: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_34, 0);  squeeze_34 = None
    unsqueeze_370: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_369, 2);  unsqueeze_369 = None
    sum_160: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_291, [0, 2])
    sub_223: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(view_51, unsqueeze_370)
    mul_1214: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(view_291, sub_223);  sub_223 = None
    sum_161: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1214, [0, 2]);  mul_1214 = None
    mul_1215: "f32[512]" = torch.ops.aten.mul.Tensor(sum_160, 0.00390625);  sum_160 = None
    unsqueeze_371: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1215, 0);  mul_1215 = None
    unsqueeze_372: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 2);  unsqueeze_371 = None
    mul_1216: "f32[512]" = torch.ops.aten.mul.Tensor(sum_161, 0.00390625)
    mul_1217: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_35, squeeze_35)
    mul_1218: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1216, mul_1217);  mul_1216 = mul_1217 = None
    unsqueeze_373: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1218, 0);  mul_1218 = None
    unsqueeze_374: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
    mul_1219: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_35, view_52);  view_52 = None
    unsqueeze_375: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1219, 0);  mul_1219 = None
    unsqueeze_376: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_375, 2);  unsqueeze_375 = None
    sub_224: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(view_51, unsqueeze_370);  view_51 = unsqueeze_370 = None
    mul_1220: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_374);  sub_224 = unsqueeze_374 = None
    sub_225: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(view_291, mul_1220);  view_291 = mul_1220 = None
    sub_226: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_225, unsqueeze_372);  sub_225 = unsqueeze_372 = None
    mul_1221: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_226, unsqueeze_376);  sub_226 = unsqueeze_376 = None
    mul_1222: "f32[512]" = torch.ops.aten.mul.Tensor(sum_161, squeeze_35);  sum_161 = squeeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_292: "f32[512, 1, 1, 1]" = torch.ops.aten.view.default(mul_1222, [512, 1, 1, 1]);  mul_1222 = None
    mul_1223: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_292, 0.0625);  view_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_293: "f32[512, 256, 1, 1]" = torch.ops.aten.view.default(mul_1221, [512, 256, 1, 1]);  mul_1221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1224: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(getitem_291, 1.7015043497085571);  getitem_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1225: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_20, 0.7071067811865476)
    erf_89: "f32[4, 256, 24, 24]" = torch.ops.aten.erf.default(mul_1225);  mul_1225 = None
    add_214: "f32[4, 256, 24, 24]" = torch.ops.aten.add.Tensor(erf_89, 1);  erf_89 = None
    mul_1226: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(add_214, 0.5);  add_214 = None
    mul_1227: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_20, convolution_20)
    mul_1228: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1227, -0.5);  mul_1227 = None
    exp_37: "f32[4, 256, 24, 24]" = torch.ops.aten.exp.default(mul_1228);  mul_1228 = None
    mul_1229: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(exp_37, 0.3989422804014327);  exp_37 = None
    mul_1230: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_20, mul_1229);  convolution_20 = mul_1229 = None
    add_215: "f32[4, 256, 24, 24]" = torch.ops.aten.add.Tensor(mul_1226, mul_1230);  mul_1226 = mul_1230 = None
    mul_1231: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1224, add_215);  mul_1224 = add_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_162: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1231, [0, 2, 3])
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_1231, mul_114, view_50, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_1231 = mul_114 = view_50 = None
    getitem_294: "f32[4, 256, 24, 24]" = convolution_backward_60[0]
    getitem_295: "f32[256, 128, 3, 3]" = convolution_backward_60[1];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_294: "f32[1, 256, 1152]" = torch.ops.aten.view.default(getitem_295, [1, 256, 1152]);  getitem_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_377: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_32, 0);  squeeze_32 = None
    unsqueeze_378: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 2);  unsqueeze_377 = None
    sum_163: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_294, [0, 2])
    sub_227: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(view_48, unsqueeze_378)
    mul_1232: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(view_294, sub_227);  sub_227 = None
    sum_164: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1232, [0, 2]);  mul_1232 = None
    mul_1233: "f32[256]" = torch.ops.aten.mul.Tensor(sum_163, 0.0008680555555555555);  sum_163 = None
    unsqueeze_379: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1233, 0);  mul_1233 = None
    unsqueeze_380: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
    mul_1234: "f32[256]" = torch.ops.aten.mul.Tensor(sum_164, 0.0008680555555555555)
    mul_1235: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_33, squeeze_33)
    mul_1236: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1234, mul_1235);  mul_1234 = mul_1235 = None
    unsqueeze_381: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1236, 0);  mul_1236 = None
    unsqueeze_382: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_381, 2);  unsqueeze_381 = None
    mul_1237: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_33, view_49);  view_49 = None
    unsqueeze_383: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1237, 0);  mul_1237 = None
    unsqueeze_384: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 2);  unsqueeze_383 = None
    sub_228: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(view_48, unsqueeze_378);  view_48 = unsqueeze_378 = None
    mul_1238: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_382);  sub_228 = unsqueeze_382 = None
    sub_229: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(view_294, mul_1238);  view_294 = mul_1238 = None
    sub_230: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(sub_229, unsqueeze_380);  sub_229 = unsqueeze_380 = None
    mul_1239: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(sub_230, unsqueeze_384);  sub_230 = unsqueeze_384 = None
    mul_1240: "f32[256]" = torch.ops.aten.mul.Tensor(sum_164, squeeze_33);  sum_164 = squeeze_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_295: "f32[256, 1, 1, 1]" = torch.ops.aten.view.default(mul_1240, [256, 1, 1, 1]);  mul_1240 = None
    mul_1241: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_295, 0.02946278254943948);  view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_296: "f32[256, 128, 3, 3]" = torch.ops.aten.view.default(mul_1239, [256, 128, 3, 3]);  mul_1239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1242: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(getitem_294, 1.7015043497085571);  getitem_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1243: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_19, 0.7071067811865476)
    erf_90: "f32[4, 256, 24, 24]" = torch.ops.aten.erf.default(mul_1243);  mul_1243 = None
    add_216: "f32[4, 256, 24, 24]" = torch.ops.aten.add.Tensor(erf_90, 1);  erf_90 = None
    mul_1244: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(add_216, 0.5);  add_216 = None
    mul_1245: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_19, convolution_19)
    mul_1246: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1245, -0.5);  mul_1245 = None
    exp_38: "f32[4, 256, 24, 24]" = torch.ops.aten.exp.default(mul_1246);  mul_1246 = None
    mul_1247: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(exp_38, 0.3989422804014327);  exp_38 = None
    mul_1248: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_19, mul_1247);  convolution_19 = mul_1247 = None
    add_217: "f32[4, 256, 24, 24]" = torch.ops.aten.add.Tensor(mul_1244, mul_1248);  mul_1244 = mul_1248 = None
    mul_1249: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1242, add_217);  mul_1242 = add_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_165: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1249, [0, 2, 3])
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_1249, mul_107, view_47, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_1249 = mul_107 = view_47 = None
    getitem_297: "f32[4, 256, 24, 24]" = convolution_backward_61[0]
    getitem_298: "f32[256, 128, 3, 3]" = convolution_backward_61[1];  convolution_backward_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_297: "f32[1, 256, 1152]" = torch.ops.aten.view.default(getitem_298, [1, 256, 1152]);  getitem_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_385: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_386: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
    sum_166: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_297, [0, 2])
    sub_231: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(view_45, unsqueeze_386)
    mul_1250: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(view_297, sub_231);  sub_231 = None
    sum_167: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1250, [0, 2]);  mul_1250 = None
    mul_1251: "f32[256]" = torch.ops.aten.mul.Tensor(sum_166, 0.0008680555555555555);  sum_166 = None
    unsqueeze_387: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1251, 0);  mul_1251 = None
    unsqueeze_388: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_387, 2);  unsqueeze_387 = None
    mul_1252: "f32[256]" = torch.ops.aten.mul.Tensor(sum_167, 0.0008680555555555555)
    mul_1253: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_1254: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1252, mul_1253);  mul_1252 = mul_1253 = None
    unsqueeze_389: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1254, 0);  mul_1254 = None
    unsqueeze_390: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 2);  unsqueeze_389 = None
    mul_1255: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_31, view_46);  view_46 = None
    unsqueeze_391: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1255, 0);  mul_1255 = None
    unsqueeze_392: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
    sub_232: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(view_45, unsqueeze_386);  view_45 = unsqueeze_386 = None
    mul_1256: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_390);  sub_232 = unsqueeze_390 = None
    sub_233: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(view_297, mul_1256);  view_297 = mul_1256 = None
    sub_234: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(sub_233, unsqueeze_388);  sub_233 = unsqueeze_388 = None
    mul_1257: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(sub_234, unsqueeze_392);  sub_234 = unsqueeze_392 = None
    mul_1258: "f32[256]" = torch.ops.aten.mul.Tensor(sum_167, squeeze_31);  sum_167 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_298: "f32[256, 1, 1, 1]" = torch.ops.aten.view.default(mul_1258, [256, 1, 1, 1]);  mul_1258 = None
    mul_1259: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_298, 0.02946278254943948);  view_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_299: "f32[256, 128, 3, 3]" = torch.ops.aten.view.default(mul_1257, [256, 128, 3, 3]);  mul_1257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1260: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(getitem_297, 1.7015043497085571);  getitem_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1261: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_18, 0.7071067811865476)
    erf_91: "f32[4, 256, 24, 24]" = torch.ops.aten.erf.default(mul_1261);  mul_1261 = None
    add_218: "f32[4, 256, 24, 24]" = torch.ops.aten.add.Tensor(erf_91, 1);  erf_91 = None
    mul_1262: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(add_218, 0.5);  add_218 = None
    mul_1263: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_18, convolution_18)
    mul_1264: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1263, -0.5);  mul_1263 = None
    exp_39: "f32[4, 256, 24, 24]" = torch.ops.aten.exp.default(mul_1264);  mul_1264 = None
    mul_1265: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(exp_39, 0.3989422804014327);  exp_39 = None
    mul_1266: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_18, mul_1265);  convolution_18 = mul_1265 = None
    add_219: "f32[4, 256, 24, 24]" = torch.ops.aten.add.Tensor(mul_1262, mul_1266);  mul_1262 = mul_1266 = None
    mul_1267: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1260, add_219);  mul_1260 = add_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_168: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1267, [0, 2, 3])
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_1267, mul_100, view_44, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1267 = mul_100 = view_44 = None
    getitem_300: "f32[4, 512, 24, 24]" = convolution_backward_62[0]
    getitem_301: "f32[256, 512, 1, 1]" = convolution_backward_62[1];  convolution_backward_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_300: "f32[1, 256, 512]" = torch.ops.aten.view.default(getitem_301, [1, 256, 512]);  getitem_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_393: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_28, 0);  squeeze_28 = None
    unsqueeze_394: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 2);  unsqueeze_393 = None
    sum_169: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_300, [0, 2])
    sub_235: "f32[1, 256, 512]" = torch.ops.aten.sub.Tensor(view_42, unsqueeze_394)
    mul_1268: "f32[1, 256, 512]" = torch.ops.aten.mul.Tensor(view_300, sub_235);  sub_235 = None
    sum_170: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1268, [0, 2]);  mul_1268 = None
    mul_1269: "f32[256]" = torch.ops.aten.mul.Tensor(sum_169, 0.001953125);  sum_169 = None
    unsqueeze_395: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1269, 0);  mul_1269 = None
    unsqueeze_396: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 2);  unsqueeze_395 = None
    mul_1270: "f32[256]" = torch.ops.aten.mul.Tensor(sum_170, 0.001953125)
    mul_1271: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_29, squeeze_29)
    mul_1272: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1270, mul_1271);  mul_1270 = mul_1271 = None
    unsqueeze_397: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1272, 0);  mul_1272 = None
    unsqueeze_398: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None
    mul_1273: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_29, view_43);  view_43 = None
    unsqueeze_399: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1273, 0);  mul_1273 = None
    unsqueeze_400: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 2);  unsqueeze_399 = None
    sub_236: "f32[1, 256, 512]" = torch.ops.aten.sub.Tensor(view_42, unsqueeze_394);  view_42 = unsqueeze_394 = None
    mul_1274: "f32[1, 256, 512]" = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_398);  sub_236 = unsqueeze_398 = None
    sub_237: "f32[1, 256, 512]" = torch.ops.aten.sub.Tensor(view_300, mul_1274);  view_300 = mul_1274 = None
    sub_238: "f32[1, 256, 512]" = torch.ops.aten.sub.Tensor(sub_237, unsqueeze_396);  sub_237 = unsqueeze_396 = None
    mul_1275: "f32[1, 256, 512]" = torch.ops.aten.mul.Tensor(sub_238, unsqueeze_400);  sub_238 = unsqueeze_400 = None
    mul_1276: "f32[256]" = torch.ops.aten.mul.Tensor(sum_170, squeeze_29);  sum_170 = squeeze_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_301: "f32[256, 1, 1, 1]" = torch.ops.aten.view.default(mul_1276, [256, 1, 1, 1]);  mul_1276 = None
    mul_1277: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_301, 0.04419417382415922);  view_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_302: "f32[256, 512, 1, 1]" = torch.ops.aten.view.default(mul_1275, [256, 512, 1, 1]);  mul_1275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_1278: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(getitem_300, 0.9805806756909201);  getitem_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1279: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1278, 1.7015043497085571);  mul_1278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1280: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_26, 0.7071067811865476)
    erf_92: "f32[4, 512, 24, 24]" = torch.ops.aten.erf.default(mul_1280);  mul_1280 = None
    add_220: "f32[4, 512, 24, 24]" = torch.ops.aten.add.Tensor(erf_92, 1);  erf_92 = None
    mul_1281: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_220, 0.5);  add_220 = None
    mul_1282: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_26, add_26)
    mul_1283: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1282, -0.5);  mul_1282 = None
    exp_40: "f32[4, 512, 24, 24]" = torch.ops.aten.exp.default(mul_1283);  mul_1283 = None
    mul_1284: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(exp_40, 0.3989422804014327);  exp_40 = None
    mul_1285: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_26, mul_1284);  add_26 = mul_1284 = None
    add_221: "f32[4, 512, 24, 24]" = torch.ops.aten.add.Tensor(mul_1281, mul_1285);  mul_1281 = mul_1285 = None
    mul_1286: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1279, add_221);  mul_1279 = add_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    add_222: "f32[4, 512, 24, 24]" = torch.ops.aten.add.Tensor(mul_1205, mul_1286);  mul_1205 = mul_1286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_1287: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_222, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    mul_1288: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1287, clone_1);  clone_1 = None
    mul_1289: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1287, primals_44);  mul_1287 = primals_44 = None
    sum_171: "f32[]" = torch.ops.aten.sum.default(mul_1288);  mul_1288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_1290: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1289, 2.0);  mul_1289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_1291: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1290, convolution_15);  convolution_15 = None
    mul_1292: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1290, sigmoid_1);  mul_1290 = sigmoid_1 = None
    sum_172: "f32[4, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1291, [2, 3], True);  mul_1291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_64: "f32[4, 512, 1, 1]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    sub_239: "f32[4, 512, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_64)
    mul_1293: "f32[4, 512, 1, 1]" = torch.ops.aten.mul.Tensor(alias_64, sub_239);  alias_64 = sub_239 = None
    mul_1294: "f32[4, 512, 1, 1]" = torch.ops.aten.mul.Tensor(sum_172, mul_1293);  sum_172 = mul_1293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_173: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1294, [0, 2, 3])
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(mul_1294, relu_1, primals_190, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1294 = primals_190 = None
    getitem_303: "f32[4, 256, 1, 1]" = convolution_backward_63[0]
    getitem_304: "f32[512, 256, 1, 1]" = convolution_backward_63[1];  convolution_backward_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_66: "f32[4, 256, 1, 1]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_67: "f32[4, 256, 1, 1]" = torch.ops.aten.alias.default(alias_66);  alias_66 = None
    le_10: "b8[4, 256, 1, 1]" = torch.ops.aten.le.Scalar(alias_67, 0);  alias_67 = None
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_10: "f32[4, 256, 1, 1]" = torch.ops.aten.where.self(le_10, scalar_tensor_10, getitem_303);  le_10 = scalar_tensor_10 = getitem_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_174: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(where_10, mean_1, primals_188, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_10 = mean_1 = primals_188 = None
    getitem_306: "f32[4, 512, 1, 1]" = convolution_backward_64[0]
    getitem_307: "f32[256, 512, 1, 1]" = convolution_backward_64[1];  convolution_backward_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_11: "f32[4, 512, 24, 24]" = torch.ops.aten.expand.default(getitem_306, [4, 512, 24, 24]);  getitem_306 = None
    div_11: "f32[4, 512, 24, 24]" = torch.ops.aten.div.Scalar(expand_11, 576);  expand_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_223: "f32[4, 512, 24, 24]" = torch.ops.aten.add.Tensor(mul_1292, div_11);  mul_1292 = div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_175: "f32[512]" = torch.ops.aten.sum.dim_IntList(add_223, [0, 2, 3])
    convolution_backward_65 = torch.ops.aten.convolution_backward.default(add_223, mul_88, view_41, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_223 = mul_88 = view_41 = None
    getitem_309: "f32[4, 256, 24, 24]" = convolution_backward_65[0]
    getitem_310: "f32[512, 256, 1, 1]" = convolution_backward_65[1];  convolution_backward_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_303: "f32[1, 512, 256]" = torch.ops.aten.view.default(getitem_310, [1, 512, 256]);  getitem_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_401: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_26, 0);  squeeze_26 = None
    unsqueeze_402: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 2);  unsqueeze_401 = None
    sum_176: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_303, [0, 2])
    sub_240: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(view_39, unsqueeze_402)
    mul_1295: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(view_303, sub_240);  sub_240 = None
    sum_177: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1295, [0, 2]);  mul_1295 = None
    mul_1296: "f32[512]" = torch.ops.aten.mul.Tensor(sum_176, 0.00390625);  sum_176 = None
    unsqueeze_403: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1296, 0);  mul_1296 = None
    unsqueeze_404: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 2);  unsqueeze_403 = None
    mul_1297: "f32[512]" = torch.ops.aten.mul.Tensor(sum_177, 0.00390625)
    mul_1298: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_27, squeeze_27)
    mul_1299: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1297, mul_1298);  mul_1297 = mul_1298 = None
    unsqueeze_405: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1299, 0);  mul_1299 = None
    unsqueeze_406: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 2);  unsqueeze_405 = None
    mul_1300: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_27, view_40);  view_40 = None
    unsqueeze_407: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1300, 0);  mul_1300 = None
    unsqueeze_408: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 2);  unsqueeze_407 = None
    sub_241: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(view_39, unsqueeze_402);  view_39 = unsqueeze_402 = None
    mul_1301: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_241, unsqueeze_406);  sub_241 = unsqueeze_406 = None
    sub_242: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(view_303, mul_1301);  view_303 = mul_1301 = None
    sub_243: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_242, unsqueeze_404);  sub_242 = unsqueeze_404 = None
    mul_1302: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_243, unsqueeze_408);  sub_243 = unsqueeze_408 = None
    mul_1303: "f32[512]" = torch.ops.aten.mul.Tensor(sum_177, squeeze_27);  sum_177 = squeeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_304: "f32[512, 1, 1, 1]" = torch.ops.aten.view.default(mul_1303, [512, 1, 1, 1]);  mul_1303 = None
    mul_1304: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_304, 0.0625);  view_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_305: "f32[512, 256, 1, 1]" = torch.ops.aten.view.default(mul_1302, [512, 256, 1, 1]);  mul_1302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1305: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(getitem_309, 1.7015043497085571);  getitem_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1306: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_14, 0.7071067811865476)
    erf_93: "f32[4, 256, 24, 24]" = torch.ops.aten.erf.default(mul_1306);  mul_1306 = None
    add_224: "f32[4, 256, 24, 24]" = torch.ops.aten.add.Tensor(erf_93, 1);  erf_93 = None
    mul_1307: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(add_224, 0.5);  add_224 = None
    mul_1308: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_14, convolution_14)
    mul_1309: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1308, -0.5);  mul_1308 = None
    exp_41: "f32[4, 256, 24, 24]" = torch.ops.aten.exp.default(mul_1309);  mul_1309 = None
    mul_1310: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(exp_41, 0.3989422804014327);  exp_41 = None
    mul_1311: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_14, mul_1310);  convolution_14 = mul_1310 = None
    add_225: "f32[4, 256, 24, 24]" = torch.ops.aten.add.Tensor(mul_1307, mul_1311);  mul_1307 = mul_1311 = None
    mul_1312: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1305, add_225);  mul_1305 = add_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_178: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1312, [0, 2, 3])
    convolution_backward_66 = torch.ops.aten.convolution_backward.default(mul_1312, mul_81, view_38, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_1312 = mul_81 = view_38 = None
    getitem_312: "f32[4, 256, 24, 24]" = convolution_backward_66[0]
    getitem_313: "f32[256, 128, 3, 3]" = convolution_backward_66[1];  convolution_backward_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_306: "f32[1, 256, 1152]" = torch.ops.aten.view.default(getitem_313, [1, 256, 1152]);  getitem_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_409: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_410: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 2);  unsqueeze_409 = None
    sum_179: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_306, [0, 2])
    sub_244: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(view_36, unsqueeze_410)
    mul_1313: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(view_306, sub_244);  sub_244 = None
    sum_180: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1313, [0, 2]);  mul_1313 = None
    mul_1314: "f32[256]" = torch.ops.aten.mul.Tensor(sum_179, 0.0008680555555555555);  sum_179 = None
    unsqueeze_411: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1314, 0);  mul_1314 = None
    unsqueeze_412: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 2);  unsqueeze_411 = None
    mul_1315: "f32[256]" = torch.ops.aten.mul.Tensor(sum_180, 0.0008680555555555555)
    mul_1316: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_1317: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1315, mul_1316);  mul_1315 = mul_1316 = None
    unsqueeze_413: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1317, 0);  mul_1317 = None
    unsqueeze_414: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 2);  unsqueeze_413 = None
    mul_1318: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_25, view_37);  view_37 = None
    unsqueeze_415: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1318, 0);  mul_1318 = None
    unsqueeze_416: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 2);  unsqueeze_415 = None
    sub_245: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(view_36, unsqueeze_410);  view_36 = unsqueeze_410 = None
    mul_1319: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(sub_245, unsqueeze_414);  sub_245 = unsqueeze_414 = None
    sub_246: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(view_306, mul_1319);  view_306 = mul_1319 = None
    sub_247: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(sub_246, unsqueeze_412);  sub_246 = unsqueeze_412 = None
    mul_1320: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(sub_247, unsqueeze_416);  sub_247 = unsqueeze_416 = None
    mul_1321: "f32[256]" = torch.ops.aten.mul.Tensor(sum_180, squeeze_25);  sum_180 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_307: "f32[256, 1, 1, 1]" = torch.ops.aten.view.default(mul_1321, [256, 1, 1, 1]);  mul_1321 = None
    mul_1322: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_307, 0.02946278254943948);  view_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_308: "f32[256, 128, 3, 3]" = torch.ops.aten.view.default(mul_1320, [256, 128, 3, 3]);  mul_1320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1323: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(getitem_312, 1.7015043497085571);  getitem_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1324: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_13, 0.7071067811865476)
    erf_94: "f32[4, 256, 24, 24]" = torch.ops.aten.erf.default(mul_1324);  mul_1324 = None
    add_226: "f32[4, 256, 24, 24]" = torch.ops.aten.add.Tensor(erf_94, 1);  erf_94 = None
    mul_1325: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(add_226, 0.5);  add_226 = None
    mul_1326: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_13, convolution_13)
    mul_1327: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1326, -0.5);  mul_1326 = None
    exp_42: "f32[4, 256, 24, 24]" = torch.ops.aten.exp.default(mul_1327);  mul_1327 = None
    mul_1328: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(exp_42, 0.3989422804014327);  exp_42 = None
    mul_1329: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_13, mul_1328);  convolution_13 = mul_1328 = None
    add_227: "f32[4, 256, 24, 24]" = torch.ops.aten.add.Tensor(mul_1325, mul_1329);  mul_1325 = mul_1329 = None
    mul_1330: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1323, add_227);  mul_1323 = add_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_181: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1330, [0, 2, 3])
    convolution_backward_67 = torch.ops.aten.convolution_backward.default(mul_1330, constant_pad_nd_2, view_35, [256], [2, 2], [0, 0], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_1330 = constant_pad_nd_2 = view_35 = None
    getitem_315: "f32[4, 256, 49, 49]" = convolution_backward_67[0]
    getitem_316: "f32[256, 128, 3, 3]" = convolution_backward_67[1];  convolution_backward_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_309: "f32[1, 256, 1152]" = torch.ops.aten.view.default(getitem_316, [1, 256, 1152]);  getitem_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_417: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_22, 0);  squeeze_22 = None
    unsqueeze_418: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 2);  unsqueeze_417 = None
    sum_182: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_309, [0, 2])
    sub_248: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(view_33, unsqueeze_418)
    mul_1331: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(view_309, sub_248);  sub_248 = None
    sum_183: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1331, [0, 2]);  mul_1331 = None
    mul_1332: "f32[256]" = torch.ops.aten.mul.Tensor(sum_182, 0.0008680555555555555);  sum_182 = None
    unsqueeze_419: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1332, 0);  mul_1332 = None
    unsqueeze_420: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 2);  unsqueeze_419 = None
    mul_1333: "f32[256]" = torch.ops.aten.mul.Tensor(sum_183, 0.0008680555555555555)
    mul_1334: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_23, squeeze_23)
    mul_1335: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1333, mul_1334);  mul_1333 = mul_1334 = None
    unsqueeze_421: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1335, 0);  mul_1335 = None
    unsqueeze_422: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 2);  unsqueeze_421 = None
    mul_1336: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_23, view_34);  view_34 = None
    unsqueeze_423: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1336, 0);  mul_1336 = None
    unsqueeze_424: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 2);  unsqueeze_423 = None
    sub_249: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(view_33, unsqueeze_418);  view_33 = unsqueeze_418 = None
    mul_1337: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(sub_249, unsqueeze_422);  sub_249 = unsqueeze_422 = None
    sub_250: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(view_309, mul_1337);  view_309 = mul_1337 = None
    sub_251: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(sub_250, unsqueeze_420);  sub_250 = unsqueeze_420 = None
    mul_1338: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(sub_251, unsqueeze_424);  sub_251 = unsqueeze_424 = None
    mul_1339: "f32[256]" = torch.ops.aten.mul.Tensor(sum_183, squeeze_23);  sum_183 = squeeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_310: "f32[256, 1, 1, 1]" = torch.ops.aten.view.default(mul_1339, [256, 1, 1, 1]);  mul_1339 = None
    mul_1340: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_310, 0.02946278254943948);  view_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_311: "f32[256, 128, 3, 3]" = torch.ops.aten.view.default(mul_1338, [256, 128, 3, 3]);  mul_1338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_7: "f32[4, 256, 48, 48]" = torch.ops.aten.constant_pad_nd.default(getitem_315, [0, -1, 0, -1]);  getitem_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1341: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(constant_pad_nd_7, 1.7015043497085571);  constant_pad_nd_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1342: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_12, 0.7071067811865476)
    erf_95: "f32[4, 256, 48, 48]" = torch.ops.aten.erf.default(mul_1342);  mul_1342 = None
    add_228: "f32[4, 256, 48, 48]" = torch.ops.aten.add.Tensor(erf_95, 1);  erf_95 = None
    mul_1343: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(add_228, 0.5);  add_228 = None
    mul_1344: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_12, convolution_12)
    mul_1345: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1344, -0.5);  mul_1344 = None
    exp_43: "f32[4, 256, 48, 48]" = torch.ops.aten.exp.default(mul_1345);  mul_1345 = None
    mul_1346: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(exp_43, 0.3989422804014327);  exp_43 = None
    mul_1347: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_12, mul_1346);  convolution_12 = mul_1346 = None
    add_229: "f32[4, 256, 48, 48]" = torch.ops.aten.add.Tensor(mul_1343, mul_1347);  mul_1343 = mul_1347 = None
    mul_1348: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1341, add_229);  mul_1341 = add_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_184: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1348, [0, 2, 3])
    convolution_backward_68 = torch.ops.aten.convolution_backward.default(mul_1348, mul_64, view_32, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1348 = view_32 = None
    getitem_318: "f32[4, 256, 48, 48]" = convolution_backward_68[0]
    getitem_319: "f32[256, 256, 1, 1]" = convolution_backward_68[1];  convolution_backward_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_312: "f32[1, 256, 256]" = torch.ops.aten.view.default(getitem_319, [1, 256, 256]);  getitem_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_425: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_20, 0);  squeeze_20 = None
    unsqueeze_426: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 2);  unsqueeze_425 = None
    sum_185: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_312, [0, 2])
    sub_252: "f32[1, 256, 256]" = torch.ops.aten.sub.Tensor(view_30, unsqueeze_426)
    mul_1349: "f32[1, 256, 256]" = torch.ops.aten.mul.Tensor(view_312, sub_252);  sub_252 = None
    sum_186: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1349, [0, 2]);  mul_1349 = None
    mul_1350: "f32[256]" = torch.ops.aten.mul.Tensor(sum_185, 0.00390625);  sum_185 = None
    unsqueeze_427: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1350, 0);  mul_1350 = None
    unsqueeze_428: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 2);  unsqueeze_427 = None
    mul_1351: "f32[256]" = torch.ops.aten.mul.Tensor(sum_186, 0.00390625)
    mul_1352: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_21, squeeze_21)
    mul_1353: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1351, mul_1352);  mul_1351 = mul_1352 = None
    unsqueeze_429: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1353, 0);  mul_1353 = None
    unsqueeze_430: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 2);  unsqueeze_429 = None
    mul_1354: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_21, view_31);  view_31 = None
    unsqueeze_431: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1354, 0);  mul_1354 = None
    unsqueeze_432: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 2);  unsqueeze_431 = None
    sub_253: "f32[1, 256, 256]" = torch.ops.aten.sub.Tensor(view_30, unsqueeze_426);  view_30 = unsqueeze_426 = None
    mul_1355: "f32[1, 256, 256]" = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_430);  sub_253 = unsqueeze_430 = None
    sub_254: "f32[1, 256, 256]" = torch.ops.aten.sub.Tensor(view_312, mul_1355);  view_312 = mul_1355 = None
    sub_255: "f32[1, 256, 256]" = torch.ops.aten.sub.Tensor(sub_254, unsqueeze_428);  sub_254 = unsqueeze_428 = None
    mul_1356: "f32[1, 256, 256]" = torch.ops.aten.mul.Tensor(sub_255, unsqueeze_432);  sub_255 = unsqueeze_432 = None
    mul_1357: "f32[256]" = torch.ops.aten.mul.Tensor(sum_186, squeeze_21);  sum_186 = squeeze_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_313: "f32[256, 1, 1, 1]" = torch.ops.aten.view.default(mul_1357, [256, 1, 1, 1]);  mul_1357 = None
    mul_1358: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_313, 0.0625);  view_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_314: "f32[256, 256, 1, 1]" = torch.ops.aten.view.default(mul_1356, [256, 256, 1, 1]);  mul_1356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_187: "f32[512]" = torch.ops.aten.sum.dim_IntList(add_222, [0, 2, 3])
    convolution_backward_69 = torch.ops.aten.convolution_backward.default(add_222, avg_pool2d, view_29, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_222 = avg_pool2d = view_29 = None
    getitem_321: "f32[4, 256, 24, 24]" = convolution_backward_69[0]
    getitem_322: "f32[512, 256, 1, 1]" = convolution_backward_69[1];  convolution_backward_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_315: "f32[1, 512, 256]" = torch.ops.aten.view.default(getitem_322, [1, 512, 256]);  getitem_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_433: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_434: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 2);  unsqueeze_433 = None
    sum_188: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_315, [0, 2])
    sub_256: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(view_27, unsqueeze_434)
    mul_1359: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(view_315, sub_256);  sub_256 = None
    sum_189: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1359, [0, 2]);  mul_1359 = None
    mul_1360: "f32[512]" = torch.ops.aten.mul.Tensor(sum_188, 0.00390625);  sum_188 = None
    unsqueeze_435: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1360, 0);  mul_1360 = None
    unsqueeze_436: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 2);  unsqueeze_435 = None
    mul_1361: "f32[512]" = torch.ops.aten.mul.Tensor(sum_189, 0.00390625)
    mul_1362: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_1363: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1361, mul_1362);  mul_1361 = mul_1362 = None
    unsqueeze_437: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1363, 0);  mul_1363 = None
    unsqueeze_438: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 2);  unsqueeze_437 = None
    mul_1364: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_19, view_28);  view_28 = None
    unsqueeze_439: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1364, 0);  mul_1364 = None
    unsqueeze_440: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 2);  unsqueeze_439 = None
    sub_257: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(view_27, unsqueeze_434);  view_27 = unsqueeze_434 = None
    mul_1365: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_257, unsqueeze_438);  sub_257 = unsqueeze_438 = None
    sub_258: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(view_315, mul_1365);  view_315 = mul_1365 = None
    sub_259: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_258, unsqueeze_436);  sub_258 = unsqueeze_436 = None
    mul_1366: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_259, unsqueeze_440);  sub_259 = unsqueeze_440 = None
    mul_1367: "f32[512]" = torch.ops.aten.mul.Tensor(sum_189, squeeze_19);  sum_189 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_316: "f32[512, 1, 1, 1]" = torch.ops.aten.view.default(mul_1367, [512, 1, 1, 1]);  mul_1367 = None
    mul_1368: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_316, 0.0625);  view_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_317: "f32[512, 256, 1, 1]" = torch.ops.aten.view.default(mul_1366, [512, 256, 1, 1]);  mul_1366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    avg_pool2d_backward_2: "f32[4, 256, 48, 48]" = torch.ops.aten.avg_pool2d_backward.default(getitem_321, mul_64, [2, 2], [2, 2], [0, 0], True, False, None);  getitem_321 = mul_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    add_230: "f32[4, 256, 48, 48]" = torch.ops.aten.add.Tensor(getitem_318, avg_pool2d_backward_2);  getitem_318 = avg_pool2d_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_1369: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(add_230, 0.9805806756909201);  add_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1370: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1369, 1.7015043497085571);  mul_1369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1371: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(add_16, 0.7071067811865476)
    erf_96: "f32[4, 256, 48, 48]" = torch.ops.aten.erf.default(mul_1371);  mul_1371 = None
    add_231: "f32[4, 256, 48, 48]" = torch.ops.aten.add.Tensor(erf_96, 1);  erf_96 = None
    mul_1372: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(add_231, 0.5);  add_231 = None
    mul_1373: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(add_16, add_16)
    mul_1374: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1373, -0.5);  mul_1373 = None
    exp_44: "f32[4, 256, 48, 48]" = torch.ops.aten.exp.default(mul_1374);  mul_1374 = None
    mul_1375: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(exp_44, 0.3989422804014327);  exp_44 = None
    mul_1376: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(add_16, mul_1375);  add_16 = mul_1375 = None
    add_232: "f32[4, 256, 48, 48]" = torch.ops.aten.add.Tensor(mul_1372, mul_1376);  mul_1372 = mul_1376 = None
    mul_1377: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1370, add_232);  mul_1370 = add_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_1378: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1377, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    mul_1379: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1378, clone);  clone = None
    mul_1380: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1378, primals_28);  mul_1378 = primals_28 = None
    sum_190: "f32[]" = torch.ops.aten.sum.default(mul_1379);  mul_1379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_1381: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1380, 2.0);  mul_1380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_1382: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1381, convolution_8);  convolution_8 = None
    mul_1383: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1381, sigmoid);  mul_1381 = sigmoid = None
    sum_191: "f32[4, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1382, [2, 3], True);  mul_1382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_68: "f32[4, 256, 1, 1]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    sub_260: "f32[4, 256, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_68)
    mul_1384: "f32[4, 256, 1, 1]" = torch.ops.aten.mul.Tensor(alias_68, sub_260);  alias_68 = sub_260 = None
    mul_1385: "f32[4, 256, 1, 1]" = torch.ops.aten.mul.Tensor(sum_191, mul_1384);  sum_191 = mul_1384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_192: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1385, [0, 2, 3])
    convolution_backward_70 = torch.ops.aten.convolution_backward.default(mul_1385, relu, primals_186, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1385 = primals_186 = None
    getitem_324: "f32[4, 128, 1, 1]" = convolution_backward_70[0]
    getitem_325: "f32[256, 128, 1, 1]" = convolution_backward_70[1];  convolution_backward_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_70: "f32[4, 128, 1, 1]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_71: "f32[4, 128, 1, 1]" = torch.ops.aten.alias.default(alias_70);  alias_70 = None
    le_11: "b8[4, 128, 1, 1]" = torch.ops.aten.le.Scalar(alias_71, 0);  alias_71 = None
    scalar_tensor_11: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_11: "f32[4, 128, 1, 1]" = torch.ops.aten.where.self(le_11, scalar_tensor_11, getitem_324);  le_11 = scalar_tensor_11 = getitem_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_193: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    convolution_backward_71 = torch.ops.aten.convolution_backward.default(where_11, mean, primals_184, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_11 = mean = primals_184 = None
    getitem_327: "f32[4, 256, 1, 1]" = convolution_backward_71[0]
    getitem_328: "f32[128, 256, 1, 1]" = convolution_backward_71[1];  convolution_backward_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_12: "f32[4, 256, 48, 48]" = torch.ops.aten.expand.default(getitem_327, [4, 256, 48, 48]);  getitem_327 = None
    div_12: "f32[4, 256, 48, 48]" = torch.ops.aten.div.Scalar(expand_12, 2304);  expand_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_233: "f32[4, 256, 48, 48]" = torch.ops.aten.add.Tensor(mul_1383, div_12);  mul_1383 = div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_194: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_233, [0, 2, 3])
    convolution_backward_72 = torch.ops.aten.convolution_backward.default(add_233, mul_52, view_26, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_233 = mul_52 = view_26 = None
    getitem_330: "f32[4, 128, 48, 48]" = convolution_backward_72[0]
    getitem_331: "f32[256, 128, 1, 1]" = convolution_backward_72[1];  convolution_backward_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_318: "f32[1, 256, 128]" = torch.ops.aten.view.default(getitem_331, [1, 256, 128]);  getitem_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_441: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_16, 0);  squeeze_16 = None
    unsqueeze_442: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 2);  unsqueeze_441 = None
    sum_195: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_318, [0, 2])
    sub_261: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(view_24, unsqueeze_442)
    mul_1386: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(view_318, sub_261);  sub_261 = None
    sum_196: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1386, [0, 2]);  mul_1386 = None
    mul_1387: "f32[256]" = torch.ops.aten.mul.Tensor(sum_195, 0.0078125);  sum_195 = None
    unsqueeze_443: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1387, 0);  mul_1387 = None
    unsqueeze_444: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 2);  unsqueeze_443 = None
    mul_1388: "f32[256]" = torch.ops.aten.mul.Tensor(sum_196, 0.0078125)
    mul_1389: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_17, squeeze_17)
    mul_1390: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1388, mul_1389);  mul_1388 = mul_1389 = None
    unsqueeze_445: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1390, 0);  mul_1390 = None
    unsqueeze_446: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 2);  unsqueeze_445 = None
    mul_1391: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_17, view_25);  view_25 = None
    unsqueeze_447: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1391, 0);  mul_1391 = None
    unsqueeze_448: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 2);  unsqueeze_447 = None
    sub_262: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(view_24, unsqueeze_442);  view_24 = unsqueeze_442 = None
    mul_1392: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(sub_262, unsqueeze_446);  sub_262 = unsqueeze_446 = None
    sub_263: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(view_318, mul_1392);  view_318 = mul_1392 = None
    sub_264: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(sub_263, unsqueeze_444);  sub_263 = unsqueeze_444 = None
    mul_1393: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_448);  sub_264 = unsqueeze_448 = None
    mul_1394: "f32[256]" = torch.ops.aten.mul.Tensor(sum_196, squeeze_17);  sum_196 = squeeze_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_319: "f32[256, 1, 1, 1]" = torch.ops.aten.view.default(mul_1394, [256, 1, 1, 1]);  mul_1394 = None
    mul_1395: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_319, 0.08838834764831845);  view_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_320: "f32[256, 128, 1, 1]" = torch.ops.aten.view.default(mul_1393, [256, 128, 1, 1]);  mul_1393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1396: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(getitem_330, 1.7015043497085571);  getitem_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1397: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_7, 0.7071067811865476)
    erf_97: "f32[4, 128, 48, 48]" = torch.ops.aten.erf.default(mul_1397);  mul_1397 = None
    add_234: "f32[4, 128, 48, 48]" = torch.ops.aten.add.Tensor(erf_97, 1);  erf_97 = None
    mul_1398: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(add_234, 0.5);  add_234 = None
    mul_1399: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_7, convolution_7)
    mul_1400: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1399, -0.5);  mul_1399 = None
    exp_45: "f32[4, 128, 48, 48]" = torch.ops.aten.exp.default(mul_1400);  mul_1400 = None
    mul_1401: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(exp_45, 0.3989422804014327);  exp_45 = None
    mul_1402: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_7, mul_1401);  convolution_7 = mul_1401 = None
    add_235: "f32[4, 128, 48, 48]" = torch.ops.aten.add.Tensor(mul_1398, mul_1402);  mul_1398 = mul_1402 = None
    mul_1403: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1396, add_235);  mul_1396 = add_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_197: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1403, [0, 2, 3])
    convolution_backward_73 = torch.ops.aten.convolution_backward.default(mul_1403, mul_45, view_23, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1403 = mul_45 = view_23 = None
    getitem_333: "f32[4, 128, 48, 48]" = convolution_backward_73[0]
    getitem_334: "f32[128, 128, 3, 3]" = convolution_backward_73[1];  convolution_backward_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_321: "f32[1, 128, 1152]" = torch.ops.aten.view.default(getitem_334, [1, 128, 1152]);  getitem_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_449: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_14, 0);  squeeze_14 = None
    unsqueeze_450: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 2);  unsqueeze_449 = None
    sum_198: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_321, [0, 2])
    sub_265: "f32[1, 128, 1152]" = torch.ops.aten.sub.Tensor(view_21, unsqueeze_450)
    mul_1404: "f32[1, 128, 1152]" = torch.ops.aten.mul.Tensor(view_321, sub_265);  sub_265 = None
    sum_199: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1404, [0, 2]);  mul_1404 = None
    mul_1405: "f32[128]" = torch.ops.aten.mul.Tensor(sum_198, 0.0008680555555555555);  sum_198 = None
    unsqueeze_451: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1405, 0);  mul_1405 = None
    unsqueeze_452: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 2);  unsqueeze_451 = None
    mul_1406: "f32[128]" = torch.ops.aten.mul.Tensor(sum_199, 0.0008680555555555555)
    mul_1407: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_15, squeeze_15)
    mul_1408: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1406, mul_1407);  mul_1406 = mul_1407 = None
    unsqueeze_453: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1408, 0);  mul_1408 = None
    unsqueeze_454: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 2);  unsqueeze_453 = None
    mul_1409: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_15, view_22);  view_22 = None
    unsqueeze_455: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1409, 0);  mul_1409 = None
    unsqueeze_456: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 2);  unsqueeze_455 = None
    sub_266: "f32[1, 128, 1152]" = torch.ops.aten.sub.Tensor(view_21, unsqueeze_450);  view_21 = unsqueeze_450 = None
    mul_1410: "f32[1, 128, 1152]" = torch.ops.aten.mul.Tensor(sub_266, unsqueeze_454);  sub_266 = unsqueeze_454 = None
    sub_267: "f32[1, 128, 1152]" = torch.ops.aten.sub.Tensor(view_321, mul_1410);  view_321 = mul_1410 = None
    sub_268: "f32[1, 128, 1152]" = torch.ops.aten.sub.Tensor(sub_267, unsqueeze_452);  sub_267 = unsqueeze_452 = None
    mul_1411: "f32[1, 128, 1152]" = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_456);  sub_268 = unsqueeze_456 = None
    mul_1412: "f32[128]" = torch.ops.aten.mul.Tensor(sum_199, squeeze_15);  sum_199 = squeeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_322: "f32[128, 1, 1, 1]" = torch.ops.aten.view.default(mul_1412, [128, 1, 1, 1]);  mul_1412 = None
    mul_1413: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_322, 0.02946278254943948);  view_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_323: "f32[128, 128, 3, 3]" = torch.ops.aten.view.default(mul_1411, [128, 128, 3, 3]);  mul_1411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1414: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(getitem_333, 1.7015043497085571);  getitem_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1415: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_6, 0.7071067811865476)
    erf_98: "f32[4, 128, 48, 48]" = torch.ops.aten.erf.default(mul_1415);  mul_1415 = None
    add_236: "f32[4, 128, 48, 48]" = torch.ops.aten.add.Tensor(erf_98, 1);  erf_98 = None
    mul_1416: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(add_236, 0.5);  add_236 = None
    mul_1417: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_6, convolution_6)
    mul_1418: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1417, -0.5);  mul_1417 = None
    exp_46: "f32[4, 128, 48, 48]" = torch.ops.aten.exp.default(mul_1418);  mul_1418 = None
    mul_1419: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(exp_46, 0.3989422804014327);  exp_46 = None
    mul_1420: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_6, mul_1419);  convolution_6 = mul_1419 = None
    add_237: "f32[4, 128, 48, 48]" = torch.ops.aten.add.Tensor(mul_1416, mul_1420);  mul_1416 = mul_1420 = None
    mul_1421: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1414, add_237);  mul_1414 = add_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_200: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1421, [0, 2, 3])
    convolution_backward_74 = torch.ops.aten.convolution_backward.default(mul_1421, mul_38, view_20, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1421 = mul_38 = view_20 = None
    getitem_336: "f32[4, 128, 48, 48]" = convolution_backward_74[0]
    getitem_337: "f32[128, 128, 3, 3]" = convolution_backward_74[1];  convolution_backward_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_324: "f32[1, 128, 1152]" = torch.ops.aten.view.default(getitem_337, [1, 128, 1152]);  getitem_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_457: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_458: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 2);  unsqueeze_457 = None
    sum_201: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_324, [0, 2])
    sub_269: "f32[1, 128, 1152]" = torch.ops.aten.sub.Tensor(view_18, unsqueeze_458)
    mul_1422: "f32[1, 128, 1152]" = torch.ops.aten.mul.Tensor(view_324, sub_269);  sub_269 = None
    sum_202: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1422, [0, 2]);  mul_1422 = None
    mul_1423: "f32[128]" = torch.ops.aten.mul.Tensor(sum_201, 0.0008680555555555555);  sum_201 = None
    unsqueeze_459: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1423, 0);  mul_1423 = None
    unsqueeze_460: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 2);  unsqueeze_459 = None
    mul_1424: "f32[128]" = torch.ops.aten.mul.Tensor(sum_202, 0.0008680555555555555)
    mul_1425: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_1426: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1424, mul_1425);  mul_1424 = mul_1425 = None
    unsqueeze_461: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1426, 0);  mul_1426 = None
    unsqueeze_462: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None
    mul_1427: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_13, view_19);  view_19 = None
    unsqueeze_463: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1427, 0);  mul_1427 = None
    unsqueeze_464: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 2);  unsqueeze_463 = None
    sub_270: "f32[1, 128, 1152]" = torch.ops.aten.sub.Tensor(view_18, unsqueeze_458);  view_18 = unsqueeze_458 = None
    mul_1428: "f32[1, 128, 1152]" = torch.ops.aten.mul.Tensor(sub_270, unsqueeze_462);  sub_270 = unsqueeze_462 = None
    sub_271: "f32[1, 128, 1152]" = torch.ops.aten.sub.Tensor(view_324, mul_1428);  view_324 = mul_1428 = None
    sub_272: "f32[1, 128, 1152]" = torch.ops.aten.sub.Tensor(sub_271, unsqueeze_460);  sub_271 = unsqueeze_460 = None
    mul_1429: "f32[1, 128, 1152]" = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_464);  sub_272 = unsqueeze_464 = None
    mul_1430: "f32[128]" = torch.ops.aten.mul.Tensor(sum_202, squeeze_13);  sum_202 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_325: "f32[128, 1, 1, 1]" = torch.ops.aten.view.default(mul_1430, [128, 1, 1, 1]);  mul_1430 = None
    mul_1431: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_325, 0.02946278254943948);  view_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_326: "f32[128, 128, 3, 3]" = torch.ops.aten.view.default(mul_1429, [128, 128, 3, 3]);  mul_1429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1432: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(getitem_336, 1.7015043497085571);  getitem_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1433: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_5, 0.7071067811865476)
    erf_99: "f32[4, 128, 48, 48]" = torch.ops.aten.erf.default(mul_1433);  mul_1433 = None
    add_238: "f32[4, 128, 48, 48]" = torch.ops.aten.add.Tensor(erf_99, 1);  erf_99 = None
    mul_1434: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(add_238, 0.5);  add_238 = None
    mul_1435: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_5, convolution_5)
    mul_1436: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1435, -0.5);  mul_1435 = None
    exp_47: "f32[4, 128, 48, 48]" = torch.ops.aten.exp.default(mul_1436);  mul_1436 = None
    mul_1437: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(exp_47, 0.3989422804014327);  exp_47 = None
    mul_1438: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_5, mul_1437);  convolution_5 = mul_1437 = None
    add_239: "f32[4, 128, 48, 48]" = torch.ops.aten.add.Tensor(mul_1434, mul_1438);  mul_1434 = mul_1438 = None
    mul_1439: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1432, add_239);  mul_1432 = add_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_203: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1439, [0, 2, 3])
    convolution_backward_75 = torch.ops.aten.convolution_backward.default(mul_1439, mul_28, view_17, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1439 = view_17 = None
    getitem_339: "f32[4, 128, 48, 48]" = convolution_backward_75[0]
    getitem_340: "f32[128, 128, 1, 1]" = convolution_backward_75[1];  convolution_backward_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_327: "f32[1, 128, 128]" = torch.ops.aten.view.default(getitem_340, [1, 128, 128]);  getitem_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_465: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_10, 0);  squeeze_10 = None
    unsqueeze_466: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 2);  unsqueeze_465 = None
    sum_204: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_327, [0, 2])
    sub_273: "f32[1, 128, 128]" = torch.ops.aten.sub.Tensor(view_15, unsqueeze_466)
    mul_1440: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_327, sub_273);  sub_273 = None
    sum_205: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1440, [0, 2]);  mul_1440 = None
    mul_1441: "f32[128]" = torch.ops.aten.mul.Tensor(sum_204, 0.0078125);  sum_204 = None
    unsqueeze_467: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1441, 0);  mul_1441 = None
    unsqueeze_468: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 2);  unsqueeze_467 = None
    mul_1442: "f32[128]" = torch.ops.aten.mul.Tensor(sum_205, 0.0078125)
    mul_1443: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_11, squeeze_11)
    mul_1444: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1442, mul_1443);  mul_1442 = mul_1443 = None
    unsqueeze_469: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1444, 0);  mul_1444 = None
    unsqueeze_470: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 2);  unsqueeze_469 = None
    mul_1445: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_11, view_16);  view_16 = None
    unsqueeze_471: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1445, 0);  mul_1445 = None
    unsqueeze_472: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 2);  unsqueeze_471 = None
    sub_274: "f32[1, 128, 128]" = torch.ops.aten.sub.Tensor(view_15, unsqueeze_466);  view_15 = unsqueeze_466 = None
    mul_1446: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(sub_274, unsqueeze_470);  sub_274 = unsqueeze_470 = None
    sub_275: "f32[1, 128, 128]" = torch.ops.aten.sub.Tensor(view_327, mul_1446);  view_327 = mul_1446 = None
    sub_276: "f32[1, 128, 128]" = torch.ops.aten.sub.Tensor(sub_275, unsqueeze_468);  sub_275 = unsqueeze_468 = None
    mul_1447: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(sub_276, unsqueeze_472);  sub_276 = unsqueeze_472 = None
    mul_1448: "f32[128]" = torch.ops.aten.mul.Tensor(sum_205, squeeze_11);  sum_205 = squeeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_328: "f32[128, 1, 1, 1]" = torch.ops.aten.view.default(mul_1448, [128, 1, 1, 1]);  mul_1448 = None
    mul_1449: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_328, 0.08838834764831845);  view_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_329: "f32[128, 128, 1, 1]" = torch.ops.aten.view.default(mul_1447, [128, 128, 1, 1]);  mul_1447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_206: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1377, [0, 2, 3])
    convolution_backward_76 = torch.ops.aten.convolution_backward.default(mul_1377, mul_28, view_14, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1377 = mul_28 = view_14 = None
    getitem_342: "f32[4, 128, 48, 48]" = convolution_backward_76[0]
    getitem_343: "f32[256, 128, 1, 1]" = convolution_backward_76[1];  convolution_backward_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    add_240: "f32[4, 128, 48, 48]" = torch.ops.aten.add.Tensor(getitem_339, getitem_342);  getitem_339 = getitem_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_330: "f32[1, 256, 128]" = torch.ops.aten.view.default(getitem_343, [1, 256, 128]);  getitem_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_473: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_8, 0);  squeeze_8 = None
    unsqueeze_474: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 2);  unsqueeze_473 = None
    sum_207: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_330, [0, 2])
    sub_277: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(view_12, unsqueeze_474)
    mul_1450: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(view_330, sub_277);  sub_277 = None
    sum_208: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1450, [0, 2]);  mul_1450 = None
    mul_1451: "f32[256]" = torch.ops.aten.mul.Tensor(sum_207, 0.0078125);  sum_207 = None
    unsqueeze_475: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1451, 0);  mul_1451 = None
    unsqueeze_476: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 2);  unsqueeze_475 = None
    mul_1452: "f32[256]" = torch.ops.aten.mul.Tensor(sum_208, 0.0078125)
    mul_1453: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_9, squeeze_9)
    mul_1454: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1452, mul_1453);  mul_1452 = mul_1453 = None
    unsqueeze_477: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1454, 0);  mul_1454 = None
    unsqueeze_478: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 2);  unsqueeze_477 = None
    mul_1455: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_9, view_13);  view_13 = None
    unsqueeze_479: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1455, 0);  mul_1455 = None
    unsqueeze_480: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 2);  unsqueeze_479 = None
    sub_278: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(view_12, unsqueeze_474);  view_12 = unsqueeze_474 = None
    mul_1456: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(sub_278, unsqueeze_478);  sub_278 = unsqueeze_478 = None
    sub_279: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(view_330, mul_1456);  view_330 = mul_1456 = None
    sub_280: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(sub_279, unsqueeze_476);  sub_279 = unsqueeze_476 = None
    mul_1457: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(sub_280, unsqueeze_480);  sub_280 = unsqueeze_480 = None
    mul_1458: "f32[256]" = torch.ops.aten.mul.Tensor(sum_208, squeeze_9);  sum_208 = squeeze_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_331: "f32[256, 1, 1, 1]" = torch.ops.aten.view.default(mul_1458, [256, 1, 1, 1]);  mul_1458 = None
    mul_1459: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_331, 0.08838834764831845);  view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_332: "f32[256, 128, 1, 1]" = torch.ops.aten.view.default(mul_1457, [256, 128, 1, 1]);  mul_1457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_1460: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(add_240, 1.0);  add_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1461: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1460, 1.7015043497085571);  mul_1460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1462: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_3, 0.7071067811865476)
    erf_100: "f32[4, 128, 48, 48]" = torch.ops.aten.erf.default(mul_1462);  mul_1462 = None
    add_241: "f32[4, 128, 48, 48]" = torch.ops.aten.add.Tensor(erf_100, 1);  erf_100 = None
    mul_1463: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(add_241, 0.5);  add_241 = None
    mul_1464: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_3, convolution_3)
    mul_1465: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1464, -0.5);  mul_1464 = None
    exp_48: "f32[4, 128, 48, 48]" = torch.ops.aten.exp.default(mul_1465);  mul_1465 = None
    mul_1466: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(exp_48, 0.3989422804014327);  exp_48 = None
    mul_1467: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_3, mul_1466);  convolution_3 = mul_1466 = None
    add_242: "f32[4, 128, 48, 48]" = torch.ops.aten.add.Tensor(mul_1463, mul_1467);  mul_1463 = mul_1467 = None
    mul_1468: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1461, add_242);  mul_1461 = add_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_209: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1468, [0, 2, 3])
    convolution_backward_77 = torch.ops.aten.convolution_backward.default(mul_1468, constant_pad_nd_1, view_11, [128], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1468 = constant_pad_nd_1 = view_11 = None
    getitem_345: "f32[4, 64, 97, 97]" = convolution_backward_77[0]
    getitem_346: "f32[128, 64, 3, 3]" = convolution_backward_77[1];  convolution_backward_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_333: "f32[1, 128, 576]" = torch.ops.aten.view.default(getitem_346, [1, 128, 576]);  getitem_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_481: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_482: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 2);  unsqueeze_481 = None
    sum_210: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_333, [0, 2])
    sub_281: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_9, unsqueeze_482)
    mul_1469: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(view_333, sub_281);  sub_281 = None
    sum_211: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1469, [0, 2]);  mul_1469 = None
    mul_1470: "f32[128]" = torch.ops.aten.mul.Tensor(sum_210, 0.001736111111111111);  sum_210 = None
    unsqueeze_483: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1470, 0);  mul_1470 = None
    unsqueeze_484: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 2);  unsqueeze_483 = None
    mul_1471: "f32[128]" = torch.ops.aten.mul.Tensor(sum_211, 0.001736111111111111)
    mul_1472: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_1473: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1471, mul_1472);  mul_1471 = mul_1472 = None
    unsqueeze_485: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1473, 0);  mul_1473 = None
    unsqueeze_486: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 2);  unsqueeze_485 = None
    mul_1474: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_7, view_10);  view_10 = None
    unsqueeze_487: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1474, 0);  mul_1474 = None
    unsqueeze_488: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 2);  unsqueeze_487 = None
    sub_282: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_9, unsqueeze_482);  view_9 = unsqueeze_482 = None
    mul_1475: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_282, unsqueeze_486);  sub_282 = unsqueeze_486 = None
    sub_283: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_333, mul_1475);  view_333 = mul_1475 = None
    sub_284: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(sub_283, unsqueeze_484);  sub_283 = unsqueeze_484 = None
    mul_1476: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_284, unsqueeze_488);  sub_284 = unsqueeze_488 = None
    mul_1477: "f32[128]" = torch.ops.aten.mul.Tensor(sum_211, squeeze_7);  sum_211 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_334: "f32[128, 1, 1, 1]" = torch.ops.aten.view.default(mul_1477, [128, 1, 1, 1]);  mul_1477 = None
    mul_1478: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_334, 0.041666666666666664);  view_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_335: "f32[128, 64, 3, 3]" = torch.ops.aten.view.default(mul_1476, [128, 64, 3, 3]);  mul_1476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_8: "f32[4, 64, 96, 96]" = torch.ops.aten.constant_pad_nd.default(getitem_345, [0, -1, 0, -1]);  getitem_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1479: "f32[4, 64, 96, 96]" = torch.ops.aten.mul.Tensor(constant_pad_nd_8, 1.7015043497085571);  constant_pad_nd_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1480: "f32[4, 64, 96, 96]" = torch.ops.aten.mul.Tensor(convolution_2, 0.7071067811865476)
    erf_101: "f32[4, 64, 96, 96]" = torch.ops.aten.erf.default(mul_1480);  mul_1480 = None
    add_243: "f32[4, 64, 96, 96]" = torch.ops.aten.add.Tensor(erf_101, 1);  erf_101 = None
    mul_1481: "f32[4, 64, 96, 96]" = torch.ops.aten.mul.Tensor(add_243, 0.5);  add_243 = None
    mul_1482: "f32[4, 64, 96, 96]" = torch.ops.aten.mul.Tensor(convolution_2, convolution_2)
    mul_1483: "f32[4, 64, 96, 96]" = torch.ops.aten.mul.Tensor(mul_1482, -0.5);  mul_1482 = None
    exp_49: "f32[4, 64, 96, 96]" = torch.ops.aten.exp.default(mul_1483);  mul_1483 = None
    mul_1484: "f32[4, 64, 96, 96]" = torch.ops.aten.mul.Tensor(exp_49, 0.3989422804014327);  exp_49 = None
    mul_1485: "f32[4, 64, 96, 96]" = torch.ops.aten.mul.Tensor(convolution_2, mul_1484);  convolution_2 = mul_1484 = None
    add_244: "f32[4, 64, 96, 96]" = torch.ops.aten.add.Tensor(mul_1481, mul_1485);  mul_1481 = mul_1485 = None
    mul_1486: "f32[4, 64, 96, 96]" = torch.ops.aten.mul.Tensor(mul_1479, add_244);  mul_1479 = add_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_212: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1486, [0, 2, 3])
    convolution_backward_78 = torch.ops.aten.convolution_backward.default(mul_1486, mul_13, view_8, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1486 = mul_13 = view_8 = None
    getitem_348: "f32[4, 32, 96, 96]" = convolution_backward_78[0]
    getitem_349: "f32[64, 32, 3, 3]" = convolution_backward_78[1];  convolution_backward_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_336: "f32[1, 64, 288]" = torch.ops.aten.view.default(getitem_349, [1, 64, 288]);  getitem_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_489: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_4, 0);  squeeze_4 = None
    unsqueeze_490: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 2);  unsqueeze_489 = None
    sum_213: "f32[64]" = torch.ops.aten.sum.dim_IntList(view_336, [0, 2])
    sub_285: "f32[1, 64, 288]" = torch.ops.aten.sub.Tensor(view_6, unsqueeze_490)
    mul_1487: "f32[1, 64, 288]" = torch.ops.aten.mul.Tensor(view_336, sub_285);  sub_285 = None
    sum_214: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1487, [0, 2]);  mul_1487 = None
    mul_1488: "f32[64]" = torch.ops.aten.mul.Tensor(sum_213, 0.003472222222222222);  sum_213 = None
    unsqueeze_491: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1488, 0);  mul_1488 = None
    unsqueeze_492: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 2);  unsqueeze_491 = None
    mul_1489: "f32[64]" = torch.ops.aten.mul.Tensor(sum_214, 0.003472222222222222)
    mul_1490: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_5, squeeze_5)
    mul_1491: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1489, mul_1490);  mul_1489 = mul_1490 = None
    unsqueeze_493: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1491, 0);  mul_1491 = None
    unsqueeze_494: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 2);  unsqueeze_493 = None
    mul_1492: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_5, view_7);  view_7 = None
    unsqueeze_495: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1492, 0);  mul_1492 = None
    unsqueeze_496: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 2);  unsqueeze_495 = None
    sub_286: "f32[1, 64, 288]" = torch.ops.aten.sub.Tensor(view_6, unsqueeze_490);  view_6 = unsqueeze_490 = None
    mul_1493: "f32[1, 64, 288]" = torch.ops.aten.mul.Tensor(sub_286, unsqueeze_494);  sub_286 = unsqueeze_494 = None
    sub_287: "f32[1, 64, 288]" = torch.ops.aten.sub.Tensor(view_336, mul_1493);  view_336 = mul_1493 = None
    sub_288: "f32[1, 64, 288]" = torch.ops.aten.sub.Tensor(sub_287, unsqueeze_492);  sub_287 = unsqueeze_492 = None
    mul_1494: "f32[1, 64, 288]" = torch.ops.aten.mul.Tensor(sub_288, unsqueeze_496);  sub_288 = unsqueeze_496 = None
    mul_1495: "f32[64]" = torch.ops.aten.mul.Tensor(sum_214, squeeze_5);  sum_214 = squeeze_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_337: "f32[64, 1, 1, 1]" = torch.ops.aten.view.default(mul_1495, [64, 1, 1, 1]);  mul_1495 = None
    mul_1496: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_337, 0.05892556509887896);  view_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_338: "f32[64, 32, 3, 3]" = torch.ops.aten.view.default(mul_1494, [64, 32, 3, 3]);  mul_1494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1497: "f32[4, 32, 96, 96]" = torch.ops.aten.mul.Tensor(getitem_348, 1.7015043497085571);  getitem_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1498: "f32[4, 32, 96, 96]" = torch.ops.aten.mul.Tensor(convolution_1, 0.7071067811865476)
    erf_102: "f32[4, 32, 96, 96]" = torch.ops.aten.erf.default(mul_1498);  mul_1498 = None
    add_245: "f32[4, 32, 96, 96]" = torch.ops.aten.add.Tensor(erf_102, 1);  erf_102 = None
    mul_1499: "f32[4, 32, 96, 96]" = torch.ops.aten.mul.Tensor(add_245, 0.5);  add_245 = None
    mul_1500: "f32[4, 32, 96, 96]" = torch.ops.aten.mul.Tensor(convolution_1, convolution_1)
    mul_1501: "f32[4, 32, 96, 96]" = torch.ops.aten.mul.Tensor(mul_1500, -0.5);  mul_1500 = None
    exp_50: "f32[4, 32, 96, 96]" = torch.ops.aten.exp.default(mul_1501);  mul_1501 = None
    mul_1502: "f32[4, 32, 96, 96]" = torch.ops.aten.mul.Tensor(exp_50, 0.3989422804014327);  exp_50 = None
    mul_1503: "f32[4, 32, 96, 96]" = torch.ops.aten.mul.Tensor(convolution_1, mul_1502);  convolution_1 = mul_1502 = None
    add_246: "f32[4, 32, 96, 96]" = torch.ops.aten.add.Tensor(mul_1499, mul_1503);  mul_1499 = mul_1503 = None
    mul_1504: "f32[4, 32, 96, 96]" = torch.ops.aten.mul.Tensor(mul_1497, add_246);  mul_1497 = add_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_215: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1504, [0, 2, 3])
    convolution_backward_79 = torch.ops.aten.convolution_backward.default(mul_1504, mul_6, view_5, [32], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1504 = mul_6 = view_5 = None
    getitem_351: "f32[4, 16, 96, 96]" = convolution_backward_79[0]
    getitem_352: "f32[32, 16, 3, 3]" = convolution_backward_79[1];  convolution_backward_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_339: "f32[1, 32, 144]" = torch.ops.aten.view.default(getitem_352, [1, 32, 144]);  getitem_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_497: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_2, 0);  squeeze_2 = None
    unsqueeze_498: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 2);  unsqueeze_497 = None
    sum_216: "f32[32]" = torch.ops.aten.sum.dim_IntList(view_339, [0, 2])
    sub_289: "f32[1, 32, 144]" = torch.ops.aten.sub.Tensor(view_3, unsqueeze_498)
    mul_1505: "f32[1, 32, 144]" = torch.ops.aten.mul.Tensor(view_339, sub_289);  sub_289 = None
    sum_217: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1505, [0, 2]);  mul_1505 = None
    mul_1506: "f32[32]" = torch.ops.aten.mul.Tensor(sum_216, 0.006944444444444444);  sum_216 = None
    unsqueeze_499: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1506, 0);  mul_1506 = None
    unsqueeze_500: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 2);  unsqueeze_499 = None
    mul_1507: "f32[32]" = torch.ops.aten.mul.Tensor(sum_217, 0.006944444444444444)
    mul_1508: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_3, squeeze_3)
    mul_1509: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1507, mul_1508);  mul_1507 = mul_1508 = None
    unsqueeze_501: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1509, 0);  mul_1509 = None
    unsqueeze_502: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 2);  unsqueeze_501 = None
    mul_1510: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_3, view_4);  view_4 = None
    unsqueeze_503: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1510, 0);  mul_1510 = None
    unsqueeze_504: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 2);  unsqueeze_503 = None
    sub_290: "f32[1, 32, 144]" = torch.ops.aten.sub.Tensor(view_3, unsqueeze_498);  view_3 = unsqueeze_498 = None
    mul_1511: "f32[1, 32, 144]" = torch.ops.aten.mul.Tensor(sub_290, unsqueeze_502);  sub_290 = unsqueeze_502 = None
    sub_291: "f32[1, 32, 144]" = torch.ops.aten.sub.Tensor(view_339, mul_1511);  view_339 = mul_1511 = None
    sub_292: "f32[1, 32, 144]" = torch.ops.aten.sub.Tensor(sub_291, unsqueeze_500);  sub_291 = unsqueeze_500 = None
    mul_1512: "f32[1, 32, 144]" = torch.ops.aten.mul.Tensor(sub_292, unsqueeze_504);  sub_292 = unsqueeze_504 = None
    mul_1513: "f32[32]" = torch.ops.aten.mul.Tensor(sum_217, squeeze_3);  sum_217 = squeeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_340: "f32[32, 1, 1, 1]" = torch.ops.aten.view.default(mul_1513, [32, 1, 1, 1]);  mul_1513 = None
    mul_1514: "f32[32, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_340, 0.08333333333333333);  view_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_341: "f32[32, 16, 3, 3]" = torch.ops.aten.view.default(mul_1512, [32, 16, 3, 3]);  mul_1512 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1515: "f32[4, 16, 96, 96]" = torch.ops.aten.mul.Tensor(getitem_351, 1.7015043497085571);  getitem_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1516: "f32[4, 16, 96, 96]" = torch.ops.aten.mul.Tensor(convolution, 0.7071067811865476)
    erf_103: "f32[4, 16, 96, 96]" = torch.ops.aten.erf.default(mul_1516);  mul_1516 = None
    add_247: "f32[4, 16, 96, 96]" = torch.ops.aten.add.Tensor(erf_103, 1);  erf_103 = None
    mul_1517: "f32[4, 16, 96, 96]" = torch.ops.aten.mul.Tensor(add_247, 0.5);  add_247 = None
    mul_1518: "f32[4, 16, 96, 96]" = torch.ops.aten.mul.Tensor(convolution, convolution)
    mul_1519: "f32[4, 16, 96, 96]" = torch.ops.aten.mul.Tensor(mul_1518, -0.5);  mul_1518 = None
    exp_51: "f32[4, 16, 96, 96]" = torch.ops.aten.exp.default(mul_1519);  mul_1519 = None
    mul_1520: "f32[4, 16, 96, 96]" = torch.ops.aten.mul.Tensor(exp_51, 0.3989422804014327);  exp_51 = None
    mul_1521: "f32[4, 16, 96, 96]" = torch.ops.aten.mul.Tensor(convolution, mul_1520);  convolution = mul_1520 = None
    add_248: "f32[4, 16, 96, 96]" = torch.ops.aten.add.Tensor(mul_1517, mul_1521);  mul_1517 = mul_1521 = None
    mul_1522: "f32[4, 16, 96, 96]" = torch.ops.aten.mul.Tensor(mul_1515, add_248);  mul_1515 = add_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_218: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1522, [0, 2, 3])
    convolution_backward_80 = torch.ops.aten.convolution_backward.default(mul_1522, constant_pad_nd, view_2, [16], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_1522 = constant_pad_nd = view_2 = None
    getitem_355: "f32[16, 3, 3, 3]" = convolution_backward_80[1];  convolution_backward_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_342: "f32[1, 16, 27]" = torch.ops.aten.view.default(getitem_355, [1, 16, 27]);  getitem_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    unsqueeze_505: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_506: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 2);  unsqueeze_505 = None
    sum_219: "f32[16]" = torch.ops.aten.sum.dim_IntList(view_342, [0, 2])
    sub_293: "f32[1, 16, 27]" = torch.ops.aten.sub.Tensor(view, unsqueeze_506)
    mul_1523: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(view_342, sub_293);  sub_293 = None
    sum_220: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1523, [0, 2]);  mul_1523 = None
    mul_1524: "f32[16]" = torch.ops.aten.mul.Tensor(sum_219, 0.037037037037037035);  sum_219 = None
    unsqueeze_507: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1524, 0);  mul_1524 = None
    unsqueeze_508: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 2);  unsqueeze_507 = None
    mul_1525: "f32[16]" = torch.ops.aten.mul.Tensor(sum_220, 0.037037037037037035)
    mul_1526: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_1527: "f32[16]" = torch.ops.aten.mul.Tensor(mul_1525, mul_1526);  mul_1525 = mul_1526 = None
    unsqueeze_509: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1527, 0);  mul_1527 = None
    unsqueeze_510: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 2);  unsqueeze_509 = None
    mul_1528: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, view_1);  view_1 = None
    unsqueeze_511: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1528, 0);  mul_1528 = None
    unsqueeze_512: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 2);  unsqueeze_511 = None
    sub_294: "f32[1, 16, 27]" = torch.ops.aten.sub.Tensor(view, unsqueeze_506);  view = unsqueeze_506 = None
    mul_1529: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(sub_294, unsqueeze_510);  sub_294 = unsqueeze_510 = None
    sub_295: "f32[1, 16, 27]" = torch.ops.aten.sub.Tensor(view_342, mul_1529);  view_342 = mul_1529 = None
    sub_296: "f32[1, 16, 27]" = torch.ops.aten.sub.Tensor(sub_295, unsqueeze_508);  sub_295 = unsqueeze_508 = None
    mul_1530: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(sub_296, unsqueeze_512);  sub_296 = unsqueeze_512 = None
    mul_1531: "f32[16]" = torch.ops.aten.mul.Tensor(sum_220, squeeze_1);  sum_220 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_343: "f32[16, 1, 1, 1]" = torch.ops.aten.view.default(mul_1531, [16, 1, 1, 1]);  mul_1531 = None
    mul_1532: "f32[16, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_343, 0.19245008972987526);  view_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_344: "f32[16, 3, 3, 3]" = torch.ops.aten.view.default(mul_1530, [16, 3, 3, 3]);  mul_1530 = None
    return pytree.tree_unflatten([addmm, view_344, mul_1532, sum_218, view_341, mul_1514, sum_215, view_338, mul_1496, sum_212, view_335, mul_1478, sum_209, view_332, mul_1459, sum_206, view_329, mul_1449, sum_203, view_326, mul_1431, sum_200, view_323, mul_1413, sum_197, view_320, mul_1395, sum_194, sum_190, view_317, mul_1368, sum_187, view_314, mul_1358, sum_184, view_311, mul_1340, sum_181, view_308, mul_1322, sum_178, view_305, mul_1304, sum_175, sum_171, view_302, mul_1277, sum_168, view_299, mul_1259, sum_165, view_296, mul_1241, sum_162, view_293, mul_1223, sum_159, sum_155, view_290, mul_1196, sum_152, view_287, mul_1186, sum_149, view_284, mul_1168, sum_146, view_281, mul_1150, sum_143, view_278, mul_1132, sum_140, sum_136, view_275, mul_1105, sum_133, view_272, mul_1087, sum_130, view_269, mul_1069, sum_127, view_266, mul_1051, sum_124, sum_120, view_263, mul_1024, sum_117, view_260, mul_1006, sum_114, view_257, mul_988, sum_111, view_254, mul_970, sum_108, sum_104, view_251, mul_943, sum_101, view_248, mul_925, sum_98, view_245, mul_907, sum_95, view_242, mul_889, sum_92, sum_88, view_239, mul_862, sum_85, view_236, mul_844, sum_82, view_233, mul_826, sum_79, view_230, mul_808, sum_76, sum_72, view_227, mul_781, sum_69, view_224, mul_763, sum_66, view_221, mul_745, sum_63, view_218, mul_727, sum_60, sum_56, view_215, mul_700, sum_53, view_212, mul_690, sum_50, view_209, mul_672, sum_47, view_206, mul_654, sum_44, view_203, mul_636, sum_41, sum_37, view_200, mul_609, sum_34, view_197, mul_591, sum_31, view_194, mul_573, sum_28, view_191, mul_555, sum_25, sum_21, view_188, mul_528, sum_18, view_185, mul_510, sum_15, view_182, mul_492, sum_12, view_179, mul_474, sum_9, sum_5, view_176, mul_456, sum_2, getitem_328, sum_193, getitem_325, sum_192, getitem_307, sum_174, getitem_304, sum_173, getitem_289, sum_158, getitem_286, sum_157, getitem_268, sum_139, getitem_265, sum_138, getitem_250, sum_123, getitem_247, sum_122, getitem_232, sum_107, getitem_229, sum_106, getitem_214, sum_91, getitem_211, sum_90, getitem_196, sum_75, getitem_193, sum_74, getitem_178, sum_59, getitem_175, sum_58, getitem_157, sum_40, getitem_154, sum_39, getitem_139, sum_24, getitem_136, sum_23, getitem_121, sum_8, getitem_118, sum_7, permute_4, view_172, None], self._out_spec)
    