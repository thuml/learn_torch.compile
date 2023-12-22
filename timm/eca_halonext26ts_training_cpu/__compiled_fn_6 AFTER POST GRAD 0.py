from __future__ import annotations



def forward(self, primals_1: "f32[24]", primals_3: "f32[32]", primals_5: "f32[64]", primals_7: "f32[64]", primals_9: "f32[64]", primals_11: "f32[256]", primals_13: "f32[256]", primals_15: "f32[64]", primals_17: "f32[64]", primals_19: "f32[256]", primals_21: "f32[128]", primals_23: "f32[128]", primals_25: "f32[512]", primals_27: "f32[512]", primals_29: "f32[128]", primals_31: "f32[128]", primals_33: "f32[512]", primals_35: "f32[256]", primals_37: "f32[256]", primals_39: "f32[1024]", primals_41: "f32[1024]", primals_43: "f32[256]", primals_47: "f32[256]", primals_49: "f32[1024]", primals_51: "f32[512]", primals_55: "f32[512]", primals_57: "f32[2048]", primals_59: "f32[2048]", primals_61: "f32[512]", primals_65: "f32[512]", primals_67: "f32[2048]", primals_69: "f32[24, 3, 3, 3]", primals_70: "f32[32, 24, 3, 3]", primals_71: "f32[64, 32, 3, 3]", primals_72: "f32[64, 64, 1, 1]", primals_73: "f32[64, 16, 3, 3]", primals_74: "f32[1, 1, 3]", primals_75: "f32[256, 64, 1, 1]", primals_76: "f32[256, 64, 1, 1]", primals_77: "f32[64, 256, 1, 1]", primals_78: "f32[64, 16, 3, 3]", primals_79: "f32[1, 1, 3]", primals_80: "f32[256, 64, 1, 1]", primals_81: "f32[128, 256, 1, 1]", primals_82: "f32[128, 16, 3, 3]", primals_83: "f32[1, 1, 5]", primals_84: "f32[512, 128, 1, 1]", primals_85: "f32[512, 256, 1, 1]", primals_86: "f32[128, 512, 1, 1]", primals_87: "f32[128, 16, 3, 3]", primals_88: "f32[1, 1, 5]", primals_89: "f32[512, 128, 1, 1]", primals_90: "f32[256, 512, 1, 1]", primals_91: "f32[256, 16, 3, 3]", primals_92: "f32[1, 1, 5]", primals_93: "f32[1024, 256, 1, 1]", primals_94: "f32[1024, 512, 1, 1]", primals_95: "f32[256, 1024, 1, 1]", primals_96: "f32[128, 256, 1, 1]", primals_97: "f32[384, 256, 1, 1]", primals_98: "f32[1024, 256, 1, 1]", primals_99: "f32[512, 1024, 1, 1]", primals_100: "f32[128, 512, 1, 1]", primals_101: "f32[640, 512, 1, 1]", primals_102: "f32[2048, 512, 1, 1]", primals_103: "f32[2048, 1024, 1, 1]", primals_104: "f32[512, 2048, 1, 1]", primals_105: "f32[128, 512, 1, 1]", primals_106: "f32[640, 512, 1, 1]", primals_107: "f32[2048, 512, 1, 1]", primals_203: "f32[8, 3, 256, 256]", convolution: "f32[8, 24, 128, 128]", squeeze_1: "f32[24]", mul_7: "f32[8, 24, 128, 128]", convolution_1: "f32[8, 32, 128, 128]", squeeze_4: "f32[32]", mul_15: "f32[8, 32, 128, 128]", convolution_2: "f32[8, 64, 128, 128]", squeeze_7: "f32[64]", mul_23: "f32[8, 64, 128, 128]", getitem_6: "f32[8, 64, 64, 64]", getitem_7: "i64[8, 64, 64, 64]", convolution_3: "f32[8, 64, 64, 64]", squeeze_10: "f32[64]", mul_31: "f32[8, 64, 64, 64]", convolution_4: "f32[8, 64, 64, 64]", squeeze_13: "f32[64]", add_24: "f32[8, 64, 64, 64]", view: "f32[8, 1, 64]", convolution_5: "f32[8, 1, 64]", mul_40: "f32[8, 64, 64, 64]", convolution_6: "f32[8, 256, 64, 64]", squeeze_16: "f32[256]", convolution_7: "f32[8, 256, 64, 64]", squeeze_19: "f32[256]", mul_55: "f32[8, 256, 64, 64]", convolution_8: "f32[8, 64, 64, 64]", squeeze_22: "f32[64]", mul_63: "f32[8, 64, 64, 64]", convolution_9: "f32[8, 64, 64, 64]", squeeze_25: "f32[64]", add_45: "f32[8, 64, 64, 64]", view_2: "f32[8, 1, 64]", convolution_10: "f32[8, 1, 64]", mul_72: "f32[8, 64, 64, 64]", convolution_11: "f32[8, 256, 64, 64]", squeeze_28: "f32[256]", mul_80: "f32[8, 256, 64, 64]", convolution_12: "f32[8, 128, 64, 64]", squeeze_31: "f32[128]", mul_88: "f32[8, 128, 64, 64]", convolution_13: "f32[8, 128, 32, 32]", squeeze_34: "f32[128]", add_61: "f32[8, 128, 32, 32]", view_4: "f32[8, 1, 128]", convolution_14: "f32[8, 1, 128]", mul_97: "f32[8, 128, 32, 32]", convolution_15: "f32[8, 512, 32, 32]", squeeze_37: "f32[512]", convolution_16: "f32[8, 512, 32, 32]", squeeze_40: "f32[512]", mul_112: "f32[8, 512, 32, 32]", convolution_17: "f32[8, 128, 32, 32]", squeeze_43: "f32[128]", mul_120: "f32[8, 128, 32, 32]", convolution_18: "f32[8, 128, 32, 32]", squeeze_46: "f32[128]", add_82: "f32[8, 128, 32, 32]", view_6: "f32[8, 1, 128]", convolution_19: "f32[8, 1, 128]", mul_129: "f32[8, 128, 32, 32]", convolution_20: "f32[8, 512, 32, 32]", squeeze_49: "f32[512]", mul_137: "f32[8, 512, 32, 32]", convolution_21: "f32[8, 256, 32, 32]", squeeze_52: "f32[256]", mul_145: "f32[8, 256, 32, 32]", convolution_22: "f32[8, 256, 16, 16]", squeeze_55: "f32[256]", add_98: "f32[8, 256, 16, 16]", view_8: "f32[8, 1, 256]", convolution_23: "f32[8, 1, 256]", mul_154: "f32[8, 256, 16, 16]", convolution_24: "f32[8, 1024, 16, 16]", squeeze_58: "f32[1024]", convolution_25: "f32[8, 1024, 16, 16]", squeeze_61: "f32[1024]", mul_169: "f32[8, 1024, 16, 16]", convolution_26: "f32[8, 256, 16, 16]", squeeze_64: "f32[256]", mul_177: "f32[8, 256, 16, 16]", view_17: "f32[16384, 16]", view_23: "f32[16384, 16]", squeeze_67: "f32[256]", mul_186: "f32[8, 256, 16, 16]", convolution_29: "f32[8, 1024, 16, 16]", squeeze_70: "f32[1024]", mul_194: "f32[8, 1024, 16, 16]", convolution_30: "f32[8, 512, 16, 16]", squeeze_73: "f32[512]", mul_202: "f32[8, 512, 16, 16]", view_42: "f32[4096, 16]", view_48: "f32[4096, 16]", squeeze_76: "f32[512]", mul_211: "f32[8, 512, 8, 8]", convolution_33: "f32[8, 2048, 8, 8]", squeeze_79: "f32[2048]", convolution_34: "f32[8, 2048, 8, 8]", squeeze_82: "f32[2048]", mul_226: "f32[8, 2048, 8, 8]", convolution_35: "f32[8, 512, 8, 8]", squeeze_85: "f32[512]", mul_234: "f32[8, 512, 8, 8]", view_67: "f32[4096, 16]", view_73: "f32[4096, 16]", squeeze_88: "f32[512]", mul_243: "f32[8, 512, 8, 8]", convolution_38: "f32[8, 2048, 8, 8]", squeeze_91: "f32[2048]", clone_51: "f32[8, 2048]", permute_34: "f32[1000, 2048]", mul_253: "f32[8, 2048, 8, 8]", unsqueeze_126: "f32[1, 2048, 1, 1]", mul_265: "f32[8, 512, 8, 8]", sub_40: "f32[8, 512, 8, 8]", permute_42: "f32[64, 144, 64]", permute_43: "f32[64, 64, 144]", alias_8: "f32[64, 1, 64, 144]", permute_47: "f32[23, 16]", permute_53: "f32[23, 16]", permute_55: "f32[64, 16, 64]", permute_56: "f32[64, 144, 16]", mul_280: "f32[8, 512, 8, 8]", unsqueeze_150: "f32[1, 512, 1, 1]", mul_292: "f32[8, 2048, 8, 8]", unsqueeze_162: "f32[1, 2048, 1, 1]", unsqueeze_174: "f32[1, 2048, 1, 1]", mul_313: "f32[8, 512, 8, 8]", sub_60: "f32[8, 512, 8, 8]", permute_68: "f32[256, 144, 16]", permute_69: "f32[256, 64, 144]", alias_9: "f32[64, 4, 16, 144]", permute_73: "f32[23, 16]", permute_79: "f32[23, 16]", permute_81: "f32[256, 16, 16]", permute_82: "f32[256, 144, 16]", mul_328: "f32[8, 512, 16, 16]", unsqueeze_198: "f32[1, 512, 1, 1]", mul_340: "f32[8, 1024, 16, 16]", unsqueeze_210: "f32[1, 1024, 1, 1]", mul_352: "f32[8, 256, 16, 16]", sub_76: "f32[8, 256, 16, 16]", permute_94: "f32[256, 144, 64]", permute_95: "f32[256, 32, 144]", alias_10: "f32[64, 4, 64, 144]", permute_99: "f32[23, 16]", permute_105: "f32[23, 16]", permute_107: "f32[256, 16, 64]", permute_108: "f32[256, 144, 16]", mul_367: "f32[8, 256, 16, 16]", unsqueeze_234: "f32[1, 256, 1, 1]", mul_379: "f32[8, 1024, 16, 16]", unsqueeze_246: "f32[1, 1024, 1, 1]", unsqueeze_258: "f32[1, 1024, 1, 1]", unsqueeze_272: "f32[1, 256, 1, 1]", mul_416: "f32[8, 256, 32, 32]", unsqueeze_284: "f32[1, 256, 1, 1]", mul_428: "f32[8, 512, 32, 32]", unsqueeze_296: "f32[1, 512, 1, 1]", unsqueeze_310: "f32[1, 128, 1, 1]", mul_456: "f32[8, 128, 32, 32]", unsqueeze_322: "f32[1, 128, 1, 1]", mul_468: "f32[8, 512, 32, 32]", unsqueeze_334: "f32[1, 512, 1, 1]", unsqueeze_346: "f32[1, 512, 1, 1]", unsqueeze_360: "f32[1, 128, 1, 1]", mul_505: "f32[8, 128, 64, 64]", unsqueeze_372: "f32[1, 128, 1, 1]", mul_517: "f32[8, 256, 64, 64]", unsqueeze_384: "f32[1, 256, 1, 1]", unsqueeze_398: "f32[1, 64, 1, 1]", mul_545: "f32[8, 64, 64, 64]", unsqueeze_410: "f32[1, 64, 1, 1]", mul_557: "f32[8, 256, 64, 64]", unsqueeze_422: "f32[1, 256, 1, 1]", unsqueeze_434: "f32[1, 256, 1, 1]", unsqueeze_448: "f32[1, 64, 1, 1]", mul_594: "f32[8, 64, 64, 64]", unsqueeze_460: "f32[1, 64, 1, 1]", mul_606: "f32[8, 64, 128, 128]", unsqueeze_472: "f32[1, 64, 1, 1]", mul_618: "f32[8, 32, 128, 128]", unsqueeze_484: "f32[1, 32, 1, 1]", mul_630: "f32[8, 24, 128, 128]", unsqueeze_496: "f32[1, 24, 1, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_4: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_24)
    mul_39: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_24, sigmoid_4);  sigmoid_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_5: "f32[8, 1, 64]" = torch.ops.aten.sigmoid.default(convolution_5);  convolution_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
    view_1: "f32[8, 64, 1, 1]" = torch.ops.aten.reshape.default(sigmoid_5, [8, -1, 1, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:91, code: return x * y.expand_as(x)
    expand: "f32[8, 64, 64, 64]" = torch.ops.aten.expand.default(view_1, [8, 64, 64, 64]);  view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_8: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_45)
    mul_71: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_45, sigmoid_8);  sigmoid_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_9: "f32[8, 1, 64]" = torch.ops.aten.sigmoid.default(convolution_10);  convolution_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
    view_3: "f32[8, 64, 1, 1]" = torch.ops.aten.reshape.default(sigmoid_9, [8, -1, 1, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:91, code: return x * y.expand_as(x)
    expand_1: "f32[8, 64, 64, 64]" = torch.ops.aten.expand.default(view_3, [8, 64, 64, 64]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_12: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(add_61)
    mul_96: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_61, sigmoid_12);  sigmoid_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_13: "f32[8, 1, 128]" = torch.ops.aten.sigmoid.default(convolution_14);  convolution_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
    view_5: "f32[8, 128, 1, 1]" = torch.ops.aten.reshape.default(sigmoid_13, [8, -1, 1, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:91, code: return x * y.expand_as(x)
    expand_2: "f32[8, 128, 32, 32]" = torch.ops.aten.expand.default(view_5, [8, 128, 32, 32]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_16: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(add_82)
    mul_128: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_82, sigmoid_16);  sigmoid_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_17: "f32[8, 1, 128]" = torch.ops.aten.sigmoid.default(convolution_19);  convolution_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
    view_7: "f32[8, 128, 1, 1]" = torch.ops.aten.reshape.default(sigmoid_17, [8, -1, 1, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:91, code: return x * y.expand_as(x)
    expand_3: "f32[8, 128, 32, 32]" = torch.ops.aten.expand.default(view_7, [8, 128, 32, 32]);  view_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_20: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_98)
    mul_153: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_98, sigmoid_20);  sigmoid_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_21: "f32[8, 1, 256]" = torch.ops.aten.sigmoid.default(convolution_23);  convolution_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
    view_9: "f32[8, 256, 1, 1]" = torch.ops.aten.reshape.default(sigmoid_21, [8, -1, 1, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:91, code: return x * y.expand_as(x)
    expand_4: "f32[8, 256, 16, 16]" = torch.ops.aten.expand.default(view_9, [8, 256, 16, 16]);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    mm_6: "f32[8, 2048]" = torch.ops.aten.mm.default(tangents_1, permute_34);  permute_34 = None
    permute_35: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_7: "f32[1000, 2048]" = torch.ops.aten.mm.default(permute_35, clone_51);  permute_35 = clone_51 = None
    permute_36: "f32[2048, 1000]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_4: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_86: "f32[1000]" = torch.ops.aten.reshape.default(sum_4, [1000]);  sum_4 = None
    permute_37: "f32[1000, 2048]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_87: "f32[8, 2048, 1, 1]" = torch.ops.aten.reshape.default(mm_6, [8, 2048, 1, 1]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand_23: "f32[8, 2048, 8, 8]" = torch.ops.aten.expand.default(view_87, [8, 2048, 8, 8]);  view_87 = None
    div_3: "f32[8, 2048, 8, 8]" = torch.ops.aten.div.Scalar(expand_23, 64);  expand_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    mul_254: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(div_3, mul_253);  div_3 = mul_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_5: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_254, [0, 2, 3])
    sub_35: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_126);  convolution_38 = unsqueeze_126 = None
    mul_255: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(mul_254, sub_35)
    sum_6: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_255, [0, 2, 3]);  mul_255 = None
    mul_256: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_5, 0.001953125)
    unsqueeze_127: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_256, 0);  mul_256 = None
    unsqueeze_128: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_127, 2);  unsqueeze_127 = None
    unsqueeze_129: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, 3);  unsqueeze_128 = None
    mul_257: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_6, 0.001953125)
    mul_258: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_259: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_257, mul_258);  mul_257 = mul_258 = None
    unsqueeze_130: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_259, 0);  mul_259 = None
    unsqueeze_131: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, 2);  unsqueeze_130 = None
    unsqueeze_132: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_131, 3);  unsqueeze_131 = None
    mul_260: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_67);  primals_67 = None
    unsqueeze_133: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_260, 0);  mul_260 = None
    unsqueeze_134: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_133, 2);  unsqueeze_133 = None
    unsqueeze_135: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, 3);  unsqueeze_134 = None
    mul_261: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_132);  sub_35 = unsqueeze_132 = None
    sub_37: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(mul_254, mul_261);  mul_261 = None
    sub_38: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(sub_37, unsqueeze_129);  sub_37 = unsqueeze_129 = None
    mul_262: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_135);  sub_38 = unsqueeze_135 = None
    mul_263: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_6, squeeze_91);  sum_6 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_262, mul_243, primals_107, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_262 = mul_243 = primals_107 = None
    getitem_70: "f32[8, 512, 8, 8]" = convolution_backward[0]
    getitem_71: "f32[2048, 512, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_266: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(getitem_70, mul_265);  getitem_70 = mul_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_7: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_266, [0, 2, 3])
    mul_267: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_266, sub_40)
    sum_8: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_267, [0, 2, 3]);  mul_267 = None
    mul_268: "f32[512]" = torch.ops.aten.mul.Tensor(sum_7, 0.001953125)
    unsqueeze_139: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_268, 0);  mul_268 = None
    unsqueeze_140: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_139, 2);  unsqueeze_139 = None
    unsqueeze_141: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, 3);  unsqueeze_140 = None
    mul_269: "f32[512]" = torch.ops.aten.mul.Tensor(sum_8, 0.001953125)
    mul_270: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_271: "f32[512]" = torch.ops.aten.mul.Tensor(mul_269, mul_270);  mul_269 = mul_270 = None
    unsqueeze_142: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_271, 0);  mul_271 = None
    unsqueeze_143: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, 2);  unsqueeze_142 = None
    unsqueeze_144: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_143, 3);  unsqueeze_143 = None
    mul_272: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_65);  primals_65 = None
    unsqueeze_145: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_272, 0);  mul_272 = None
    unsqueeze_146: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_145, 2);  unsqueeze_145 = None
    unsqueeze_147: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, 3);  unsqueeze_146 = None
    mul_273: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_144);  sub_40 = unsqueeze_144 = None
    sub_42: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(mul_266, mul_273);  mul_266 = mul_273 = None
    sub_43: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_42, unsqueeze_141);  sub_42 = unsqueeze_141 = None
    mul_274: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_147);  sub_43 = unsqueeze_147 = None
    mul_275: "f32[512]" = torch.ops.aten.mul.Tensor(sum_8, squeeze_88);  sum_8 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:206, code: out = out.permute(0, 3, 1, 4, 2).contiguous().view(
    view_88: "f32[4096, 1, 8, 1, 8]" = torch.ops.aten.reshape.default(mul_274, [4096, 1, 8, 1, 8]);  mul_274 = None
    permute_40: "f32[4096, 8, 8, 1, 1]" = torch.ops.aten.permute.default(view_88, [0, 2, 4, 1, 3]);  view_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:205, code: out = out.reshape(-1, self.block_size_ds, self.block_size_ds, num_h_blocks, num_w_blocks)
    view_89: "f32[64, 64, 64, 1]" = torch.ops.aten.reshape.default(permute_40, [64, 64, 64, 1]);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:203, code: out = (attn @ v).transpose(1, 3)  # B * num_heads, dim_head_v, block_size ** 2, num_blocks
    permute_41: "f32[64, 1, 64, 64]" = torch.ops.aten.permute.default(view_89, [0, 3, 2, 1]);  view_89 = None
    view_90: "f32[64, 64, 64]" = torch.ops.aten.reshape.default(permute_41, [64, 64, 64]);  permute_41 = None
    bmm_6: "f32[64, 144, 64]" = torch.ops.aten.bmm.default(permute_42, view_90);  permute_42 = None
    bmm_7: "f32[64, 64, 144]" = torch.ops.aten.bmm.default(view_90, permute_43);  view_90 = permute_43 = None
    view_91: "f32[64, 1, 144, 64]" = torch.ops.aten.reshape.default(bmm_6, [64, 1, 144, 64]);  bmm_6 = None
    view_92: "f32[64, 1, 64, 144]" = torch.ops.aten.reshape.default(bmm_7, [64, 1, 64, 144]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:201, code: attn = attn.softmax(dim=-1)
    mul_276: "f32[64, 1, 64, 144]" = torch.ops.aten.mul.Tensor(view_92, alias_8);  view_92 = None
    sum_9: "f32[64, 1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_276, [-1], True)
    mul_277: "f32[64, 1, 64, 144]" = torch.ops.aten.mul.Tensor(alias_8, sum_9);  alias_8 = sum_9 = None
    sub_44: "f32[64, 1, 64, 144]" = torch.ops.aten.sub.Tensor(mul_276, mul_277);  mul_276 = mul_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:93, code: rel_logits = rel_logits.reshape(B, BB, HW, -1)
    view_93: "f32[64, 8, 8, 12, 12]" = torch.ops.aten.reshape.default(sub_44, [64, 8, 8, 12, 12])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:58, code: return x.permute(permute_mask)
    permute_44: "f32[64, 8, 12, 8, 12]" = torch.ops.aten.permute.default(view_93, [0, 2, 4, 1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:57, code: x = x.reshape(B, H, 1, W, win_size).expand(-1, -1, win_size, -1, -1)
    sum_10: "f32[64, 8, 1, 8, 12]" = torch.ops.aten.sum.dim_IntList(permute_44, [2], True);  permute_44 = None
    view_94: "f32[512, 8, 12]" = torch.ops.aten.reshape.default(sum_10, [512, 8, 12]);  sum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:54, code: x = x_pad[:, :W, win_size - 1:]
    full_default_2: "f32[512, 8, 23]" = torch.ops.aten.full.default([512, 8, 23], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter: "f32[512, 8, 23]" = torch.ops.aten.slice_scatter.default(full_default_2, view_94, 2, 11, 9223372036854775807);  view_94 = None
    full_default_3: "f32[512, 9, 23]" = torch.ops.aten.full.default([512, 9, 23], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_1: "f32[512, 9, 23]" = torch.ops.aten.slice_scatter.default(full_default_3, slice_scatter, 1, 0, 8);  slice_scatter = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:53, code: x_pad = x_pad.reshape(-1, W + 1, rel_size)
    view_95: "f32[512, 207]" = torch.ops.aten.reshape.default(slice_scatter_1, [512, 207]);  slice_scatter_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:50, code: x_pad = F.pad(x_pad, [0, rel_size - W])
    constant_pad_nd_15: "f32[512, 192]" = torch.ops.aten.constant_pad_nd.default(view_95, [0, -15]);  view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:49, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_96: "f32[512, 8, 24]" = torch.ops.aten.reshape.default(constant_pad_nd_15, [512, 8, 24]);  constant_pad_nd_15 = None
    constant_pad_nd_16: "f32[512, 8, 23]" = torch.ops.aten.constant_pad_nd.default(view_96, [0, -1]);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:46, code: x = x.reshape(-1, W, rel_size)
    view_97: "f32[64, 8, 8, 23]" = torch.ops.aten.reshape.default(constant_pad_nd_16, [64, 8, 8, 23]);  constant_pad_nd_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:45, code: x = (q @ rel_k.transpose(-1, -2))
    view_98: "f32[4096, 23]" = torch.ops.aten.reshape.default(view_97, [4096, 23]);  view_97 = None
    permute_45: "f32[23, 4096]" = torch.ops.aten.permute.default(view_98, [1, 0])
    mm_8: "f32[23, 16]" = torch.ops.aten.mm.default(permute_45, view_73);  permute_45 = view_73 = None
    permute_46: "f32[16, 23]" = torch.ops.aten.permute.default(mm_8, [1, 0]);  mm_8 = None
    mm_9: "f32[4096, 16]" = torch.ops.aten.mm.default(view_98, permute_47);  view_98 = permute_47 = None
    view_99: "f32[64, 8, 8, 16]" = torch.ops.aten.reshape.default(mm_9, [64, 8, 8, 16]);  mm_9 = None
    permute_48: "f32[23, 16]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:89, code: q = q.transpose(1, 2)
    permute_49: "f32[64, 8, 8, 16]" = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:58, code: return x.permute(permute_mask)
    permute_50: "f32[64, 8, 12, 8, 12]" = torch.ops.aten.permute.default(view_93, [0, 1, 3, 2, 4]);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:57, code: x = x.reshape(B, H, 1, W, win_size).expand(-1, -1, win_size, -1, -1)
    sum_11: "f32[64, 8, 1, 8, 12]" = torch.ops.aten.sum.dim_IntList(permute_50, [2], True);  permute_50 = None
    view_100: "f32[512, 8, 12]" = torch.ops.aten.reshape.default(sum_11, [512, 8, 12]);  sum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:54, code: x = x_pad[:, :W, win_size - 1:]
    slice_scatter_3: "f32[512, 8, 23]" = torch.ops.aten.slice_scatter.default(full_default_2, view_100, 2, 11, 9223372036854775807);  full_default_2 = view_100 = None
    slice_scatter_4: "f32[512, 9, 23]" = torch.ops.aten.slice_scatter.default(full_default_3, slice_scatter_3, 1, 0, 8);  full_default_3 = slice_scatter_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:53, code: x_pad = x_pad.reshape(-1, W + 1, rel_size)
    view_101: "f32[512, 207]" = torch.ops.aten.reshape.default(slice_scatter_4, [512, 207]);  slice_scatter_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:50, code: x_pad = F.pad(x_pad, [0, rel_size - W])
    constant_pad_nd_17: "f32[512, 192]" = torch.ops.aten.constant_pad_nd.default(view_101, [0, -15]);  view_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:49, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_102: "f32[512, 8, 24]" = torch.ops.aten.reshape.default(constant_pad_nd_17, [512, 8, 24]);  constant_pad_nd_17 = None
    constant_pad_nd_18: "f32[512, 8, 23]" = torch.ops.aten.constant_pad_nd.default(view_102, [0, -1]);  view_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:46, code: x = x.reshape(-1, W, rel_size)
    view_103: "f32[64, 8, 8, 23]" = torch.ops.aten.reshape.default(constant_pad_nd_18, [64, 8, 8, 23]);  constant_pad_nd_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:45, code: x = (q @ rel_k.transpose(-1, -2))
    view_104: "f32[4096, 23]" = torch.ops.aten.reshape.default(view_103, [4096, 23]);  view_103 = None
    permute_51: "f32[23, 4096]" = torch.ops.aten.permute.default(view_104, [1, 0])
    mm_10: "f32[23, 16]" = torch.ops.aten.mm.default(permute_51, view_67);  permute_51 = view_67 = None
    permute_52: "f32[16, 23]" = torch.ops.aten.permute.default(mm_10, [1, 0]);  mm_10 = None
    mm_11: "f32[4096, 16]" = torch.ops.aten.mm.default(view_104, permute_53);  view_104 = permute_53 = None
    view_105: "f32[64, 8, 8, 16]" = torch.ops.aten.reshape.default(mm_11, [64, 8, 8, 16]);  mm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:45, code: x = (q @ rel_k.transpose(-1, -2))
    add_171: "f32[64, 8, 8, 16]" = torch.ops.aten.add.Tensor(permute_49, view_105);  permute_49 = view_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:45, code: x = (q @ rel_k.transpose(-1, -2))
    permute_54: "f32[23, 16]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:85, code: q = q.reshape(-1, self.block_size, self.block_size, self.dim_head)
    clone_52: "f32[64, 8, 8, 16]" = torch.ops.aten.clone.default(add_171, memory_format = torch.contiguous_format);  add_171 = None
    view_106: "f32[64, 1, 64, 16]" = torch.ops.aten.reshape.default(clone_52, [64, 1, 64, 16]);  clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:199, code: attn = (q @ k.transpose(-1, -2)) * self.scale + self.pos_embed(q)
    mul_278: "f32[64, 1, 64, 144]" = torch.ops.aten.mul.Tensor(sub_44, 0.25);  sub_44 = None
    view_107: "f32[64, 64, 144]" = torch.ops.aten.reshape.default(mul_278, [64, 64, 144]);  mul_278 = None
    bmm_8: "f32[64, 16, 144]" = torch.ops.aten.bmm.default(permute_55, view_107);  permute_55 = None
    bmm_9: "f32[64, 64, 16]" = torch.ops.aten.bmm.default(view_107, permute_56);  view_107 = permute_56 = None
    view_108: "f32[64, 1, 16, 144]" = torch.ops.aten.reshape.default(bmm_8, [64, 1, 16, 144]);  bmm_8 = None
    view_109: "f32[64, 1, 64, 16]" = torch.ops.aten.reshape.default(bmm_9, [64, 1, 64, 16]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:199, code: attn = (q @ k.transpose(-1, -2)) * self.scale + self.pos_embed(q)
    add_172: "f32[64, 1, 64, 16]" = torch.ops.aten.add.Tensor(view_106, view_109);  view_106 = view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:199, code: attn = (q @ k.transpose(-1, -2)) * self.scale + self.pos_embed(q)
    permute_57: "f32[64, 1, 144, 16]" = torch.ops.aten.permute.default(view_108, [0, 1, 3, 2]);  view_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:193, code: k, v = torch.split(kv, [self.dim_head_qk, self.dim_head_v], dim=-1)
    cat: "f32[64, 1, 144, 80]" = torch.ops.aten.cat.default([permute_57, view_91], 3);  permute_57 = view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:192, code: B * self.num_heads, self.dim_head_qk + self.dim_head_v, num_blocks, -1).permute(0, 2, 3, 1)
    permute_58: "f32[64, 80, 1, 144]" = torch.ops.aten.permute.default(cat, [0, 3, 1, 2]);  cat = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:191, code: kv = kv.unfold(2, self.win_size, self.block_size).unfold(3, self.win_size, self.block_size).reshape(
    clone_53: "f32[64, 80, 1, 144]" = torch.ops.aten.clone.default(permute_58, memory_format = torch.contiguous_format);  permute_58 = None
    view_110: "f32[8, 640, 1, 1, 12, 12]" = torch.ops.aten.reshape.default(clone_53, [8, 640, 1, 1, 12, 12]);  clone_53 = None
    iota: "i32[12]" = torch.ops.prims.iota.default(12, start = 0, step = 1, dtype = torch.int32, device = device(type='cpu'), requires_grad = False)
    unfold_6: "i32[1, 12]" = torch.ops.aten.unfold.default(iota, 0, 12, 8);  iota = None
    view_111: "i32[12]" = torch.ops.aten.reshape.default(unfold_6, [12]);  unfold_6 = None
    permute_59: "f32[8, 640, 1, 1, 12, 12]" = torch.ops.aten.permute.default(view_110, [0, 1, 2, 3, 5, 4]);  view_110 = None
    view_112: "f32[8, 640, 1, 12, 12]" = torch.ops.aten.reshape.default(permute_59, [8, 640, 1, 12, 12]);  permute_59 = None
    full_default_8: "f32[8, 640, 1, 12, 12]" = torch.ops.aten.full.default([8, 640, 1, 12, 12], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put: "f32[8, 640, 1, 12, 12]" = torch.ops.prims._unsafe_index_put_.default(full_default_8, [None, None, None, view_111], view_112, True);  full_default_8 = view_112 = None
    permute_60: "f32[8, 640, 1, 12, 12]" = torch.ops.aten.permute.default(_unsafe_index_put, [0, 1, 2, 4, 3]);  _unsafe_index_put = None
    view_114: "f32[8, 640, 12, 12]" = torch.ops.aten.reshape.default(permute_60, [8, 640, 12, 12]);  permute_60 = None
    full_default_9: "f32[8, 640, 12, 12]" = torch.ops.aten.full.default([8, 640, 12, 12], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_1: "f32[8, 640, 12, 12]" = torch.ops.prims._unsafe_index_put_.default(full_default_9, [None, None, view_111], view_114, True);  full_default_9 = view_111 = view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:190, code: kv = F.pad(kv, [self.halo_size, self.halo_size, self.halo_size, self.halo_size])
    constant_pad_nd_19: "f32[8, 640, 8, 8]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_1, [-2, -2, -2, -2]);  _unsafe_index_put_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:186, code: kv = self.kv(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(constant_pad_nd_19, mul_234, primals_106, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  constant_pad_nd_19 = primals_106 = None
    getitem_73: "f32[8, 512, 8, 8]" = convolution_backward_1[0]
    getitem_74: "f32[640, 512, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:183, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1, num_blocks).transpose(1, 3)
    permute_61: "f32[64, 16, 64, 1]" = torch.ops.aten.permute.default(add_172, [0, 3, 2, 1]);  add_172 = None
    view_115: "f32[64, 16, 8, 8, 1, 1]" = torch.ops.aten.reshape.default(permute_61, [64, 16, 8, 8, 1, 1]);  permute_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:181, code: num_h_blocks, self.block_size_ds, num_w_blocks, self.block_size_ds).permute(0, 1, 3, 5, 2, 4)
    permute_62: "f32[64, 16, 1, 8, 1, 8]" = torch.ops.aten.permute.default(view_115, [0, 1, 4, 2, 5, 3]);  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:179, code: q = q.reshape(
    clone_54: "f32[64, 16, 1, 8, 1, 8]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    view_116: "f32[8, 128, 8, 8]" = torch.ops.aten.reshape.default(clone_54, [8, 128, 8, 8]);  clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:177, code: q = self.q(x)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(view_116, mul_234, primals_105, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_116 = mul_234 = primals_105 = None
    getitem_76: "f32[8, 512, 8, 8]" = convolution_backward_2[0]
    getitem_77: "f32[128, 512, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:177, code: q = self.q(x)
    add_173: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(getitem_73, getitem_76);  getitem_73 = getitem_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_281: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_173, mul_280);  add_173 = mul_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_12: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_281, [0, 2, 3])
    sub_46: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_150);  convolution_35 = unsqueeze_150 = None
    mul_282: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_281, sub_46)
    sum_13: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_282, [0, 2, 3]);  mul_282 = None
    mul_283: "f32[512]" = torch.ops.aten.mul.Tensor(sum_12, 0.001953125)
    unsqueeze_151: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_283, 0);  mul_283 = None
    unsqueeze_152: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_151, 2);  unsqueeze_151 = None
    unsqueeze_153: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, 3);  unsqueeze_152 = None
    mul_284: "f32[512]" = torch.ops.aten.mul.Tensor(sum_13, 0.001953125)
    mul_285: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_286: "f32[512]" = torch.ops.aten.mul.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None
    unsqueeze_154: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_286, 0);  mul_286 = None
    unsqueeze_155: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, 2);  unsqueeze_154 = None
    unsqueeze_156: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_155, 3);  unsqueeze_155 = None
    mul_287: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_61);  primals_61 = None
    unsqueeze_157: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_287, 0);  mul_287 = None
    unsqueeze_158: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_157, 2);  unsqueeze_157 = None
    unsqueeze_159: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, 3);  unsqueeze_158 = None
    mul_288: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_156);  sub_46 = unsqueeze_156 = None
    sub_48: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(mul_281, mul_288);  mul_281 = mul_288 = None
    sub_49: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_48, unsqueeze_153);  sub_48 = unsqueeze_153 = None
    mul_289: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_159);  sub_49 = unsqueeze_159 = None
    mul_290: "f32[512]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_85);  sum_13 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_289, mul_226, primals_104, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_289 = mul_226 = primals_104 = None
    getitem_79: "f32[8, 2048, 8, 8]" = convolution_backward_3[0]
    getitem_80: "f32[512, 2048, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_175: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(mul_254, getitem_79);  mul_254 = getitem_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    mul_293: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(add_175, mul_292);  add_175 = mul_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_14: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_293, [0, 2, 3])
    sub_51: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_162);  convolution_34 = unsqueeze_162 = None
    mul_294: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(mul_293, sub_51)
    sum_15: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_294, [0, 2, 3]);  mul_294 = None
    mul_295: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_14, 0.001953125)
    unsqueeze_163: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_295, 0);  mul_295 = None
    unsqueeze_164: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_163, 2);  unsqueeze_163 = None
    unsqueeze_165: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, 3);  unsqueeze_164 = None
    mul_296: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_15, 0.001953125)
    mul_297: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_298: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_296, mul_297);  mul_296 = mul_297 = None
    unsqueeze_166: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_298, 0);  mul_298 = None
    unsqueeze_167: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, 2);  unsqueeze_166 = None
    unsqueeze_168: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 3);  unsqueeze_167 = None
    mul_299: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_59);  primals_59 = None
    unsqueeze_169: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_299, 0);  mul_299 = None
    unsqueeze_170: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_169, 2);  unsqueeze_169 = None
    unsqueeze_171: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, 3);  unsqueeze_170 = None
    mul_300: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_168);  sub_51 = unsqueeze_168 = None
    sub_53: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(mul_293, mul_300);  mul_300 = None
    sub_54: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(sub_53, unsqueeze_165);  sub_53 = None
    mul_301: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_171);  sub_54 = unsqueeze_171 = None
    mul_302: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_82);  sum_15 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_301, mul_194, primals_103, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_301 = primals_103 = None
    getitem_82: "f32[8, 1024, 16, 16]" = convolution_backward_4[0]
    getitem_83: "f32[2048, 1024, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_55: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_174);  convolution_33 = unsqueeze_174 = None
    mul_303: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(mul_293, sub_55)
    sum_17: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_303, [0, 2, 3]);  mul_303 = None
    mul_305: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_17, 0.001953125)
    mul_306: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_307: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_305, mul_306);  mul_305 = mul_306 = None
    unsqueeze_178: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_307, 0);  mul_307 = None
    unsqueeze_179: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, 2);  unsqueeze_178 = None
    unsqueeze_180: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_179, 3);  unsqueeze_179 = None
    mul_308: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_57);  primals_57 = None
    unsqueeze_181: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_308, 0);  mul_308 = None
    unsqueeze_182: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_181, 2);  unsqueeze_181 = None
    unsqueeze_183: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, 3);  unsqueeze_182 = None
    mul_309: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_180);  sub_55 = unsqueeze_180 = None
    sub_57: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(mul_293, mul_309);  mul_293 = mul_309 = None
    sub_58: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(sub_57, unsqueeze_165);  sub_57 = unsqueeze_165 = None
    mul_310: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_183);  sub_58 = unsqueeze_183 = None
    mul_311: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_79);  sum_17 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_310, mul_211, primals_102, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_310 = mul_211 = primals_102 = None
    getitem_85: "f32[8, 512, 8, 8]" = convolution_backward_5[0]
    getitem_86: "f32[2048, 512, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_314: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(getitem_85, mul_313);  getitem_85 = mul_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_18: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_314, [0, 2, 3])
    mul_315: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_314, sub_60)
    sum_19: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_315, [0, 2, 3]);  mul_315 = None
    mul_316: "f32[512]" = torch.ops.aten.mul.Tensor(sum_18, 0.001953125)
    unsqueeze_187: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_316, 0);  mul_316 = None
    unsqueeze_188: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_187, 2);  unsqueeze_187 = None
    unsqueeze_189: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, 3);  unsqueeze_188 = None
    mul_317: "f32[512]" = torch.ops.aten.mul.Tensor(sum_19, 0.001953125)
    mul_318: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_319: "f32[512]" = torch.ops.aten.mul.Tensor(mul_317, mul_318);  mul_317 = mul_318 = None
    unsqueeze_190: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_319, 0);  mul_319 = None
    unsqueeze_191: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, 2);  unsqueeze_190 = None
    unsqueeze_192: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_191, 3);  unsqueeze_191 = None
    mul_320: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_55);  primals_55 = None
    unsqueeze_193: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_320, 0);  mul_320 = None
    unsqueeze_194: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_193, 2);  unsqueeze_193 = None
    unsqueeze_195: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, 3);  unsqueeze_194 = None
    mul_321: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_192);  sub_60 = unsqueeze_192 = None
    sub_62: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(mul_314, mul_321);  mul_314 = mul_321 = None
    sub_63: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_62, unsqueeze_189);  sub_62 = unsqueeze_189 = None
    mul_322: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_195);  sub_63 = unsqueeze_195 = None
    mul_323: "f32[512]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_76);  sum_19 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:206, code: out = out.permute(0, 3, 1, 4, 2).contiguous().view(
    view_117: "f32[4096, 2, 4, 2, 4]" = torch.ops.aten.reshape.default(mul_322, [4096, 2, 4, 2, 4]);  mul_322 = None
    permute_66: "f32[4096, 4, 4, 2, 2]" = torch.ops.aten.permute.default(view_117, [0, 2, 4, 1, 3]);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:205, code: out = out.reshape(-1, self.block_size_ds, self.block_size_ds, num_h_blocks, num_w_blocks)
    clone_55: "f32[4096, 4, 4, 2, 2]" = torch.ops.aten.clone.default(permute_66, memory_format = torch.contiguous_format);  permute_66 = None
    view_118: "f32[64, 64, 16, 4]" = torch.ops.aten.reshape.default(clone_55, [64, 64, 16, 4]);  clone_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:203, code: out = (attn @ v).transpose(1, 3)  # B * num_heads, dim_head_v, block_size ** 2, num_blocks
    permute_67: "f32[64, 4, 16, 64]" = torch.ops.aten.permute.default(view_118, [0, 3, 2, 1]);  view_118 = None
    clone_56: "f32[64, 4, 16, 64]" = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None
    view_119: "f32[256, 16, 64]" = torch.ops.aten.reshape.default(clone_56, [256, 16, 64]);  clone_56 = None
    bmm_10: "f32[256, 144, 64]" = torch.ops.aten.bmm.default(permute_68, view_119);  permute_68 = None
    bmm_11: "f32[256, 16, 144]" = torch.ops.aten.bmm.default(view_119, permute_69);  view_119 = permute_69 = None
    view_120: "f32[64, 4, 144, 64]" = torch.ops.aten.reshape.default(bmm_10, [64, 4, 144, 64]);  bmm_10 = None
    view_121: "f32[64, 4, 16, 144]" = torch.ops.aten.reshape.default(bmm_11, [64, 4, 16, 144]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:201, code: attn = attn.softmax(dim=-1)
    mul_324: "f32[64, 4, 16, 144]" = torch.ops.aten.mul.Tensor(view_121, alias_9);  view_121 = None
    sum_20: "f32[64, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_324, [-1], True)
    mul_325: "f32[64, 4, 16, 144]" = torch.ops.aten.mul.Tensor(alias_9, sum_20);  alias_9 = sum_20 = None
    sub_64: "f32[64, 4, 16, 144]" = torch.ops.aten.sub.Tensor(mul_324, mul_325);  mul_324 = mul_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:93, code: rel_logits = rel_logits.reshape(B, BB, HW, -1)
    view_122: "f32[256, 4, 4, 12, 12]" = torch.ops.aten.reshape.default(sub_64, [256, 4, 4, 12, 12])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:58, code: return x.permute(permute_mask)
    permute_70: "f32[256, 4, 12, 4, 12]" = torch.ops.aten.permute.default(view_122, [0, 2, 4, 1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:57, code: x = x.reshape(B, H, 1, W, win_size).expand(-1, -1, win_size, -1, -1)
    sum_21: "f32[256, 4, 1, 4, 12]" = torch.ops.aten.sum.dim_IntList(permute_70, [2], True);  permute_70 = None
    view_123: "f32[1024, 4, 12]" = torch.ops.aten.reshape.default(sum_21, [1024, 4, 12]);  sum_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:54, code: x = x_pad[:, :W, win_size - 1:]
    full_default_13: "f32[1024, 4, 23]" = torch.ops.aten.full.default([1024, 4, 23], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_6: "f32[1024, 4, 23]" = torch.ops.aten.slice_scatter.default(full_default_13, view_123, 2, 11, 9223372036854775807);  view_123 = None
    full_default_14: "f32[1024, 5, 23]" = torch.ops.aten.full.default([1024, 5, 23], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_7: "f32[1024, 5, 23]" = torch.ops.aten.slice_scatter.default(full_default_14, slice_scatter_6, 1, 0, 4);  slice_scatter_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:53, code: x_pad = x_pad.reshape(-1, W + 1, rel_size)
    view_124: "f32[1024, 115]" = torch.ops.aten.reshape.default(slice_scatter_7, [1024, 115]);  slice_scatter_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:50, code: x_pad = F.pad(x_pad, [0, rel_size - W])
    constant_pad_nd_20: "f32[1024, 96]" = torch.ops.aten.constant_pad_nd.default(view_124, [0, -19]);  view_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:49, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_125: "f32[1024, 4, 24]" = torch.ops.aten.reshape.default(constant_pad_nd_20, [1024, 4, 24]);  constant_pad_nd_20 = None
    constant_pad_nd_21: "f32[1024, 4, 23]" = torch.ops.aten.constant_pad_nd.default(view_125, [0, -1]);  view_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:46, code: x = x.reshape(-1, W, rel_size)
    view_126: "f32[256, 4, 4, 23]" = torch.ops.aten.reshape.default(constant_pad_nd_21, [256, 4, 4, 23]);  constant_pad_nd_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:45, code: x = (q @ rel_k.transpose(-1, -2))
    view_127: "f32[4096, 23]" = torch.ops.aten.reshape.default(view_126, [4096, 23]);  view_126 = None
    permute_71: "f32[23, 4096]" = torch.ops.aten.permute.default(view_127, [1, 0])
    mm_12: "f32[23, 16]" = torch.ops.aten.mm.default(permute_71, view_48);  permute_71 = view_48 = None
    permute_72: "f32[16, 23]" = torch.ops.aten.permute.default(mm_12, [1, 0]);  mm_12 = None
    mm_13: "f32[4096, 16]" = torch.ops.aten.mm.default(view_127, permute_73);  view_127 = permute_73 = None
    view_128: "f32[256, 4, 4, 16]" = torch.ops.aten.reshape.default(mm_13, [256, 4, 4, 16]);  mm_13 = None
    permute_74: "f32[23, 16]" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:89, code: q = q.transpose(1, 2)
    permute_75: "f32[256, 4, 4, 16]" = torch.ops.aten.permute.default(view_128, [0, 2, 1, 3]);  view_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:58, code: return x.permute(permute_mask)
    permute_76: "f32[256, 4, 12, 4, 12]" = torch.ops.aten.permute.default(view_122, [0, 1, 3, 2, 4]);  view_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:57, code: x = x.reshape(B, H, 1, W, win_size).expand(-1, -1, win_size, -1, -1)
    sum_22: "f32[256, 4, 1, 4, 12]" = torch.ops.aten.sum.dim_IntList(permute_76, [2], True);  permute_76 = None
    view_129: "f32[1024, 4, 12]" = torch.ops.aten.reshape.default(sum_22, [1024, 4, 12]);  sum_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:54, code: x = x_pad[:, :W, win_size - 1:]
    slice_scatter_9: "f32[1024, 4, 23]" = torch.ops.aten.slice_scatter.default(full_default_13, view_129, 2, 11, 9223372036854775807);  full_default_13 = view_129 = None
    slice_scatter_10: "f32[1024, 5, 23]" = torch.ops.aten.slice_scatter.default(full_default_14, slice_scatter_9, 1, 0, 4);  full_default_14 = slice_scatter_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:53, code: x_pad = x_pad.reshape(-1, W + 1, rel_size)
    view_130: "f32[1024, 115]" = torch.ops.aten.reshape.default(slice_scatter_10, [1024, 115]);  slice_scatter_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:50, code: x_pad = F.pad(x_pad, [0, rel_size - W])
    constant_pad_nd_22: "f32[1024, 96]" = torch.ops.aten.constant_pad_nd.default(view_130, [0, -19]);  view_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:49, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_131: "f32[1024, 4, 24]" = torch.ops.aten.reshape.default(constant_pad_nd_22, [1024, 4, 24]);  constant_pad_nd_22 = None
    constant_pad_nd_23: "f32[1024, 4, 23]" = torch.ops.aten.constant_pad_nd.default(view_131, [0, -1]);  view_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:46, code: x = x.reshape(-1, W, rel_size)
    view_132: "f32[256, 4, 4, 23]" = torch.ops.aten.reshape.default(constant_pad_nd_23, [256, 4, 4, 23]);  constant_pad_nd_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:45, code: x = (q @ rel_k.transpose(-1, -2))
    view_133: "f32[4096, 23]" = torch.ops.aten.reshape.default(view_132, [4096, 23]);  view_132 = None
    permute_77: "f32[23, 4096]" = torch.ops.aten.permute.default(view_133, [1, 0])
    mm_14: "f32[23, 16]" = torch.ops.aten.mm.default(permute_77, view_42);  permute_77 = view_42 = None
    permute_78: "f32[16, 23]" = torch.ops.aten.permute.default(mm_14, [1, 0]);  mm_14 = None
    mm_15: "f32[4096, 16]" = torch.ops.aten.mm.default(view_133, permute_79);  view_133 = permute_79 = None
    view_134: "f32[256, 4, 4, 16]" = torch.ops.aten.reshape.default(mm_15, [256, 4, 4, 16]);  mm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:45, code: x = (q @ rel_k.transpose(-1, -2))
    add_178: "f32[256, 4, 4, 16]" = torch.ops.aten.add.Tensor(permute_75, view_134);  permute_75 = view_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:45, code: x = (q @ rel_k.transpose(-1, -2))
    permute_80: "f32[23, 16]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:85, code: q = q.reshape(-1, self.block_size, self.block_size, self.dim_head)
    clone_57: "f32[256, 4, 4, 16]" = torch.ops.aten.clone.default(add_178, memory_format = torch.contiguous_format);  add_178 = None
    view_135: "f32[64, 4, 16, 16]" = torch.ops.aten.reshape.default(clone_57, [64, 4, 16, 16]);  clone_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:199, code: attn = (q @ k.transpose(-1, -2)) * self.scale + self.pos_embed(q)
    mul_326: "f32[64, 4, 16, 144]" = torch.ops.aten.mul.Tensor(sub_64, 0.25);  sub_64 = None
    view_136: "f32[256, 16, 144]" = torch.ops.aten.reshape.default(mul_326, [256, 16, 144]);  mul_326 = None
    bmm_12: "f32[256, 16, 144]" = torch.ops.aten.bmm.default(permute_81, view_136);  permute_81 = None
    bmm_13: "f32[256, 16, 16]" = torch.ops.aten.bmm.default(view_136, permute_82);  view_136 = permute_82 = None
    view_137: "f32[64, 4, 16, 144]" = torch.ops.aten.reshape.default(bmm_12, [64, 4, 16, 144]);  bmm_12 = None
    view_138: "f32[64, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_13, [64, 4, 16, 16]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:199, code: attn = (q @ k.transpose(-1, -2)) * self.scale + self.pos_embed(q)
    add_179: "f32[64, 4, 16, 16]" = torch.ops.aten.add.Tensor(view_135, view_138);  view_135 = view_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:199, code: attn = (q @ k.transpose(-1, -2)) * self.scale + self.pos_embed(q)
    permute_83: "f32[64, 4, 144, 16]" = torch.ops.aten.permute.default(view_137, [0, 1, 3, 2]);  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:193, code: k, v = torch.split(kv, [self.dim_head_qk, self.dim_head_v], dim=-1)
    cat_1: "f32[64, 4, 144, 80]" = torch.ops.aten.cat.default([permute_83, view_120], 3);  permute_83 = view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:192, code: B * self.num_heads, self.dim_head_qk + self.dim_head_v, num_blocks, -1).permute(0, 2, 3, 1)
    permute_84: "f32[64, 80, 4, 144]" = torch.ops.aten.permute.default(cat_1, [0, 3, 1, 2]);  cat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:191, code: kv = kv.unfold(2, self.win_size, self.block_size).unfold(3, self.win_size, self.block_size).reshape(
    clone_58: "f32[64, 80, 4, 144]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
    view_139: "f32[8, 640, 2, 2, 12, 12]" = torch.ops.aten.reshape.default(clone_58, [8, 640, 2, 2, 12, 12]);  clone_58 = None
    iota_2: "i32[20]" = torch.ops.prims.iota.default(20, start = 0, step = 1, dtype = torch.int32, device = device(type='cpu'), requires_grad = False)
    unfold_8: "i32[2, 12]" = torch.ops.aten.unfold.default(iota_2, 0, 12, 8);  iota_2 = None
    clone_59: "i32[2, 12]" = torch.ops.aten.clone.default(unfold_8, memory_format = torch.contiguous_format);  unfold_8 = None
    view_140: "i32[24]" = torch.ops.aten.reshape.default(clone_59, [24]);  clone_59 = None
    permute_85: "f32[8, 640, 2, 2, 12, 12]" = torch.ops.aten.permute.default(view_139, [0, 1, 2, 3, 5, 4]);  view_139 = None
    clone_60: "f32[8, 640, 2, 2, 12, 12]" = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None
    view_141: "f32[8, 640, 2, 24, 12]" = torch.ops.aten.reshape.default(clone_60, [8, 640, 2, 24, 12]);  clone_60 = None
    full_default_19: "f32[8, 640, 2, 20, 12]" = torch.ops.aten.full.default([8, 640, 2, 20, 12], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_2: "f32[8, 640, 2, 20, 12]" = torch.ops.prims._unsafe_index_put_.default(full_default_19, [None, None, None, view_140], view_141, True);  full_default_19 = view_141 = None
    permute_86: "f32[8, 640, 2, 12, 20]" = torch.ops.aten.permute.default(_unsafe_index_put_2, [0, 1, 2, 4, 3]);  _unsafe_index_put_2 = None
    clone_62: "f32[8, 640, 2, 12, 20]" = torch.ops.aten.clone.default(permute_86, memory_format = torch.contiguous_format);  permute_86 = None
    view_143: "f32[8, 640, 24, 20]" = torch.ops.aten.reshape.default(clone_62, [8, 640, 24, 20]);  clone_62 = None
    full_default_20: "f32[8, 640, 20, 20]" = torch.ops.aten.full.default([8, 640, 20, 20], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_3: "f32[8, 640, 20, 20]" = torch.ops.prims._unsafe_index_put_.default(full_default_20, [None, None, view_140], view_143, True);  full_default_20 = view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:190, code: kv = F.pad(kv, [self.halo_size, self.halo_size, self.halo_size, self.halo_size])
    constant_pad_nd_24: "f32[8, 640, 16, 16]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_3, [-2, -2, -2, -2]);  _unsafe_index_put_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:186, code: kv = self.kv(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(constant_pad_nd_24, mul_202, primals_101, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  constant_pad_nd_24 = primals_101 = None
    getitem_88: "f32[8, 512, 16, 16]" = convolution_backward_6[0]
    getitem_89: "f32[640, 512, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:183, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1, num_blocks).transpose(1, 3)
    permute_87: "f32[64, 16, 16, 4]" = torch.ops.aten.permute.default(add_179, [0, 3, 2, 1]);  add_179 = None
    view_144: "f32[64, 16, 4, 4, 2, 2]" = torch.ops.aten.reshape.default(permute_87, [64, 16, 4, 4, 2, 2]);  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:181, code: num_h_blocks, self.block_size_ds, num_w_blocks, self.block_size_ds).permute(0, 1, 3, 5, 2, 4)
    permute_88: "f32[64, 16, 2, 4, 2, 4]" = torch.ops.aten.permute.default(view_144, [0, 1, 4, 2, 5, 3]);  view_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:179, code: q = q.reshape(
    clone_63: "f32[64, 16, 2, 4, 2, 4]" = torch.ops.aten.clone.default(permute_88, memory_format = torch.contiguous_format);  permute_88 = None
    view_145: "f32[8, 128, 8, 8]" = torch.ops.aten.reshape.default(clone_63, [8, 128, 8, 8]);  clone_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:177, code: q = self.q(x)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(view_145, mul_202, primals_100, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_145 = mul_202 = primals_100 = None
    getitem_91: "f32[8, 512, 16, 16]" = convolution_backward_7[0]
    getitem_92: "f32[128, 512, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:177, code: q = self.q(x)
    add_180: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(getitem_88, getitem_91);  getitem_88 = getitem_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_329: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(add_180, mul_328);  add_180 = mul_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_23: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_329, [0, 2, 3])
    sub_66: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_198);  convolution_30 = unsqueeze_198 = None
    mul_330: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_329, sub_66)
    sum_24: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_330, [0, 2, 3]);  mul_330 = None
    mul_331: "f32[512]" = torch.ops.aten.mul.Tensor(sum_23, 0.00048828125)
    unsqueeze_199: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_331, 0);  mul_331 = None
    unsqueeze_200: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_199, 2);  unsqueeze_199 = None
    unsqueeze_201: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, 3);  unsqueeze_200 = None
    mul_332: "f32[512]" = torch.ops.aten.mul.Tensor(sum_24, 0.00048828125)
    mul_333: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_334: "f32[512]" = torch.ops.aten.mul.Tensor(mul_332, mul_333);  mul_332 = mul_333 = None
    unsqueeze_202: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_334, 0);  mul_334 = None
    unsqueeze_203: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, 2);  unsqueeze_202 = None
    unsqueeze_204: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 3);  unsqueeze_203 = None
    mul_335: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_51);  primals_51 = None
    unsqueeze_205: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_335, 0);  mul_335 = None
    unsqueeze_206: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 2);  unsqueeze_205 = None
    unsqueeze_207: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, 3);  unsqueeze_206 = None
    mul_336: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_204);  sub_66 = unsqueeze_204 = None
    sub_68: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(mul_329, mul_336);  mul_329 = mul_336 = None
    sub_69: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_68, unsqueeze_201);  sub_68 = unsqueeze_201 = None
    mul_337: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_207);  sub_69 = unsqueeze_207 = None
    mul_338: "f32[512]" = torch.ops.aten.mul.Tensor(sum_24, squeeze_73);  sum_24 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_337, mul_194, primals_99, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_337 = mul_194 = primals_99 = None
    getitem_94: "f32[8, 1024, 16, 16]" = convolution_backward_8[0]
    getitem_95: "f32[512, 1024, 1, 1]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_182: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(getitem_82, getitem_94);  getitem_82 = getitem_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    mul_341: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(add_182, mul_340);  add_182 = mul_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_25: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_341, [0, 2, 3])
    sub_71: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_210);  convolution_29 = unsqueeze_210 = None
    mul_342: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_341, sub_71)
    sum_26: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_342, [0, 2, 3]);  mul_342 = None
    mul_343: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_25, 0.00048828125)
    unsqueeze_211: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_343, 0);  mul_343 = None
    unsqueeze_212: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 2);  unsqueeze_211 = None
    unsqueeze_213: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, 3);  unsqueeze_212 = None
    mul_344: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_26, 0.00048828125)
    mul_345: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_346: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
    unsqueeze_214: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_346, 0);  mul_346 = None
    unsqueeze_215: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, 2);  unsqueeze_214 = None
    unsqueeze_216: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 3);  unsqueeze_215 = None
    mul_347: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_49);  primals_49 = None
    unsqueeze_217: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_347, 0);  mul_347 = None
    unsqueeze_218: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 2);  unsqueeze_217 = None
    unsqueeze_219: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 3);  unsqueeze_218 = None
    mul_348: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_216);  sub_71 = unsqueeze_216 = None
    sub_73: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(mul_341, mul_348);  mul_348 = None
    sub_74: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_73, unsqueeze_213);  sub_73 = unsqueeze_213 = None
    mul_349: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_219);  sub_74 = unsqueeze_219 = None
    mul_350: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_26, squeeze_70);  sum_26 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_349, mul_186, primals_98, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_349 = mul_186 = primals_98 = None
    getitem_97: "f32[8, 256, 16, 16]" = convolution_backward_9[0]
    getitem_98: "f32[1024, 256, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    full_default_23: "f32[8, 256, 16, 16]" = torch.ops.aten.full.default([8, 256, 16, 16], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    mul_353: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_97, mul_352);  getitem_97 = mul_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_27: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_353, [0, 2, 3])
    mul_354: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_353, sub_76)
    sum_28: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_354, [0, 2, 3]);  mul_354 = None
    mul_355: "f32[256]" = torch.ops.aten.mul.Tensor(sum_27, 0.00048828125)
    unsqueeze_223: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_355, 0);  mul_355 = None
    unsqueeze_224: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 2);  unsqueeze_223 = None
    unsqueeze_225: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 3);  unsqueeze_224 = None
    mul_356: "f32[256]" = torch.ops.aten.mul.Tensor(sum_28, 0.00048828125)
    mul_357: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_358: "f32[256]" = torch.ops.aten.mul.Tensor(mul_356, mul_357);  mul_356 = mul_357 = None
    unsqueeze_226: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_358, 0);  mul_358 = None
    unsqueeze_227: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 2);  unsqueeze_226 = None
    unsqueeze_228: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 3);  unsqueeze_227 = None
    mul_359: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_47);  primals_47 = None
    unsqueeze_229: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_359, 0);  mul_359 = None
    unsqueeze_230: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 2);  unsqueeze_229 = None
    unsqueeze_231: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 3);  unsqueeze_230 = None
    mul_360: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_228);  sub_76 = unsqueeze_228 = None
    sub_78: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(mul_353, mul_360);  mul_353 = mul_360 = None
    sub_79: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_78, unsqueeze_225);  sub_78 = unsqueeze_225 = None
    mul_361: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_231);  sub_79 = unsqueeze_231 = None
    mul_362: "f32[256]" = torch.ops.aten.mul.Tensor(sum_28, squeeze_67);  sum_28 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:206, code: out = out.permute(0, 3, 1, 4, 2).contiguous().view(
    view_146: "f32[2048, 2, 8, 2, 8]" = torch.ops.aten.reshape.default(mul_361, [2048, 2, 8, 2, 8]);  mul_361 = None
    permute_92: "f32[2048, 8, 8, 2, 2]" = torch.ops.aten.permute.default(view_146, [0, 2, 4, 1, 3]);  view_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:205, code: out = out.reshape(-1, self.block_size_ds, self.block_size_ds, num_h_blocks, num_w_blocks)
    clone_64: "f32[2048, 8, 8, 2, 2]" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
    view_147: "f32[64, 32, 64, 4]" = torch.ops.aten.reshape.default(clone_64, [64, 32, 64, 4]);  clone_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:203, code: out = (attn @ v).transpose(1, 3)  # B * num_heads, dim_head_v, block_size ** 2, num_blocks
    permute_93: "f32[64, 4, 64, 32]" = torch.ops.aten.permute.default(view_147, [0, 3, 2, 1]);  view_147 = None
    clone_65: "f32[64, 4, 64, 32]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
    view_148: "f32[256, 64, 32]" = torch.ops.aten.reshape.default(clone_65, [256, 64, 32]);  clone_65 = None
    bmm_14: "f32[256, 144, 32]" = torch.ops.aten.bmm.default(permute_94, view_148);  permute_94 = None
    bmm_15: "f32[256, 64, 144]" = torch.ops.aten.bmm.default(view_148, permute_95);  view_148 = permute_95 = None
    view_149: "f32[64, 4, 144, 32]" = torch.ops.aten.reshape.default(bmm_14, [64, 4, 144, 32]);  bmm_14 = None
    view_150: "f32[64, 4, 64, 144]" = torch.ops.aten.reshape.default(bmm_15, [64, 4, 64, 144]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:201, code: attn = attn.softmax(dim=-1)
    mul_363: "f32[64, 4, 64, 144]" = torch.ops.aten.mul.Tensor(view_150, alias_10);  view_150 = None
    sum_29: "f32[64, 4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_363, [-1], True)
    mul_364: "f32[64, 4, 64, 144]" = torch.ops.aten.mul.Tensor(alias_10, sum_29);  alias_10 = sum_29 = None
    sub_80: "f32[64, 4, 64, 144]" = torch.ops.aten.sub.Tensor(mul_363, mul_364);  mul_363 = mul_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:93, code: rel_logits = rel_logits.reshape(B, BB, HW, -1)
    view_151: "f32[256, 8, 8, 12, 12]" = torch.ops.aten.reshape.default(sub_80, [256, 8, 8, 12, 12])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:58, code: return x.permute(permute_mask)
    permute_96: "f32[256, 8, 12, 8, 12]" = torch.ops.aten.permute.default(view_151, [0, 2, 4, 1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:57, code: x = x.reshape(B, H, 1, W, win_size).expand(-1, -1, win_size, -1, -1)
    sum_30: "f32[256, 8, 1, 8, 12]" = torch.ops.aten.sum.dim_IntList(permute_96, [2], True);  permute_96 = None
    view_152: "f32[2048, 8, 12]" = torch.ops.aten.reshape.default(sum_30, [2048, 8, 12]);  sum_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:54, code: x = x_pad[:, :W, win_size - 1:]
    full_default_24: "f32[2048, 8, 23]" = torch.ops.aten.full.default([2048, 8, 23], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_12: "f32[2048, 8, 23]" = torch.ops.aten.slice_scatter.default(full_default_24, view_152, 2, 11, 9223372036854775807);  view_152 = None
    full_default_25: "f32[2048, 9, 23]" = torch.ops.aten.full.default([2048, 9, 23], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_13: "f32[2048, 9, 23]" = torch.ops.aten.slice_scatter.default(full_default_25, slice_scatter_12, 1, 0, 8);  slice_scatter_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:53, code: x_pad = x_pad.reshape(-1, W + 1, rel_size)
    view_153: "f32[2048, 207]" = torch.ops.aten.reshape.default(slice_scatter_13, [2048, 207]);  slice_scatter_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:50, code: x_pad = F.pad(x_pad, [0, rel_size - W])
    constant_pad_nd_25: "f32[2048, 192]" = torch.ops.aten.constant_pad_nd.default(view_153, [0, -15]);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:49, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_154: "f32[2048, 8, 24]" = torch.ops.aten.reshape.default(constant_pad_nd_25, [2048, 8, 24]);  constant_pad_nd_25 = None
    constant_pad_nd_26: "f32[2048, 8, 23]" = torch.ops.aten.constant_pad_nd.default(view_154, [0, -1]);  view_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:46, code: x = x.reshape(-1, W, rel_size)
    view_155: "f32[256, 8, 8, 23]" = torch.ops.aten.reshape.default(constant_pad_nd_26, [256, 8, 8, 23]);  constant_pad_nd_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:45, code: x = (q @ rel_k.transpose(-1, -2))
    view_156: "f32[16384, 23]" = torch.ops.aten.reshape.default(view_155, [16384, 23]);  view_155 = None
    permute_97: "f32[23, 16384]" = torch.ops.aten.permute.default(view_156, [1, 0])
    mm_16: "f32[23, 16]" = torch.ops.aten.mm.default(permute_97, view_23);  permute_97 = view_23 = None
    permute_98: "f32[16, 23]" = torch.ops.aten.permute.default(mm_16, [1, 0]);  mm_16 = None
    mm_17: "f32[16384, 16]" = torch.ops.aten.mm.default(view_156, permute_99);  view_156 = permute_99 = None
    view_157: "f32[256, 8, 8, 16]" = torch.ops.aten.reshape.default(mm_17, [256, 8, 8, 16]);  mm_17 = None
    permute_100: "f32[23, 16]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:89, code: q = q.transpose(1, 2)
    permute_101: "f32[256, 8, 8, 16]" = torch.ops.aten.permute.default(view_157, [0, 2, 1, 3]);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:58, code: return x.permute(permute_mask)
    permute_102: "f32[256, 8, 12, 8, 12]" = torch.ops.aten.permute.default(view_151, [0, 1, 3, 2, 4]);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:57, code: x = x.reshape(B, H, 1, W, win_size).expand(-1, -1, win_size, -1, -1)
    sum_31: "f32[256, 8, 1, 8, 12]" = torch.ops.aten.sum.dim_IntList(permute_102, [2], True);  permute_102 = None
    view_158: "f32[2048, 8, 12]" = torch.ops.aten.reshape.default(sum_31, [2048, 8, 12]);  sum_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:54, code: x = x_pad[:, :W, win_size - 1:]
    slice_scatter_15: "f32[2048, 8, 23]" = torch.ops.aten.slice_scatter.default(full_default_24, view_158, 2, 11, 9223372036854775807);  full_default_24 = view_158 = None
    slice_scatter_16: "f32[2048, 9, 23]" = torch.ops.aten.slice_scatter.default(full_default_25, slice_scatter_15, 1, 0, 8);  full_default_25 = slice_scatter_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:53, code: x_pad = x_pad.reshape(-1, W + 1, rel_size)
    view_159: "f32[2048, 207]" = torch.ops.aten.reshape.default(slice_scatter_16, [2048, 207]);  slice_scatter_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:50, code: x_pad = F.pad(x_pad, [0, rel_size - W])
    constant_pad_nd_27: "f32[2048, 192]" = torch.ops.aten.constant_pad_nd.default(view_159, [0, -15]);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:49, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_160: "f32[2048, 8, 24]" = torch.ops.aten.reshape.default(constant_pad_nd_27, [2048, 8, 24]);  constant_pad_nd_27 = None
    constant_pad_nd_28: "f32[2048, 8, 23]" = torch.ops.aten.constant_pad_nd.default(view_160, [0, -1]);  view_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:46, code: x = x.reshape(-1, W, rel_size)
    view_161: "f32[256, 8, 8, 23]" = torch.ops.aten.reshape.default(constant_pad_nd_28, [256, 8, 8, 23]);  constant_pad_nd_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:45, code: x = (q @ rel_k.transpose(-1, -2))
    view_162: "f32[16384, 23]" = torch.ops.aten.reshape.default(view_161, [16384, 23]);  view_161 = None
    permute_103: "f32[23, 16384]" = torch.ops.aten.permute.default(view_162, [1, 0])
    mm_18: "f32[23, 16]" = torch.ops.aten.mm.default(permute_103, view_17);  permute_103 = view_17 = None
    permute_104: "f32[16, 23]" = torch.ops.aten.permute.default(mm_18, [1, 0]);  mm_18 = None
    mm_19: "f32[16384, 16]" = torch.ops.aten.mm.default(view_162, permute_105);  view_162 = permute_105 = None
    view_163: "f32[256, 8, 8, 16]" = torch.ops.aten.reshape.default(mm_19, [256, 8, 8, 16]);  mm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:45, code: x = (q @ rel_k.transpose(-1, -2))
    add_185: "f32[256, 8, 8, 16]" = torch.ops.aten.add.Tensor(permute_101, view_163);  permute_101 = view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:45, code: x = (q @ rel_k.transpose(-1, -2))
    permute_106: "f32[23, 16]" = torch.ops.aten.permute.default(permute_104, [1, 0]);  permute_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:85, code: q = q.reshape(-1, self.block_size, self.block_size, self.dim_head)
    clone_66: "f32[256, 8, 8, 16]" = torch.ops.aten.clone.default(add_185, memory_format = torch.contiguous_format);  add_185 = None
    view_164: "f32[64, 4, 64, 16]" = torch.ops.aten.reshape.default(clone_66, [64, 4, 64, 16]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:199, code: attn = (q @ k.transpose(-1, -2)) * self.scale + self.pos_embed(q)
    mul_365: "f32[64, 4, 64, 144]" = torch.ops.aten.mul.Tensor(sub_80, 0.25);  sub_80 = None
    view_165: "f32[256, 64, 144]" = torch.ops.aten.reshape.default(mul_365, [256, 64, 144]);  mul_365 = None
    bmm_16: "f32[256, 16, 144]" = torch.ops.aten.bmm.default(permute_107, view_165);  permute_107 = None
    bmm_17: "f32[256, 64, 16]" = torch.ops.aten.bmm.default(view_165, permute_108);  view_165 = permute_108 = None
    view_166: "f32[64, 4, 16, 144]" = torch.ops.aten.reshape.default(bmm_16, [64, 4, 16, 144]);  bmm_16 = None
    view_167: "f32[64, 4, 64, 16]" = torch.ops.aten.reshape.default(bmm_17, [64, 4, 64, 16]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:199, code: attn = (q @ k.transpose(-1, -2)) * self.scale + self.pos_embed(q)
    add_186: "f32[64, 4, 64, 16]" = torch.ops.aten.add.Tensor(view_164, view_167);  view_164 = view_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:199, code: attn = (q @ k.transpose(-1, -2)) * self.scale + self.pos_embed(q)
    permute_109: "f32[64, 4, 144, 16]" = torch.ops.aten.permute.default(view_166, [0, 1, 3, 2]);  view_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:193, code: k, v = torch.split(kv, [self.dim_head_qk, self.dim_head_v], dim=-1)
    cat_2: "f32[64, 4, 144, 48]" = torch.ops.aten.cat.default([permute_109, view_149], 3);  permute_109 = view_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:192, code: B * self.num_heads, self.dim_head_qk + self.dim_head_v, num_blocks, -1).permute(0, 2, 3, 1)
    permute_110: "f32[64, 48, 4, 144]" = torch.ops.aten.permute.default(cat_2, [0, 3, 1, 2]);  cat_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:191, code: kv = kv.unfold(2, self.win_size, self.block_size).unfold(3, self.win_size, self.block_size).reshape(
    clone_67: "f32[64, 48, 4, 144]" = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None
    view_168: "f32[8, 384, 2, 2, 12, 12]" = torch.ops.aten.reshape.default(clone_67, [8, 384, 2, 2, 12, 12]);  clone_67 = None
    permute_111: "f32[8, 384, 2, 2, 12, 12]" = torch.ops.aten.permute.default(view_168, [0, 1, 2, 3, 5, 4]);  view_168 = None
    clone_69: "f32[8, 384, 2, 2, 12, 12]" = torch.ops.aten.clone.default(permute_111, memory_format = torch.contiguous_format);  permute_111 = None
    view_170: "f32[8, 384, 2, 24, 12]" = torch.ops.aten.reshape.default(clone_69, [8, 384, 2, 24, 12]);  clone_69 = None
    full_default_30: "f32[8, 384, 2, 20, 12]" = torch.ops.aten.full.default([8, 384, 2, 20, 12], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_4: "f32[8, 384, 2, 20, 12]" = torch.ops.prims._unsafe_index_put_.default(full_default_30, [None, None, None, view_140], view_170, True);  full_default_30 = view_170 = None
    permute_112: "f32[8, 384, 2, 12, 20]" = torch.ops.aten.permute.default(_unsafe_index_put_4, [0, 1, 2, 4, 3]);  _unsafe_index_put_4 = None
    clone_71: "f32[8, 384, 2, 12, 20]" = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
    view_172: "f32[8, 384, 24, 20]" = torch.ops.aten.reshape.default(clone_71, [8, 384, 24, 20]);  clone_71 = None
    full_default_31: "f32[8, 384, 20, 20]" = torch.ops.aten.full.default([8, 384, 20, 20], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_5: "f32[8, 384, 20, 20]" = torch.ops.prims._unsafe_index_put_.default(full_default_31, [None, None, view_140], view_172, True);  full_default_31 = view_140 = view_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:190, code: kv = F.pad(kv, [self.halo_size, self.halo_size, self.halo_size, self.halo_size])
    constant_pad_nd_29: "f32[8, 384, 16, 16]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_5, [-2, -2, -2, -2]);  _unsafe_index_put_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:186, code: kv = self.kv(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(constant_pad_nd_29, mul_177, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  constant_pad_nd_29 = primals_97 = None
    getitem_100: "f32[8, 256, 16, 16]" = convolution_backward_10[0]
    getitem_101: "f32[384, 256, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:183, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1, num_blocks).transpose(1, 3)
    permute_113: "f32[64, 16, 64, 4]" = torch.ops.aten.permute.default(add_186, [0, 3, 2, 1]);  add_186 = None
    view_173: "f32[64, 16, 8, 8, 2, 2]" = torch.ops.aten.reshape.default(permute_113, [64, 16, 8, 8, 2, 2]);  permute_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:181, code: num_h_blocks, self.block_size_ds, num_w_blocks, self.block_size_ds).permute(0, 1, 3, 5, 2, 4)
    permute_114: "f32[64, 16, 2, 8, 2, 8]" = torch.ops.aten.permute.default(view_173, [0, 1, 4, 2, 5, 3]);  view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:179, code: q = q.reshape(
    clone_72: "f32[64, 16, 2, 8, 2, 8]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
    view_174: "f32[8, 128, 16, 16]" = torch.ops.aten.reshape.default(clone_72, [8, 128, 16, 16]);  clone_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:177, code: q = self.q(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(view_174, mul_177, primals_96, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_174 = mul_177 = primals_96 = None
    getitem_103: "f32[8, 256, 16, 16]" = convolution_backward_11[0]
    getitem_104: "f32[128, 256, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:177, code: q = self.q(x)
    add_187: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(getitem_100, getitem_103);  getitem_100 = getitem_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_368: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_187, mul_367);  add_187 = mul_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_32: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_368, [0, 2, 3])
    sub_82: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_234);  convolution_26 = unsqueeze_234 = None
    mul_369: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_368, sub_82)
    sum_33: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_369, [0, 2, 3]);  mul_369 = None
    mul_370: "f32[256]" = torch.ops.aten.mul.Tensor(sum_32, 0.00048828125)
    unsqueeze_235: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_370, 0);  mul_370 = None
    unsqueeze_236: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 2);  unsqueeze_235 = None
    unsqueeze_237: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 3);  unsqueeze_236 = None
    mul_371: "f32[256]" = torch.ops.aten.mul.Tensor(sum_33, 0.00048828125)
    mul_372: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_373: "f32[256]" = torch.ops.aten.mul.Tensor(mul_371, mul_372);  mul_371 = mul_372 = None
    unsqueeze_238: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_373, 0);  mul_373 = None
    unsqueeze_239: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 2);  unsqueeze_238 = None
    unsqueeze_240: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 3);  unsqueeze_239 = None
    mul_374: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_43);  primals_43 = None
    unsqueeze_241: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_374, 0);  mul_374 = None
    unsqueeze_242: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 2);  unsqueeze_241 = None
    unsqueeze_243: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 3);  unsqueeze_242 = None
    mul_375: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_240);  sub_82 = unsqueeze_240 = None
    sub_84: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(mul_368, mul_375);  mul_368 = mul_375 = None
    sub_85: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_84, unsqueeze_237);  sub_84 = unsqueeze_237 = None
    mul_376: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_243);  sub_85 = unsqueeze_243 = None
    mul_377: "f32[256]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_64);  sum_33 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_376, mul_169, primals_95, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_376 = mul_169 = primals_95 = None
    getitem_106: "f32[8, 1024, 16, 16]" = convolution_backward_12[0]
    getitem_107: "f32[256, 1024, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_189: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_341, getitem_106);  mul_341 = getitem_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    mul_380: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(add_189, mul_379);  add_189 = mul_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_34: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_380, [0, 2, 3])
    sub_87: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_246);  convolution_25 = unsqueeze_246 = None
    mul_381: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_380, sub_87)
    sum_35: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_381, [0, 2, 3]);  mul_381 = None
    mul_382: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_34, 0.00048828125)
    unsqueeze_247: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_382, 0);  mul_382 = None
    unsqueeze_248: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 2);  unsqueeze_247 = None
    unsqueeze_249: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 3);  unsqueeze_248 = None
    mul_383: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_35, 0.00048828125)
    mul_384: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_385: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_383, mul_384);  mul_383 = mul_384 = None
    unsqueeze_250: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_385, 0);  mul_385 = None
    unsqueeze_251: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 2);  unsqueeze_250 = None
    unsqueeze_252: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 3);  unsqueeze_251 = None
    mul_386: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_41);  primals_41 = None
    unsqueeze_253: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_386, 0);  mul_386 = None
    unsqueeze_254: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 2);  unsqueeze_253 = None
    unsqueeze_255: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 3);  unsqueeze_254 = None
    mul_387: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_252);  sub_87 = unsqueeze_252 = None
    sub_89: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(mul_380, mul_387);  mul_387 = None
    sub_90: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_89, unsqueeze_249);  sub_89 = None
    mul_388: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_255);  sub_90 = unsqueeze_255 = None
    mul_389: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_61);  sum_35 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_388, mul_137, primals_94, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_388 = primals_94 = None
    getitem_109: "f32[8, 512, 32, 32]" = convolution_backward_13[0]
    getitem_110: "f32[1024, 512, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_91: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_258);  convolution_24 = unsqueeze_258 = None
    mul_390: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_380, sub_91)
    sum_37: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_390, [0, 2, 3]);  mul_390 = None
    mul_392: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_37, 0.00048828125)
    mul_393: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_394: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_392, mul_393);  mul_392 = mul_393 = None
    unsqueeze_262: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_394, 0);  mul_394 = None
    unsqueeze_263: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, 2);  unsqueeze_262 = None
    unsqueeze_264: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 3);  unsqueeze_263 = None
    mul_395: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_39);  primals_39 = None
    unsqueeze_265: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_395, 0);  mul_395 = None
    unsqueeze_266: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 2);  unsqueeze_265 = None
    unsqueeze_267: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 3);  unsqueeze_266 = None
    mul_396: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_264);  sub_91 = unsqueeze_264 = None
    sub_93: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(mul_380, mul_396);  mul_380 = mul_396 = None
    sub_94: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_93, unsqueeze_249);  sub_93 = unsqueeze_249 = None
    mul_397: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_267);  sub_94 = unsqueeze_267 = None
    mul_398: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_58);  sum_37 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_397, mul_154, primals_93, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_397 = mul_154 = primals_93 = None
    getitem_112: "f32[8, 256, 16, 16]" = convolution_backward_14[0]
    getitem_113: "f32[1024, 256, 1, 1]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:91, code: return x * y.expand_as(x)
    mul_399: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_112, mul_153);  mul_153 = None
    mul_400: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_112, expand_4);  getitem_112 = expand_4 = None
    sum_38: "f32[8, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_399, [2, 3], True);  mul_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
    view_175: "f32[8, 1, 256]" = torch.ops.aten.reshape.default(sum_38, [8, 1, 256]);  sum_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_95: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(1, sigmoid_21)
    mul_401: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(sigmoid_21, sub_95);  sigmoid_21 = sub_95 = None
    mul_402: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(view_175, mul_401);  view_175 = mul_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:86, code: y = self.conv(y)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_402, view_8, primals_92, [0], [1], [2], [1], False, [0], 1, [True, True, False]);  mul_402 = view_8 = primals_92 = None
    getitem_115: "f32[8, 1, 256]" = convolution_backward_15[0]
    getitem_116: "f32[1, 1, 5]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    view_176: "f32[8, 256]" = torch.ops.aten.reshape.default(getitem_115, [8, 256]);  getitem_115 = None
    unsqueeze_268: "f32[8, 256, 1]" = torch.ops.aten.unsqueeze.default(view_176, 2);  view_176 = None
    unsqueeze_269: "f32[8, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, 3);  unsqueeze_268 = None
    expand_24: "f32[8, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_269, [8, 256, 16, 16]);  unsqueeze_269 = None
    div_4: "f32[8, 256, 16, 16]" = torch.ops.aten.div.Scalar(expand_24, 256);  expand_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    add_191: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_400, div_4);  mul_400 = div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_42: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_98)
    sub_96: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(full_default_23, sigmoid_42);  full_default_23 = None
    mul_403: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_98, sub_96);  add_98 = sub_96 = None
    add_192: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Scalar(mul_403, 1);  mul_403 = None
    mul_404: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_42, add_192);  sigmoid_42 = add_192 = None
    mul_405: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_191, mul_404);  add_191 = mul_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_39: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_405, [0, 2, 3])
    sub_97: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_272);  convolution_22 = unsqueeze_272 = None
    mul_406: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_405, sub_97)
    sum_40: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_406, [0, 2, 3]);  mul_406 = None
    mul_407: "f32[256]" = torch.ops.aten.mul.Tensor(sum_39, 0.00048828125)
    unsqueeze_273: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_407, 0);  mul_407 = None
    unsqueeze_274: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 2);  unsqueeze_273 = None
    unsqueeze_275: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 3);  unsqueeze_274 = None
    mul_408: "f32[256]" = torch.ops.aten.mul.Tensor(sum_40, 0.00048828125)
    mul_409: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_410: "f32[256]" = torch.ops.aten.mul.Tensor(mul_408, mul_409);  mul_408 = mul_409 = None
    unsqueeze_276: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_410, 0);  mul_410 = None
    unsqueeze_277: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 2);  unsqueeze_276 = None
    unsqueeze_278: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 3);  unsqueeze_277 = None
    mul_411: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_37);  primals_37 = None
    unsqueeze_279: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_411, 0);  mul_411 = None
    unsqueeze_280: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 2);  unsqueeze_279 = None
    unsqueeze_281: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 3);  unsqueeze_280 = None
    mul_412: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_278);  sub_97 = unsqueeze_278 = None
    sub_99: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(mul_405, mul_412);  mul_405 = mul_412 = None
    sub_100: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_99, unsqueeze_275);  sub_99 = unsqueeze_275 = None
    mul_413: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_281);  sub_100 = unsqueeze_281 = None
    mul_414: "f32[256]" = torch.ops.aten.mul.Tensor(sum_40, squeeze_55);  sum_40 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_413, mul_145, primals_91, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False]);  mul_413 = mul_145 = primals_91 = None
    getitem_118: "f32[8, 256, 32, 32]" = convolution_backward_16[0]
    getitem_119: "f32[256, 16, 3, 3]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_417: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_118, mul_416);  getitem_118 = mul_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_41: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_417, [0, 2, 3])
    sub_102: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_284);  convolution_21 = unsqueeze_284 = None
    mul_418: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_417, sub_102)
    sum_42: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_418, [0, 2, 3]);  mul_418 = None
    mul_419: "f32[256]" = torch.ops.aten.mul.Tensor(sum_41, 0.0001220703125)
    unsqueeze_285: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_419, 0);  mul_419 = None
    unsqueeze_286: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 2);  unsqueeze_285 = None
    unsqueeze_287: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 3);  unsqueeze_286 = None
    mul_420: "f32[256]" = torch.ops.aten.mul.Tensor(sum_42, 0.0001220703125)
    mul_421: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_422: "f32[256]" = torch.ops.aten.mul.Tensor(mul_420, mul_421);  mul_420 = mul_421 = None
    unsqueeze_288: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_422, 0);  mul_422 = None
    unsqueeze_289: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 2);  unsqueeze_288 = None
    unsqueeze_290: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 3);  unsqueeze_289 = None
    mul_423: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_35);  primals_35 = None
    unsqueeze_291: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_423, 0);  mul_423 = None
    unsqueeze_292: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 2);  unsqueeze_291 = None
    unsqueeze_293: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 3);  unsqueeze_292 = None
    mul_424: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_290);  sub_102 = unsqueeze_290 = None
    sub_104: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(mul_417, mul_424);  mul_417 = mul_424 = None
    sub_105: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(sub_104, unsqueeze_287);  sub_104 = unsqueeze_287 = None
    mul_425: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_293);  sub_105 = unsqueeze_293 = None
    mul_426: "f32[256]" = torch.ops.aten.mul.Tensor(sum_42, squeeze_52);  sum_42 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_425, mul_137, primals_90, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_425 = mul_137 = primals_90 = None
    getitem_121: "f32[8, 512, 32, 32]" = convolution_backward_17[0]
    getitem_122: "f32[256, 512, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_194: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(getitem_109, getitem_121);  getitem_109 = getitem_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    mul_429: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_194, mul_428);  add_194 = mul_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_43: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_429, [0, 2, 3])
    sub_107: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_296);  convolution_20 = unsqueeze_296 = None
    mul_430: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_429, sub_107)
    sum_44: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_430, [0, 2, 3]);  mul_430 = None
    mul_431: "f32[512]" = torch.ops.aten.mul.Tensor(sum_43, 0.0001220703125)
    unsqueeze_297: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_431, 0);  mul_431 = None
    unsqueeze_298: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 2);  unsqueeze_297 = None
    unsqueeze_299: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 3);  unsqueeze_298 = None
    mul_432: "f32[512]" = torch.ops.aten.mul.Tensor(sum_44, 0.0001220703125)
    mul_433: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_434: "f32[512]" = torch.ops.aten.mul.Tensor(mul_432, mul_433);  mul_432 = mul_433 = None
    unsqueeze_300: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_434, 0);  mul_434 = None
    unsqueeze_301: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 2);  unsqueeze_300 = None
    unsqueeze_302: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 3);  unsqueeze_301 = None
    mul_435: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_33);  primals_33 = None
    unsqueeze_303: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_435, 0);  mul_435 = None
    unsqueeze_304: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 2);  unsqueeze_303 = None
    unsqueeze_305: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, 3);  unsqueeze_304 = None
    mul_436: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_302);  sub_107 = unsqueeze_302 = None
    sub_109: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(mul_429, mul_436);  mul_436 = None
    sub_110: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(sub_109, unsqueeze_299);  sub_109 = unsqueeze_299 = None
    mul_437: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_305);  sub_110 = unsqueeze_305 = None
    mul_438: "f32[512]" = torch.ops.aten.mul.Tensor(sum_44, squeeze_49);  sum_44 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_437, mul_129, primals_89, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_437 = mul_129 = primals_89 = None
    getitem_124: "f32[8, 128, 32, 32]" = convolution_backward_18[0]
    getitem_125: "f32[512, 128, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:91, code: return x * y.expand_as(x)
    mul_439: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_124, mul_128);  mul_128 = None
    mul_440: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_124, expand_3);  getitem_124 = expand_3 = None
    sum_45: "f32[8, 128, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_439, [2, 3], True);  mul_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
    view_177: "f32[8, 1, 128]" = torch.ops.aten.reshape.default(sum_45, [8, 1, 128]);  sum_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_111: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(1, sigmoid_17)
    mul_441: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(sigmoid_17, sub_111);  sigmoid_17 = sub_111 = None
    mul_442: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(view_177, mul_441);  view_177 = mul_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:86, code: y = self.conv(y)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_442, view_6, primals_88, [0], [1], [2], [1], False, [0], 1, [True, True, False]);  mul_442 = view_6 = primals_88 = None
    getitem_127: "f32[8, 1, 128]" = convolution_backward_19[0]
    getitem_128: "f32[1, 1, 5]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    view_178: "f32[8, 128]" = torch.ops.aten.reshape.default(getitem_127, [8, 128]);  getitem_127 = None
    unsqueeze_306: "f32[8, 128, 1]" = torch.ops.aten.unsqueeze.default(view_178, 2);  view_178 = None
    unsqueeze_307: "f32[8, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, 3);  unsqueeze_306 = None
    expand_25: "f32[8, 128, 32, 32]" = torch.ops.aten.expand.default(unsqueeze_307, [8, 128, 32, 32]);  unsqueeze_307 = None
    div_5: "f32[8, 128, 32, 32]" = torch.ops.aten.div.Scalar(expand_25, 1024);  expand_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    add_196: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_440, div_5);  mul_440 = div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_45: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(add_82)
    full_default_37: "f32[8, 128, 32, 32]" = torch.ops.aten.full.default([8, 128, 32, 32], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_112: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(full_default_37, sigmoid_45)
    mul_443: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_82, sub_112);  add_82 = sub_112 = None
    add_197: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Scalar(mul_443, 1);  mul_443 = None
    mul_444: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_45, add_197);  sigmoid_45 = add_197 = None
    mul_445: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_196, mul_444);  add_196 = mul_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_46: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_445, [0, 2, 3])
    sub_113: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_310);  convolution_18 = unsqueeze_310 = None
    mul_446: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_445, sub_113)
    sum_47: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_446, [0, 2, 3]);  mul_446 = None
    mul_447: "f32[128]" = torch.ops.aten.mul.Tensor(sum_46, 0.0001220703125)
    unsqueeze_311: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_447, 0);  mul_447 = None
    unsqueeze_312: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 2);  unsqueeze_311 = None
    unsqueeze_313: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, 3);  unsqueeze_312 = None
    mul_448: "f32[128]" = torch.ops.aten.mul.Tensor(sum_47, 0.0001220703125)
    mul_449: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_450: "f32[128]" = torch.ops.aten.mul.Tensor(mul_448, mul_449);  mul_448 = mul_449 = None
    unsqueeze_314: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_450, 0);  mul_450 = None
    unsqueeze_315: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 2);  unsqueeze_314 = None
    unsqueeze_316: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 3);  unsqueeze_315 = None
    mul_451: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_31);  primals_31 = None
    unsqueeze_317: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_451, 0);  mul_451 = None
    unsqueeze_318: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 2);  unsqueeze_317 = None
    unsqueeze_319: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, 3);  unsqueeze_318 = None
    mul_452: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_316);  sub_113 = unsqueeze_316 = None
    sub_115: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(mul_445, mul_452);  mul_445 = mul_452 = None
    sub_116: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_115, unsqueeze_313);  sub_115 = unsqueeze_313 = None
    mul_453: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_319);  sub_116 = unsqueeze_319 = None
    mul_454: "f32[128]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_46);  sum_47 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_453, mul_120, primals_87, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_453 = mul_120 = primals_87 = None
    getitem_130: "f32[8, 128, 32, 32]" = convolution_backward_20[0]
    getitem_131: "f32[128, 16, 3, 3]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_457: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_130, mul_456);  getitem_130 = mul_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_48: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_457, [0, 2, 3])
    sub_118: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_322);  convolution_17 = unsqueeze_322 = None
    mul_458: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_457, sub_118)
    sum_49: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_458, [0, 2, 3]);  mul_458 = None
    mul_459: "f32[128]" = torch.ops.aten.mul.Tensor(sum_48, 0.0001220703125)
    unsqueeze_323: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_459, 0);  mul_459 = None
    unsqueeze_324: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 2);  unsqueeze_323 = None
    unsqueeze_325: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, 3);  unsqueeze_324 = None
    mul_460: "f32[128]" = torch.ops.aten.mul.Tensor(sum_49, 0.0001220703125)
    mul_461: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_462: "f32[128]" = torch.ops.aten.mul.Tensor(mul_460, mul_461);  mul_460 = mul_461 = None
    unsqueeze_326: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_462, 0);  mul_462 = None
    unsqueeze_327: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 2);  unsqueeze_326 = None
    unsqueeze_328: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 3);  unsqueeze_327 = None
    mul_463: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_29);  primals_29 = None
    unsqueeze_329: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_463, 0);  mul_463 = None
    unsqueeze_330: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 2);  unsqueeze_329 = None
    unsqueeze_331: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, 3);  unsqueeze_330 = None
    mul_464: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_328);  sub_118 = unsqueeze_328 = None
    sub_120: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(mul_457, mul_464);  mul_457 = mul_464 = None
    sub_121: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_120, unsqueeze_325);  sub_120 = unsqueeze_325 = None
    mul_465: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_331);  sub_121 = unsqueeze_331 = None
    mul_466: "f32[128]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_43);  sum_49 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_465, mul_112, primals_86, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_465 = mul_112 = primals_86 = None
    getitem_133: "f32[8, 512, 32, 32]" = convolution_backward_21[0]
    getitem_134: "f32[128, 512, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_199: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_429, getitem_133);  mul_429 = getitem_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    mul_469: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_199, mul_468);  add_199 = mul_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_50: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_469, [0, 2, 3])
    sub_123: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_334);  convolution_16 = unsqueeze_334 = None
    mul_470: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_469, sub_123)
    sum_51: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_470, [0, 2, 3]);  mul_470 = None
    mul_471: "f32[512]" = torch.ops.aten.mul.Tensor(sum_50, 0.0001220703125)
    unsqueeze_335: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_471, 0);  mul_471 = None
    unsqueeze_336: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 2);  unsqueeze_335 = None
    unsqueeze_337: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, 3);  unsqueeze_336 = None
    mul_472: "f32[512]" = torch.ops.aten.mul.Tensor(sum_51, 0.0001220703125)
    mul_473: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_474: "f32[512]" = torch.ops.aten.mul.Tensor(mul_472, mul_473);  mul_472 = mul_473 = None
    unsqueeze_338: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_474, 0);  mul_474 = None
    unsqueeze_339: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 2);  unsqueeze_338 = None
    unsqueeze_340: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 3);  unsqueeze_339 = None
    mul_475: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_27);  primals_27 = None
    unsqueeze_341: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_475, 0);  mul_475 = None
    unsqueeze_342: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 2);  unsqueeze_341 = None
    unsqueeze_343: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 3);  unsqueeze_342 = None
    mul_476: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_340);  sub_123 = unsqueeze_340 = None
    sub_125: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(mul_469, mul_476);  mul_476 = None
    sub_126: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(sub_125, unsqueeze_337);  sub_125 = None
    mul_477: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_343);  sub_126 = unsqueeze_343 = None
    mul_478: "f32[512]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_40);  sum_51 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_477, mul_80, primals_85, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_477 = primals_85 = None
    getitem_136: "f32[8, 256, 64, 64]" = convolution_backward_22[0]
    getitem_137: "f32[512, 256, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_127: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_346);  convolution_15 = unsqueeze_346 = None
    mul_479: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_469, sub_127)
    sum_53: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_479, [0, 2, 3]);  mul_479 = None
    mul_481: "f32[512]" = torch.ops.aten.mul.Tensor(sum_53, 0.0001220703125)
    mul_482: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_483: "f32[512]" = torch.ops.aten.mul.Tensor(mul_481, mul_482);  mul_481 = mul_482 = None
    unsqueeze_350: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_483, 0);  mul_483 = None
    unsqueeze_351: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 2);  unsqueeze_350 = None
    unsqueeze_352: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 3);  unsqueeze_351 = None
    mul_484: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_25);  primals_25 = None
    unsqueeze_353: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_484, 0);  mul_484 = None
    unsqueeze_354: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 2);  unsqueeze_353 = None
    unsqueeze_355: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, 3);  unsqueeze_354 = None
    mul_485: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_352);  sub_127 = unsqueeze_352 = None
    sub_129: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(mul_469, mul_485);  mul_469 = mul_485 = None
    sub_130: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(sub_129, unsqueeze_337);  sub_129 = unsqueeze_337 = None
    mul_486: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_355);  sub_130 = unsqueeze_355 = None
    mul_487: "f32[512]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_37);  sum_53 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_486, mul_97, primals_84, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_486 = mul_97 = primals_84 = None
    getitem_139: "f32[8, 128, 32, 32]" = convolution_backward_23[0]
    getitem_140: "f32[512, 128, 1, 1]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:91, code: return x * y.expand_as(x)
    mul_488: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_139, mul_96);  mul_96 = None
    mul_489: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_139, expand_2);  getitem_139 = expand_2 = None
    sum_54: "f32[8, 128, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_488, [2, 3], True);  mul_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
    view_179: "f32[8, 1, 128]" = torch.ops.aten.reshape.default(sum_54, [8, 1, 128]);  sum_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_131: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(1, sigmoid_13)
    mul_490: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(sigmoid_13, sub_131);  sigmoid_13 = sub_131 = None
    mul_491: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(view_179, mul_490);  view_179 = mul_490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:86, code: y = self.conv(y)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_491, view_4, primals_83, [0], [1], [2], [1], False, [0], 1, [True, True, False]);  mul_491 = view_4 = primals_83 = None
    getitem_142: "f32[8, 1, 128]" = convolution_backward_24[0]
    getitem_143: "f32[1, 1, 5]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    view_180: "f32[8, 128]" = torch.ops.aten.reshape.default(getitem_142, [8, 128]);  getitem_142 = None
    unsqueeze_356: "f32[8, 128, 1]" = torch.ops.aten.unsqueeze.default(view_180, 2);  view_180 = None
    unsqueeze_357: "f32[8, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 3);  unsqueeze_356 = None
    expand_26: "f32[8, 128, 32, 32]" = torch.ops.aten.expand.default(unsqueeze_357, [8, 128, 32, 32]);  unsqueeze_357 = None
    div_6: "f32[8, 128, 32, 32]" = torch.ops.aten.div.Scalar(expand_26, 1024);  expand_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    add_201: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_489, div_6);  mul_489 = div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_48: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(add_61)
    sub_132: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(full_default_37, sigmoid_48);  full_default_37 = None
    mul_492: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_61, sub_132);  add_61 = sub_132 = None
    add_202: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Scalar(mul_492, 1);  mul_492 = None
    mul_493: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_48, add_202);  sigmoid_48 = add_202 = None
    mul_494: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_201, mul_493);  add_201 = mul_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_55: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_494, [0, 2, 3])
    sub_133: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_360);  convolution_13 = unsqueeze_360 = None
    mul_495: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_494, sub_133)
    sum_56: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_495, [0, 2, 3]);  mul_495 = None
    mul_496: "f32[128]" = torch.ops.aten.mul.Tensor(sum_55, 0.0001220703125)
    unsqueeze_361: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_496, 0);  mul_496 = None
    unsqueeze_362: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
    unsqueeze_363: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 3);  unsqueeze_362 = None
    mul_497: "f32[128]" = torch.ops.aten.mul.Tensor(sum_56, 0.0001220703125)
    mul_498: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_499: "f32[128]" = torch.ops.aten.mul.Tensor(mul_497, mul_498);  mul_497 = mul_498 = None
    unsqueeze_364: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_499, 0);  mul_499 = None
    unsqueeze_365: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, 2);  unsqueeze_364 = None
    unsqueeze_366: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 3);  unsqueeze_365 = None
    mul_500: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_23);  primals_23 = None
    unsqueeze_367: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_500, 0);  mul_500 = None
    unsqueeze_368: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 2);  unsqueeze_367 = None
    unsqueeze_369: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 3);  unsqueeze_368 = None
    mul_501: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_366);  sub_133 = unsqueeze_366 = None
    sub_135: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(mul_494, mul_501);  mul_494 = mul_501 = None
    sub_136: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_135, unsqueeze_363);  sub_135 = unsqueeze_363 = None
    mul_502: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_369);  sub_136 = unsqueeze_369 = None
    mul_503: "f32[128]" = torch.ops.aten.mul.Tensor(sum_56, squeeze_34);  sum_56 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_502, mul_88, primals_82, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_502 = mul_88 = primals_82 = None
    getitem_145: "f32[8, 128, 64, 64]" = convolution_backward_25[0]
    getitem_146: "f32[128, 16, 3, 3]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_506: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_145, mul_505);  getitem_145 = mul_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_57: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_506, [0, 2, 3])
    sub_138: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_372);  convolution_12 = unsqueeze_372 = None
    mul_507: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_506, sub_138)
    sum_58: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_507, [0, 2, 3]);  mul_507 = None
    mul_508: "f32[128]" = torch.ops.aten.mul.Tensor(sum_57, 3.0517578125e-05)
    unsqueeze_373: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_508, 0);  mul_508 = None
    unsqueeze_374: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
    unsqueeze_375: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 3);  unsqueeze_374 = None
    mul_509: "f32[128]" = torch.ops.aten.mul.Tensor(sum_58, 3.0517578125e-05)
    mul_510: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_511: "f32[128]" = torch.ops.aten.mul.Tensor(mul_509, mul_510);  mul_509 = mul_510 = None
    unsqueeze_376: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_511, 0);  mul_511 = None
    unsqueeze_377: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 2);  unsqueeze_376 = None
    unsqueeze_378: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 3);  unsqueeze_377 = None
    mul_512: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_21);  primals_21 = None
    unsqueeze_379: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_512, 0);  mul_512 = None
    unsqueeze_380: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
    unsqueeze_381: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 3);  unsqueeze_380 = None
    mul_513: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_378);  sub_138 = unsqueeze_378 = None
    sub_140: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(mul_506, mul_513);  mul_506 = mul_513 = None
    sub_141: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(sub_140, unsqueeze_375);  sub_140 = unsqueeze_375 = None
    mul_514: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_381);  sub_141 = unsqueeze_381 = None
    mul_515: "f32[128]" = torch.ops.aten.mul.Tensor(sum_58, squeeze_31);  sum_58 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_514, mul_80, primals_81, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_514 = mul_80 = primals_81 = None
    getitem_148: "f32[8, 256, 64, 64]" = convolution_backward_26[0]
    getitem_149: "f32[128, 256, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_204: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(getitem_136, getitem_148);  getitem_136 = getitem_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    mul_518: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_204, mul_517);  add_204 = mul_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_59: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_518, [0, 2, 3])
    sub_143: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_384);  convolution_11 = unsqueeze_384 = None
    mul_519: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_518, sub_143)
    sum_60: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_519, [0, 2, 3]);  mul_519 = None
    mul_520: "f32[256]" = torch.ops.aten.mul.Tensor(sum_59, 3.0517578125e-05)
    unsqueeze_385: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_520, 0);  mul_520 = None
    unsqueeze_386: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
    unsqueeze_387: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 3);  unsqueeze_386 = None
    mul_521: "f32[256]" = torch.ops.aten.mul.Tensor(sum_60, 3.0517578125e-05)
    mul_522: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_523: "f32[256]" = torch.ops.aten.mul.Tensor(mul_521, mul_522);  mul_521 = mul_522 = None
    unsqueeze_388: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_523, 0);  mul_523 = None
    unsqueeze_389: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 2);  unsqueeze_388 = None
    unsqueeze_390: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 3);  unsqueeze_389 = None
    mul_524: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_19);  primals_19 = None
    unsqueeze_391: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_524, 0);  mul_524 = None
    unsqueeze_392: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
    unsqueeze_393: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 3);  unsqueeze_392 = None
    mul_525: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_390);  sub_143 = unsqueeze_390 = None
    sub_145: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(mul_518, mul_525);  mul_525 = None
    sub_146: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(sub_145, unsqueeze_387);  sub_145 = unsqueeze_387 = None
    mul_526: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_146, unsqueeze_393);  sub_146 = unsqueeze_393 = None
    mul_527: "f32[256]" = torch.ops.aten.mul.Tensor(sum_60, squeeze_28);  sum_60 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_526, mul_72, primals_80, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_526 = mul_72 = primals_80 = None
    getitem_151: "f32[8, 64, 64, 64]" = convolution_backward_27[0]
    getitem_152: "f32[256, 64, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:91, code: return x * y.expand_as(x)
    mul_528: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_151, mul_71);  mul_71 = None
    mul_529: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_151, expand_1);  getitem_151 = expand_1 = None
    sum_61: "f32[8, 64, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_528, [2, 3], True);  mul_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
    view_181: "f32[8, 1, 64]" = torch.ops.aten.reshape.default(sum_61, [8, 1, 64]);  sum_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_147: "f32[8, 1, 64]" = torch.ops.aten.sub.Tensor(1, sigmoid_9)
    mul_530: "f32[8, 1, 64]" = torch.ops.aten.mul.Tensor(sigmoid_9, sub_147);  sigmoid_9 = sub_147 = None
    mul_531: "f32[8, 1, 64]" = torch.ops.aten.mul.Tensor(view_181, mul_530);  view_181 = mul_530 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:86, code: y = self.conv(y)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_531, view_2, primals_79, [0], [1], [1], [1], False, [0], 1, [True, True, False]);  mul_531 = view_2 = primals_79 = None
    getitem_154: "f32[8, 1, 64]" = convolution_backward_28[0]
    getitem_155: "f32[1, 1, 3]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    view_182: "f32[8, 64]" = torch.ops.aten.reshape.default(getitem_154, [8, 64]);  getitem_154 = None
    unsqueeze_394: "f32[8, 64, 1]" = torch.ops.aten.unsqueeze.default(view_182, 2);  view_182 = None
    unsqueeze_395: "f32[8, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 3);  unsqueeze_394 = None
    expand_27: "f32[8, 64, 64, 64]" = torch.ops.aten.expand.default(unsqueeze_395, [8, 64, 64, 64]);  unsqueeze_395 = None
    div_7: "f32[8, 64, 64, 64]" = torch.ops.aten.div.Scalar(expand_27, 4096);  expand_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    add_206: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_529, div_7);  mul_529 = div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_51: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_45)
    full_default_43: "f32[8, 64, 64, 64]" = torch.ops.aten.full.default([8, 64, 64, 64], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_148: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(full_default_43, sigmoid_51)
    mul_532: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_45, sub_148);  add_45 = sub_148 = None
    add_207: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Scalar(mul_532, 1);  mul_532 = None
    mul_533: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_51, add_207);  sigmoid_51 = add_207 = None
    mul_534: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_206, mul_533);  add_206 = mul_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_62: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_534, [0, 2, 3])
    sub_149: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_398);  convolution_9 = unsqueeze_398 = None
    mul_535: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_534, sub_149)
    sum_63: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_535, [0, 2, 3]);  mul_535 = None
    mul_536: "f32[64]" = torch.ops.aten.mul.Tensor(sum_62, 3.0517578125e-05)
    unsqueeze_399: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_536, 0);  mul_536 = None
    unsqueeze_400: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 2);  unsqueeze_399 = None
    unsqueeze_401: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, 3);  unsqueeze_400 = None
    mul_537: "f32[64]" = torch.ops.aten.mul.Tensor(sum_63, 3.0517578125e-05)
    mul_538: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_539: "f32[64]" = torch.ops.aten.mul.Tensor(mul_537, mul_538);  mul_537 = mul_538 = None
    unsqueeze_402: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_539, 0);  mul_539 = None
    unsqueeze_403: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, 2);  unsqueeze_402 = None
    unsqueeze_404: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 3);  unsqueeze_403 = None
    mul_540: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_17);  primals_17 = None
    unsqueeze_405: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_540, 0);  mul_540 = None
    unsqueeze_406: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 2);  unsqueeze_405 = None
    unsqueeze_407: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 3);  unsqueeze_406 = None
    mul_541: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_404);  sub_149 = unsqueeze_404 = None
    sub_151: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(mul_534, mul_541);  mul_534 = mul_541 = None
    sub_152: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_151, unsqueeze_401);  sub_151 = unsqueeze_401 = None
    mul_542: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_407);  sub_152 = unsqueeze_407 = None
    mul_543: "f32[64]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_25);  sum_63 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_542, mul_63, primals_78, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False]);  mul_542 = mul_63 = primals_78 = None
    getitem_157: "f32[8, 64, 64, 64]" = convolution_backward_29[0]
    getitem_158: "f32[64, 16, 3, 3]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_546: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_157, mul_545);  getitem_157 = mul_545 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_64: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_546, [0, 2, 3])
    sub_154: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_410);  convolution_8 = unsqueeze_410 = None
    mul_547: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_546, sub_154)
    sum_65: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_547, [0, 2, 3]);  mul_547 = None
    mul_548: "f32[64]" = torch.ops.aten.mul.Tensor(sum_64, 3.0517578125e-05)
    unsqueeze_411: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_548, 0);  mul_548 = None
    unsqueeze_412: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 2);  unsqueeze_411 = None
    unsqueeze_413: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, 3);  unsqueeze_412 = None
    mul_549: "f32[64]" = torch.ops.aten.mul.Tensor(sum_65, 3.0517578125e-05)
    mul_550: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_551: "f32[64]" = torch.ops.aten.mul.Tensor(mul_549, mul_550);  mul_549 = mul_550 = None
    unsqueeze_414: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_551, 0);  mul_551 = None
    unsqueeze_415: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, 2);  unsqueeze_414 = None
    unsqueeze_416: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 3);  unsqueeze_415 = None
    mul_552: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_15);  primals_15 = None
    unsqueeze_417: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_552, 0);  mul_552 = None
    unsqueeze_418: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 2);  unsqueeze_417 = None
    unsqueeze_419: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 3);  unsqueeze_418 = None
    mul_553: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_416);  sub_154 = unsqueeze_416 = None
    sub_156: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(mul_546, mul_553);  mul_546 = mul_553 = None
    sub_157: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_156, unsqueeze_413);  sub_156 = unsqueeze_413 = None
    mul_554: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_419);  sub_157 = unsqueeze_419 = None
    mul_555: "f32[64]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_22);  sum_65 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_554, mul_55, primals_77, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_554 = mul_55 = primals_77 = None
    getitem_160: "f32[8, 256, 64, 64]" = convolution_backward_30[0]
    getitem_161: "f32[64, 256, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_209: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_518, getitem_160);  mul_518 = getitem_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    mul_558: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_209, mul_557);  add_209 = mul_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_66: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_558, [0, 2, 3])
    sub_159: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_422);  convolution_7 = unsqueeze_422 = None
    mul_559: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_558, sub_159)
    sum_67: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_559, [0, 2, 3]);  mul_559 = None
    mul_560: "f32[256]" = torch.ops.aten.mul.Tensor(sum_66, 3.0517578125e-05)
    unsqueeze_423: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_560, 0);  mul_560 = None
    unsqueeze_424: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 2);  unsqueeze_423 = None
    unsqueeze_425: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 3);  unsqueeze_424 = None
    mul_561: "f32[256]" = torch.ops.aten.mul.Tensor(sum_67, 3.0517578125e-05)
    mul_562: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_563: "f32[256]" = torch.ops.aten.mul.Tensor(mul_561, mul_562);  mul_561 = mul_562 = None
    unsqueeze_426: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_563, 0);  mul_563 = None
    unsqueeze_427: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 2);  unsqueeze_426 = None
    unsqueeze_428: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 3);  unsqueeze_427 = None
    mul_564: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_13);  primals_13 = None
    unsqueeze_429: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_564, 0);  mul_564 = None
    unsqueeze_430: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 2);  unsqueeze_429 = None
    unsqueeze_431: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 3);  unsqueeze_430 = None
    mul_565: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_428);  sub_159 = unsqueeze_428 = None
    sub_161: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(mul_558, mul_565);  mul_565 = None
    sub_162: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(sub_161, unsqueeze_425);  sub_161 = None
    mul_566: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_431);  sub_162 = unsqueeze_431 = None
    mul_567: "f32[256]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_19);  sum_67 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_566, getitem_6, primals_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_566 = primals_76 = None
    getitem_163: "f32[8, 64, 64, 64]" = convolution_backward_31[0]
    getitem_164: "f32[256, 64, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_163: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_434);  convolution_6 = unsqueeze_434 = None
    mul_568: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_558, sub_163)
    sum_69: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_568, [0, 2, 3]);  mul_568 = None
    mul_570: "f32[256]" = torch.ops.aten.mul.Tensor(sum_69, 3.0517578125e-05)
    mul_571: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_572: "f32[256]" = torch.ops.aten.mul.Tensor(mul_570, mul_571);  mul_570 = mul_571 = None
    unsqueeze_438: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_572, 0);  mul_572 = None
    unsqueeze_439: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 2);  unsqueeze_438 = None
    unsqueeze_440: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 3);  unsqueeze_439 = None
    mul_573: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_11);  primals_11 = None
    unsqueeze_441: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_573, 0);  mul_573 = None
    unsqueeze_442: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 2);  unsqueeze_441 = None
    unsqueeze_443: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 3);  unsqueeze_442 = None
    mul_574: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_163, unsqueeze_440);  sub_163 = unsqueeze_440 = None
    sub_165: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(mul_558, mul_574);  mul_558 = mul_574 = None
    sub_166: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(sub_165, unsqueeze_425);  sub_165 = unsqueeze_425 = None
    mul_575: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_443);  sub_166 = unsqueeze_443 = None
    mul_576: "f32[256]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_16);  sum_69 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_575, mul_40, primals_75, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_575 = mul_40 = primals_75 = None
    getitem_166: "f32[8, 64, 64, 64]" = convolution_backward_32[0]
    getitem_167: "f32[256, 64, 1, 1]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:91, code: return x * y.expand_as(x)
    mul_577: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_166, mul_39);  mul_39 = None
    mul_578: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_166, expand);  getitem_166 = expand = None
    sum_70: "f32[8, 64, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_577, [2, 3], True);  mul_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
    view_183: "f32[8, 1, 64]" = torch.ops.aten.reshape.default(sum_70, [8, 1, 64]);  sum_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_167: "f32[8, 1, 64]" = torch.ops.aten.sub.Tensor(1, sigmoid_5)
    mul_579: "f32[8, 1, 64]" = torch.ops.aten.mul.Tensor(sigmoid_5, sub_167);  sigmoid_5 = sub_167 = None
    mul_580: "f32[8, 1, 64]" = torch.ops.aten.mul.Tensor(view_183, mul_579);  view_183 = mul_579 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:86, code: y = self.conv(y)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_580, view, primals_74, [0], [1], [1], [1], False, [0], 1, [True, True, False]);  mul_580 = view = primals_74 = None
    getitem_169: "f32[8, 1, 64]" = convolution_backward_33[0]
    getitem_170: "f32[1, 1, 3]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    view_184: "f32[8, 64]" = torch.ops.aten.reshape.default(getitem_169, [8, 64]);  getitem_169 = None
    unsqueeze_444: "f32[8, 64, 1]" = torch.ops.aten.unsqueeze.default(view_184, 2);  view_184 = None
    unsqueeze_445: "f32[8, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, 3);  unsqueeze_444 = None
    expand_28: "f32[8, 64, 64, 64]" = torch.ops.aten.expand.default(unsqueeze_445, [8, 64, 64, 64]);  unsqueeze_445 = None
    div_8: "f32[8, 64, 64, 64]" = torch.ops.aten.div.Scalar(expand_28, 4096);  expand_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    add_211: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_578, div_8);  mul_578 = div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_54: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_24)
    sub_168: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(full_default_43, sigmoid_54);  full_default_43 = None
    mul_581: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_24, sub_168);  add_24 = sub_168 = None
    add_212: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Scalar(mul_581, 1);  mul_581 = None
    mul_582: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_54, add_212);  sigmoid_54 = add_212 = None
    mul_583: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_211, mul_582);  add_211 = mul_582 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_71: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_583, [0, 2, 3])
    sub_169: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_448);  convolution_4 = unsqueeze_448 = None
    mul_584: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_583, sub_169)
    sum_72: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_584, [0, 2, 3]);  mul_584 = None
    mul_585: "f32[64]" = torch.ops.aten.mul.Tensor(sum_71, 3.0517578125e-05)
    unsqueeze_449: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_585, 0);  mul_585 = None
    unsqueeze_450: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 2);  unsqueeze_449 = None
    unsqueeze_451: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 3);  unsqueeze_450 = None
    mul_586: "f32[64]" = torch.ops.aten.mul.Tensor(sum_72, 3.0517578125e-05)
    mul_587: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_588: "f32[64]" = torch.ops.aten.mul.Tensor(mul_586, mul_587);  mul_586 = mul_587 = None
    unsqueeze_452: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_588, 0);  mul_588 = None
    unsqueeze_453: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 2);  unsqueeze_452 = None
    unsqueeze_454: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 3);  unsqueeze_453 = None
    mul_589: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_9);  primals_9 = None
    unsqueeze_455: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_589, 0);  mul_589 = None
    unsqueeze_456: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 2);  unsqueeze_455 = None
    unsqueeze_457: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, 3);  unsqueeze_456 = None
    mul_590: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_454);  sub_169 = unsqueeze_454 = None
    sub_171: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(mul_583, mul_590);  mul_583 = mul_590 = None
    sub_172: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_171, unsqueeze_451);  sub_171 = unsqueeze_451 = None
    mul_591: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_457);  sub_172 = unsqueeze_457 = None
    mul_592: "f32[64]" = torch.ops.aten.mul.Tensor(sum_72, squeeze_13);  sum_72 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_591, mul_31, primals_73, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False]);  mul_591 = mul_31 = primals_73 = None
    getitem_172: "f32[8, 64, 64, 64]" = convolution_backward_34[0]
    getitem_173: "f32[64, 16, 3, 3]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_595: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_172, mul_594);  getitem_172 = mul_594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_73: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_595, [0, 2, 3])
    sub_174: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_460);  convolution_3 = unsqueeze_460 = None
    mul_596: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_595, sub_174)
    sum_74: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_596, [0, 2, 3]);  mul_596 = None
    mul_597: "f32[64]" = torch.ops.aten.mul.Tensor(sum_73, 3.0517578125e-05)
    unsqueeze_461: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_597, 0);  mul_597 = None
    unsqueeze_462: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None
    unsqueeze_463: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 3);  unsqueeze_462 = None
    mul_598: "f32[64]" = torch.ops.aten.mul.Tensor(sum_74, 3.0517578125e-05)
    mul_599: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_600: "f32[64]" = torch.ops.aten.mul.Tensor(mul_598, mul_599);  mul_598 = mul_599 = None
    unsqueeze_464: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_600, 0);  mul_600 = None
    unsqueeze_465: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 2);  unsqueeze_464 = None
    unsqueeze_466: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 3);  unsqueeze_465 = None
    mul_601: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_7);  primals_7 = None
    unsqueeze_467: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_601, 0);  mul_601 = None
    unsqueeze_468: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 2);  unsqueeze_467 = None
    unsqueeze_469: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, 3);  unsqueeze_468 = None
    mul_602: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_174, unsqueeze_466);  sub_174 = unsqueeze_466 = None
    sub_176: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(mul_595, mul_602);  mul_595 = mul_602 = None
    sub_177: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_176, unsqueeze_463);  sub_176 = unsqueeze_463 = None
    mul_603: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_469);  sub_177 = unsqueeze_469 = None
    mul_604: "f32[64]" = torch.ops.aten.mul.Tensor(sum_74, squeeze_10);  sum_74 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_603, getitem_6, primals_72, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_603 = getitem_6 = primals_72 = None
    getitem_175: "f32[8, 64, 64, 64]" = convolution_backward_35[0]
    getitem_176: "f32[64, 64, 1, 1]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_214: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(getitem_163, getitem_175);  getitem_163 = getitem_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:1245, code: x = self.stem(x)
    max_pool2d_with_indices_backward: "f32[8, 64, 128, 128]" = torch.ops.aten.max_pool2d_with_indices_backward.default(add_214, mul_23, [3, 3], [2, 2], [1, 1], [1, 1], False, getitem_7);  add_214 = mul_23 = getitem_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_607: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(max_pool2d_with_indices_backward, mul_606);  max_pool2d_with_indices_backward = mul_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_75: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_607, [0, 2, 3])
    sub_179: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_472);  convolution_2 = unsqueeze_472 = None
    mul_608: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_607, sub_179)
    sum_76: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_608, [0, 2, 3]);  mul_608 = None
    mul_609: "f32[64]" = torch.ops.aten.mul.Tensor(sum_75, 7.62939453125e-06)
    unsqueeze_473: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_609, 0);  mul_609 = None
    unsqueeze_474: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 2);  unsqueeze_473 = None
    unsqueeze_475: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, 3);  unsqueeze_474 = None
    mul_610: "f32[64]" = torch.ops.aten.mul.Tensor(sum_76, 7.62939453125e-06)
    mul_611: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_612: "f32[64]" = torch.ops.aten.mul.Tensor(mul_610, mul_611);  mul_610 = mul_611 = None
    unsqueeze_476: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_612, 0);  mul_612 = None
    unsqueeze_477: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 2);  unsqueeze_476 = None
    unsqueeze_478: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 3);  unsqueeze_477 = None
    mul_613: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_5);  primals_5 = None
    unsqueeze_479: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_613, 0);  mul_613 = None
    unsqueeze_480: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 2);  unsqueeze_479 = None
    unsqueeze_481: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, 3);  unsqueeze_480 = None
    mul_614: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_478);  sub_179 = unsqueeze_478 = None
    sub_181: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(mul_607, mul_614);  mul_607 = mul_614 = None
    sub_182: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(sub_181, unsqueeze_475);  sub_181 = unsqueeze_475 = None
    mul_615: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_481);  sub_182 = unsqueeze_481 = None
    mul_616: "f32[64]" = torch.ops.aten.mul.Tensor(sum_76, squeeze_7);  sum_76 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_615, mul_15, primals_71, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_615 = mul_15 = primals_71 = None
    getitem_178: "f32[8, 32, 128, 128]" = convolution_backward_36[0]
    getitem_179: "f32[64, 32, 3, 3]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_619: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(getitem_178, mul_618);  getitem_178 = mul_618 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_77: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_619, [0, 2, 3])
    sub_184: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_484);  convolution_1 = unsqueeze_484 = None
    mul_620: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(mul_619, sub_184)
    sum_78: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_620, [0, 2, 3]);  mul_620 = None
    mul_621: "f32[32]" = torch.ops.aten.mul.Tensor(sum_77, 7.62939453125e-06)
    unsqueeze_485: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_621, 0);  mul_621 = None
    unsqueeze_486: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 2);  unsqueeze_485 = None
    unsqueeze_487: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 3);  unsqueeze_486 = None
    mul_622: "f32[32]" = torch.ops.aten.mul.Tensor(sum_78, 7.62939453125e-06)
    mul_623: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_624: "f32[32]" = torch.ops.aten.mul.Tensor(mul_622, mul_623);  mul_622 = mul_623 = None
    unsqueeze_488: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_624, 0);  mul_624 = None
    unsqueeze_489: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 2);  unsqueeze_488 = None
    unsqueeze_490: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 3);  unsqueeze_489 = None
    mul_625: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_3);  primals_3 = None
    unsqueeze_491: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_625, 0);  mul_625 = None
    unsqueeze_492: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 2);  unsqueeze_491 = None
    unsqueeze_493: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, 3);  unsqueeze_492 = None
    mul_626: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_490);  sub_184 = unsqueeze_490 = None
    sub_186: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(mul_619, mul_626);  mul_619 = mul_626 = None
    sub_187: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(sub_186, unsqueeze_487);  sub_186 = unsqueeze_487 = None
    mul_627: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_493);  sub_187 = unsqueeze_493 = None
    mul_628: "f32[32]" = torch.ops.aten.mul.Tensor(sum_78, squeeze_4);  sum_78 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_627, mul_7, primals_70, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_627 = mul_7 = primals_70 = None
    getitem_181: "f32[8, 24, 128, 128]" = convolution_backward_37[0]
    getitem_182: "f32[32, 24, 3, 3]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_631: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(getitem_181, mul_630);  getitem_181 = mul_630 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_79: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_631, [0, 2, 3])
    sub_189: "f32[8, 24, 128, 128]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_496);  convolution = unsqueeze_496 = None
    mul_632: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(mul_631, sub_189)
    sum_80: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_632, [0, 2, 3]);  mul_632 = None
    mul_633: "f32[24]" = torch.ops.aten.mul.Tensor(sum_79, 7.62939453125e-06)
    unsqueeze_497: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_633, 0);  mul_633 = None
    unsqueeze_498: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 2);  unsqueeze_497 = None
    unsqueeze_499: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 3);  unsqueeze_498 = None
    mul_634: "f32[24]" = torch.ops.aten.mul.Tensor(sum_80, 7.62939453125e-06)
    mul_635: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_636: "f32[24]" = torch.ops.aten.mul.Tensor(mul_634, mul_635);  mul_634 = mul_635 = None
    unsqueeze_500: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_636, 0);  mul_636 = None
    unsqueeze_501: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 2);  unsqueeze_500 = None
    unsqueeze_502: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 3);  unsqueeze_501 = None
    mul_637: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_1);  primals_1 = None
    unsqueeze_503: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_637, 0);  mul_637 = None
    unsqueeze_504: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 2);  unsqueeze_503 = None
    unsqueeze_505: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, 3);  unsqueeze_504 = None
    mul_638: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_502);  sub_189 = unsqueeze_502 = None
    sub_191: "f32[8, 24, 128, 128]" = torch.ops.aten.sub.Tensor(mul_631, mul_638);  mul_631 = mul_638 = None
    sub_192: "f32[8, 24, 128, 128]" = torch.ops.aten.sub.Tensor(sub_191, unsqueeze_499);  sub_191 = unsqueeze_499 = None
    mul_639: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_505);  sub_192 = unsqueeze_505 = None
    mul_640: "f32[24]" = torch.ops.aten.mul.Tensor(sum_80, squeeze_1);  sum_80 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_639, primals_203, primals_69, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_639 = primals_203 = primals_69 = None
    getitem_185: "f32[24, 3, 3, 3]" = convolution_backward_38[1];  convolution_backward_38 = None
    return [mul_640, sum_79, mul_628, sum_77, mul_616, sum_75, mul_604, sum_73, mul_592, sum_71, mul_576, sum_66, mul_567, sum_66, mul_555, sum_64, mul_543, sum_62, mul_527, sum_59, mul_515, sum_57, mul_503, sum_55, mul_487, sum_50, mul_478, sum_50, mul_466, sum_48, mul_454, sum_46, mul_438, sum_43, mul_426, sum_41, mul_414, sum_39, mul_398, sum_34, mul_389, sum_34, mul_377, sum_32, permute_106, permute_100, mul_362, sum_27, mul_350, sum_25, mul_338, sum_23, permute_80, permute_74, mul_323, sum_18, mul_311, sum_14, mul_302, sum_14, mul_290, sum_12, permute_54, permute_48, mul_275, sum_7, mul_263, sum_5, getitem_185, getitem_182, getitem_179, getitem_176, getitem_173, getitem_170, getitem_167, getitem_164, getitem_161, getitem_158, getitem_155, getitem_152, getitem_149, getitem_146, getitem_143, getitem_140, getitem_137, getitem_134, getitem_131, getitem_128, getitem_125, getitem_122, getitem_119, getitem_116, getitem_113, getitem_110, getitem_107, getitem_104, getitem_101, getitem_98, getitem_95, getitem_92, getitem_89, getitem_86, getitem_83, getitem_80, getitem_77, getitem_74, getitem_71, permute_37, view_86, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    