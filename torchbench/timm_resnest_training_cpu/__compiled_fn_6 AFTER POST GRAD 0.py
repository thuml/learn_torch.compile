from __future__ import annotations



def forward(self, primals_1: "f32[32, 3, 3, 3]", primals_2: "f32[32]", primals_4: "f32[32, 32, 3, 3]", primals_5: "f32[32]", primals_7: "f32[64, 32, 3, 3]", primals_8: "f32[64]", primals_10: "f32[64, 64, 1, 1]", primals_11: "f32[64]", primals_13: "f32[128, 32, 3, 3]", primals_14: "f32[128]", primals_16: "f32[32, 64, 1, 1]", primals_18: "f32[32]", primals_20: "f32[128, 32, 1, 1]", primals_22: "f32[256, 64, 1, 1]", primals_23: "f32[256]", primals_25: "f32[256, 64, 1, 1]", primals_26: "f32[256]", primals_28: "f32[128, 256, 1, 1]", primals_29: "f32[128]", primals_31: "f32[256, 64, 3, 3]", primals_32: "f32[256]", primals_34: "f32[64, 128, 1, 1]", primals_36: "f32[64]", primals_38: "f32[256, 64, 1, 1]", primals_40: "f32[512, 128, 1, 1]", primals_41: "f32[512]", primals_43: "f32[512, 256, 1, 1]", primals_44: "f32[512]", primals_46: "f32[256, 512, 1, 1]", primals_47: "f32[256]", primals_49: "f32[512, 128, 3, 3]", primals_50: "f32[512]", primals_52: "f32[128, 256, 1, 1]", primals_54: "f32[128]", primals_56: "f32[512, 128, 1, 1]", primals_58: "f32[1024, 256, 1, 1]", primals_59: "f32[1024]", primals_61: "f32[1024, 512, 1, 1]", primals_62: "f32[1024]", primals_64: "f32[512, 1024, 1, 1]", primals_65: "f32[512]", primals_67: "f32[1024, 256, 3, 3]", primals_68: "f32[1024]", primals_70: "f32[256, 512, 1, 1]", primals_72: "f32[256]", primals_74: "f32[1024, 256, 1, 1]", primals_76: "f32[2048, 512, 1, 1]", primals_77: "f32[2048]", primals_79: "f32[2048, 1024, 1, 1]", primals_80: "f32[2048]", primals_84: "f32[32]", primals_85: "f32[32]", primals_87: "f32[32]", primals_88: "f32[32]", primals_90: "f32[64]", primals_91: "f32[64]", primals_93: "f32[64]", primals_94: "f32[64]", primals_96: "f32[128]", primals_97: "f32[128]", primals_99: "f32[32]", primals_100: "f32[32]", primals_102: "f32[256]", primals_103: "f32[256]", primals_105: "f32[256]", primals_106: "f32[256]", primals_108: "f32[128]", primals_109: "f32[128]", primals_111: "f32[256]", primals_112: "f32[256]", primals_114: "f32[64]", primals_115: "f32[64]", primals_117: "f32[512]", primals_118: "f32[512]", primals_120: "f32[512]", primals_121: "f32[512]", primals_123: "f32[256]", primals_124: "f32[256]", primals_126: "f32[512]", primals_127: "f32[512]", primals_129: "f32[128]", primals_130: "f32[128]", primals_132: "f32[1024]", primals_133: "f32[1024]", primals_135: "f32[1024]", primals_136: "f32[1024]", primals_138: "f32[512]", primals_139: "f32[512]", primals_141: "f32[1024]", primals_142: "f32[1024]", primals_144: "f32[256]", primals_145: "f32[256]", primals_147: "f32[2048]", primals_148: "f32[2048]", primals_150: "f32[2048]", primals_151: "f32[2048]", primals_153: "f32[4, 3, 224, 224]", convolution: "f32[4, 32, 112, 112]", relu: "f32[4, 32, 112, 112]", convolution_1: "f32[4, 32, 112, 112]", relu_1: "f32[4, 32, 112, 112]", convolution_2: "f32[4, 64, 112, 112]", relu_2: "f32[4, 64, 112, 112]", getitem: "f32[4, 64, 56, 56]", getitem_1: "i64[4, 64, 56, 56]", convolution_3: "f32[4, 64, 56, 56]", relu_3: "f32[4, 64, 56, 56]", convolution_4: "f32[4, 128, 56, 56]", relu_4: "f32[4, 128, 56, 56]", mean: "f32[4, 64, 1, 1]", convolution_5: "f32[4, 32, 1, 1]", relu_5: "f32[4, 32, 1, 1]", div: "f32[4, 2, 1, 64]", sum_3: "f32[4, 64, 56, 56]", convolution_7: "f32[4, 256, 56, 56]", convolution_8: "f32[4, 256, 56, 56]", relu_6: "f32[4, 256, 56, 56]", convolution_9: "f32[4, 128, 56, 56]", relu_7: "f32[4, 128, 56, 56]", convolution_10: "f32[4, 256, 56, 56]", relu_8: "f32[4, 256, 56, 56]", mean_1: "f32[4, 128, 1, 1]", convolution_11: "f32[4, 64, 1, 1]", relu_9: "f32[4, 64, 1, 1]", div_1: "f32[4, 2, 1, 128]", sum_6: "f32[4, 128, 56, 56]", avg_pool2d: "f32[4, 128, 28, 28]", convolution_13: "f32[4, 512, 28, 28]", avg_pool2d_1: "f32[4, 256, 28, 28]", convolution_14: "f32[4, 512, 28, 28]", relu_10: "f32[4, 512, 28, 28]", convolution_15: "f32[4, 256, 28, 28]", relu_11: "f32[4, 256, 28, 28]", convolution_16: "f32[4, 512, 28, 28]", relu_12: "f32[4, 512, 28, 28]", mean_2: "f32[4, 256, 1, 1]", convolution_17: "f32[4, 128, 1, 1]", relu_13: "f32[4, 128, 1, 1]", div_2: "f32[4, 2, 1, 256]", sum_9: "f32[4, 256, 28, 28]", avg_pool2d_2: "f32[4, 256, 14, 14]", convolution_19: "f32[4, 1024, 14, 14]", avg_pool2d_3: "f32[4, 512, 14, 14]", convolution_20: "f32[4, 1024, 14, 14]", relu_14: "f32[4, 1024, 14, 14]", convolution_21: "f32[4, 512, 14, 14]", relu_15: "f32[4, 512, 14, 14]", convolution_22: "f32[4, 1024, 14, 14]", relu_16: "f32[4, 1024, 14, 14]", mean_3: "f32[4, 512, 1, 1]", convolution_23: "f32[4, 256, 1, 1]", relu_17: "f32[4, 256, 1, 1]", div_3: "f32[4, 2, 1, 512]", sum_12: "f32[4, 512, 14, 14]", avg_pool2d_4: "f32[4, 512, 7, 7]", convolution_25: "f32[4, 2048, 7, 7]", avg_pool2d_5: "f32[4, 1024, 7, 7]", convolution_26: "f32[4, 2048, 7, 7]", view_24: "f32[4, 2048]", permute_5: "f32[1000, 2048]", le: "b8[4, 2048, 7, 7]", tangents_1: "f32[4, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_1: "f32[4, 2, 64, 56, 56]" = torch.ops.aten.reshape.default(relu_4, [4, 2, 64, 56, 56])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_3: "f32[4, 128]" = torch.ops.aten.reshape.default(div, [4, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_4: "f32[4, 128, 1, 1]" = torch.ops.aten.reshape.default(view_3, [4, -1, 1, 1]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_5: "f32[4, 2, 64, 1, 1]" = torch.ops.aten.reshape.default(view_4, [4, 2, 64, 1, 1]);  view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_7: "f32[4, 2, 128, 56, 56]" = torch.ops.aten.reshape.default(relu_8, [4, 2, 128, 56, 56])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_9: "f32[4, 256]" = torch.ops.aten.reshape.default(div_1, [4, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_10: "f32[4, 256, 1, 1]" = torch.ops.aten.reshape.default(view_9, [4, -1, 1, 1]);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_11: "f32[4, 2, 128, 1, 1]" = torch.ops.aten.reshape.default(view_10, [4, 2, 128, 1, 1]);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_13: "f32[4, 2, 256, 28, 28]" = torch.ops.aten.reshape.default(relu_12, [4, 2, 256, 28, 28])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_15: "f32[4, 512]" = torch.ops.aten.reshape.default(div_2, [4, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_16: "f32[4, 512, 1, 1]" = torch.ops.aten.reshape.default(view_15, [4, -1, 1, 1]);  view_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_17: "f32[4, 2, 256, 1, 1]" = torch.ops.aten.reshape.default(view_16, [4, 2, 256, 1, 1]);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_19: "f32[4, 2, 512, 14, 14]" = torch.ops.aten.reshape.default(relu_16, [4, 2, 512, 14, 14])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_21: "f32[4, 1024]" = torch.ops.aten.reshape.default(div_3, [4, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_22: "f32[4, 1024, 1, 1]" = torch.ops.aten.reshape.default(view_21, [4, -1, 1, 1]);  view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_23: "f32[4, 2, 512, 1, 1]" = torch.ops.aten.reshape.default(view_22, [4, 2, 512, 1, 1]);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:538, code: return x if pre_logits else self.fc(x)
    mm: "f32[4, 2048]" = torch.ops.aten.mm.default(tangents_1, permute_5);  permute_5 = None
    permute_6: "f32[1000, 4]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 2048]" = torch.ops.aten.mm.default(permute_6, view_24);  permute_6 = view_24 = None
    permute_7: "f32[2048, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_13: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_25: "f32[1000]" = torch.ops.aten.reshape.default(sum_13, [1000]);  sum_13 = None
    permute_8: "f32[1000, 2048]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_26: "f32[4, 2048, 1, 1]" = torch.ops.aten.reshape.default(mm, [4, 2048, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[4, 2048, 7, 7]" = torch.ops.aten.expand.default(view_26, [4, 2048, 7, 7]);  view_26 = None
    div_4: "f32[4, 2048, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where: "f32[4, 2048, 7, 7]" = torch.ops.aten.where.self(le, full_default, div_4);  le = div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:113, code: shortcut = self.downsample(x)
    add_50: "f32[2048]" = torch.ops.aten.add.Tensor(primals_151, 1e-05);  primals_151 = None
    rsqrt: "f32[2048]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    unsqueeze_184: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(primals_150, 0);  primals_150 = None
    unsqueeze_185: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, 2);  unsqueeze_184 = None
    unsqueeze_186: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_185, 3);  unsqueeze_185 = None
    sum_14: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_27: "f32[4, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_186);  convolution_26 = unsqueeze_186 = None
    mul_73: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_27);  sub_27 = None
    sum_15: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_73, [0, 2, 3]);  mul_73 = None
    mul_78: "f32[2048]" = torch.ops.aten.mul.Tensor(rsqrt, primals_80);  primals_80 = None
    unsqueeze_193: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_78, 0);  mul_78 = None
    unsqueeze_194: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_193, 2);  unsqueeze_193 = None
    unsqueeze_195: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, 3);  unsqueeze_194 = None
    mul_79: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where, unsqueeze_195);  unsqueeze_195 = None
    mul_80: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_15, rsqrt);  sum_15 = rsqrt = None
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_79, avg_pool2d_5, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_79 = avg_pool2d_5 = primals_79 = None
    getitem_2: "f32[4, 1024, 7, 7]" = convolution_backward[0]
    getitem_3: "f32[2048, 1024, 1, 1]" = convolution_backward[1];  convolution_backward = None
    avg_pool2d_backward: "f32[4, 1024, 14, 14]" = torch.ops.aten.avg_pool2d_backward.default(getitem_2, relu_14, [2, 2], [2, 2], [0, 0], True, False, None);  getitem_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_51: "f32[2048]" = torch.ops.aten.add.Tensor(primals_148, 1e-05);  primals_148 = None
    rsqrt_1: "f32[2048]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
    unsqueeze_196: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(primals_147, 0);  primals_147 = None
    unsqueeze_197: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, 2);  unsqueeze_196 = None
    unsqueeze_198: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 3);  unsqueeze_197 = None
    sub_28: "f32[4, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_198);  convolution_25 = unsqueeze_198 = None
    mul_81: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_28);  sub_28 = None
    sum_17: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_81, [0, 2, 3]);  mul_81 = None
    mul_86: "f32[2048]" = torch.ops.aten.mul.Tensor(rsqrt_1, primals_77);  primals_77 = None
    unsqueeze_205: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_86, 0);  mul_86 = None
    unsqueeze_206: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 2);  unsqueeze_205 = None
    unsqueeze_207: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, 3);  unsqueeze_206 = None
    mul_87: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where, unsqueeze_207);  where = unsqueeze_207 = None
    mul_88: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_17, rsqrt_1);  sum_17 = rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_87, avg_pool2d_4, primals_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_87 = avg_pool2d_4 = primals_76 = None
    getitem_5: "f32[4, 512, 7, 7]" = convolution_backward_1[0]
    getitem_6: "f32[2048, 512, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:107, code: out = self.avd_last(out)
    avg_pool2d_backward_1: "f32[4, 512, 14, 14]" = torch.ops.aten.avg_pool2d_backward.default(getitem_5, sum_12, [3, 3], [2, 2], [1, 1], False, True, None);  getitem_5 = sum_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_208: "f32[4, 1, 512, 14, 14]" = torch.ops.aten.unsqueeze.default(avg_pool2d_backward_1, 1);  avg_pool2d_backward_1 = None
    expand_1: "f32[4, 2, 512, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_208, [4, 2, 512, 14, 14]);  unsqueeze_208 = None
    mul_89: "f32[4, 2, 512, 14, 14]" = torch.ops.aten.mul.Tensor(expand_1, view_19);  view_19 = None
    mul_90: "f32[4, 2, 512, 14, 14]" = torch.ops.aten.mul.Tensor(expand_1, view_23);  expand_1 = view_23 = None
    sum_18: "f32[4, 2, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_89, [3, 4], True);  mul_89 = None
    view_27: "f32[4, 1024, 1, 1]" = torch.ops.aten.reshape.default(sum_18, [4, 1024, 1, 1]);  sum_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_28: "f32[4, 1024]" = torch.ops.aten.reshape.default(view_27, [4, 1024]);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_29: "f32[4, 2, 1, 512]" = torch.ops.aten.reshape.default(view_28, [4, 2, 1, 512]);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    mul_91: "f32[4, 2, 1, 512]" = torch.ops.aten.mul.Tensor(view_29, div_3);  view_29 = None
    sum_19: "f32[4, 1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_91, [1], True)
    mul_92: "f32[4, 2, 1, 512]" = torch.ops.aten.mul.Tensor(div_3, sum_19);  div_3 = sum_19 = None
    sub_29: "f32[4, 2, 1, 512]" = torch.ops.aten.sub.Tensor(mul_91, mul_92);  mul_91 = mul_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_9: "f32[4, 1, 2, 512]" = torch.ops.aten.permute.default(sub_29, [0, 2, 1, 3]);  sub_29 = None
    view_30: "f32[4, 1024, 1, 1]" = torch.ops.aten.reshape.default(permute_9, [4, 1024, 1, 1]);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(view_30, relu_17, primals_74, [1024], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_30 = primals_74 = None
    getitem_8: "f32[4, 256, 1, 1]" = convolution_backward_2[0]
    getitem_9: "f32[1024, 256, 1, 1]" = convolution_backward_2[1]
    getitem_10: "f32[1024]" = convolution_backward_2[2];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    le_1: "b8[4, 256, 1, 1]" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
    where_1: "f32[4, 256, 1, 1]" = torch.ops.aten.where.self(le_1, full_default, getitem_8);  le_1 = getitem_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_52: "f32[256]" = torch.ops.aten.add.Tensor(primals_145, 1e-05);  primals_145 = None
    rsqrt_2: "f32[256]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    unsqueeze_209: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_144, 0);  primals_144 = None
    unsqueeze_210: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 2);  unsqueeze_209 = None
    unsqueeze_211: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, 3);  unsqueeze_210 = None
    sum_20: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_30: "f32[4, 256, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_211);  convolution_23 = unsqueeze_211 = None
    mul_93: "f32[4, 256, 1, 1]" = torch.ops.aten.mul.Tensor(where_1, sub_30);  sub_30 = None
    sum_21: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_93, [0, 2, 3]);  mul_93 = None
    mul_98: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_2, primals_72);  primals_72 = None
    unsqueeze_218: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_98, 0);  mul_98 = None
    unsqueeze_219: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 2);  unsqueeze_218 = None
    unsqueeze_220: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_219, 3);  unsqueeze_219 = None
    mul_99: "f32[4, 256, 1, 1]" = torch.ops.aten.mul.Tensor(where_1, unsqueeze_220);  where_1 = unsqueeze_220 = None
    mul_100: "f32[256]" = torch.ops.aten.mul.Tensor(sum_21, rsqrt_2);  sum_21 = rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_99, mean_3, primals_70, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_99 = mean_3 = primals_70 = None
    getitem_11: "f32[4, 512, 1, 1]" = convolution_backward_3[0]
    getitem_12: "f32[256, 512, 1, 1]" = convolution_backward_3[1]
    getitem_13: "f32[256]" = convolution_backward_3[2];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_2: "f32[4, 512, 14, 14]" = torch.ops.aten.expand.default(getitem_11, [4, 512, 14, 14]);  getitem_11 = None
    div_5: "f32[4, 512, 14, 14]" = torch.ops.aten.div.Scalar(expand_2, 196);  expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_221: "f32[4, 1, 512, 14, 14]" = torch.ops.aten.unsqueeze.default(div_5, 1);  div_5 = None
    expand_3: "f32[4, 2, 512, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_221, [4, 2, 512, 14, 14]);  unsqueeze_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_53: "f32[4, 2, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_90, expand_3);  mul_90 = expand_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_31: "f32[4, 1024, 14, 14]" = torch.ops.aten.reshape.default(add_53, [4, 1024, 14, 14]);  add_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    le_2: "b8[4, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
    where_2: "f32[4, 1024, 14, 14]" = torch.ops.aten.where.self(le_2, full_default, view_31);  le_2 = view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_54: "f32[1024]" = torch.ops.aten.add.Tensor(primals_142, 1e-05);  primals_142 = None
    rsqrt_3: "f32[1024]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
    unsqueeze_222: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(primals_141, 0);  primals_141 = None
    unsqueeze_223: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, 2);  unsqueeze_222 = None
    unsqueeze_224: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 3);  unsqueeze_223 = None
    sum_22: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_31: "f32[4, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_224);  convolution_22 = unsqueeze_224 = None
    mul_101: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_2, sub_31);  sub_31 = None
    sum_23: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_101, [0, 2, 3]);  mul_101 = None
    mul_106: "f32[1024]" = torch.ops.aten.mul.Tensor(rsqrt_3, primals_68);  primals_68 = None
    unsqueeze_231: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_106, 0);  mul_106 = None
    unsqueeze_232: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_231, 2);  unsqueeze_231 = None
    unsqueeze_233: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 3);  unsqueeze_232 = None
    mul_107: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_2, unsqueeze_233);  where_2 = unsqueeze_233 = None
    mul_108: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_23, rsqrt_3);  sum_23 = rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_107, relu_15, primals_67, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_107 = primals_67 = None
    getitem_14: "f32[4, 512, 14, 14]" = convolution_backward_4[0]
    getitem_15: "f32[1024, 256, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    le_3: "b8[4, 512, 14, 14]" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
    where_3: "f32[4, 512, 14, 14]" = torch.ops.aten.where.self(le_3, full_default, getitem_14);  le_3 = getitem_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_55: "f32[512]" = torch.ops.aten.add.Tensor(primals_139, 1e-05);  primals_139 = None
    rsqrt_4: "f32[512]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
    unsqueeze_234: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_138, 0);  primals_138 = None
    unsqueeze_235: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, 2);  unsqueeze_234 = None
    unsqueeze_236: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 3);  unsqueeze_235 = None
    sum_24: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_32: "f32[4, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_236);  convolution_21 = unsqueeze_236 = None
    mul_109: "f32[4, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_3, sub_32);  sub_32 = None
    sum_25: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_109, [0, 2, 3]);  mul_109 = None
    mul_114: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_4, primals_65);  primals_65 = None
    unsqueeze_243: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_114, 0);  mul_114 = None
    unsqueeze_244: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_243, 2);  unsqueeze_243 = None
    unsqueeze_245: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 3);  unsqueeze_244 = None
    mul_115: "f32[4, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_3, unsqueeze_245);  where_3 = unsqueeze_245 = None
    mul_116: "f32[512]" = torch.ops.aten.mul.Tensor(sum_25, rsqrt_4);  sum_25 = rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_115, relu_14, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_115 = primals_64 = None
    getitem_17: "f32[4, 1024, 14, 14]" = convolution_backward_5[0]
    getitem_18: "f32[512, 1024, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_56: "f32[4, 1024, 14, 14]" = torch.ops.aten.add.Tensor(avg_pool2d_backward, getitem_17);  avg_pool2d_backward = getitem_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    le_4: "b8[4, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
    where_4: "f32[4, 1024, 14, 14]" = torch.ops.aten.where.self(le_4, full_default, add_56);  le_4 = add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:113, code: shortcut = self.downsample(x)
    add_57: "f32[1024]" = torch.ops.aten.add.Tensor(primals_136, 1e-05);  primals_136 = None
    rsqrt_5: "f32[1024]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
    unsqueeze_246: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(primals_135, 0);  primals_135 = None
    unsqueeze_247: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, 2);  unsqueeze_246 = None
    unsqueeze_248: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 3);  unsqueeze_247 = None
    sum_26: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_33: "f32[4, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_248);  convolution_20 = unsqueeze_248 = None
    mul_117: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_4, sub_33);  sub_33 = None
    sum_27: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_117, [0, 2, 3]);  mul_117 = None
    mul_122: "f32[1024]" = torch.ops.aten.mul.Tensor(rsqrt_5, primals_62);  primals_62 = None
    unsqueeze_255: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_122, 0);  mul_122 = None
    unsqueeze_256: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 2);  unsqueeze_255 = None
    unsqueeze_257: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, 3);  unsqueeze_256 = None
    mul_123: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_4, unsqueeze_257);  unsqueeze_257 = None
    mul_124: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_27, rsqrt_5);  sum_27 = rsqrt_5 = None
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_123, avg_pool2d_3, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_123 = avg_pool2d_3 = primals_61 = None
    getitem_20: "f32[4, 512, 14, 14]" = convolution_backward_6[0]
    getitem_21: "f32[1024, 512, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    avg_pool2d_backward_2: "f32[4, 512, 28, 28]" = torch.ops.aten.avg_pool2d_backward.default(getitem_20, relu_10, [2, 2], [2, 2], [0, 0], True, False, None);  getitem_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_58: "f32[1024]" = torch.ops.aten.add.Tensor(primals_133, 1e-05);  primals_133 = None
    rsqrt_6: "f32[1024]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    unsqueeze_258: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(primals_132, 0);  primals_132 = None
    unsqueeze_259: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, 2);  unsqueeze_258 = None
    unsqueeze_260: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 3);  unsqueeze_259 = None
    sub_34: "f32[4, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_260);  convolution_19 = unsqueeze_260 = None
    mul_125: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_4, sub_34);  sub_34 = None
    sum_29: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_125, [0, 2, 3]);  mul_125 = None
    mul_130: "f32[1024]" = torch.ops.aten.mul.Tensor(rsqrt_6, primals_59);  primals_59 = None
    unsqueeze_267: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_130, 0);  mul_130 = None
    unsqueeze_268: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 2);  unsqueeze_267 = None
    unsqueeze_269: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, 3);  unsqueeze_268 = None
    mul_131: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_4, unsqueeze_269);  where_4 = unsqueeze_269 = None
    mul_132: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_29, rsqrt_6);  sum_29 = rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_131, avg_pool2d_2, primals_58, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_131 = avg_pool2d_2 = primals_58 = None
    getitem_23: "f32[4, 256, 14, 14]" = convolution_backward_7[0]
    getitem_24: "f32[1024, 256, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:107, code: out = self.avd_last(out)
    avg_pool2d_backward_3: "f32[4, 256, 28, 28]" = torch.ops.aten.avg_pool2d_backward.default(getitem_23, sum_9, [3, 3], [2, 2], [1, 1], False, True, None);  getitem_23 = sum_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_270: "f32[4, 1, 256, 28, 28]" = torch.ops.aten.unsqueeze.default(avg_pool2d_backward_3, 1);  avg_pool2d_backward_3 = None
    expand_4: "f32[4, 2, 256, 28, 28]" = torch.ops.aten.expand.default(unsqueeze_270, [4, 2, 256, 28, 28]);  unsqueeze_270 = None
    mul_133: "f32[4, 2, 256, 28, 28]" = torch.ops.aten.mul.Tensor(expand_4, view_13);  view_13 = None
    mul_134: "f32[4, 2, 256, 28, 28]" = torch.ops.aten.mul.Tensor(expand_4, view_17);  expand_4 = view_17 = None
    sum_30: "f32[4, 2, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_133, [3, 4], True);  mul_133 = None
    view_32: "f32[4, 512, 1, 1]" = torch.ops.aten.reshape.default(sum_30, [4, 512, 1, 1]);  sum_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_33: "f32[4, 512]" = torch.ops.aten.reshape.default(view_32, [4, 512]);  view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_34: "f32[4, 2, 1, 256]" = torch.ops.aten.reshape.default(view_33, [4, 2, 1, 256]);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    mul_135: "f32[4, 2, 1, 256]" = torch.ops.aten.mul.Tensor(view_34, div_2);  view_34 = None
    sum_31: "f32[4, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(mul_135, [1], True)
    mul_136: "f32[4, 2, 1, 256]" = torch.ops.aten.mul.Tensor(div_2, sum_31);  div_2 = sum_31 = None
    sub_35: "f32[4, 2, 1, 256]" = torch.ops.aten.sub.Tensor(mul_135, mul_136);  mul_135 = mul_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_10: "f32[4, 1, 2, 256]" = torch.ops.aten.permute.default(sub_35, [0, 2, 1, 3]);  sub_35 = None
    view_35: "f32[4, 512, 1, 1]" = torch.ops.aten.reshape.default(permute_10, [4, 512, 1, 1]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(view_35, relu_13, primals_56, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_35 = primals_56 = None
    getitem_26: "f32[4, 128, 1, 1]" = convolution_backward_8[0]
    getitem_27: "f32[512, 128, 1, 1]" = convolution_backward_8[1]
    getitem_28: "f32[512]" = convolution_backward_8[2];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    le_5: "b8[4, 128, 1, 1]" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
    where_5: "f32[4, 128, 1, 1]" = torch.ops.aten.where.self(le_5, full_default, getitem_26);  le_5 = getitem_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_59: "f32[128]" = torch.ops.aten.add.Tensor(primals_130, 1e-05);  primals_130 = None
    rsqrt_7: "f32[128]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
    unsqueeze_271: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_129, 0);  primals_129 = None
    unsqueeze_272: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 2);  unsqueeze_271 = None
    unsqueeze_273: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 3);  unsqueeze_272 = None
    sum_32: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_36: "f32[4, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_273);  convolution_17 = unsqueeze_273 = None
    mul_137: "f32[4, 128, 1, 1]" = torch.ops.aten.mul.Tensor(where_5, sub_36);  sub_36 = None
    sum_33: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_137, [0, 2, 3]);  mul_137 = None
    mul_142: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_7, primals_54);  primals_54 = None
    unsqueeze_280: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_142, 0);  mul_142 = None
    unsqueeze_281: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 2);  unsqueeze_280 = None
    unsqueeze_282: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 3);  unsqueeze_281 = None
    mul_143: "f32[4, 128, 1, 1]" = torch.ops.aten.mul.Tensor(where_5, unsqueeze_282);  where_5 = unsqueeze_282 = None
    mul_144: "f32[128]" = torch.ops.aten.mul.Tensor(sum_33, rsqrt_7);  sum_33 = rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_143, mean_2, primals_52, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_143 = mean_2 = primals_52 = None
    getitem_29: "f32[4, 256, 1, 1]" = convolution_backward_9[0]
    getitem_30: "f32[128, 256, 1, 1]" = convolution_backward_9[1]
    getitem_31: "f32[128]" = convolution_backward_9[2];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_5: "f32[4, 256, 28, 28]" = torch.ops.aten.expand.default(getitem_29, [4, 256, 28, 28]);  getitem_29 = None
    div_6: "f32[4, 256, 28, 28]" = torch.ops.aten.div.Scalar(expand_5, 784);  expand_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_283: "f32[4, 1, 256, 28, 28]" = torch.ops.aten.unsqueeze.default(div_6, 1);  div_6 = None
    expand_6: "f32[4, 2, 256, 28, 28]" = torch.ops.aten.expand.default(unsqueeze_283, [4, 2, 256, 28, 28]);  unsqueeze_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_60: "f32[4, 2, 256, 28, 28]" = torch.ops.aten.add.Tensor(mul_134, expand_6);  mul_134 = expand_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_36: "f32[4, 512, 28, 28]" = torch.ops.aten.reshape.default(add_60, [4, 512, 28, 28]);  add_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    le_6: "b8[4, 512, 28, 28]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
    where_6: "f32[4, 512, 28, 28]" = torch.ops.aten.where.self(le_6, full_default, view_36);  le_6 = view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_61: "f32[512]" = torch.ops.aten.add.Tensor(primals_127, 1e-05);  primals_127 = None
    rsqrt_8: "f32[512]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
    unsqueeze_284: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_126, 0);  primals_126 = None
    unsqueeze_285: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 2);  unsqueeze_284 = None
    unsqueeze_286: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 3);  unsqueeze_285 = None
    sum_34: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_37: "f32[4, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_286);  convolution_16 = unsqueeze_286 = None
    mul_145: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_6, sub_37);  sub_37 = None
    sum_35: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_145, [0, 2, 3]);  mul_145 = None
    mul_150: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_8, primals_50);  primals_50 = None
    unsqueeze_293: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_150, 0);  mul_150 = None
    unsqueeze_294: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 2);  unsqueeze_293 = None
    unsqueeze_295: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, 3);  unsqueeze_294 = None
    mul_151: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_6, unsqueeze_295);  where_6 = unsqueeze_295 = None
    mul_152: "f32[512]" = torch.ops.aten.mul.Tensor(sum_35, rsqrt_8);  sum_35 = rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_151, relu_11, primals_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_151 = primals_49 = None
    getitem_32: "f32[4, 256, 28, 28]" = convolution_backward_10[0]
    getitem_33: "f32[512, 128, 3, 3]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    le_7: "b8[4, 256, 28, 28]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
    where_7: "f32[4, 256, 28, 28]" = torch.ops.aten.where.self(le_7, full_default, getitem_32);  le_7 = getitem_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_62: "f32[256]" = torch.ops.aten.add.Tensor(primals_124, 1e-05);  primals_124 = None
    rsqrt_9: "f32[256]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    unsqueeze_296: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_123, 0);  primals_123 = None
    unsqueeze_297: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 2);  unsqueeze_296 = None
    unsqueeze_298: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 3);  unsqueeze_297 = None
    sum_36: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_38: "f32[4, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_298);  convolution_15 = unsqueeze_298 = None
    mul_153: "f32[4, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_7, sub_38);  sub_38 = None
    sum_37: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_153, [0, 2, 3]);  mul_153 = None
    mul_158: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_9, primals_47);  primals_47 = None
    unsqueeze_305: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_158, 0);  mul_158 = None
    unsqueeze_306: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 2);  unsqueeze_305 = None
    unsqueeze_307: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, 3);  unsqueeze_306 = None
    mul_159: "f32[4, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_7, unsqueeze_307);  where_7 = unsqueeze_307 = None
    mul_160: "f32[256]" = torch.ops.aten.mul.Tensor(sum_37, rsqrt_9);  sum_37 = rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_159, relu_10, primals_46, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_159 = primals_46 = None
    getitem_35: "f32[4, 512, 28, 28]" = convolution_backward_11[0]
    getitem_36: "f32[256, 512, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_63: "f32[4, 512, 28, 28]" = torch.ops.aten.add.Tensor(avg_pool2d_backward_2, getitem_35);  avg_pool2d_backward_2 = getitem_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    le_8: "b8[4, 512, 28, 28]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
    where_8: "f32[4, 512, 28, 28]" = torch.ops.aten.where.self(le_8, full_default, add_63);  le_8 = add_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:113, code: shortcut = self.downsample(x)
    add_64: "f32[512]" = torch.ops.aten.add.Tensor(primals_121, 1e-05);  primals_121 = None
    rsqrt_10: "f32[512]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
    unsqueeze_308: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_120, 0);  primals_120 = None
    unsqueeze_309: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 2);  unsqueeze_308 = None
    unsqueeze_310: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 3);  unsqueeze_309 = None
    sum_38: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_39: "f32[4, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_310);  convolution_14 = unsqueeze_310 = None
    mul_161: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_8, sub_39);  sub_39 = None
    sum_39: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_161, [0, 2, 3]);  mul_161 = None
    mul_166: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_10, primals_44);  primals_44 = None
    unsqueeze_317: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_166, 0);  mul_166 = None
    unsqueeze_318: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 2);  unsqueeze_317 = None
    unsqueeze_319: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, 3);  unsqueeze_318 = None
    mul_167: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_8, unsqueeze_319);  unsqueeze_319 = None
    mul_168: "f32[512]" = torch.ops.aten.mul.Tensor(sum_39, rsqrt_10);  sum_39 = rsqrt_10 = None
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_167, avg_pool2d_1, primals_43, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_167 = avg_pool2d_1 = primals_43 = None
    getitem_38: "f32[4, 256, 28, 28]" = convolution_backward_12[0]
    getitem_39: "f32[512, 256, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    avg_pool2d_backward_4: "f32[4, 256, 56, 56]" = torch.ops.aten.avg_pool2d_backward.default(getitem_38, relu_6, [2, 2], [2, 2], [0, 0], True, False, None);  getitem_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_65: "f32[512]" = torch.ops.aten.add.Tensor(primals_118, 1e-05);  primals_118 = None
    rsqrt_11: "f32[512]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
    unsqueeze_320: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_117, 0);  primals_117 = None
    unsqueeze_321: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 2);  unsqueeze_320 = None
    unsqueeze_322: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 3);  unsqueeze_321 = None
    sub_40: "f32[4, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_322);  convolution_13 = unsqueeze_322 = None
    mul_169: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_8, sub_40);  sub_40 = None
    sum_41: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_169, [0, 2, 3]);  mul_169 = None
    mul_174: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_11, primals_41);  primals_41 = None
    unsqueeze_329: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_174, 0);  mul_174 = None
    unsqueeze_330: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 2);  unsqueeze_329 = None
    unsqueeze_331: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, 3);  unsqueeze_330 = None
    mul_175: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_8, unsqueeze_331);  where_8 = unsqueeze_331 = None
    mul_176: "f32[512]" = torch.ops.aten.mul.Tensor(sum_41, rsqrt_11);  sum_41 = rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_175, avg_pool2d, primals_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_175 = avg_pool2d = primals_40 = None
    getitem_41: "f32[4, 128, 28, 28]" = convolution_backward_13[0]
    getitem_42: "f32[512, 128, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:107, code: out = self.avd_last(out)
    avg_pool2d_backward_5: "f32[4, 128, 56, 56]" = torch.ops.aten.avg_pool2d_backward.default(getitem_41, sum_6, [3, 3], [2, 2], [1, 1], False, True, None);  getitem_41 = sum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_332: "f32[4, 1, 128, 56, 56]" = torch.ops.aten.unsqueeze.default(avg_pool2d_backward_5, 1);  avg_pool2d_backward_5 = None
    expand_7: "f32[4, 2, 128, 56, 56]" = torch.ops.aten.expand.default(unsqueeze_332, [4, 2, 128, 56, 56]);  unsqueeze_332 = None
    mul_177: "f32[4, 2, 128, 56, 56]" = torch.ops.aten.mul.Tensor(expand_7, view_7);  view_7 = None
    mul_178: "f32[4, 2, 128, 56, 56]" = torch.ops.aten.mul.Tensor(expand_7, view_11);  expand_7 = view_11 = None
    sum_42: "f32[4, 2, 128, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_177, [3, 4], True);  mul_177 = None
    view_37: "f32[4, 256, 1, 1]" = torch.ops.aten.reshape.default(sum_42, [4, 256, 1, 1]);  sum_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_38: "f32[4, 256]" = torch.ops.aten.reshape.default(view_37, [4, 256]);  view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_39: "f32[4, 2, 1, 128]" = torch.ops.aten.reshape.default(view_38, [4, 2, 1, 128]);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    mul_179: "f32[4, 2, 1, 128]" = torch.ops.aten.mul.Tensor(view_39, div_1);  view_39 = None
    sum_43: "f32[4, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_179, [1], True)
    mul_180: "f32[4, 2, 1, 128]" = torch.ops.aten.mul.Tensor(div_1, sum_43);  div_1 = sum_43 = None
    sub_41: "f32[4, 2, 1, 128]" = torch.ops.aten.sub.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_11: "f32[4, 1, 2, 128]" = torch.ops.aten.permute.default(sub_41, [0, 2, 1, 3]);  sub_41 = None
    view_40: "f32[4, 256, 1, 1]" = torch.ops.aten.reshape.default(permute_11, [4, 256, 1, 1]);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(view_40, relu_9, primals_38, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_40 = primals_38 = None
    getitem_44: "f32[4, 64, 1, 1]" = convolution_backward_14[0]
    getitem_45: "f32[256, 64, 1, 1]" = convolution_backward_14[1]
    getitem_46: "f32[256]" = convolution_backward_14[2];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    le_9: "b8[4, 64, 1, 1]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
    where_9: "f32[4, 64, 1, 1]" = torch.ops.aten.where.self(le_9, full_default, getitem_44);  le_9 = getitem_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_66: "f32[64]" = torch.ops.aten.add.Tensor(primals_115, 1e-05);  primals_115 = None
    rsqrt_12: "f32[64]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    unsqueeze_333: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_114, 0);  primals_114 = None
    unsqueeze_334: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 2);  unsqueeze_333 = None
    unsqueeze_335: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 3);  unsqueeze_334 = None
    sum_44: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_42: "f32[4, 64, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_335);  convolution_11 = unsqueeze_335 = None
    mul_181: "f32[4, 64, 1, 1]" = torch.ops.aten.mul.Tensor(where_9, sub_42);  sub_42 = None
    sum_45: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_181, [0, 2, 3]);  mul_181 = None
    mul_186: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_12, primals_36);  primals_36 = None
    unsqueeze_342: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_186, 0);  mul_186 = None
    unsqueeze_343: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 2);  unsqueeze_342 = None
    unsqueeze_344: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 3);  unsqueeze_343 = None
    mul_187: "f32[4, 64, 1, 1]" = torch.ops.aten.mul.Tensor(where_9, unsqueeze_344);  where_9 = unsqueeze_344 = None
    mul_188: "f32[64]" = torch.ops.aten.mul.Tensor(sum_45, rsqrt_12);  sum_45 = rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_187, mean_1, primals_34, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_187 = mean_1 = primals_34 = None
    getitem_47: "f32[4, 128, 1, 1]" = convolution_backward_15[0]
    getitem_48: "f32[64, 128, 1, 1]" = convolution_backward_15[1]
    getitem_49: "f32[64]" = convolution_backward_15[2];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_8: "f32[4, 128, 56, 56]" = torch.ops.aten.expand.default(getitem_47, [4, 128, 56, 56]);  getitem_47 = None
    div_7: "f32[4, 128, 56, 56]" = torch.ops.aten.div.Scalar(expand_8, 3136);  expand_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_345: "f32[4, 1, 128, 56, 56]" = torch.ops.aten.unsqueeze.default(div_7, 1);  div_7 = None
    expand_9: "f32[4, 2, 128, 56, 56]" = torch.ops.aten.expand.default(unsqueeze_345, [4, 2, 128, 56, 56]);  unsqueeze_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_67: "f32[4, 2, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_178, expand_9);  mul_178 = expand_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_41: "f32[4, 256, 56, 56]" = torch.ops.aten.reshape.default(add_67, [4, 256, 56, 56]);  add_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    le_10: "b8[4, 256, 56, 56]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
    where_10: "f32[4, 256, 56, 56]" = torch.ops.aten.where.self(le_10, full_default, view_41);  le_10 = view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_68: "f32[256]" = torch.ops.aten.add.Tensor(primals_112, 1e-05);  primals_112 = None
    rsqrt_13: "f32[256]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    unsqueeze_346: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_111, 0);  primals_111 = None
    unsqueeze_347: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 2);  unsqueeze_346 = None
    unsqueeze_348: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 3);  unsqueeze_347 = None
    sum_46: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_43: "f32[4, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_348);  convolution_10 = unsqueeze_348 = None
    mul_189: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_10, sub_43);  sub_43 = None
    sum_47: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_189, [0, 2, 3]);  mul_189 = None
    mul_194: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_13, primals_32);  primals_32 = None
    unsqueeze_355: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_194, 0);  mul_194 = None
    unsqueeze_356: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 2);  unsqueeze_355 = None
    unsqueeze_357: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 3);  unsqueeze_356 = None
    mul_195: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_10, unsqueeze_357);  where_10 = unsqueeze_357 = None
    mul_196: "f32[256]" = torch.ops.aten.mul.Tensor(sum_47, rsqrt_13);  sum_47 = rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_195, relu_7, primals_31, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_195 = primals_31 = None
    getitem_50: "f32[4, 128, 56, 56]" = convolution_backward_16[0]
    getitem_51: "f32[256, 64, 3, 3]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    le_11: "b8[4, 128, 56, 56]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
    where_11: "f32[4, 128, 56, 56]" = torch.ops.aten.where.self(le_11, full_default, getitem_50);  le_11 = getitem_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_69: "f32[128]" = torch.ops.aten.add.Tensor(primals_109, 1e-05);  primals_109 = None
    rsqrt_14: "f32[128]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
    unsqueeze_358: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_108, 0);  primals_108 = None
    unsqueeze_359: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 2);  unsqueeze_358 = None
    unsqueeze_360: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 3);  unsqueeze_359 = None
    sum_48: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_44: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_360);  convolution_9 = unsqueeze_360 = None
    mul_197: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_11, sub_44);  sub_44 = None
    sum_49: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_197, [0, 2, 3]);  mul_197 = None
    mul_202: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_14, primals_29);  primals_29 = None
    unsqueeze_367: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_202, 0);  mul_202 = None
    unsqueeze_368: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 2);  unsqueeze_367 = None
    unsqueeze_369: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 3);  unsqueeze_368 = None
    mul_203: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_11, unsqueeze_369);  where_11 = unsqueeze_369 = None
    mul_204: "f32[128]" = torch.ops.aten.mul.Tensor(sum_49, rsqrt_14);  sum_49 = rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_203, relu_6, primals_28, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_203 = primals_28 = None
    getitem_53: "f32[4, 256, 56, 56]" = convolution_backward_17[0]
    getitem_54: "f32[128, 256, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_70: "f32[4, 256, 56, 56]" = torch.ops.aten.add.Tensor(avg_pool2d_backward_4, getitem_53);  avg_pool2d_backward_4 = getitem_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    le_12: "b8[4, 256, 56, 56]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
    where_12: "f32[4, 256, 56, 56]" = torch.ops.aten.where.self(le_12, full_default, add_70);  le_12 = add_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:113, code: shortcut = self.downsample(x)
    add_71: "f32[256]" = torch.ops.aten.add.Tensor(primals_106, 1e-05);  primals_106 = None
    rsqrt_15: "f32[256]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
    unsqueeze_370: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_105, 0);  primals_105 = None
    unsqueeze_371: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 2);  unsqueeze_370 = None
    unsqueeze_372: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 3);  unsqueeze_371 = None
    sum_50: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_45: "f32[4, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_372);  convolution_8 = unsqueeze_372 = None
    mul_205: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_12, sub_45);  sub_45 = None
    sum_51: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_205, [0, 2, 3]);  mul_205 = None
    mul_210: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_15, primals_26);  primals_26 = None
    unsqueeze_379: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_210, 0);  mul_210 = None
    unsqueeze_380: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
    unsqueeze_381: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 3);  unsqueeze_380 = None
    mul_211: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_12, unsqueeze_381);  unsqueeze_381 = None
    mul_212: "f32[256]" = torch.ops.aten.mul.Tensor(sum_51, rsqrt_15);  sum_51 = rsqrt_15 = None
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_211, getitem, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_211 = primals_25 = None
    getitem_56: "f32[4, 64, 56, 56]" = convolution_backward_18[0]
    getitem_57: "f32[256, 64, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_72: "f32[256]" = torch.ops.aten.add.Tensor(primals_103, 1e-05);  primals_103 = None
    rsqrt_16: "f32[256]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
    unsqueeze_382: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_102, 0);  primals_102 = None
    unsqueeze_383: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 2);  unsqueeze_382 = None
    unsqueeze_384: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 3);  unsqueeze_383 = None
    sub_46: "f32[4, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_384);  convolution_7 = unsqueeze_384 = None
    mul_213: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_12, sub_46);  sub_46 = None
    sum_53: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_213, [0, 2, 3]);  mul_213 = None
    mul_218: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_16, primals_23);  primals_23 = None
    unsqueeze_391: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_218, 0);  mul_218 = None
    unsqueeze_392: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
    unsqueeze_393: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 3);  unsqueeze_392 = None
    mul_219: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_12, unsqueeze_393);  where_12 = unsqueeze_393 = None
    mul_220: "f32[256]" = torch.ops.aten.mul.Tensor(sum_53, rsqrt_16);  sum_53 = rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_219, sum_3, primals_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_219 = sum_3 = primals_22 = None
    getitem_59: "f32[4, 64, 56, 56]" = convolution_backward_19[0]
    getitem_60: "f32[256, 64, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_394: "f32[4, 1, 64, 56, 56]" = torch.ops.aten.unsqueeze.default(getitem_59, 1);  getitem_59 = None
    expand_10: "f32[4, 2, 64, 56, 56]" = torch.ops.aten.expand.default(unsqueeze_394, [4, 2, 64, 56, 56]);  unsqueeze_394 = None
    mul_221: "f32[4, 2, 64, 56, 56]" = torch.ops.aten.mul.Tensor(expand_10, view_1);  view_1 = None
    mul_222: "f32[4, 2, 64, 56, 56]" = torch.ops.aten.mul.Tensor(expand_10, view_5);  expand_10 = view_5 = None
    sum_54: "f32[4, 2, 64, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_221, [3, 4], True);  mul_221 = None
    view_42: "f32[4, 128, 1, 1]" = torch.ops.aten.reshape.default(sum_54, [4, 128, 1, 1]);  sum_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_43: "f32[4, 128]" = torch.ops.aten.reshape.default(view_42, [4, 128]);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_44: "f32[4, 2, 1, 64]" = torch.ops.aten.reshape.default(view_43, [4, 2, 1, 64]);  view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    mul_223: "f32[4, 2, 1, 64]" = torch.ops.aten.mul.Tensor(view_44, div);  view_44 = None
    sum_55: "f32[4, 1, 1, 64]" = torch.ops.aten.sum.dim_IntList(mul_223, [1], True)
    mul_224: "f32[4, 2, 1, 64]" = torch.ops.aten.mul.Tensor(div, sum_55);  div = sum_55 = None
    sub_47: "f32[4, 2, 1, 64]" = torch.ops.aten.sub.Tensor(mul_223, mul_224);  mul_223 = mul_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_12: "f32[4, 1, 2, 64]" = torch.ops.aten.permute.default(sub_47, [0, 2, 1, 3]);  sub_47 = None
    view_45: "f32[4, 128, 1, 1]" = torch.ops.aten.reshape.default(permute_12, [4, 128, 1, 1]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(view_45, relu_5, primals_20, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_45 = primals_20 = None
    getitem_62: "f32[4, 32, 1, 1]" = convolution_backward_20[0]
    getitem_63: "f32[128, 32, 1, 1]" = convolution_backward_20[1]
    getitem_64: "f32[128]" = convolution_backward_20[2];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    le_13: "b8[4, 32, 1, 1]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
    where_13: "f32[4, 32, 1, 1]" = torch.ops.aten.where.self(le_13, full_default, getitem_62);  le_13 = getitem_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_73: "f32[32]" = torch.ops.aten.add.Tensor(primals_100, 1e-05);  primals_100 = None
    rsqrt_17: "f32[32]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    unsqueeze_395: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(primals_99, 0);  primals_99 = None
    unsqueeze_396: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 2);  unsqueeze_395 = None
    unsqueeze_397: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, 3);  unsqueeze_396 = None
    sum_56: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_48: "f32[4, 32, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_397);  convolution_5 = unsqueeze_397 = None
    mul_225: "f32[4, 32, 1, 1]" = torch.ops.aten.mul.Tensor(where_13, sub_48);  sub_48 = None
    sum_57: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_225, [0, 2, 3]);  mul_225 = None
    mul_230: "f32[32]" = torch.ops.aten.mul.Tensor(rsqrt_17, primals_18);  primals_18 = None
    unsqueeze_404: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_230, 0);  mul_230 = None
    unsqueeze_405: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 2);  unsqueeze_404 = None
    unsqueeze_406: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 3);  unsqueeze_405 = None
    mul_231: "f32[4, 32, 1, 1]" = torch.ops.aten.mul.Tensor(where_13, unsqueeze_406);  where_13 = unsqueeze_406 = None
    mul_232: "f32[32]" = torch.ops.aten.mul.Tensor(sum_57, rsqrt_17);  sum_57 = rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_231, mean, primals_16, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_231 = mean = primals_16 = None
    getitem_65: "f32[4, 64, 1, 1]" = convolution_backward_21[0]
    getitem_66: "f32[32, 64, 1, 1]" = convolution_backward_21[1]
    getitem_67: "f32[32]" = convolution_backward_21[2];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_11: "f32[4, 64, 56, 56]" = torch.ops.aten.expand.default(getitem_65, [4, 64, 56, 56]);  getitem_65 = None
    div_8: "f32[4, 64, 56, 56]" = torch.ops.aten.div.Scalar(expand_11, 3136);  expand_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_407: "f32[4, 1, 64, 56, 56]" = torch.ops.aten.unsqueeze.default(div_8, 1);  div_8 = None
    expand_12: "f32[4, 2, 64, 56, 56]" = torch.ops.aten.expand.default(unsqueeze_407, [4, 2, 64, 56, 56]);  unsqueeze_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_74: "f32[4, 2, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_222, expand_12);  mul_222 = expand_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_46: "f32[4, 128, 56, 56]" = torch.ops.aten.reshape.default(add_74, [4, 128, 56, 56]);  add_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    le_14: "b8[4, 128, 56, 56]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
    where_14: "f32[4, 128, 56, 56]" = torch.ops.aten.where.self(le_14, full_default, view_46);  le_14 = view_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_75: "f32[128]" = torch.ops.aten.add.Tensor(primals_97, 1e-05);  primals_97 = None
    rsqrt_18: "f32[128]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
    unsqueeze_408: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_96, 0);  primals_96 = None
    unsqueeze_409: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, 2);  unsqueeze_408 = None
    unsqueeze_410: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 3);  unsqueeze_409 = None
    sum_58: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_49: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_410);  convolution_4 = unsqueeze_410 = None
    mul_233: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_14, sub_49);  sub_49 = None
    sum_59: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_233, [0, 2, 3]);  mul_233 = None
    mul_238: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_18, primals_14);  primals_14 = None
    unsqueeze_417: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_238, 0);  mul_238 = None
    unsqueeze_418: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 2);  unsqueeze_417 = None
    unsqueeze_419: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 3);  unsqueeze_418 = None
    mul_239: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_14, unsqueeze_419);  where_14 = unsqueeze_419 = None
    mul_240: "f32[128]" = torch.ops.aten.mul.Tensor(sum_59, rsqrt_18);  sum_59 = rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_239, relu_3, primals_13, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_239 = primals_13 = None
    getitem_68: "f32[4, 64, 56, 56]" = convolution_backward_22[0]
    getitem_69: "f32[128, 32, 3, 3]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    le_15: "b8[4, 64, 56, 56]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
    where_15: "f32[4, 64, 56, 56]" = torch.ops.aten.where.self(le_15, full_default, getitem_68);  le_15 = getitem_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_76: "f32[64]" = torch.ops.aten.add.Tensor(primals_94, 1e-05);  primals_94 = None
    rsqrt_19: "f32[64]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
    unsqueeze_420: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_93, 0);  primals_93 = None
    unsqueeze_421: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, 2);  unsqueeze_420 = None
    unsqueeze_422: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 3);  unsqueeze_421 = None
    sum_60: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_50: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_422);  convolution_3 = unsqueeze_422 = None
    mul_241: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_15, sub_50);  sub_50 = None
    sum_61: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_241, [0, 2, 3]);  mul_241 = None
    mul_246: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_19, primals_11);  primals_11 = None
    unsqueeze_429: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_246, 0);  mul_246 = None
    unsqueeze_430: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 2);  unsqueeze_429 = None
    unsqueeze_431: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 3);  unsqueeze_430 = None
    mul_247: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_15, unsqueeze_431);  where_15 = unsqueeze_431 = None
    mul_248: "f32[64]" = torch.ops.aten.mul.Tensor(sum_61, rsqrt_19);  sum_61 = rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_247, getitem, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_247 = getitem = primals_10 = None
    getitem_71: "f32[4, 64, 56, 56]" = convolution_backward_23[0]
    getitem_72: "f32[64, 64, 1, 1]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_77: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(getitem_56, getitem_71);  getitem_56 = getitem_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:523, code: x = self.maxpool(x)
    max_pool2d_with_indices_backward: "f32[4, 64, 112, 112]" = torch.ops.aten.max_pool2d_with_indices_backward.default(add_77, relu_2, [3, 3], [2, 2], [1, 1], [1, 1], False, getitem_1);  add_77 = getitem_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:522, code: x = self.act1(x)
    le_16: "b8[4, 64, 112, 112]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
    where_16: "f32[4, 64, 112, 112]" = torch.ops.aten.where.self(le_16, full_default, max_pool2d_with_indices_backward);  le_16 = max_pool2d_with_indices_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:521, code: x = self.bn1(x)
    add_78: "f32[64]" = torch.ops.aten.add.Tensor(primals_91, 1e-05);  primals_91 = None
    rsqrt_20: "f32[64]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    unsqueeze_432: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_90, 0);  primals_90 = None
    unsqueeze_433: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, 2);  unsqueeze_432 = None
    unsqueeze_434: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 3);  unsqueeze_433 = None
    sum_62: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_51: "f32[4, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_434);  convolution_2 = unsqueeze_434 = None
    mul_249: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_16, sub_51);  sub_51 = None
    sum_63: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_249, [0, 2, 3]);  mul_249 = None
    mul_254: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_20, primals_8);  primals_8 = None
    unsqueeze_441: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_254, 0);  mul_254 = None
    unsqueeze_442: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 2);  unsqueeze_441 = None
    unsqueeze_443: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 3);  unsqueeze_442 = None
    mul_255: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_16, unsqueeze_443);  where_16 = unsqueeze_443 = None
    mul_256: "f32[64]" = torch.ops.aten.mul.Tensor(sum_63, rsqrt_20);  sum_63 = rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:520, code: x = self.conv1(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_255, relu_1, primals_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_255 = primals_7 = None
    getitem_74: "f32[4, 32, 112, 112]" = convolution_backward_24[0]
    getitem_75: "f32[64, 32, 3, 3]" = convolution_backward_24[1];  convolution_backward_24 = None
    le_17: "b8[4, 32, 112, 112]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
    where_17: "f32[4, 32, 112, 112]" = torch.ops.aten.where.self(le_17, full_default, getitem_74);  le_17 = getitem_74 = None
    add_79: "f32[32]" = torch.ops.aten.add.Tensor(primals_88, 1e-05);  primals_88 = None
    rsqrt_21: "f32[32]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
    unsqueeze_444: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(primals_87, 0);  primals_87 = None
    unsqueeze_445: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, 2);  unsqueeze_444 = None
    unsqueeze_446: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 3);  unsqueeze_445 = None
    sum_64: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_52: "f32[4, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_446);  convolution_1 = unsqueeze_446 = None
    mul_257: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_17, sub_52);  sub_52 = None
    sum_65: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_257, [0, 2, 3]);  mul_257 = None
    mul_262: "f32[32]" = torch.ops.aten.mul.Tensor(rsqrt_21, primals_5);  primals_5 = None
    unsqueeze_453: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_262, 0);  mul_262 = None
    unsqueeze_454: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 2);  unsqueeze_453 = None
    unsqueeze_455: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, 3);  unsqueeze_454 = None
    mul_263: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_17, unsqueeze_455);  where_17 = unsqueeze_455 = None
    mul_264: "f32[32]" = torch.ops.aten.mul.Tensor(sum_65, rsqrt_21);  sum_65 = rsqrt_21 = None
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_263, relu, primals_4, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_263 = primals_4 = None
    getitem_77: "f32[4, 32, 112, 112]" = convolution_backward_25[0]
    getitem_78: "f32[32, 32, 3, 3]" = convolution_backward_25[1];  convolution_backward_25 = None
    le_18: "b8[4, 32, 112, 112]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
    where_18: "f32[4, 32, 112, 112]" = torch.ops.aten.where.self(le_18, full_default, getitem_77);  le_18 = full_default = getitem_77 = None
    add_80: "f32[32]" = torch.ops.aten.add.Tensor(primals_85, 1e-05);  primals_85 = None
    rsqrt_22: "f32[32]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
    unsqueeze_456: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(primals_84, 0);  primals_84 = None
    unsqueeze_457: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, 2);  unsqueeze_456 = None
    unsqueeze_458: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 3);  unsqueeze_457 = None
    sum_66: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    sub_53: "f32[4, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_458);  convolution = unsqueeze_458 = None
    mul_265: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_18, sub_53);  sub_53 = None
    sum_67: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_265, [0, 2, 3]);  mul_265 = None
    mul_270: "f32[32]" = torch.ops.aten.mul.Tensor(rsqrt_22, primals_2);  primals_2 = None
    unsqueeze_465: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_270, 0);  mul_270 = None
    unsqueeze_466: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 2);  unsqueeze_465 = None
    unsqueeze_467: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 3);  unsqueeze_466 = None
    mul_271: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_18, unsqueeze_467);  where_18 = unsqueeze_467 = None
    mul_272: "f32[32]" = torch.ops.aten.mul.Tensor(sum_67, rsqrt_22);  sum_67 = rsqrt_22 = None
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_271, primals_153, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_271 = primals_153 = primals_1 = None
    getitem_81: "f32[32, 3, 3, 3]" = convolution_backward_26[1];  convolution_backward_26 = None
    return [getitem_81, mul_272, sum_66, getitem_78, mul_264, sum_64, getitem_75, mul_256, sum_62, getitem_72, mul_248, sum_60, getitem_69, mul_240, sum_58, getitem_66, getitem_67, mul_232, sum_56, getitem_63, getitem_64, getitem_60, mul_220, sum_50, getitem_57, mul_212, sum_50, getitem_54, mul_204, sum_48, getitem_51, mul_196, sum_46, getitem_48, getitem_49, mul_188, sum_44, getitem_45, getitem_46, getitem_42, mul_176, sum_38, getitem_39, mul_168, sum_38, getitem_36, mul_160, sum_36, getitem_33, mul_152, sum_34, getitem_30, getitem_31, mul_144, sum_32, getitem_27, getitem_28, getitem_24, mul_132, sum_26, getitem_21, mul_124, sum_26, getitem_18, mul_116, sum_24, getitem_15, mul_108, sum_22, getitem_12, getitem_13, mul_100, sum_20, getitem_9, getitem_10, getitem_6, mul_88, sum_14, getitem_3, mul_80, sum_14, permute_8, view_25, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    