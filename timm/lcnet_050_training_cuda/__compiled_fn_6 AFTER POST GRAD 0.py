from __future__ import annotations



def forward(self, primals_1: "f32[8]", primals_3: "f32[8]", primals_5: "f32[16]", primals_7: "f32[16]", primals_9: "f32[32]", primals_11: "f32[32]", primals_13: "f32[32]", primals_15: "f32[32]", primals_17: "f32[64]", primals_19: "f32[64]", primals_21: "f32[64]", primals_23: "f32[64]", primals_25: "f32[128]", primals_27: "f32[128]", primals_29: "f32[128]", primals_31: "f32[128]", primals_33: "f32[128]", primals_35: "f32[128]", primals_37: "f32[128]", primals_39: "f32[128]", primals_41: "f32[128]", primals_43: "f32[128]", primals_45: "f32[128]", primals_47: "f32[128]", primals_49: "f32[256]", primals_51: "f32[256]", primals_53: "f32[256]", primals_57: "f32[8, 3, 3, 3]", primals_58: "f32[8, 1, 3, 3]", primals_59: "f32[16, 8, 1, 1]", primals_60: "f32[16, 1, 3, 3]", primals_61: "f32[32, 16, 1, 1]", primals_62: "f32[32, 1, 3, 3]", primals_63: "f32[32, 32, 1, 1]", primals_64: "f32[32, 1, 3, 3]", primals_65: "f32[64, 32, 1, 1]", primals_66: "f32[64, 1, 3, 3]", primals_67: "f32[64, 64, 1, 1]", primals_68: "f32[64, 1, 3, 3]", primals_69: "f32[128, 64, 1, 1]", primals_70: "f32[128, 1, 5, 5]", primals_71: "f32[128, 128, 1, 1]", primals_72: "f32[128, 1, 5, 5]", primals_73: "f32[128, 128, 1, 1]", primals_74: "f32[128, 1, 5, 5]", primals_75: "f32[128, 128, 1, 1]", primals_76: "f32[128, 1, 5, 5]", primals_77: "f32[128, 128, 1, 1]", primals_78: "f32[128, 1, 5, 5]", primals_79: "f32[128, 128, 1, 1]", primals_80: "f32[128, 1, 5, 5]", primals_81: "f32[32, 128, 1, 1]", primals_83: "f32[128, 32, 1, 1]", primals_85: "f32[256, 128, 1, 1]", primals_86: "f32[256, 1, 5, 5]", primals_87: "f32[64, 256, 1, 1]", primals_89: "f32[256, 64, 1, 1]", primals_91: "f32[256, 256, 1, 1]", primals_92: "f32[1280, 256, 1, 1]", primals_175: "f32[8, 3, 224, 224]", convolution: "f32[8, 8, 112, 112]", squeeze_1: "f32[8]", clone: "f32[8, 8, 112, 112]", div: "f32[8, 8, 112, 112]", convolution_1: "f32[8, 8, 112, 112]", squeeze_4: "f32[8]", clone_1: "f32[8, 8, 112, 112]", div_1: "f32[8, 8, 112, 112]", convolution_2: "f32[8, 16, 112, 112]", squeeze_7: "f32[16]", clone_2: "f32[8, 16, 112, 112]", div_2: "f32[8, 16, 112, 112]", convolution_3: "f32[8, 16, 56, 56]", squeeze_10: "f32[16]", clone_3: "f32[8, 16, 56, 56]", div_3: "f32[8, 16, 56, 56]", convolution_4: "f32[8, 32, 56, 56]", squeeze_13: "f32[32]", clone_4: "f32[8, 32, 56, 56]", div_4: "f32[8, 32, 56, 56]", convolution_5: "f32[8, 32, 56, 56]", squeeze_16: "f32[32]", clone_5: "f32[8, 32, 56, 56]", div_5: "f32[8, 32, 56, 56]", convolution_6: "f32[8, 32, 56, 56]", squeeze_19: "f32[32]", clone_6: "f32[8, 32, 56, 56]", div_6: "f32[8, 32, 56, 56]", convolution_7: "f32[8, 32, 28, 28]", squeeze_22: "f32[32]", clone_7: "f32[8, 32, 28, 28]", div_7: "f32[8, 32, 28, 28]", convolution_8: "f32[8, 64, 28, 28]", squeeze_25: "f32[64]", clone_8: "f32[8, 64, 28, 28]", div_8: "f32[8, 64, 28, 28]", convolution_9: "f32[8, 64, 28, 28]", squeeze_28: "f32[64]", clone_9: "f32[8, 64, 28, 28]", div_9: "f32[8, 64, 28, 28]", convolution_10: "f32[8, 64, 28, 28]", squeeze_31: "f32[64]", clone_10: "f32[8, 64, 28, 28]", div_10: "f32[8, 64, 28, 28]", convolution_11: "f32[8, 64, 14, 14]", squeeze_34: "f32[64]", clone_11: "f32[8, 64, 14, 14]", div_11: "f32[8, 64, 14, 14]", convolution_12: "f32[8, 128, 14, 14]", squeeze_37: "f32[128]", clone_12: "f32[8, 128, 14, 14]", div_12: "f32[8, 128, 14, 14]", convolution_13: "f32[8, 128, 14, 14]", squeeze_40: "f32[128]", clone_13: "f32[8, 128, 14, 14]", div_13: "f32[8, 128, 14, 14]", convolution_14: "f32[8, 128, 14, 14]", squeeze_43: "f32[128]", clone_14: "f32[8, 128, 14, 14]", div_14: "f32[8, 128, 14, 14]", convolution_15: "f32[8, 128, 14, 14]", squeeze_46: "f32[128]", clone_15: "f32[8, 128, 14, 14]", div_15: "f32[8, 128, 14, 14]", convolution_16: "f32[8, 128, 14, 14]", squeeze_49: "f32[128]", clone_16: "f32[8, 128, 14, 14]", div_16: "f32[8, 128, 14, 14]", convolution_17: "f32[8, 128, 14, 14]", squeeze_52: "f32[128]", clone_17: "f32[8, 128, 14, 14]", div_17: "f32[8, 128, 14, 14]", convolution_18: "f32[8, 128, 14, 14]", squeeze_55: "f32[128]", clone_18: "f32[8, 128, 14, 14]", div_18: "f32[8, 128, 14, 14]", convolution_19: "f32[8, 128, 14, 14]", squeeze_58: "f32[128]", clone_19: "f32[8, 128, 14, 14]", div_19: "f32[8, 128, 14, 14]", convolution_20: "f32[8, 128, 14, 14]", squeeze_61: "f32[128]", clone_20: "f32[8, 128, 14, 14]", div_20: "f32[8, 128, 14, 14]", convolution_21: "f32[8, 128, 14, 14]", squeeze_64: "f32[128]", clone_21: "f32[8, 128, 14, 14]", div_21: "f32[8, 128, 14, 14]", convolution_22: "f32[8, 128, 14, 14]", squeeze_67: "f32[128]", clone_22: "f32[8, 128, 14, 14]", div_22: "f32[8, 128, 14, 14]", convolution_23: "f32[8, 128, 7, 7]", squeeze_70: "f32[128]", clone_23: "f32[8, 128, 7, 7]", div_23: "f32[8, 128, 7, 7]", mean: "f32[8, 128, 1, 1]", relu: "f32[8, 32, 1, 1]", div_24: "f32[8, 128, 1, 1]", mul_192: "f32[8, 128, 7, 7]", convolution_26: "f32[8, 256, 7, 7]", squeeze_73: "f32[256]", clone_24: "f32[8, 256, 7, 7]", div_25: "f32[8, 256, 7, 7]", convolution_27: "f32[8, 256, 7, 7]", squeeze_76: "f32[256]", clone_25: "f32[8, 256, 7, 7]", div_26: "f32[8, 256, 7, 7]", mean_1: "f32[8, 256, 1, 1]", relu_1: "f32[8, 64, 1, 1]", div_27: "f32[8, 256, 1, 1]", mul_209: "f32[8, 256, 7, 7]", convolution_30: "f32[8, 256, 7, 7]", squeeze_79: "f32[256]", clone_26: "f32[8, 256, 7, 7]", mean_2: "f32[8, 256, 1, 1]", convolution_31: "f32[8, 1280, 1, 1]", view_1: "f32[8, 1280]", permute_1: "f32[1000, 1280]", unsqueeze_110: "f32[1, 256, 1, 1]", bitwise_and: "b8[8, 256, 1, 1]", unsqueeze_122: "f32[1, 256, 1, 1]", unsqueeze_134: "f32[1, 256, 1, 1]", bitwise_and_1: "b8[8, 128, 1, 1]", unsqueeze_146: "f32[1, 128, 1, 1]", unsqueeze_158: "f32[1, 128, 1, 1]", unsqueeze_170: "f32[1, 128, 1, 1]", unsqueeze_182: "f32[1, 128, 1, 1]", unsqueeze_194: "f32[1, 128, 1, 1]", unsqueeze_206: "f32[1, 128, 1, 1]", unsqueeze_218: "f32[1, 128, 1, 1]", unsqueeze_230: "f32[1, 128, 1, 1]", unsqueeze_242: "f32[1, 128, 1, 1]", unsqueeze_254: "f32[1, 128, 1, 1]", unsqueeze_266: "f32[1, 128, 1, 1]", unsqueeze_278: "f32[1, 128, 1, 1]", unsqueeze_290: "f32[1, 64, 1, 1]", unsqueeze_302: "f32[1, 64, 1, 1]", unsqueeze_314: "f32[1, 64, 1, 1]", unsqueeze_326: "f32[1, 64, 1, 1]", unsqueeze_338: "f32[1, 32, 1, 1]", unsqueeze_350: "f32[1, 32, 1, 1]", unsqueeze_362: "f32[1, 32, 1, 1]", unsqueeze_374: "f32[1, 32, 1, 1]", unsqueeze_386: "f32[1, 16, 1, 1]", unsqueeze_398: "f32[1, 16, 1, 1]", unsqueeze_410: "f32[1, 8, 1, 1]", unsqueeze_422: "f32[1, 8, 1, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/linear.py:19, code: return F.linear(input, self.weight, self.bias)
    mm: "f32[8, 1280]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 1280]" = torch.ops.aten.mm.default(permute_2, view_1);  permute_2 = view_1 = None
    permute_3: "f32[1280, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_2: "f32[1000]" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 1280]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:147, code: x = self.flatten(x)
    view_3: "f32[8, 1280, 1, 1]" = torch.ops.aten.reshape.default(mm, [8, 1280, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:146, code: x = self.act2(x)
    lt: "b8[8, 1280, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_31, -3)
    le: "b8[8, 1280, 1, 1]" = torch.ops.aten.le.Scalar(convolution_31, 3)
    div_30: "f32[8, 1280, 1, 1]" = torch.ops.aten.div.Tensor(convolution_31, 3);  convolution_31 = None
    add_165: "f32[8, 1280, 1, 1]" = torch.ops.aten.add.Tensor(div_30, 0.5);  div_30 = None
    mul_219: "f32[8, 1280, 1, 1]" = torch.ops.aten.mul.Tensor(view_3, add_165);  add_165 = None
    where: "f32[8, 1280, 1, 1]" = torch.ops.aten.where.self(le, mul_219, view_3);  le = mul_219 = view_3 = None
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where_1: "f32[8, 1280, 1, 1]" = torch.ops.aten.where.self(lt, full_default, where);  lt = where = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:145, code: x = self.conv_head(x)
    sum_2: "f32[1280]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    convolution_backward = torch.ops.aten.convolution_backward.default(where_1, mean_2, primals_92, [1280], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_1 = mean_2 = primals_92 = None
    getitem_54: "f32[8, 256, 1, 1]" = convolution_backward[0]
    getitem_55: "f32[1280, 256, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 256, 7, 7]" = torch.ops.aten.expand.default(getitem_54, [8, 256, 7, 7]);  getitem_54 = None
    div_31: "f32[8, 256, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_1: "b8[8, 256, 7, 7]" = torch.ops.aten.lt.Scalar(clone_26, -3)
    le_1: "b8[8, 256, 7, 7]" = torch.ops.aten.le.Scalar(clone_26, 3)
    div_32: "f32[8, 256, 7, 7]" = torch.ops.aten.div.Tensor(clone_26, 3);  clone_26 = None
    add_166: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(div_32, 0.5);  div_32 = None
    mul_220: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(div_31, add_166);  add_166 = None
    where_2: "f32[8, 256, 7, 7]" = torch.ops.aten.where.self(le_1, mul_220, div_31);  le_1 = mul_220 = div_31 = None
    where_3: "f32[8, 256, 7, 7]" = torch.ops.aten.where.self(lt_1, full_default, where_2);  lt_1 = where_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_3: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_27: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_110);  convolution_30 = unsqueeze_110 = None
    mul_221: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_27)
    sum_4: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_221, [0, 2, 3]);  mul_221 = None
    mul_222: "f32[256]" = torch.ops.aten.mul.Tensor(sum_3, 0.002551020408163265)
    unsqueeze_111: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_222, 0);  mul_222 = None
    unsqueeze_112: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_111, 2);  unsqueeze_111 = None
    unsqueeze_113: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, 3);  unsqueeze_112 = None
    mul_223: "f32[256]" = torch.ops.aten.mul.Tensor(sum_4, 0.002551020408163265)
    mul_224: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_225: "f32[256]" = torch.ops.aten.mul.Tensor(mul_223, mul_224);  mul_223 = mul_224 = None
    unsqueeze_114: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_225, 0);  mul_225 = None
    unsqueeze_115: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, 2);  unsqueeze_114 = None
    unsqueeze_116: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_115, 3);  unsqueeze_115 = None
    mul_226: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_53);  primals_53 = None
    unsqueeze_117: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_226, 0);  mul_226 = None
    unsqueeze_118: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_117, 2);  unsqueeze_117 = None
    unsqueeze_119: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, 3);  unsqueeze_118 = None
    mul_227: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_116);  sub_27 = unsqueeze_116 = None
    sub_29: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(where_3, mul_227);  where_3 = mul_227 = None
    sub_30: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(sub_29, unsqueeze_113);  sub_29 = unsqueeze_113 = None
    mul_228: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_119);  sub_30 = unsqueeze_119 = None
    mul_229: "f32[256]" = torch.ops.aten.mul.Tensor(sum_4, squeeze_79);  sum_4 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_228, mul_209, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_228 = mul_209 = primals_91 = None
    getitem_57: "f32[8, 256, 7, 7]" = convolution_backward_1[0]
    getitem_58: "f32[256, 256, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_230: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_57, div_26);  div_26 = None
    mul_231: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_57, div_27);  getitem_57 = div_27 = None
    sum_5: "f32[8, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_230, [2, 3], True);  mul_230 = None
    mul_232: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(sum_5, 0.16666666666666666);  sum_5 = None
    where_4: "f32[8, 256, 1, 1]" = torch.ops.aten.where.self(bitwise_and, mul_232, full_default);  bitwise_and = mul_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    sum_6: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(where_4, relu_1, primals_89, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_4 = primals_89 = None
    getitem_60: "f32[8, 64, 1, 1]" = convolution_backward_2[0]
    getitem_61: "f32[256, 64, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    le_2: "b8[8, 64, 1, 1]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
    where_5: "f32[8, 64, 1, 1]" = torch.ops.aten.where.self(le_2, full_default, getitem_60);  le_2 = getitem_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    sum_7: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(where_5, mean_1, primals_87, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_5 = mean_1 = primals_87 = None
    getitem_63: "f32[8, 256, 1, 1]" = convolution_backward_3[0]
    getitem_64: "f32[64, 256, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_1: "f32[8, 256, 7, 7]" = torch.ops.aten.expand.default(getitem_63, [8, 256, 7, 7]);  getitem_63 = None
    div_33: "f32[8, 256, 7, 7]" = torch.ops.aten.div.Scalar(expand_1, 49);  expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_167: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(mul_231, div_33);  mul_231 = div_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_3: "b8[8, 256, 7, 7]" = torch.ops.aten.lt.Scalar(clone_25, -3)
    le_3: "b8[8, 256, 7, 7]" = torch.ops.aten.le.Scalar(clone_25, 3)
    div_34: "f32[8, 256, 7, 7]" = torch.ops.aten.div.Tensor(clone_25, 3);  clone_25 = None
    add_168: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(div_34, 0.5);  div_34 = None
    mul_233: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(add_167, add_168);  add_168 = None
    where_6: "f32[8, 256, 7, 7]" = torch.ops.aten.where.self(le_3, mul_233, add_167);  le_3 = mul_233 = add_167 = None
    where_7: "f32[8, 256, 7, 7]" = torch.ops.aten.where.self(lt_3, full_default, where_6);  lt_3 = where_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_8: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_31: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_122);  convolution_27 = unsqueeze_122 = None
    mul_234: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, sub_31)
    sum_9: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_234, [0, 2, 3]);  mul_234 = None
    mul_235: "f32[256]" = torch.ops.aten.mul.Tensor(sum_8, 0.002551020408163265)
    unsqueeze_123: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_235, 0);  mul_235 = None
    unsqueeze_124: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_123, 2);  unsqueeze_123 = None
    unsqueeze_125: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, 3);  unsqueeze_124 = None
    mul_236: "f32[256]" = torch.ops.aten.mul.Tensor(sum_9, 0.002551020408163265)
    mul_237: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_238: "f32[256]" = torch.ops.aten.mul.Tensor(mul_236, mul_237);  mul_236 = mul_237 = None
    unsqueeze_126: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_238, 0);  mul_238 = None
    unsqueeze_127: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, 2);  unsqueeze_126 = None
    unsqueeze_128: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_127, 3);  unsqueeze_127 = None
    mul_239: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_51);  primals_51 = None
    unsqueeze_129: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_239, 0);  mul_239 = None
    unsqueeze_130: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_129, 2);  unsqueeze_129 = None
    unsqueeze_131: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, 3);  unsqueeze_130 = None
    mul_240: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_128);  sub_31 = unsqueeze_128 = None
    sub_33: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(where_7, mul_240);  where_7 = mul_240 = None
    sub_34: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(sub_33, unsqueeze_125);  sub_33 = unsqueeze_125 = None
    mul_241: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_131);  sub_34 = unsqueeze_131 = None
    mul_242: "f32[256]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_76);  sum_9 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_241, div_25, primals_86, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 256, [True, True, False]);  mul_241 = div_25 = primals_86 = None
    getitem_66: "f32[8, 256, 7, 7]" = convolution_backward_4[0]
    getitem_67: "f32[256, 1, 5, 5]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_4: "b8[8, 256, 7, 7]" = torch.ops.aten.lt.Scalar(clone_24, -3)
    le_4: "b8[8, 256, 7, 7]" = torch.ops.aten.le.Scalar(clone_24, 3)
    div_35: "f32[8, 256, 7, 7]" = torch.ops.aten.div.Tensor(clone_24, 3);  clone_24 = None
    add_169: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(div_35, 0.5);  div_35 = None
    mul_243: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_66, add_169);  add_169 = None
    where_8: "f32[8, 256, 7, 7]" = torch.ops.aten.where.self(le_4, mul_243, getitem_66);  le_4 = mul_243 = getitem_66 = None
    where_9: "f32[8, 256, 7, 7]" = torch.ops.aten.where.self(lt_4, full_default, where_8);  lt_4 = where_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_10: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_35: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_134);  convolution_26 = unsqueeze_134 = None
    mul_244: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(where_9, sub_35)
    sum_11: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_244, [0, 2, 3]);  mul_244 = None
    mul_245: "f32[256]" = torch.ops.aten.mul.Tensor(sum_10, 0.002551020408163265)
    unsqueeze_135: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_245, 0);  mul_245 = None
    unsqueeze_136: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_135, 2);  unsqueeze_135 = None
    unsqueeze_137: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, 3);  unsqueeze_136 = None
    mul_246: "f32[256]" = torch.ops.aten.mul.Tensor(sum_11, 0.002551020408163265)
    mul_247: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_248: "f32[256]" = torch.ops.aten.mul.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
    unsqueeze_138: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_248, 0);  mul_248 = None
    unsqueeze_139: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, 2);  unsqueeze_138 = None
    unsqueeze_140: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_139, 3);  unsqueeze_139 = None
    mul_249: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_49);  primals_49 = None
    unsqueeze_141: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_249, 0);  mul_249 = None
    unsqueeze_142: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_141, 2);  unsqueeze_141 = None
    unsqueeze_143: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, 3);  unsqueeze_142 = None
    mul_250: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_140);  sub_35 = unsqueeze_140 = None
    sub_37: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(where_9, mul_250);  where_9 = mul_250 = None
    sub_38: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(sub_37, unsqueeze_137);  sub_37 = unsqueeze_137 = None
    mul_251: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_143);  sub_38 = unsqueeze_143 = None
    mul_252: "f32[256]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_73);  sum_11 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_251, mul_192, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_251 = mul_192 = primals_85 = None
    getitem_69: "f32[8, 128, 7, 7]" = convolution_backward_5[0]
    getitem_70: "f32[256, 128, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_253: "f32[8, 128, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_69, div_23);  div_23 = None
    mul_254: "f32[8, 128, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_69, div_24);  getitem_69 = div_24 = None
    sum_12: "f32[8, 128, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_253, [2, 3], True);  mul_253 = None
    mul_255: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sum_12, 0.16666666666666666);  sum_12 = None
    where_10: "f32[8, 128, 1, 1]" = torch.ops.aten.where.self(bitwise_and_1, mul_255, full_default);  bitwise_and_1 = mul_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    sum_13: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(where_10, relu, primals_83, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_10 = primals_83 = None
    getitem_72: "f32[8, 32, 1, 1]" = convolution_backward_6[0]
    getitem_73: "f32[128, 32, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    le_5: "b8[8, 32, 1, 1]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
    where_11: "f32[8, 32, 1, 1]" = torch.ops.aten.where.self(le_5, full_default, getitem_72);  le_5 = getitem_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    sum_14: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(where_11, mean, primals_81, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_11 = mean = primals_81 = None
    getitem_75: "f32[8, 128, 1, 1]" = convolution_backward_7[0]
    getitem_76: "f32[32, 128, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_2: "f32[8, 128, 7, 7]" = torch.ops.aten.expand.default(getitem_75, [8, 128, 7, 7]);  getitem_75 = None
    div_36: "f32[8, 128, 7, 7]" = torch.ops.aten.div.Scalar(expand_2, 49);  expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_170: "f32[8, 128, 7, 7]" = torch.ops.aten.add.Tensor(mul_254, div_36);  mul_254 = div_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_6: "b8[8, 128, 7, 7]" = torch.ops.aten.lt.Scalar(clone_23, -3)
    le_6: "b8[8, 128, 7, 7]" = torch.ops.aten.le.Scalar(clone_23, 3)
    div_37: "f32[8, 128, 7, 7]" = torch.ops.aten.div.Tensor(clone_23, 3);  clone_23 = None
    add_171: "f32[8, 128, 7, 7]" = torch.ops.aten.add.Tensor(div_37, 0.5);  div_37 = None
    mul_256: "f32[8, 128, 7, 7]" = torch.ops.aten.mul.Tensor(add_170, add_171);  add_171 = None
    where_12: "f32[8, 128, 7, 7]" = torch.ops.aten.where.self(le_6, mul_256, add_170);  le_6 = mul_256 = add_170 = None
    where_13: "f32[8, 128, 7, 7]" = torch.ops.aten.where.self(lt_6, full_default, where_12);  lt_6 = where_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_15: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_39: "f32[8, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_146);  convolution_23 = unsqueeze_146 = None
    mul_257: "f32[8, 128, 7, 7]" = torch.ops.aten.mul.Tensor(where_13, sub_39)
    sum_16: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_257, [0, 2, 3]);  mul_257 = None
    mul_258: "f32[128]" = torch.ops.aten.mul.Tensor(sum_15, 0.002551020408163265)
    unsqueeze_147: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_258, 0);  mul_258 = None
    unsqueeze_148: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_147, 2);  unsqueeze_147 = None
    unsqueeze_149: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, 3);  unsqueeze_148 = None
    mul_259: "f32[128]" = torch.ops.aten.mul.Tensor(sum_16, 0.002551020408163265)
    mul_260: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_261: "f32[128]" = torch.ops.aten.mul.Tensor(mul_259, mul_260);  mul_259 = mul_260 = None
    unsqueeze_150: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_261, 0);  mul_261 = None
    unsqueeze_151: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, 2);  unsqueeze_150 = None
    unsqueeze_152: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_151, 3);  unsqueeze_151 = None
    mul_262: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_47);  primals_47 = None
    unsqueeze_153: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_262, 0);  mul_262 = None
    unsqueeze_154: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_153, 2);  unsqueeze_153 = None
    unsqueeze_155: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, 3);  unsqueeze_154 = None
    mul_263: "f32[8, 128, 7, 7]" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_152);  sub_39 = unsqueeze_152 = None
    sub_41: "f32[8, 128, 7, 7]" = torch.ops.aten.sub.Tensor(where_13, mul_263);  where_13 = mul_263 = None
    sub_42: "f32[8, 128, 7, 7]" = torch.ops.aten.sub.Tensor(sub_41, unsqueeze_149);  sub_41 = unsqueeze_149 = None
    mul_264: "f32[8, 128, 7, 7]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_155);  sub_42 = unsqueeze_155 = None
    mul_265: "f32[128]" = torch.ops.aten.mul.Tensor(sum_16, squeeze_70);  sum_16 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_264, div_22, primals_80, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 128, [True, True, False]);  mul_264 = div_22 = primals_80 = None
    getitem_78: "f32[8, 128, 14, 14]" = convolution_backward_8[0]
    getitem_79: "f32[128, 1, 5, 5]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_7: "b8[8, 128, 14, 14]" = torch.ops.aten.lt.Scalar(clone_22, -3)
    le_7: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(clone_22, 3)
    div_38: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(clone_22, 3);  clone_22 = None
    add_172: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(div_38, 0.5);  div_38 = None
    mul_266: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_78, add_172);  add_172 = None
    where_14: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_7, mul_266, getitem_78);  le_7 = mul_266 = getitem_78 = None
    where_15: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(lt_7, full_default, where_14);  lt_7 = where_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_17: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_43: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_158);  convolution_22 = unsqueeze_158 = None
    mul_267: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, sub_43)
    sum_18: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_267, [0, 2, 3]);  mul_267 = None
    mul_268: "f32[128]" = torch.ops.aten.mul.Tensor(sum_17, 0.0006377551020408163)
    unsqueeze_159: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_268, 0);  mul_268 = None
    unsqueeze_160: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_159, 2);  unsqueeze_159 = None
    unsqueeze_161: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, 3);  unsqueeze_160 = None
    mul_269: "f32[128]" = torch.ops.aten.mul.Tensor(sum_18, 0.0006377551020408163)
    mul_270: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_271: "f32[128]" = torch.ops.aten.mul.Tensor(mul_269, mul_270);  mul_269 = mul_270 = None
    unsqueeze_162: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_271, 0);  mul_271 = None
    unsqueeze_163: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, 2);  unsqueeze_162 = None
    unsqueeze_164: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_163, 3);  unsqueeze_163 = None
    mul_272: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_45);  primals_45 = None
    unsqueeze_165: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_272, 0);  mul_272 = None
    unsqueeze_166: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_165, 2);  unsqueeze_165 = None
    unsqueeze_167: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, 3);  unsqueeze_166 = None
    mul_273: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_164);  sub_43 = unsqueeze_164 = None
    sub_45: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_15, mul_273);  where_15 = mul_273 = None
    sub_46: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_45, unsqueeze_161);  sub_45 = unsqueeze_161 = None
    mul_274: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_167);  sub_46 = unsqueeze_167 = None
    mul_275: "f32[128]" = torch.ops.aten.mul.Tensor(sum_18, squeeze_67);  sum_18 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_274, div_21, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_274 = div_21 = primals_79 = None
    getitem_81: "f32[8, 128, 14, 14]" = convolution_backward_9[0]
    getitem_82: "f32[128, 128, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_8: "b8[8, 128, 14, 14]" = torch.ops.aten.lt.Scalar(clone_21, -3)
    le_8: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(clone_21, 3)
    div_39: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(clone_21, 3);  clone_21 = None
    add_173: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(div_39, 0.5);  div_39 = None
    mul_276: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_81, add_173);  add_173 = None
    where_16: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_8, mul_276, getitem_81);  le_8 = mul_276 = getitem_81 = None
    where_17: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(lt_8, full_default, where_16);  lt_8 = where_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_19: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_47: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_170);  convolution_21 = unsqueeze_170 = None
    mul_277: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_17, sub_47)
    sum_20: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_277, [0, 2, 3]);  mul_277 = None
    mul_278: "f32[128]" = torch.ops.aten.mul.Tensor(sum_19, 0.0006377551020408163)
    unsqueeze_171: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_278, 0);  mul_278 = None
    unsqueeze_172: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_171, 2);  unsqueeze_171 = None
    unsqueeze_173: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, 3);  unsqueeze_172 = None
    mul_279: "f32[128]" = torch.ops.aten.mul.Tensor(sum_20, 0.0006377551020408163)
    mul_280: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_281: "f32[128]" = torch.ops.aten.mul.Tensor(mul_279, mul_280);  mul_279 = mul_280 = None
    unsqueeze_174: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_281, 0);  mul_281 = None
    unsqueeze_175: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, 2);  unsqueeze_174 = None
    unsqueeze_176: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_175, 3);  unsqueeze_175 = None
    mul_282: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_43);  primals_43 = None
    unsqueeze_177: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_282, 0);  mul_282 = None
    unsqueeze_178: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_177, 2);  unsqueeze_177 = None
    unsqueeze_179: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, 3);  unsqueeze_178 = None
    mul_283: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_176);  sub_47 = unsqueeze_176 = None
    sub_49: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_17, mul_283);  where_17 = mul_283 = None
    sub_50: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_49, unsqueeze_173);  sub_49 = unsqueeze_173 = None
    mul_284: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_179);  sub_50 = unsqueeze_179 = None
    mul_285: "f32[128]" = torch.ops.aten.mul.Tensor(sum_20, squeeze_64);  sum_20 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_284, div_20, primals_78, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 128, [True, True, False]);  mul_284 = div_20 = primals_78 = None
    getitem_84: "f32[8, 128, 14, 14]" = convolution_backward_10[0]
    getitem_85: "f32[128, 1, 5, 5]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_9: "b8[8, 128, 14, 14]" = torch.ops.aten.lt.Scalar(clone_20, -3)
    le_9: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(clone_20, 3)
    div_40: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(clone_20, 3);  clone_20 = None
    add_174: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(div_40, 0.5);  div_40 = None
    mul_286: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_84, add_174);  add_174 = None
    where_18: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_9, mul_286, getitem_84);  le_9 = mul_286 = getitem_84 = None
    where_19: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(lt_9, full_default, where_18);  lt_9 = where_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_21: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_51: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_182);  convolution_20 = unsqueeze_182 = None
    mul_287: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_19, sub_51)
    sum_22: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_287, [0, 2, 3]);  mul_287 = None
    mul_288: "f32[128]" = torch.ops.aten.mul.Tensor(sum_21, 0.0006377551020408163)
    unsqueeze_183: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_288, 0);  mul_288 = None
    unsqueeze_184: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_183, 2);  unsqueeze_183 = None
    unsqueeze_185: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, 3);  unsqueeze_184 = None
    mul_289: "f32[128]" = torch.ops.aten.mul.Tensor(sum_22, 0.0006377551020408163)
    mul_290: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_291: "f32[128]" = torch.ops.aten.mul.Tensor(mul_289, mul_290);  mul_289 = mul_290 = None
    unsqueeze_186: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_291, 0);  mul_291 = None
    unsqueeze_187: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, 2);  unsqueeze_186 = None
    unsqueeze_188: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_187, 3);  unsqueeze_187 = None
    mul_292: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_41);  primals_41 = None
    unsqueeze_189: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_292, 0);  mul_292 = None
    unsqueeze_190: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_189, 2);  unsqueeze_189 = None
    unsqueeze_191: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, 3);  unsqueeze_190 = None
    mul_293: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_188);  sub_51 = unsqueeze_188 = None
    sub_53: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_19, mul_293);  where_19 = mul_293 = None
    sub_54: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_53, unsqueeze_185);  sub_53 = unsqueeze_185 = None
    mul_294: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_191);  sub_54 = unsqueeze_191 = None
    mul_295: "f32[128]" = torch.ops.aten.mul.Tensor(sum_22, squeeze_61);  sum_22 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_294, div_19, primals_77, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_294 = div_19 = primals_77 = None
    getitem_87: "f32[8, 128, 14, 14]" = convolution_backward_11[0]
    getitem_88: "f32[128, 128, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_10: "b8[8, 128, 14, 14]" = torch.ops.aten.lt.Scalar(clone_19, -3)
    le_10: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(clone_19, 3)
    div_41: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(clone_19, 3);  clone_19 = None
    add_175: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(div_41, 0.5);  div_41 = None
    mul_296: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_87, add_175);  add_175 = None
    where_20: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_10, mul_296, getitem_87);  le_10 = mul_296 = getitem_87 = None
    where_21: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(lt_10, full_default, where_20);  lt_10 = where_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_23: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_55: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_194);  convolution_19 = unsqueeze_194 = None
    mul_297: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_21, sub_55)
    sum_24: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_297, [0, 2, 3]);  mul_297 = None
    mul_298: "f32[128]" = torch.ops.aten.mul.Tensor(sum_23, 0.0006377551020408163)
    unsqueeze_195: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_298, 0);  mul_298 = None
    unsqueeze_196: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_195, 2);  unsqueeze_195 = None
    unsqueeze_197: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, 3);  unsqueeze_196 = None
    mul_299: "f32[128]" = torch.ops.aten.mul.Tensor(sum_24, 0.0006377551020408163)
    mul_300: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_301: "f32[128]" = torch.ops.aten.mul.Tensor(mul_299, mul_300);  mul_299 = mul_300 = None
    unsqueeze_198: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_301, 0);  mul_301 = None
    unsqueeze_199: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, 2);  unsqueeze_198 = None
    unsqueeze_200: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_199, 3);  unsqueeze_199 = None
    mul_302: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_39);  primals_39 = None
    unsqueeze_201: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_302, 0);  mul_302 = None
    unsqueeze_202: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_201, 2);  unsqueeze_201 = None
    unsqueeze_203: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, 3);  unsqueeze_202 = None
    mul_303: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_200);  sub_55 = unsqueeze_200 = None
    sub_57: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_21, mul_303);  where_21 = mul_303 = None
    sub_58: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_57, unsqueeze_197);  sub_57 = unsqueeze_197 = None
    mul_304: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_203);  sub_58 = unsqueeze_203 = None
    mul_305: "f32[128]" = torch.ops.aten.mul.Tensor(sum_24, squeeze_58);  sum_24 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_304, div_18, primals_76, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 128, [True, True, False]);  mul_304 = div_18 = primals_76 = None
    getitem_90: "f32[8, 128, 14, 14]" = convolution_backward_12[0]
    getitem_91: "f32[128, 1, 5, 5]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_11: "b8[8, 128, 14, 14]" = torch.ops.aten.lt.Scalar(clone_18, -3)
    le_11: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(clone_18, 3)
    div_42: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(clone_18, 3);  clone_18 = None
    add_176: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(div_42, 0.5);  div_42 = None
    mul_306: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_90, add_176);  add_176 = None
    where_22: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_11, mul_306, getitem_90);  le_11 = mul_306 = getitem_90 = None
    where_23: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(lt_11, full_default, where_22);  lt_11 = where_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_25: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_59: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_206);  convolution_18 = unsqueeze_206 = None
    mul_307: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_23, sub_59)
    sum_26: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_307, [0, 2, 3]);  mul_307 = None
    mul_308: "f32[128]" = torch.ops.aten.mul.Tensor(sum_25, 0.0006377551020408163)
    unsqueeze_207: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_308, 0);  mul_308 = None
    unsqueeze_208: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_207, 2);  unsqueeze_207 = None
    unsqueeze_209: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, 3);  unsqueeze_208 = None
    mul_309: "f32[128]" = torch.ops.aten.mul.Tensor(sum_26, 0.0006377551020408163)
    mul_310: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_311: "f32[128]" = torch.ops.aten.mul.Tensor(mul_309, mul_310);  mul_309 = mul_310 = None
    unsqueeze_210: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_311, 0);  mul_311 = None
    unsqueeze_211: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, 2);  unsqueeze_210 = None
    unsqueeze_212: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 3);  unsqueeze_211 = None
    mul_312: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_37);  primals_37 = None
    unsqueeze_213: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_312, 0);  mul_312 = None
    unsqueeze_214: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 2);  unsqueeze_213 = None
    unsqueeze_215: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, 3);  unsqueeze_214 = None
    mul_313: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_212);  sub_59 = unsqueeze_212 = None
    sub_61: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_23, mul_313);  where_23 = mul_313 = None
    sub_62: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_61, unsqueeze_209);  sub_61 = unsqueeze_209 = None
    mul_314: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_215);  sub_62 = unsqueeze_215 = None
    mul_315: "f32[128]" = torch.ops.aten.mul.Tensor(sum_26, squeeze_55);  sum_26 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_314, div_17, primals_75, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_314 = div_17 = primals_75 = None
    getitem_93: "f32[8, 128, 14, 14]" = convolution_backward_13[0]
    getitem_94: "f32[128, 128, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_12: "b8[8, 128, 14, 14]" = torch.ops.aten.lt.Scalar(clone_17, -3)
    le_12: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(clone_17, 3)
    div_43: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(clone_17, 3);  clone_17 = None
    add_177: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(div_43, 0.5);  div_43 = None
    mul_316: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_93, add_177);  add_177 = None
    where_24: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_12, mul_316, getitem_93);  le_12 = mul_316 = getitem_93 = None
    where_25: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(lt_12, full_default, where_24);  lt_12 = where_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_27: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_63: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_218);  convolution_17 = unsqueeze_218 = None
    mul_317: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_25, sub_63)
    sum_28: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_317, [0, 2, 3]);  mul_317 = None
    mul_318: "f32[128]" = torch.ops.aten.mul.Tensor(sum_27, 0.0006377551020408163)
    unsqueeze_219: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_318, 0);  mul_318 = None
    unsqueeze_220: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_219, 2);  unsqueeze_219 = None
    unsqueeze_221: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, 3);  unsqueeze_220 = None
    mul_319: "f32[128]" = torch.ops.aten.mul.Tensor(sum_28, 0.0006377551020408163)
    mul_320: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_321: "f32[128]" = torch.ops.aten.mul.Tensor(mul_319, mul_320);  mul_319 = mul_320 = None
    unsqueeze_222: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_321, 0);  mul_321 = None
    unsqueeze_223: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, 2);  unsqueeze_222 = None
    unsqueeze_224: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 3);  unsqueeze_223 = None
    mul_322: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_35);  primals_35 = None
    unsqueeze_225: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_322, 0);  mul_322 = None
    unsqueeze_226: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_225, 2);  unsqueeze_225 = None
    unsqueeze_227: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 3);  unsqueeze_226 = None
    mul_323: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_224);  sub_63 = unsqueeze_224 = None
    sub_65: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_25, mul_323);  where_25 = mul_323 = None
    sub_66: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_65, unsqueeze_221);  sub_65 = unsqueeze_221 = None
    mul_324: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_227);  sub_66 = unsqueeze_227 = None
    mul_325: "f32[128]" = torch.ops.aten.mul.Tensor(sum_28, squeeze_52);  sum_28 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_324, div_16, primals_74, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 128, [True, True, False]);  mul_324 = div_16 = primals_74 = None
    getitem_96: "f32[8, 128, 14, 14]" = convolution_backward_14[0]
    getitem_97: "f32[128, 1, 5, 5]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_13: "b8[8, 128, 14, 14]" = torch.ops.aten.lt.Scalar(clone_16, -3)
    le_13: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(clone_16, 3)
    div_44: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(clone_16, 3);  clone_16 = None
    add_178: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(div_44, 0.5);  div_44 = None
    mul_326: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_96, add_178);  add_178 = None
    where_26: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_13, mul_326, getitem_96);  le_13 = mul_326 = getitem_96 = None
    where_27: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(lt_13, full_default, where_26);  lt_13 = where_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_29: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_67: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_230);  convolution_16 = unsqueeze_230 = None
    mul_327: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_27, sub_67)
    sum_30: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_327, [0, 2, 3]);  mul_327 = None
    mul_328: "f32[128]" = torch.ops.aten.mul.Tensor(sum_29, 0.0006377551020408163)
    unsqueeze_231: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_328, 0);  mul_328 = None
    unsqueeze_232: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_231, 2);  unsqueeze_231 = None
    unsqueeze_233: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 3);  unsqueeze_232 = None
    mul_329: "f32[128]" = torch.ops.aten.mul.Tensor(sum_30, 0.0006377551020408163)
    mul_330: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_331: "f32[128]" = torch.ops.aten.mul.Tensor(mul_329, mul_330);  mul_329 = mul_330 = None
    unsqueeze_234: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_331, 0);  mul_331 = None
    unsqueeze_235: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, 2);  unsqueeze_234 = None
    unsqueeze_236: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 3);  unsqueeze_235 = None
    mul_332: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_33);  primals_33 = None
    unsqueeze_237: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_332, 0);  mul_332 = None
    unsqueeze_238: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_237, 2);  unsqueeze_237 = None
    unsqueeze_239: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 3);  unsqueeze_238 = None
    mul_333: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_236);  sub_67 = unsqueeze_236 = None
    sub_69: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_27, mul_333);  where_27 = mul_333 = None
    sub_70: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_69, unsqueeze_233);  sub_69 = unsqueeze_233 = None
    mul_334: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_239);  sub_70 = unsqueeze_239 = None
    mul_335: "f32[128]" = torch.ops.aten.mul.Tensor(sum_30, squeeze_49);  sum_30 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_334, div_15, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_334 = div_15 = primals_73 = None
    getitem_99: "f32[8, 128, 14, 14]" = convolution_backward_15[0]
    getitem_100: "f32[128, 128, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_14: "b8[8, 128, 14, 14]" = torch.ops.aten.lt.Scalar(clone_15, -3)
    le_14: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(clone_15, 3)
    div_45: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(clone_15, 3);  clone_15 = None
    add_179: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(div_45, 0.5);  div_45 = None
    mul_336: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_99, add_179);  add_179 = None
    where_28: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_14, mul_336, getitem_99);  le_14 = mul_336 = getitem_99 = None
    where_29: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(lt_14, full_default, where_28);  lt_14 = where_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_31: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    sub_71: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_242);  convolution_15 = unsqueeze_242 = None
    mul_337: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_29, sub_71)
    sum_32: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_337, [0, 2, 3]);  mul_337 = None
    mul_338: "f32[128]" = torch.ops.aten.mul.Tensor(sum_31, 0.0006377551020408163)
    unsqueeze_243: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_338, 0);  mul_338 = None
    unsqueeze_244: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_243, 2);  unsqueeze_243 = None
    unsqueeze_245: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 3);  unsqueeze_244 = None
    mul_339: "f32[128]" = torch.ops.aten.mul.Tensor(sum_32, 0.0006377551020408163)
    mul_340: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_341: "f32[128]" = torch.ops.aten.mul.Tensor(mul_339, mul_340);  mul_339 = mul_340 = None
    unsqueeze_246: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_341, 0);  mul_341 = None
    unsqueeze_247: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, 2);  unsqueeze_246 = None
    unsqueeze_248: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 3);  unsqueeze_247 = None
    mul_342: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_31);  primals_31 = None
    unsqueeze_249: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_342, 0);  mul_342 = None
    unsqueeze_250: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_249, 2);  unsqueeze_249 = None
    unsqueeze_251: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 3);  unsqueeze_250 = None
    mul_343: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_248);  sub_71 = unsqueeze_248 = None
    sub_73: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_29, mul_343);  where_29 = mul_343 = None
    sub_74: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_73, unsqueeze_245);  sub_73 = unsqueeze_245 = None
    mul_344: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_251);  sub_74 = unsqueeze_251 = None
    mul_345: "f32[128]" = torch.ops.aten.mul.Tensor(sum_32, squeeze_46);  sum_32 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_344, div_14, primals_72, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 128, [True, True, False]);  mul_344 = div_14 = primals_72 = None
    getitem_102: "f32[8, 128, 14, 14]" = convolution_backward_16[0]
    getitem_103: "f32[128, 1, 5, 5]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_15: "b8[8, 128, 14, 14]" = torch.ops.aten.lt.Scalar(clone_14, -3)
    le_15: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(clone_14, 3)
    div_46: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(clone_14, 3);  clone_14 = None
    add_180: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(div_46, 0.5);  div_46 = None
    mul_346: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_102, add_180);  add_180 = None
    where_30: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_15, mul_346, getitem_102);  le_15 = mul_346 = getitem_102 = None
    where_31: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(lt_15, full_default, where_30);  lt_15 = where_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_33: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_75: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_254);  convolution_14 = unsqueeze_254 = None
    mul_347: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_31, sub_75)
    sum_34: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_347, [0, 2, 3]);  mul_347 = None
    mul_348: "f32[128]" = torch.ops.aten.mul.Tensor(sum_33, 0.0006377551020408163)
    unsqueeze_255: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_348, 0);  mul_348 = None
    unsqueeze_256: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 2);  unsqueeze_255 = None
    unsqueeze_257: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, 3);  unsqueeze_256 = None
    mul_349: "f32[128]" = torch.ops.aten.mul.Tensor(sum_34, 0.0006377551020408163)
    mul_350: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_351: "f32[128]" = torch.ops.aten.mul.Tensor(mul_349, mul_350);  mul_349 = mul_350 = None
    unsqueeze_258: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_351, 0);  mul_351 = None
    unsqueeze_259: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, 2);  unsqueeze_258 = None
    unsqueeze_260: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 3);  unsqueeze_259 = None
    mul_352: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_29);  primals_29 = None
    unsqueeze_261: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_352, 0);  mul_352 = None
    unsqueeze_262: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 2);  unsqueeze_261 = None
    unsqueeze_263: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, 3);  unsqueeze_262 = None
    mul_353: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_260);  sub_75 = unsqueeze_260 = None
    sub_77: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_31, mul_353);  where_31 = mul_353 = None
    sub_78: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_77, unsqueeze_257);  sub_77 = unsqueeze_257 = None
    mul_354: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_263);  sub_78 = unsqueeze_263 = None
    mul_355: "f32[128]" = torch.ops.aten.mul.Tensor(sum_34, squeeze_43);  sum_34 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_354, div_13, primals_71, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_354 = div_13 = primals_71 = None
    getitem_105: "f32[8, 128, 14, 14]" = convolution_backward_17[0]
    getitem_106: "f32[128, 128, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_16: "b8[8, 128, 14, 14]" = torch.ops.aten.lt.Scalar(clone_13, -3)
    le_16: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(clone_13, 3)
    div_47: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(clone_13, 3);  clone_13 = None
    add_181: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(div_47, 0.5);  div_47 = None
    mul_356: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_105, add_181);  add_181 = None
    where_32: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_16, mul_356, getitem_105);  le_16 = mul_356 = getitem_105 = None
    where_33: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(lt_16, full_default, where_32);  lt_16 = where_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_35: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
    sub_79: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_266);  convolution_13 = unsqueeze_266 = None
    mul_357: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_33, sub_79)
    sum_36: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_357, [0, 2, 3]);  mul_357 = None
    mul_358: "f32[128]" = torch.ops.aten.mul.Tensor(sum_35, 0.0006377551020408163)
    unsqueeze_267: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_358, 0);  mul_358 = None
    unsqueeze_268: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 2);  unsqueeze_267 = None
    unsqueeze_269: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, 3);  unsqueeze_268 = None
    mul_359: "f32[128]" = torch.ops.aten.mul.Tensor(sum_36, 0.0006377551020408163)
    mul_360: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_361: "f32[128]" = torch.ops.aten.mul.Tensor(mul_359, mul_360);  mul_359 = mul_360 = None
    unsqueeze_270: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_361, 0);  mul_361 = None
    unsqueeze_271: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 2);  unsqueeze_270 = None
    unsqueeze_272: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 3);  unsqueeze_271 = None
    mul_362: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_27);  primals_27 = None
    unsqueeze_273: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_362, 0);  mul_362 = None
    unsqueeze_274: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 2);  unsqueeze_273 = None
    unsqueeze_275: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 3);  unsqueeze_274 = None
    mul_363: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_272);  sub_79 = unsqueeze_272 = None
    sub_81: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_33, mul_363);  where_33 = mul_363 = None
    sub_82: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_81, unsqueeze_269);  sub_81 = unsqueeze_269 = None
    mul_364: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_275);  sub_82 = unsqueeze_275 = None
    mul_365: "f32[128]" = torch.ops.aten.mul.Tensor(sum_36, squeeze_40);  sum_36 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_364, div_12, primals_70, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 128, [True, True, False]);  mul_364 = div_12 = primals_70 = None
    getitem_108: "f32[8, 128, 14, 14]" = convolution_backward_18[0]
    getitem_109: "f32[128, 1, 5, 5]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_17: "b8[8, 128, 14, 14]" = torch.ops.aten.lt.Scalar(clone_12, -3)
    le_17: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(clone_12, 3)
    div_48: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(clone_12, 3);  clone_12 = None
    add_182: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(div_48, 0.5);  div_48 = None
    mul_366: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_108, add_182);  add_182 = None
    where_34: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_17, mul_366, getitem_108);  le_17 = mul_366 = getitem_108 = None
    where_35: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(lt_17, full_default, where_34);  lt_17 = where_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_37: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
    sub_83: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_278);  convolution_12 = unsqueeze_278 = None
    mul_367: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_35, sub_83)
    sum_38: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_367, [0, 2, 3]);  mul_367 = None
    mul_368: "f32[128]" = torch.ops.aten.mul.Tensor(sum_37, 0.0006377551020408163)
    unsqueeze_279: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_368, 0);  mul_368 = None
    unsqueeze_280: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 2);  unsqueeze_279 = None
    unsqueeze_281: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 3);  unsqueeze_280 = None
    mul_369: "f32[128]" = torch.ops.aten.mul.Tensor(sum_38, 0.0006377551020408163)
    mul_370: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_371: "f32[128]" = torch.ops.aten.mul.Tensor(mul_369, mul_370);  mul_369 = mul_370 = None
    unsqueeze_282: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_371, 0);  mul_371 = None
    unsqueeze_283: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, 2);  unsqueeze_282 = None
    unsqueeze_284: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 3);  unsqueeze_283 = None
    mul_372: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_25);  primals_25 = None
    unsqueeze_285: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_372, 0);  mul_372 = None
    unsqueeze_286: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 2);  unsqueeze_285 = None
    unsqueeze_287: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 3);  unsqueeze_286 = None
    mul_373: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_284);  sub_83 = unsqueeze_284 = None
    sub_85: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_35, mul_373);  where_35 = mul_373 = None
    sub_86: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_85, unsqueeze_281);  sub_85 = unsqueeze_281 = None
    mul_374: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_287);  sub_86 = unsqueeze_287 = None
    mul_375: "f32[128]" = torch.ops.aten.mul.Tensor(sum_38, squeeze_37);  sum_38 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_374, div_11, primals_69, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_374 = div_11 = primals_69 = None
    getitem_111: "f32[8, 64, 14, 14]" = convolution_backward_19[0]
    getitem_112: "f32[128, 64, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_18: "b8[8, 64, 14, 14]" = torch.ops.aten.lt.Scalar(clone_11, -3)
    le_18: "b8[8, 64, 14, 14]" = torch.ops.aten.le.Scalar(clone_11, 3)
    div_49: "f32[8, 64, 14, 14]" = torch.ops.aten.div.Tensor(clone_11, 3);  clone_11 = None
    add_183: "f32[8, 64, 14, 14]" = torch.ops.aten.add.Tensor(div_49, 0.5);  div_49 = None
    mul_376: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_111, add_183);  add_183 = None
    where_36: "f32[8, 64, 14, 14]" = torch.ops.aten.where.self(le_18, mul_376, getitem_111);  le_18 = mul_376 = getitem_111 = None
    where_37: "f32[8, 64, 14, 14]" = torch.ops.aten.where.self(lt_18, full_default, where_36);  lt_18 = where_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_39: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
    sub_87: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_290);  convolution_11 = unsqueeze_290 = None
    mul_377: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(where_37, sub_87)
    sum_40: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_377, [0, 2, 3]);  mul_377 = None
    mul_378: "f32[64]" = torch.ops.aten.mul.Tensor(sum_39, 0.0006377551020408163)
    unsqueeze_291: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_378, 0);  mul_378 = None
    unsqueeze_292: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 2);  unsqueeze_291 = None
    unsqueeze_293: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 3);  unsqueeze_292 = None
    mul_379: "f32[64]" = torch.ops.aten.mul.Tensor(sum_40, 0.0006377551020408163)
    mul_380: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_381: "f32[64]" = torch.ops.aten.mul.Tensor(mul_379, mul_380);  mul_379 = mul_380 = None
    unsqueeze_294: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_381, 0);  mul_381 = None
    unsqueeze_295: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, 2);  unsqueeze_294 = None
    unsqueeze_296: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 3);  unsqueeze_295 = None
    mul_382: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_23);  primals_23 = None
    unsqueeze_297: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_382, 0);  mul_382 = None
    unsqueeze_298: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 2);  unsqueeze_297 = None
    unsqueeze_299: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 3);  unsqueeze_298 = None
    mul_383: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_296);  sub_87 = unsqueeze_296 = None
    sub_89: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(where_37, mul_383);  where_37 = mul_383 = None
    sub_90: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(sub_89, unsqueeze_293);  sub_89 = unsqueeze_293 = None
    mul_384: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_299);  sub_90 = unsqueeze_299 = None
    mul_385: "f32[64]" = torch.ops.aten.mul.Tensor(sum_40, squeeze_34);  sum_40 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_384, div_10, primals_68, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False]);  mul_384 = div_10 = primals_68 = None
    getitem_114: "f32[8, 64, 28, 28]" = convolution_backward_20[0]
    getitem_115: "f32[64, 1, 3, 3]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_19: "b8[8, 64, 28, 28]" = torch.ops.aten.lt.Scalar(clone_10, -3)
    le_19: "b8[8, 64, 28, 28]" = torch.ops.aten.le.Scalar(clone_10, 3)
    div_50: "f32[8, 64, 28, 28]" = torch.ops.aten.div.Tensor(clone_10, 3);  clone_10 = None
    add_184: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(div_50, 0.5);  div_50 = None
    mul_386: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_114, add_184);  add_184 = None
    where_38: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(le_19, mul_386, getitem_114);  le_19 = mul_386 = getitem_114 = None
    where_39: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(lt_19, full_default, where_38);  lt_19 = where_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_41: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
    sub_91: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_302);  convolution_10 = unsqueeze_302 = None
    mul_387: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(where_39, sub_91)
    sum_42: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_387, [0, 2, 3]);  mul_387 = None
    mul_388: "f32[64]" = torch.ops.aten.mul.Tensor(sum_41, 0.00015943877551020407)
    unsqueeze_303: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_388, 0);  mul_388 = None
    unsqueeze_304: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 2);  unsqueeze_303 = None
    unsqueeze_305: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, 3);  unsqueeze_304 = None
    mul_389: "f32[64]" = torch.ops.aten.mul.Tensor(sum_42, 0.00015943877551020407)
    mul_390: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_391: "f32[64]" = torch.ops.aten.mul.Tensor(mul_389, mul_390);  mul_389 = mul_390 = None
    unsqueeze_306: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_391, 0);  mul_391 = None
    unsqueeze_307: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, 2);  unsqueeze_306 = None
    unsqueeze_308: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 3);  unsqueeze_307 = None
    mul_392: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_21);  primals_21 = None
    unsqueeze_309: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_392, 0);  mul_392 = None
    unsqueeze_310: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 2);  unsqueeze_309 = None
    unsqueeze_311: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 3);  unsqueeze_310 = None
    mul_393: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_308);  sub_91 = unsqueeze_308 = None
    sub_93: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(where_39, mul_393);  where_39 = mul_393 = None
    sub_94: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(sub_93, unsqueeze_305);  sub_93 = unsqueeze_305 = None
    mul_394: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_311);  sub_94 = unsqueeze_311 = None
    mul_395: "f32[64]" = torch.ops.aten.mul.Tensor(sum_42, squeeze_31);  sum_42 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_394, div_9, primals_67, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_394 = div_9 = primals_67 = None
    getitem_117: "f32[8, 64, 28, 28]" = convolution_backward_21[0]
    getitem_118: "f32[64, 64, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_20: "b8[8, 64, 28, 28]" = torch.ops.aten.lt.Scalar(clone_9, -3)
    le_20: "b8[8, 64, 28, 28]" = torch.ops.aten.le.Scalar(clone_9, 3)
    div_51: "f32[8, 64, 28, 28]" = torch.ops.aten.div.Tensor(clone_9, 3);  clone_9 = None
    add_185: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(div_51, 0.5);  div_51 = None
    mul_396: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_117, add_185);  add_185 = None
    where_40: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(le_20, mul_396, getitem_117);  le_20 = mul_396 = getitem_117 = None
    where_41: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(lt_20, full_default, where_40);  lt_20 = where_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_43: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_41, [0, 2, 3])
    sub_95: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_314);  convolution_9 = unsqueeze_314 = None
    mul_397: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(where_41, sub_95)
    sum_44: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_397, [0, 2, 3]);  mul_397 = None
    mul_398: "f32[64]" = torch.ops.aten.mul.Tensor(sum_43, 0.00015943877551020407)
    unsqueeze_315: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_398, 0);  mul_398 = None
    unsqueeze_316: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 2);  unsqueeze_315 = None
    unsqueeze_317: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, 3);  unsqueeze_316 = None
    mul_399: "f32[64]" = torch.ops.aten.mul.Tensor(sum_44, 0.00015943877551020407)
    mul_400: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_401: "f32[64]" = torch.ops.aten.mul.Tensor(mul_399, mul_400);  mul_399 = mul_400 = None
    unsqueeze_318: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_401, 0);  mul_401 = None
    unsqueeze_319: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, 2);  unsqueeze_318 = None
    unsqueeze_320: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 3);  unsqueeze_319 = None
    mul_402: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_19);  primals_19 = None
    unsqueeze_321: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_402, 0);  mul_402 = None
    unsqueeze_322: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 2);  unsqueeze_321 = None
    unsqueeze_323: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 3);  unsqueeze_322 = None
    mul_403: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_320);  sub_95 = unsqueeze_320 = None
    sub_97: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(where_41, mul_403);  where_41 = mul_403 = None
    sub_98: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(sub_97, unsqueeze_317);  sub_97 = unsqueeze_317 = None
    mul_404: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_323);  sub_98 = unsqueeze_323 = None
    mul_405: "f32[64]" = torch.ops.aten.mul.Tensor(sum_44, squeeze_28);  sum_44 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_404, div_8, primals_66, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False]);  mul_404 = div_8 = primals_66 = None
    getitem_120: "f32[8, 64, 28, 28]" = convolution_backward_22[0]
    getitem_121: "f32[64, 1, 3, 3]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_21: "b8[8, 64, 28, 28]" = torch.ops.aten.lt.Scalar(clone_8, -3)
    le_21: "b8[8, 64, 28, 28]" = torch.ops.aten.le.Scalar(clone_8, 3)
    div_52: "f32[8, 64, 28, 28]" = torch.ops.aten.div.Tensor(clone_8, 3);  clone_8 = None
    add_186: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(div_52, 0.5);  div_52 = None
    mul_406: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_120, add_186);  add_186 = None
    where_42: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(le_21, mul_406, getitem_120);  le_21 = mul_406 = getitem_120 = None
    where_43: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(lt_21, full_default, where_42);  lt_21 = where_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_45: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
    sub_99: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_326);  convolution_8 = unsqueeze_326 = None
    mul_407: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(where_43, sub_99)
    sum_46: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_407, [0, 2, 3]);  mul_407 = None
    mul_408: "f32[64]" = torch.ops.aten.mul.Tensor(sum_45, 0.00015943877551020407)
    unsqueeze_327: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_408, 0);  mul_408 = None
    unsqueeze_328: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 2);  unsqueeze_327 = None
    unsqueeze_329: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 3);  unsqueeze_328 = None
    mul_409: "f32[64]" = torch.ops.aten.mul.Tensor(sum_46, 0.00015943877551020407)
    mul_410: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_411: "f32[64]" = torch.ops.aten.mul.Tensor(mul_409, mul_410);  mul_409 = mul_410 = None
    unsqueeze_330: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_411, 0);  mul_411 = None
    unsqueeze_331: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, 2);  unsqueeze_330 = None
    unsqueeze_332: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 3);  unsqueeze_331 = None
    mul_412: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_17);  primals_17 = None
    unsqueeze_333: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_412, 0);  mul_412 = None
    unsqueeze_334: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 2);  unsqueeze_333 = None
    unsqueeze_335: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 3);  unsqueeze_334 = None
    mul_413: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_332);  sub_99 = unsqueeze_332 = None
    sub_101: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(where_43, mul_413);  where_43 = mul_413 = None
    sub_102: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(sub_101, unsqueeze_329);  sub_101 = unsqueeze_329 = None
    mul_414: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_335);  sub_102 = unsqueeze_335 = None
    mul_415: "f32[64]" = torch.ops.aten.mul.Tensor(sum_46, squeeze_25);  sum_46 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_414, div_7, primals_65, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_414 = div_7 = primals_65 = None
    getitem_123: "f32[8, 32, 28, 28]" = convolution_backward_23[0]
    getitem_124: "f32[64, 32, 1, 1]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_22: "b8[8, 32, 28, 28]" = torch.ops.aten.lt.Scalar(clone_7, -3)
    le_22: "b8[8, 32, 28, 28]" = torch.ops.aten.le.Scalar(clone_7, 3)
    div_53: "f32[8, 32, 28, 28]" = torch.ops.aten.div.Tensor(clone_7, 3);  clone_7 = None
    add_187: "f32[8, 32, 28, 28]" = torch.ops.aten.add.Tensor(div_53, 0.5);  div_53 = None
    mul_416: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_123, add_187);  add_187 = None
    where_44: "f32[8, 32, 28, 28]" = torch.ops.aten.where.self(le_22, mul_416, getitem_123);  le_22 = mul_416 = getitem_123 = None
    where_45: "f32[8, 32, 28, 28]" = torch.ops.aten.where.self(lt_22, full_default, where_44);  lt_22 = where_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_47: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_45, [0, 2, 3])
    sub_103: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_338);  convolution_7 = unsqueeze_338 = None
    mul_417: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(where_45, sub_103)
    sum_48: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_417, [0, 2, 3]);  mul_417 = None
    mul_418: "f32[32]" = torch.ops.aten.mul.Tensor(sum_47, 0.00015943877551020407)
    unsqueeze_339: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_418, 0);  mul_418 = None
    unsqueeze_340: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 2);  unsqueeze_339 = None
    unsqueeze_341: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, 3);  unsqueeze_340 = None
    mul_419: "f32[32]" = torch.ops.aten.mul.Tensor(sum_48, 0.00015943877551020407)
    mul_420: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_421: "f32[32]" = torch.ops.aten.mul.Tensor(mul_419, mul_420);  mul_419 = mul_420 = None
    unsqueeze_342: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_421, 0);  mul_421 = None
    unsqueeze_343: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 2);  unsqueeze_342 = None
    unsqueeze_344: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 3);  unsqueeze_343 = None
    mul_422: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_15);  primals_15 = None
    unsqueeze_345: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_422, 0);  mul_422 = None
    unsqueeze_346: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_345, 2);  unsqueeze_345 = None
    unsqueeze_347: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 3);  unsqueeze_346 = None
    mul_423: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_344);  sub_103 = unsqueeze_344 = None
    sub_105: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(where_45, mul_423);  where_45 = mul_423 = None
    sub_106: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(sub_105, unsqueeze_341);  sub_105 = unsqueeze_341 = None
    mul_424: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_347);  sub_106 = unsqueeze_347 = None
    mul_425: "f32[32]" = torch.ops.aten.mul.Tensor(sum_48, squeeze_22);  sum_48 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_424, div_6, primals_64, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_424 = div_6 = primals_64 = None
    getitem_126: "f32[8, 32, 56, 56]" = convolution_backward_24[0]
    getitem_127: "f32[32, 1, 3, 3]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_23: "b8[8, 32, 56, 56]" = torch.ops.aten.lt.Scalar(clone_6, -3)
    le_23: "b8[8, 32, 56, 56]" = torch.ops.aten.le.Scalar(clone_6, 3)
    div_54: "f32[8, 32, 56, 56]" = torch.ops.aten.div.Tensor(clone_6, 3);  clone_6 = None
    add_188: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(div_54, 0.5);  div_54 = None
    mul_426: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_126, add_188);  add_188 = None
    where_46: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(le_23, mul_426, getitem_126);  le_23 = mul_426 = getitem_126 = None
    where_47: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(lt_23, full_default, where_46);  lt_23 = where_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_49: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_47, [0, 2, 3])
    sub_107: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_350);  convolution_6 = unsqueeze_350 = None
    mul_427: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(where_47, sub_107)
    sum_50: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_427, [0, 2, 3]);  mul_427 = None
    mul_428: "f32[32]" = torch.ops.aten.mul.Tensor(sum_49, 3.985969387755102e-05)
    unsqueeze_351: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_428, 0);  mul_428 = None
    unsqueeze_352: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 2);  unsqueeze_351 = None
    unsqueeze_353: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, 3);  unsqueeze_352 = None
    mul_429: "f32[32]" = torch.ops.aten.mul.Tensor(sum_50, 3.985969387755102e-05)
    mul_430: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_431: "f32[32]" = torch.ops.aten.mul.Tensor(mul_429, mul_430);  mul_429 = mul_430 = None
    unsqueeze_354: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_431, 0);  mul_431 = None
    unsqueeze_355: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, 2);  unsqueeze_354 = None
    unsqueeze_356: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 3);  unsqueeze_355 = None
    mul_432: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_13);  primals_13 = None
    unsqueeze_357: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_432, 0);  mul_432 = None
    unsqueeze_358: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 2);  unsqueeze_357 = None
    unsqueeze_359: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 3);  unsqueeze_358 = None
    mul_433: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_356);  sub_107 = unsqueeze_356 = None
    sub_109: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(where_47, mul_433);  where_47 = mul_433 = None
    sub_110: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(sub_109, unsqueeze_353);  sub_109 = unsqueeze_353 = None
    mul_434: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_359);  sub_110 = unsqueeze_359 = None
    mul_435: "f32[32]" = torch.ops.aten.mul.Tensor(sum_50, squeeze_19);  sum_50 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_434, div_5, primals_63, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_434 = div_5 = primals_63 = None
    getitem_129: "f32[8, 32, 56, 56]" = convolution_backward_25[0]
    getitem_130: "f32[32, 32, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_24: "b8[8, 32, 56, 56]" = torch.ops.aten.lt.Scalar(clone_5, -3)
    le_24: "b8[8, 32, 56, 56]" = torch.ops.aten.le.Scalar(clone_5, 3)
    div_55: "f32[8, 32, 56, 56]" = torch.ops.aten.div.Tensor(clone_5, 3);  clone_5 = None
    add_189: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(div_55, 0.5);  div_55 = None
    mul_436: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_129, add_189);  add_189 = None
    where_48: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(le_24, mul_436, getitem_129);  le_24 = mul_436 = getitem_129 = None
    where_49: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(lt_24, full_default, where_48);  lt_24 = where_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_51: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_49, [0, 2, 3])
    sub_111: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_362);  convolution_5 = unsqueeze_362 = None
    mul_437: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(where_49, sub_111)
    sum_52: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_437, [0, 2, 3]);  mul_437 = None
    mul_438: "f32[32]" = torch.ops.aten.mul.Tensor(sum_51, 3.985969387755102e-05)
    unsqueeze_363: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_438, 0);  mul_438 = None
    unsqueeze_364: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 2);  unsqueeze_363 = None
    unsqueeze_365: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, 3);  unsqueeze_364 = None
    mul_439: "f32[32]" = torch.ops.aten.mul.Tensor(sum_52, 3.985969387755102e-05)
    mul_440: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_441: "f32[32]" = torch.ops.aten.mul.Tensor(mul_439, mul_440);  mul_439 = mul_440 = None
    unsqueeze_366: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_441, 0);  mul_441 = None
    unsqueeze_367: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, 2);  unsqueeze_366 = None
    unsqueeze_368: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 3);  unsqueeze_367 = None
    mul_442: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_11);  primals_11 = None
    unsqueeze_369: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_442, 0);  mul_442 = None
    unsqueeze_370: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_369, 2);  unsqueeze_369 = None
    unsqueeze_371: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 3);  unsqueeze_370 = None
    mul_443: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_368);  sub_111 = unsqueeze_368 = None
    sub_113: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(where_49, mul_443);  where_49 = mul_443 = None
    sub_114: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(sub_113, unsqueeze_365);  sub_113 = unsqueeze_365 = None
    mul_444: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_371);  sub_114 = unsqueeze_371 = None
    mul_445: "f32[32]" = torch.ops.aten.mul.Tensor(sum_52, squeeze_16);  sum_52 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_444, div_4, primals_62, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_444 = div_4 = primals_62 = None
    getitem_132: "f32[8, 32, 56, 56]" = convolution_backward_26[0]
    getitem_133: "f32[32, 1, 3, 3]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_25: "b8[8, 32, 56, 56]" = torch.ops.aten.lt.Scalar(clone_4, -3)
    le_25: "b8[8, 32, 56, 56]" = torch.ops.aten.le.Scalar(clone_4, 3)
    div_56: "f32[8, 32, 56, 56]" = torch.ops.aten.div.Tensor(clone_4, 3);  clone_4 = None
    add_190: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(div_56, 0.5);  div_56 = None
    mul_446: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_132, add_190);  add_190 = None
    where_50: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(le_25, mul_446, getitem_132);  le_25 = mul_446 = getitem_132 = None
    where_51: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(lt_25, full_default, where_50);  lt_25 = where_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_53: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_51, [0, 2, 3])
    sub_115: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_374);  convolution_4 = unsqueeze_374 = None
    mul_447: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(where_51, sub_115)
    sum_54: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_447, [0, 2, 3]);  mul_447 = None
    mul_448: "f32[32]" = torch.ops.aten.mul.Tensor(sum_53, 3.985969387755102e-05)
    unsqueeze_375: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_448, 0);  mul_448 = None
    unsqueeze_376: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_375, 2);  unsqueeze_375 = None
    unsqueeze_377: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 3);  unsqueeze_376 = None
    mul_449: "f32[32]" = torch.ops.aten.mul.Tensor(sum_54, 3.985969387755102e-05)
    mul_450: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_451: "f32[32]" = torch.ops.aten.mul.Tensor(mul_449, mul_450);  mul_449 = mul_450 = None
    unsqueeze_378: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_451, 0);  mul_451 = None
    unsqueeze_379: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, 2);  unsqueeze_378 = None
    unsqueeze_380: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 3);  unsqueeze_379 = None
    mul_452: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_9);  primals_9 = None
    unsqueeze_381: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_452, 0);  mul_452 = None
    unsqueeze_382: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_381, 2);  unsqueeze_381 = None
    unsqueeze_383: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 3);  unsqueeze_382 = None
    mul_453: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_380);  sub_115 = unsqueeze_380 = None
    sub_117: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(where_51, mul_453);  where_51 = mul_453 = None
    sub_118: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(sub_117, unsqueeze_377);  sub_117 = unsqueeze_377 = None
    mul_454: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_383);  sub_118 = unsqueeze_383 = None
    mul_455: "f32[32]" = torch.ops.aten.mul.Tensor(sum_54, squeeze_13);  sum_54 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_454, div_3, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_454 = div_3 = primals_61 = None
    getitem_135: "f32[8, 16, 56, 56]" = convolution_backward_27[0]
    getitem_136: "f32[32, 16, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_26: "b8[8, 16, 56, 56]" = torch.ops.aten.lt.Scalar(clone_3, -3)
    le_26: "b8[8, 16, 56, 56]" = torch.ops.aten.le.Scalar(clone_3, 3)
    div_57: "f32[8, 16, 56, 56]" = torch.ops.aten.div.Tensor(clone_3, 3);  clone_3 = None
    add_191: "f32[8, 16, 56, 56]" = torch.ops.aten.add.Tensor(div_57, 0.5);  div_57 = None
    mul_456: "f32[8, 16, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_135, add_191);  add_191 = None
    where_52: "f32[8, 16, 56, 56]" = torch.ops.aten.where.self(le_26, mul_456, getitem_135);  le_26 = mul_456 = getitem_135 = None
    where_53: "f32[8, 16, 56, 56]" = torch.ops.aten.where.self(lt_26, full_default, where_52);  lt_26 = where_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_55: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_53, [0, 2, 3])
    sub_119: "f32[8, 16, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_386);  convolution_3 = unsqueeze_386 = None
    mul_457: "f32[8, 16, 56, 56]" = torch.ops.aten.mul.Tensor(where_53, sub_119)
    sum_56: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_457, [0, 2, 3]);  mul_457 = None
    mul_458: "f32[16]" = torch.ops.aten.mul.Tensor(sum_55, 3.985969387755102e-05)
    unsqueeze_387: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_458, 0);  mul_458 = None
    unsqueeze_388: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_387, 2);  unsqueeze_387 = None
    unsqueeze_389: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 3);  unsqueeze_388 = None
    mul_459: "f32[16]" = torch.ops.aten.mul.Tensor(sum_56, 3.985969387755102e-05)
    mul_460: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_461: "f32[16]" = torch.ops.aten.mul.Tensor(mul_459, mul_460);  mul_459 = mul_460 = None
    unsqueeze_390: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_461, 0);  mul_461 = None
    unsqueeze_391: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, 2);  unsqueeze_390 = None
    unsqueeze_392: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 3);  unsqueeze_391 = None
    mul_462: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_7);  primals_7 = None
    unsqueeze_393: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_462, 0);  mul_462 = None
    unsqueeze_394: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 2);  unsqueeze_393 = None
    unsqueeze_395: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 3);  unsqueeze_394 = None
    mul_463: "f32[8, 16, 56, 56]" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_392);  sub_119 = unsqueeze_392 = None
    sub_121: "f32[8, 16, 56, 56]" = torch.ops.aten.sub.Tensor(where_53, mul_463);  where_53 = mul_463 = None
    sub_122: "f32[8, 16, 56, 56]" = torch.ops.aten.sub.Tensor(sub_121, unsqueeze_389);  sub_121 = unsqueeze_389 = None
    mul_464: "f32[8, 16, 56, 56]" = torch.ops.aten.mul.Tensor(sub_122, unsqueeze_395);  sub_122 = unsqueeze_395 = None
    mul_465: "f32[16]" = torch.ops.aten.mul.Tensor(sum_56, squeeze_10);  sum_56 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_464, div_2, primals_60, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False]);  mul_464 = div_2 = primals_60 = None
    getitem_138: "f32[8, 16, 112, 112]" = convolution_backward_28[0]
    getitem_139: "f32[16, 1, 3, 3]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_27: "b8[8, 16, 112, 112]" = torch.ops.aten.lt.Scalar(clone_2, -3)
    le_27: "b8[8, 16, 112, 112]" = torch.ops.aten.le.Scalar(clone_2, 3)
    div_58: "f32[8, 16, 112, 112]" = torch.ops.aten.div.Tensor(clone_2, 3);  clone_2 = None
    add_192: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(div_58, 0.5);  div_58 = None
    mul_466: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_138, add_192);  add_192 = None
    where_54: "f32[8, 16, 112, 112]" = torch.ops.aten.where.self(le_27, mul_466, getitem_138);  le_27 = mul_466 = getitem_138 = None
    where_55: "f32[8, 16, 112, 112]" = torch.ops.aten.where.self(lt_27, full_default, where_54);  lt_27 = where_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_57: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_55, [0, 2, 3])
    sub_123: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_398);  convolution_2 = unsqueeze_398 = None
    mul_467: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(where_55, sub_123)
    sum_58: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_467, [0, 2, 3]);  mul_467 = None
    mul_468: "f32[16]" = torch.ops.aten.mul.Tensor(sum_57, 9.964923469387754e-06)
    unsqueeze_399: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_468, 0);  mul_468 = None
    unsqueeze_400: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 2);  unsqueeze_399 = None
    unsqueeze_401: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, 3);  unsqueeze_400 = None
    mul_469: "f32[16]" = torch.ops.aten.mul.Tensor(sum_58, 9.964923469387754e-06)
    mul_470: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_471: "f32[16]" = torch.ops.aten.mul.Tensor(mul_469, mul_470);  mul_469 = mul_470 = None
    unsqueeze_402: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_471, 0);  mul_471 = None
    unsqueeze_403: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, 2);  unsqueeze_402 = None
    unsqueeze_404: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 3);  unsqueeze_403 = None
    mul_472: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_5);  primals_5 = None
    unsqueeze_405: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_472, 0);  mul_472 = None
    unsqueeze_406: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 2);  unsqueeze_405 = None
    unsqueeze_407: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 3);  unsqueeze_406 = None
    mul_473: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_404);  sub_123 = unsqueeze_404 = None
    sub_125: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(where_55, mul_473);  where_55 = mul_473 = None
    sub_126: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_125, unsqueeze_401);  sub_125 = unsqueeze_401 = None
    mul_474: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_407);  sub_126 = unsqueeze_407 = None
    mul_475: "f32[16]" = torch.ops.aten.mul.Tensor(sum_58, squeeze_7);  sum_58 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_474, div_1, primals_59, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_474 = div_1 = primals_59 = None
    getitem_141: "f32[8, 8, 112, 112]" = convolution_backward_29[0]
    getitem_142: "f32[16, 8, 1, 1]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_28: "b8[8, 8, 112, 112]" = torch.ops.aten.lt.Scalar(clone_1, -3)
    le_28: "b8[8, 8, 112, 112]" = torch.ops.aten.le.Scalar(clone_1, 3)
    div_59: "f32[8, 8, 112, 112]" = torch.ops.aten.div.Tensor(clone_1, 3);  clone_1 = None
    add_193: "f32[8, 8, 112, 112]" = torch.ops.aten.add.Tensor(div_59, 0.5);  div_59 = None
    mul_476: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_141, add_193);  add_193 = None
    where_56: "f32[8, 8, 112, 112]" = torch.ops.aten.where.self(le_28, mul_476, getitem_141);  le_28 = mul_476 = getitem_141 = None
    where_57: "f32[8, 8, 112, 112]" = torch.ops.aten.where.self(lt_28, full_default, where_56);  lt_28 = where_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_59: "f32[8]" = torch.ops.aten.sum.dim_IntList(where_57, [0, 2, 3])
    sub_127: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_410);  convolution_1 = unsqueeze_410 = None
    mul_477: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(where_57, sub_127)
    sum_60: "f32[8]" = torch.ops.aten.sum.dim_IntList(mul_477, [0, 2, 3]);  mul_477 = None
    mul_478: "f32[8]" = torch.ops.aten.mul.Tensor(sum_59, 9.964923469387754e-06)
    unsqueeze_411: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(mul_478, 0);  mul_478 = None
    unsqueeze_412: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 2);  unsqueeze_411 = None
    unsqueeze_413: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, 3);  unsqueeze_412 = None
    mul_479: "f32[8]" = torch.ops.aten.mul.Tensor(sum_60, 9.964923469387754e-06)
    mul_480: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_481: "f32[8]" = torch.ops.aten.mul.Tensor(mul_479, mul_480);  mul_479 = mul_480 = None
    unsqueeze_414: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(mul_481, 0);  mul_481 = None
    unsqueeze_415: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, 2);  unsqueeze_414 = None
    unsqueeze_416: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 3);  unsqueeze_415 = None
    mul_482: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_3);  primals_3 = None
    unsqueeze_417: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(mul_482, 0);  mul_482 = None
    unsqueeze_418: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 2);  unsqueeze_417 = None
    unsqueeze_419: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 3);  unsqueeze_418 = None
    mul_483: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_416);  sub_127 = unsqueeze_416 = None
    sub_129: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(where_57, mul_483);  where_57 = mul_483 = None
    sub_130: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(sub_129, unsqueeze_413);  sub_129 = unsqueeze_413 = None
    mul_484: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_419);  sub_130 = unsqueeze_419 = None
    mul_485: "f32[8]" = torch.ops.aten.mul.Tensor(sum_60, squeeze_4);  sum_60 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_484, div, primals_58, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_484 = div = primals_58 = None
    getitem_144: "f32[8, 8, 112, 112]" = convolution_backward_30[0]
    getitem_145: "f32[8, 1, 3, 3]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_29: "b8[8, 8, 112, 112]" = torch.ops.aten.lt.Scalar(clone, -3)
    le_29: "b8[8, 8, 112, 112]" = torch.ops.aten.le.Scalar(clone, 3)
    div_60: "f32[8, 8, 112, 112]" = torch.ops.aten.div.Tensor(clone, 3);  clone = None
    add_194: "f32[8, 8, 112, 112]" = torch.ops.aten.add.Tensor(div_60, 0.5);  div_60 = None
    mul_486: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_144, add_194);  add_194 = None
    where_58: "f32[8, 8, 112, 112]" = torch.ops.aten.where.self(le_29, mul_486, getitem_144);  le_29 = mul_486 = getitem_144 = None
    where_59: "f32[8, 8, 112, 112]" = torch.ops.aten.where.self(lt_29, full_default, where_58);  lt_29 = full_default = where_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_61: "f32[8]" = torch.ops.aten.sum.dim_IntList(where_59, [0, 2, 3])
    sub_131: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_422);  convolution = unsqueeze_422 = None
    mul_487: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(where_59, sub_131)
    sum_62: "f32[8]" = torch.ops.aten.sum.dim_IntList(mul_487, [0, 2, 3]);  mul_487 = None
    mul_488: "f32[8]" = torch.ops.aten.mul.Tensor(sum_61, 9.964923469387754e-06)
    unsqueeze_423: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(mul_488, 0);  mul_488 = None
    unsqueeze_424: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 2);  unsqueeze_423 = None
    unsqueeze_425: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 3);  unsqueeze_424 = None
    mul_489: "f32[8]" = torch.ops.aten.mul.Tensor(sum_62, 9.964923469387754e-06)
    mul_490: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_491: "f32[8]" = torch.ops.aten.mul.Tensor(mul_489, mul_490);  mul_489 = mul_490 = None
    unsqueeze_426: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(mul_491, 0);  mul_491 = None
    unsqueeze_427: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 2);  unsqueeze_426 = None
    unsqueeze_428: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 3);  unsqueeze_427 = None
    mul_492: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_1);  primals_1 = None
    unsqueeze_429: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(mul_492, 0);  mul_492 = None
    unsqueeze_430: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 2);  unsqueeze_429 = None
    unsqueeze_431: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 3);  unsqueeze_430 = None
    mul_493: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_428);  sub_131 = unsqueeze_428 = None
    sub_133: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(where_59, mul_493);  where_59 = mul_493 = None
    sub_134: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(sub_133, unsqueeze_425);  sub_133 = unsqueeze_425 = None
    mul_494: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_431);  sub_134 = unsqueeze_431 = None
    mul_495: "f32[8]" = torch.ops.aten.mul.Tensor(sum_62, squeeze_1);  sum_62 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:135, code: x = self.conv_stem(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_494, primals_175, primals_57, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_494 = primals_175 = primals_57 = None
    getitem_148: "f32[8, 3, 3, 3]" = convolution_backward_31[1];  convolution_backward_31 = None
    return [mul_495, sum_61, mul_485, sum_59, mul_475, sum_57, mul_465, sum_55, mul_455, sum_53, mul_445, sum_51, mul_435, sum_49, mul_425, sum_47, mul_415, sum_45, mul_405, sum_43, mul_395, sum_41, mul_385, sum_39, mul_375, sum_37, mul_365, sum_35, mul_355, sum_33, mul_345, sum_31, mul_335, sum_29, mul_325, sum_27, mul_315, sum_25, mul_305, sum_23, mul_295, sum_21, mul_285, sum_19, mul_275, sum_17, mul_265, sum_15, mul_252, sum_10, mul_242, sum_8, mul_229, sum_3, permute_4, view_2, getitem_148, getitem_145, getitem_142, getitem_139, getitem_136, getitem_133, getitem_130, getitem_127, getitem_124, getitem_121, getitem_118, getitem_115, getitem_112, getitem_109, getitem_106, getitem_103, getitem_100, getitem_97, getitem_94, getitem_91, getitem_88, getitem_85, getitem_82, getitem_79, getitem_76, sum_14, getitem_73, sum_13, getitem_70, getitem_67, getitem_64, sum_7, getitem_61, sum_6, getitem_58, getitem_55, sum_2, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    