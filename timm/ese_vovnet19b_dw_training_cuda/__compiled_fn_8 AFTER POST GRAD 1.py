from __future__ import annotations



def forward(self, primals_1: "f32[64]", primals_3: "f32[64]", primals_5: "f32[64]", primals_7: "f32[128]", primals_9: "f32[128]", primals_11: "f32[128]", primals_13: "f32[128]", primals_15: "f32[256]", primals_17: "f32[160]", primals_19: "f32[160]", primals_21: "f32[160]", primals_23: "f32[160]", primals_25: "f32[512]", primals_27: "f32[192]", primals_29: "f32[192]", primals_31: "f32[192]", primals_33: "f32[192]", primals_35: "f32[768]", primals_37: "f32[224]", primals_39: "f32[224]", primals_41: "f32[224]", primals_43: "f32[224]", primals_45: "f32[1024]", primals_47: "f32[64, 3, 3, 3]", primals_48: "f32[64, 1, 3, 3]", primals_49: "f32[64, 64, 1, 1]", primals_50: "f32[64, 1, 3, 3]", primals_51: "f32[64, 64, 1, 1]", primals_52: "f32[128, 64, 1, 1]", primals_53: "f32[128, 1, 3, 3]", primals_54: "f32[128, 128, 1, 1]", primals_55: "f32[128, 1, 3, 3]", primals_56: "f32[128, 128, 1, 1]", primals_57: "f32[128, 1, 3, 3]", primals_58: "f32[128, 128, 1, 1]", primals_59: "f32[256, 448, 1, 1]", primals_60: "f32[256, 256, 1, 1]", primals_62: "f32[160, 256, 1, 1]", primals_63: "f32[160, 1, 3, 3]", primals_64: "f32[160, 160, 1, 1]", primals_65: "f32[160, 1, 3, 3]", primals_66: "f32[160, 160, 1, 1]", primals_67: "f32[160, 1, 3, 3]", primals_68: "f32[160, 160, 1, 1]", primals_69: "f32[512, 736, 1, 1]", primals_70: "f32[512, 512, 1, 1]", primals_72: "f32[192, 512, 1, 1]", primals_73: "f32[192, 1, 3, 3]", primals_74: "f32[192, 192, 1, 1]", primals_75: "f32[192, 1, 3, 3]", primals_76: "f32[192, 192, 1, 1]", primals_77: "f32[192, 1, 3, 3]", primals_78: "f32[192, 192, 1, 1]", primals_79: "f32[768, 1088, 1, 1]", primals_80: "f32[768, 768, 1, 1]", primals_82: "f32[224, 768, 1, 1]", primals_83: "f32[224, 1, 3, 3]", primals_84: "f32[224, 224, 1, 1]", primals_85: "f32[224, 1, 3, 3]", primals_86: "f32[224, 224, 1, 1]", primals_87: "f32[224, 1, 3, 3]", primals_88: "f32[224, 224, 1, 1]", primals_89: "f32[1024, 1440, 1, 1]", primals_90: "f32[1024, 1024, 1, 1]", primals_163: "f32[8, 3, 224, 224]", convolution: "f32[8, 64, 112, 112]", squeeze_1: "f32[64]", relu: "f32[8, 64, 112, 112]", convolution_1: "f32[8, 64, 112, 112]", convolution_2: "f32[8, 64, 112, 112]", squeeze_4: "f32[64]", relu_1: "f32[8, 64, 112, 112]", convolution_3: "f32[8, 64, 56, 56]", convolution_4: "f32[8, 64, 56, 56]", squeeze_7: "f32[64]", relu_2: "f32[8, 64, 56, 56]", convolution_5: "f32[8, 128, 56, 56]", squeeze_10: "f32[128]", relu_3: "f32[8, 128, 56, 56]", convolution_6: "f32[8, 128, 56, 56]", convolution_7: "f32[8, 128, 56, 56]", squeeze_13: "f32[128]", relu_4: "f32[8, 128, 56, 56]", convolution_8: "f32[8, 128, 56, 56]", convolution_9: "f32[8, 128, 56, 56]", squeeze_16: "f32[128]", relu_5: "f32[8, 128, 56, 56]", convolution_10: "f32[8, 128, 56, 56]", convolution_11: "f32[8, 128, 56, 56]", squeeze_19: "f32[128]", cat: "f32[8, 448, 56, 56]", convolution_12: "f32[8, 256, 56, 56]", squeeze_22: "f32[256]", relu_7: "f32[8, 256, 56, 56]", mean: "f32[8, 256, 1, 1]", div: "f32[8, 256, 1, 1]", mul_56: "f32[8, 256, 56, 56]", getitem_16: "f32[8, 256, 28, 28]", getitem_17: "i64[8, 256, 28, 28]", convolution_14: "f32[8, 160, 28, 28]", squeeze_25: "f32[160]", relu_8: "f32[8, 160, 28, 28]", convolution_15: "f32[8, 160, 28, 28]", convolution_16: "f32[8, 160, 28, 28]", squeeze_28: "f32[160]", relu_9: "f32[8, 160, 28, 28]", convolution_17: "f32[8, 160, 28, 28]", convolution_18: "f32[8, 160, 28, 28]", squeeze_31: "f32[160]", relu_10: "f32[8, 160, 28, 28]", convolution_19: "f32[8, 160, 28, 28]", convolution_20: "f32[8, 160, 28, 28]", squeeze_34: "f32[160]", cat_1: "f32[8, 736, 28, 28]", convolution_21: "f32[8, 512, 28, 28]", squeeze_37: "f32[512]", relu_12: "f32[8, 512, 28, 28]", mean_1: "f32[8, 512, 1, 1]", div_1: "f32[8, 512, 1, 1]", mul_92: "f32[8, 512, 28, 28]", getitem_28: "f32[8, 512, 14, 14]", getitem_29: "i64[8, 512, 14, 14]", convolution_23: "f32[8, 192, 14, 14]", squeeze_40: "f32[192]", relu_13: "f32[8, 192, 14, 14]", convolution_24: "f32[8, 192, 14, 14]", convolution_25: "f32[8, 192, 14, 14]", squeeze_43: "f32[192]", relu_14: "f32[8, 192, 14, 14]", convolution_26: "f32[8, 192, 14, 14]", convolution_27: "f32[8, 192, 14, 14]", squeeze_46: "f32[192]", relu_15: "f32[8, 192, 14, 14]", convolution_28: "f32[8, 192, 14, 14]", convolution_29: "f32[8, 192, 14, 14]", squeeze_49: "f32[192]", cat_2: "f32[8, 1088, 14, 14]", convolution_30: "f32[8, 768, 14, 14]", squeeze_52: "f32[768]", relu_17: "f32[8, 768, 14, 14]", mean_2: "f32[8, 768, 1, 1]", div_2: "f32[8, 768, 1, 1]", mul_128: "f32[8, 768, 14, 14]", getitem_40: "f32[8, 768, 7, 7]", getitem_41: "i64[8, 768, 7, 7]", convolution_32: "f32[8, 224, 7, 7]", squeeze_55: "f32[224]", relu_18: "f32[8, 224, 7, 7]", convolution_33: "f32[8, 224, 7, 7]", convolution_34: "f32[8, 224, 7, 7]", squeeze_58: "f32[224]", relu_19: "f32[8, 224, 7, 7]", convolution_35: "f32[8, 224, 7, 7]", convolution_36: "f32[8, 224, 7, 7]", squeeze_61: "f32[224]", relu_20: "f32[8, 224, 7, 7]", convolution_37: "f32[8, 224, 7, 7]", convolution_38: "f32[8, 224, 7, 7]", squeeze_64: "f32[224]", cat_3: "f32[8, 1440, 7, 7]", convolution_39: "f32[8, 1024, 7, 7]", squeeze_67: "f32[1024]", relu_22: "f32[8, 1024, 7, 7]", mean_3: "f32[8, 1024, 1, 1]", div_3: "f32[8, 1024, 1, 1]", clone: "f32[8, 1024]", permute_1: "f32[1000, 1024]", bitwise_and: "b8[8, 1024, 1, 1]", unsqueeze_94: "f32[1, 1024, 1, 1]", le_1: "b8[8, 224, 7, 7]", unsqueeze_106: "f32[1, 224, 1, 1]", unsqueeze_118: "f32[1, 224, 1, 1]", unsqueeze_130: "f32[1, 224, 1, 1]", unsqueeze_142: "f32[1, 224, 1, 1]", bitwise_and_1: "b8[8, 768, 1, 1]", unsqueeze_154: "f32[1, 768, 1, 1]", le_6: "b8[8, 192, 14, 14]", unsqueeze_166: "f32[1, 192, 1, 1]", unsqueeze_178: "f32[1, 192, 1, 1]", unsqueeze_190: "f32[1, 192, 1, 1]", unsqueeze_202: "f32[1, 192, 1, 1]", bitwise_and_2: "b8[8, 512, 1, 1]", unsqueeze_214: "f32[1, 512, 1, 1]", le_11: "b8[8, 160, 28, 28]", unsqueeze_226: "f32[1, 160, 1, 1]", unsqueeze_238: "f32[1, 160, 1, 1]", unsqueeze_250: "f32[1, 160, 1, 1]", unsqueeze_262: "f32[1, 160, 1, 1]", bitwise_and_3: "b8[8, 256, 1, 1]", unsqueeze_274: "f32[1, 256, 1, 1]", le_16: "b8[8, 128, 56, 56]", unsqueeze_286: "f32[1, 128, 1, 1]", unsqueeze_298: "f32[1, 128, 1, 1]", unsqueeze_310: "f32[1, 128, 1, 1]", unsqueeze_322: "f32[1, 128, 1, 1]", unsqueeze_334: "f32[1, 64, 1, 1]", unsqueeze_346: "f32[1, 64, 1, 1]", unsqueeze_358: "f32[1, 64, 1, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    mm: "f32[8, 1024]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 1024]" = torch.ops.aten.mm.default(permute_2, clone);  permute_2 = clone = None
    permute_3: "f32[1024, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 1024]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_2: "f32[8, 1024, 1, 1]" = torch.ops.aten.reshape.default(mm, [8, 1024, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 1024, 7, 7]" = torch.ops.aten.expand.default(view_2, [8, 1024, 7, 7]);  view_2 = None
    div_4: "f32[8, 1024, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:71, code: return x * self.gate(x_se)
    mul_165: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(div_4, relu_22)
    mul_166: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(div_4, div_3);  div_4 = div_3 = None
    sum_2: "f32[8, 1024, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_165, [2, 3], True);  mul_165 = None
    mul_167: "f32[8, 1024, 1, 1]" = torch.ops.aten.mul.Tensor(sum_2, 0.16666666666666666);  sum_2 = None
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where: "f32[8, 1024, 1, 1]" = torch.ops.aten.where.self(bitwise_and, mul_167, full_default);  bitwise_and = mul_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:70, code: x_se = self.fc(x_se)
    sum_3: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    convolution_backward = torch.ops.aten.convolution_backward.default(where, mean_3, primals_90, [1024], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where = mean_3 = primals_90 = None
    getitem_52: "f32[8, 1024, 1, 1]" = convolution_backward[0]
    getitem_53: "f32[1024, 1024, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:66, code: x_se = x.mean((2, 3), keepdim=True)
    expand_1: "f32[8, 1024, 7, 7]" = torch.ops.aten.expand.default(getitem_52, [8, 1024, 7, 7]);  getitem_52 = None
    div_5: "f32[8, 1024, 7, 7]" = torch.ops.aten.div.Scalar(expand_1, 49);  expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:66, code: x_se = x.mean((2, 3), keepdim=True)
    add_119: "f32[8, 1024, 7, 7]" = torch.ops.aten.add.Tensor(mul_166, div_5);  mul_166 = div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le: "b8[8, 1024, 7, 7]" = torch.ops.aten.le.Scalar(relu_22, 0);  relu_22 = None
    where_1: "f32[8, 1024, 7, 7]" = torch.ops.aten.where.self(le, full_default, add_119);  le = add_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_4: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_23: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_94);  convolution_39 = unsqueeze_94 = None
    mul_168: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, sub_23)
    sum_5: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_168, [0, 2, 3]);  mul_168 = None
    mul_169: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_4, 0.002551020408163265)
    unsqueeze_95: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_169, 0);  mul_169 = None
    unsqueeze_96: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_95, 2);  unsqueeze_95 = None
    unsqueeze_97: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, 3);  unsqueeze_96 = None
    mul_170: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_5, 0.002551020408163265)
    mul_171: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_172: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_170, mul_171);  mul_170 = mul_171 = None
    unsqueeze_98: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_172, 0);  mul_172 = None
    unsqueeze_99: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, 2);  unsqueeze_98 = None
    unsqueeze_100: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_99, 3);  unsqueeze_99 = None
    mul_173: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_45);  primals_45 = None
    unsqueeze_101: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_173, 0);  mul_173 = None
    unsqueeze_102: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_101, 2);  unsqueeze_101 = None
    unsqueeze_103: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, 3);  unsqueeze_102 = None
    mul_174: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_100);  sub_23 = unsqueeze_100 = None
    sub_25: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(where_1, mul_174);  where_1 = mul_174 = None
    sub_26: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(sub_25, unsqueeze_97);  sub_25 = unsqueeze_97 = None
    mul_175: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_103);  sub_26 = unsqueeze_103 = None
    mul_176: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_67);  sum_5 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_175, cat_3, primals_89, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_175 = cat_3 = primals_89 = None
    getitem_55: "f32[8, 1440, 7, 7]" = convolution_backward_1[0]
    getitem_56: "f32[1024, 1440, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    slice_1: "f32[8, 768, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_55, 1, 0, 768)
    slice_2: "f32[8, 224, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_55, 1, 768, 992)
    slice_3: "f32[8, 224, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_55, 1, 992, 1216)
    slice_4: "f32[8, 224, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_55, 1, 1216, 1440);  getitem_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_2: "f32[8, 224, 7, 7]" = torch.ops.aten.where.self(le_1, full_default, slice_4);  le_1 = slice_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_6: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_27: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_106);  convolution_38 = unsqueeze_106 = None
    mul_177: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_27)
    sum_7: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_177, [0, 2, 3]);  mul_177 = None
    mul_178: "f32[224]" = torch.ops.aten.mul.Tensor(sum_6, 0.002551020408163265)
    unsqueeze_107: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_178, 0);  mul_178 = None
    unsqueeze_108: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_107, 2);  unsqueeze_107 = None
    unsqueeze_109: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, 3);  unsqueeze_108 = None
    mul_179: "f32[224]" = torch.ops.aten.mul.Tensor(sum_7, 0.002551020408163265)
    mul_180: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_181: "f32[224]" = torch.ops.aten.mul.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
    unsqueeze_110: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_181, 0);  mul_181 = None
    unsqueeze_111: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, 2);  unsqueeze_110 = None
    unsqueeze_112: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_111, 3);  unsqueeze_111 = None
    mul_182: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_43);  primals_43 = None
    unsqueeze_113: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_182, 0);  mul_182 = None
    unsqueeze_114: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_113, 2);  unsqueeze_113 = None
    unsqueeze_115: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, 3);  unsqueeze_114 = None
    mul_183: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_112);  sub_27 = unsqueeze_112 = None
    sub_29: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(where_2, mul_183);  where_2 = mul_183 = None
    sub_30: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(sub_29, unsqueeze_109);  sub_29 = unsqueeze_109 = None
    mul_184: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_115);  sub_30 = unsqueeze_115 = None
    mul_185: "f32[224]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_64);  sum_7 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_184, convolution_37, primals_88, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_184 = convolution_37 = primals_88 = None
    getitem_58: "f32[8, 224, 7, 7]" = convolution_backward_2[0]
    getitem_59: "f32[224, 224, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(getitem_58, relu_20, primals_87, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 224, [True, True, False]);  getitem_58 = primals_87 = None
    getitem_61: "f32[8, 224, 7, 7]" = convolution_backward_3[0]
    getitem_62: "f32[224, 1, 3, 3]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    add_120: "f32[8, 224, 7, 7]" = torch.ops.aten.add.Tensor(slice_3, getitem_61);  slice_3 = getitem_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_2: "b8[8, 224, 7, 7]" = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None
    where_3: "f32[8, 224, 7, 7]" = torch.ops.aten.where.self(le_2, full_default, add_120);  le_2 = add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_8: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_31: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_118);  convolution_36 = unsqueeze_118 = None
    mul_186: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_31)
    sum_9: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_186, [0, 2, 3]);  mul_186 = None
    mul_187: "f32[224]" = torch.ops.aten.mul.Tensor(sum_8, 0.002551020408163265)
    unsqueeze_119: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_187, 0);  mul_187 = None
    unsqueeze_120: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_119, 2);  unsqueeze_119 = None
    unsqueeze_121: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, 3);  unsqueeze_120 = None
    mul_188: "f32[224]" = torch.ops.aten.mul.Tensor(sum_9, 0.002551020408163265)
    mul_189: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_190: "f32[224]" = torch.ops.aten.mul.Tensor(mul_188, mul_189);  mul_188 = mul_189 = None
    unsqueeze_122: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_190, 0);  mul_190 = None
    unsqueeze_123: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, 2);  unsqueeze_122 = None
    unsqueeze_124: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_123, 3);  unsqueeze_123 = None
    mul_191: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_41);  primals_41 = None
    unsqueeze_125: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_191, 0);  mul_191 = None
    unsqueeze_126: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_125, 2);  unsqueeze_125 = None
    unsqueeze_127: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, 3);  unsqueeze_126 = None
    mul_192: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_124);  sub_31 = unsqueeze_124 = None
    sub_33: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(where_3, mul_192);  where_3 = mul_192 = None
    sub_34: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(sub_33, unsqueeze_121);  sub_33 = unsqueeze_121 = None
    mul_193: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_127);  sub_34 = unsqueeze_127 = None
    mul_194: "f32[224]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_61);  sum_9 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_193, convolution_35, primals_86, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_193 = convolution_35 = primals_86 = None
    getitem_64: "f32[8, 224, 7, 7]" = convolution_backward_4[0]
    getitem_65: "f32[224, 224, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(getitem_64, relu_19, primals_85, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 224, [True, True, False]);  getitem_64 = primals_85 = None
    getitem_67: "f32[8, 224, 7, 7]" = convolution_backward_5[0]
    getitem_68: "f32[224, 1, 3, 3]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    add_121: "f32[8, 224, 7, 7]" = torch.ops.aten.add.Tensor(slice_2, getitem_67);  slice_2 = getitem_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_3: "b8[8, 224, 7, 7]" = torch.ops.aten.le.Scalar(relu_19, 0);  relu_19 = None
    where_4: "f32[8, 224, 7, 7]" = torch.ops.aten.where.self(le_3, full_default, add_121);  le_3 = add_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_10: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_35: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_130);  convolution_34 = unsqueeze_130 = None
    mul_195: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, sub_35)
    sum_11: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_195, [0, 2, 3]);  mul_195 = None
    mul_196: "f32[224]" = torch.ops.aten.mul.Tensor(sum_10, 0.002551020408163265)
    unsqueeze_131: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_196, 0);  mul_196 = None
    unsqueeze_132: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_131, 2);  unsqueeze_131 = None
    unsqueeze_133: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, 3);  unsqueeze_132 = None
    mul_197: "f32[224]" = torch.ops.aten.mul.Tensor(sum_11, 0.002551020408163265)
    mul_198: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_199: "f32[224]" = torch.ops.aten.mul.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
    unsqueeze_134: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_199, 0);  mul_199 = None
    unsqueeze_135: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, 2);  unsqueeze_134 = None
    unsqueeze_136: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_135, 3);  unsqueeze_135 = None
    mul_200: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_39);  primals_39 = None
    unsqueeze_137: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_200, 0);  mul_200 = None
    unsqueeze_138: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_137, 2);  unsqueeze_137 = None
    unsqueeze_139: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, 3);  unsqueeze_138 = None
    mul_201: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_136);  sub_35 = unsqueeze_136 = None
    sub_37: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(where_4, mul_201);  where_4 = mul_201 = None
    sub_38: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(sub_37, unsqueeze_133);  sub_37 = unsqueeze_133 = None
    mul_202: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_139);  sub_38 = unsqueeze_139 = None
    mul_203: "f32[224]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_58);  sum_11 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_202, convolution_33, primals_84, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_202 = convolution_33 = primals_84 = None
    getitem_70: "f32[8, 224, 7, 7]" = convolution_backward_6[0]
    getitem_71: "f32[224, 224, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(getitem_70, relu_18, primals_83, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 224, [True, True, False]);  getitem_70 = primals_83 = None
    getitem_73: "f32[8, 224, 7, 7]" = convolution_backward_7[0]
    getitem_74: "f32[224, 1, 3, 3]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_4: "b8[8, 224, 7, 7]" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
    where_5: "f32[8, 224, 7, 7]" = torch.ops.aten.where.self(le_4, full_default, getitem_73);  le_4 = getitem_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_12: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_39: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_142);  convolution_32 = unsqueeze_142 = None
    mul_204: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_5, sub_39)
    sum_13: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_204, [0, 2, 3]);  mul_204 = None
    mul_205: "f32[224]" = torch.ops.aten.mul.Tensor(sum_12, 0.002551020408163265)
    unsqueeze_143: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_205, 0);  mul_205 = None
    unsqueeze_144: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_143, 2);  unsqueeze_143 = None
    unsqueeze_145: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, 3);  unsqueeze_144 = None
    mul_206: "f32[224]" = torch.ops.aten.mul.Tensor(sum_13, 0.002551020408163265)
    mul_207: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_208: "f32[224]" = torch.ops.aten.mul.Tensor(mul_206, mul_207);  mul_206 = mul_207 = None
    unsqueeze_146: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_208, 0);  mul_208 = None
    unsqueeze_147: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, 2);  unsqueeze_146 = None
    unsqueeze_148: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_147, 3);  unsqueeze_147 = None
    mul_209: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_37);  primals_37 = None
    unsqueeze_149: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_209, 0);  mul_209 = None
    unsqueeze_150: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_149, 2);  unsqueeze_149 = None
    unsqueeze_151: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, 3);  unsqueeze_150 = None
    mul_210: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_148);  sub_39 = unsqueeze_148 = None
    sub_41: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(where_5, mul_210);  where_5 = mul_210 = None
    sub_42: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(sub_41, unsqueeze_145);  sub_41 = unsqueeze_145 = None
    mul_211: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_151);  sub_42 = unsqueeze_151 = None
    mul_212: "f32[224]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_55);  sum_13 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_211, getitem_40, primals_82, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_211 = getitem_40 = primals_82 = None
    getitem_76: "f32[8, 768, 7, 7]" = convolution_backward_8[0]
    getitem_77: "f32[224, 768, 1, 1]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_122: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(slice_1, getitem_76);  slice_1 = getitem_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:145, code: x = self.pool(x)
    max_pool2d_with_indices_backward: "f32[8, 768, 14, 14]" = torch.ops.aten.max_pool2d_with_indices_backward.default(add_122, mul_128, [3, 3], [2, 2], [0, 0], [1, 1], True, getitem_41);  add_122 = mul_128 = getitem_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:71, code: return x * self.gate(x_se)
    mul_213: "f32[8, 768, 14, 14]" = torch.ops.aten.mul.Tensor(max_pool2d_with_indices_backward, relu_17)
    mul_214: "f32[8, 768, 14, 14]" = torch.ops.aten.mul.Tensor(max_pool2d_with_indices_backward, div_2);  max_pool2d_with_indices_backward = div_2 = None
    sum_14: "f32[8, 768, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_213, [2, 3], True);  mul_213 = None
    mul_215: "f32[8, 768, 1, 1]" = torch.ops.aten.mul.Tensor(sum_14, 0.16666666666666666);  sum_14 = None
    where_6: "f32[8, 768, 1, 1]" = torch.ops.aten.where.self(bitwise_and_1, mul_215, full_default);  bitwise_and_1 = mul_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:70, code: x_se = self.fc(x_se)
    sum_15: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(where_6, mean_2, primals_80, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_6 = mean_2 = primals_80 = None
    getitem_79: "f32[8, 768, 1, 1]" = convolution_backward_9[0]
    getitem_80: "f32[768, 768, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:66, code: x_se = x.mean((2, 3), keepdim=True)
    expand_2: "f32[8, 768, 14, 14]" = torch.ops.aten.expand.default(getitem_79, [8, 768, 14, 14]);  getitem_79 = None
    div_6: "f32[8, 768, 14, 14]" = torch.ops.aten.div.Scalar(expand_2, 196);  expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:66, code: x_se = x.mean((2, 3), keepdim=True)
    add_123: "f32[8, 768, 14, 14]" = torch.ops.aten.add.Tensor(mul_214, div_6);  mul_214 = div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_5: "b8[8, 768, 14, 14]" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
    where_7: "f32[8, 768, 14, 14]" = torch.ops.aten.where.self(le_5, full_default, add_123);  le_5 = add_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_16: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_43: "f32[8, 768, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_154);  convolution_30 = unsqueeze_154 = None
    mul_216: "f32[8, 768, 14, 14]" = torch.ops.aten.mul.Tensor(where_7, sub_43)
    sum_17: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_216, [0, 2, 3]);  mul_216 = None
    mul_217: "f32[768]" = torch.ops.aten.mul.Tensor(sum_16, 0.0006377551020408163)
    unsqueeze_155: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_217, 0);  mul_217 = None
    unsqueeze_156: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_155, 2);  unsqueeze_155 = None
    unsqueeze_157: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, 3);  unsqueeze_156 = None
    mul_218: "f32[768]" = torch.ops.aten.mul.Tensor(sum_17, 0.0006377551020408163)
    mul_219: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_220: "f32[768]" = torch.ops.aten.mul.Tensor(mul_218, mul_219);  mul_218 = mul_219 = None
    unsqueeze_158: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_220, 0);  mul_220 = None
    unsqueeze_159: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, 2);  unsqueeze_158 = None
    unsqueeze_160: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_159, 3);  unsqueeze_159 = None
    mul_221: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_35);  primals_35 = None
    unsqueeze_161: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_221, 0);  mul_221 = None
    unsqueeze_162: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_161, 2);  unsqueeze_161 = None
    unsqueeze_163: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, 3);  unsqueeze_162 = None
    mul_222: "f32[8, 768, 14, 14]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_160);  sub_43 = unsqueeze_160 = None
    sub_45: "f32[8, 768, 14, 14]" = torch.ops.aten.sub.Tensor(where_7, mul_222);  where_7 = mul_222 = None
    sub_46: "f32[8, 768, 14, 14]" = torch.ops.aten.sub.Tensor(sub_45, unsqueeze_157);  sub_45 = unsqueeze_157 = None
    mul_223: "f32[8, 768, 14, 14]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_163);  sub_46 = unsqueeze_163 = None
    mul_224: "f32[768]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_52);  sum_17 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_223, cat_2, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_223 = cat_2 = primals_79 = None
    getitem_82: "f32[8, 1088, 14, 14]" = convolution_backward_10[0]
    getitem_83: "f32[768, 1088, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    slice_5: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_82, 1, 0, 512)
    slice_6: "f32[8, 192, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_82, 1, 512, 704)
    slice_7: "f32[8, 192, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_82, 1, 704, 896)
    slice_8: "f32[8, 192, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_82, 1, 896, 1088);  getitem_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_8: "f32[8, 192, 14, 14]" = torch.ops.aten.where.self(le_6, full_default, slice_8);  le_6 = slice_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_18: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_47: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_166);  convolution_29 = unsqueeze_166 = None
    mul_225: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_8, sub_47)
    sum_19: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_225, [0, 2, 3]);  mul_225 = None
    mul_226: "f32[192]" = torch.ops.aten.mul.Tensor(sum_18, 0.0006377551020408163)
    unsqueeze_167: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_226, 0);  mul_226 = None
    unsqueeze_168: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 2);  unsqueeze_167 = None
    unsqueeze_169: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, 3);  unsqueeze_168 = None
    mul_227: "f32[192]" = torch.ops.aten.mul.Tensor(sum_19, 0.0006377551020408163)
    mul_228: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_229: "f32[192]" = torch.ops.aten.mul.Tensor(mul_227, mul_228);  mul_227 = mul_228 = None
    unsqueeze_170: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_229, 0);  mul_229 = None
    unsqueeze_171: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, 2);  unsqueeze_170 = None
    unsqueeze_172: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_171, 3);  unsqueeze_171 = None
    mul_230: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_33);  primals_33 = None
    unsqueeze_173: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_230, 0);  mul_230 = None
    unsqueeze_174: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_173, 2);  unsqueeze_173 = None
    unsqueeze_175: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, 3);  unsqueeze_174 = None
    mul_231: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_172);  sub_47 = unsqueeze_172 = None
    sub_49: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(where_8, mul_231);  where_8 = mul_231 = None
    sub_50: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(sub_49, unsqueeze_169);  sub_49 = unsqueeze_169 = None
    mul_232: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_175);  sub_50 = unsqueeze_175 = None
    mul_233: "f32[192]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_49);  sum_19 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_232, convolution_28, primals_78, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_232 = convolution_28 = primals_78 = None
    getitem_85: "f32[8, 192, 14, 14]" = convolution_backward_11[0]
    getitem_86: "f32[192, 192, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(getitem_85, relu_15, primals_77, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False]);  getitem_85 = primals_77 = None
    getitem_88: "f32[8, 192, 14, 14]" = convolution_backward_12[0]
    getitem_89: "f32[192, 1, 3, 3]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    add_124: "f32[8, 192, 14, 14]" = torch.ops.aten.add.Tensor(slice_7, getitem_88);  slice_7 = getitem_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_7: "b8[8, 192, 14, 14]" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
    where_9: "f32[8, 192, 14, 14]" = torch.ops.aten.where.self(le_7, full_default, add_124);  le_7 = add_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_20: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_51: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_178);  convolution_27 = unsqueeze_178 = None
    mul_234: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_9, sub_51)
    sum_21: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_234, [0, 2, 3]);  mul_234 = None
    mul_235: "f32[192]" = torch.ops.aten.mul.Tensor(sum_20, 0.0006377551020408163)
    unsqueeze_179: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_235, 0);  mul_235 = None
    unsqueeze_180: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_179, 2);  unsqueeze_179 = None
    unsqueeze_181: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, 3);  unsqueeze_180 = None
    mul_236: "f32[192]" = torch.ops.aten.mul.Tensor(sum_21, 0.0006377551020408163)
    mul_237: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_238: "f32[192]" = torch.ops.aten.mul.Tensor(mul_236, mul_237);  mul_236 = mul_237 = None
    unsqueeze_182: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_238, 0);  mul_238 = None
    unsqueeze_183: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, 2);  unsqueeze_182 = None
    unsqueeze_184: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_183, 3);  unsqueeze_183 = None
    mul_239: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_31);  primals_31 = None
    unsqueeze_185: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_239, 0);  mul_239 = None
    unsqueeze_186: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_185, 2);  unsqueeze_185 = None
    unsqueeze_187: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, 3);  unsqueeze_186 = None
    mul_240: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_184);  sub_51 = unsqueeze_184 = None
    sub_53: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(where_9, mul_240);  where_9 = mul_240 = None
    sub_54: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(sub_53, unsqueeze_181);  sub_53 = unsqueeze_181 = None
    mul_241: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_187);  sub_54 = unsqueeze_187 = None
    mul_242: "f32[192]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_46);  sum_21 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_241, convolution_26, primals_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_241 = convolution_26 = primals_76 = None
    getitem_91: "f32[8, 192, 14, 14]" = convolution_backward_13[0]
    getitem_92: "f32[192, 192, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(getitem_91, relu_14, primals_75, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False]);  getitem_91 = primals_75 = None
    getitem_94: "f32[8, 192, 14, 14]" = convolution_backward_14[0]
    getitem_95: "f32[192, 1, 3, 3]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    add_125: "f32[8, 192, 14, 14]" = torch.ops.aten.add.Tensor(slice_6, getitem_94);  slice_6 = getitem_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_8: "b8[8, 192, 14, 14]" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
    where_10: "f32[8, 192, 14, 14]" = torch.ops.aten.where.self(le_8, full_default, add_125);  le_8 = add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_22: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_55: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_190);  convolution_25 = unsqueeze_190 = None
    mul_243: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_10, sub_55)
    sum_23: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_243, [0, 2, 3]);  mul_243 = None
    mul_244: "f32[192]" = torch.ops.aten.mul.Tensor(sum_22, 0.0006377551020408163)
    unsqueeze_191: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_244, 0);  mul_244 = None
    unsqueeze_192: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_191, 2);  unsqueeze_191 = None
    unsqueeze_193: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, 3);  unsqueeze_192 = None
    mul_245: "f32[192]" = torch.ops.aten.mul.Tensor(sum_23, 0.0006377551020408163)
    mul_246: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_247: "f32[192]" = torch.ops.aten.mul.Tensor(mul_245, mul_246);  mul_245 = mul_246 = None
    unsqueeze_194: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_247, 0);  mul_247 = None
    unsqueeze_195: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, 2);  unsqueeze_194 = None
    unsqueeze_196: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_195, 3);  unsqueeze_195 = None
    mul_248: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_29);  primals_29 = None
    unsqueeze_197: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_248, 0);  mul_248 = None
    unsqueeze_198: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 2);  unsqueeze_197 = None
    unsqueeze_199: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, 3);  unsqueeze_198 = None
    mul_249: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_196);  sub_55 = unsqueeze_196 = None
    sub_57: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(where_10, mul_249);  where_10 = mul_249 = None
    sub_58: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(sub_57, unsqueeze_193);  sub_57 = unsqueeze_193 = None
    mul_250: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_199);  sub_58 = unsqueeze_199 = None
    mul_251: "f32[192]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_43);  sum_23 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_250, convolution_24, primals_74, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_250 = convolution_24 = primals_74 = None
    getitem_97: "f32[8, 192, 14, 14]" = convolution_backward_15[0]
    getitem_98: "f32[192, 192, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(getitem_97, relu_13, primals_73, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False]);  getitem_97 = primals_73 = None
    getitem_100: "f32[8, 192, 14, 14]" = convolution_backward_16[0]
    getitem_101: "f32[192, 1, 3, 3]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_9: "b8[8, 192, 14, 14]" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
    where_11: "f32[8, 192, 14, 14]" = torch.ops.aten.where.self(le_9, full_default, getitem_100);  le_9 = getitem_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_24: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_59: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_202);  convolution_23 = unsqueeze_202 = None
    mul_252: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_11, sub_59)
    sum_25: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_252, [0, 2, 3]);  mul_252 = None
    mul_253: "f32[192]" = torch.ops.aten.mul.Tensor(sum_24, 0.0006377551020408163)
    unsqueeze_203: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_253, 0);  mul_253 = None
    unsqueeze_204: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 2);  unsqueeze_203 = None
    unsqueeze_205: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, 3);  unsqueeze_204 = None
    mul_254: "f32[192]" = torch.ops.aten.mul.Tensor(sum_25, 0.0006377551020408163)
    mul_255: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_256: "f32[192]" = torch.ops.aten.mul.Tensor(mul_254, mul_255);  mul_254 = mul_255 = None
    unsqueeze_206: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_256, 0);  mul_256 = None
    unsqueeze_207: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, 2);  unsqueeze_206 = None
    unsqueeze_208: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_207, 3);  unsqueeze_207 = None
    mul_257: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_27);  primals_27 = None
    unsqueeze_209: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_257, 0);  mul_257 = None
    unsqueeze_210: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 2);  unsqueeze_209 = None
    unsqueeze_211: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, 3);  unsqueeze_210 = None
    mul_258: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_208);  sub_59 = unsqueeze_208 = None
    sub_61: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(where_11, mul_258);  where_11 = mul_258 = None
    sub_62: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(sub_61, unsqueeze_205);  sub_61 = unsqueeze_205 = None
    mul_259: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_211);  sub_62 = unsqueeze_211 = None
    mul_260: "f32[192]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_40);  sum_25 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_259, getitem_28, primals_72, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_259 = getitem_28 = primals_72 = None
    getitem_103: "f32[8, 512, 14, 14]" = convolution_backward_17[0]
    getitem_104: "f32[192, 512, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_126: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(slice_5, getitem_103);  slice_5 = getitem_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:145, code: x = self.pool(x)
    max_pool2d_with_indices_backward_1: "f32[8, 512, 28, 28]" = torch.ops.aten.max_pool2d_with_indices_backward.default(add_126, mul_92, [3, 3], [2, 2], [0, 0], [1, 1], True, getitem_29);  add_126 = mul_92 = getitem_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:71, code: return x * self.gate(x_se)
    mul_261: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(max_pool2d_with_indices_backward_1, relu_12)
    mul_262: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(max_pool2d_with_indices_backward_1, div_1);  max_pool2d_with_indices_backward_1 = div_1 = None
    sum_26: "f32[8, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_261, [2, 3], True);  mul_261 = None
    mul_263: "f32[8, 512, 1, 1]" = torch.ops.aten.mul.Tensor(sum_26, 0.16666666666666666);  sum_26 = None
    where_12: "f32[8, 512, 1, 1]" = torch.ops.aten.where.self(bitwise_and_2, mul_263, full_default);  bitwise_and_2 = mul_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:70, code: x_se = self.fc(x_se)
    sum_27: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(where_12, mean_1, primals_70, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_12 = mean_1 = primals_70 = None
    getitem_106: "f32[8, 512, 1, 1]" = convolution_backward_18[0]
    getitem_107: "f32[512, 512, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:66, code: x_se = x.mean((2, 3), keepdim=True)
    expand_3: "f32[8, 512, 28, 28]" = torch.ops.aten.expand.default(getitem_106, [8, 512, 28, 28]);  getitem_106 = None
    div_7: "f32[8, 512, 28, 28]" = torch.ops.aten.div.Scalar(expand_3, 784);  expand_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:66, code: x_se = x.mean((2, 3), keepdim=True)
    add_127: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_262, div_7);  mul_262 = div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_10: "b8[8, 512, 28, 28]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
    where_13: "f32[8, 512, 28, 28]" = torch.ops.aten.where.self(le_10, full_default, add_127);  le_10 = add_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_28: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_63: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_214);  convolution_21 = unsqueeze_214 = None
    mul_264: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_13, sub_63)
    sum_29: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_264, [0, 2, 3]);  mul_264 = None
    mul_265: "f32[512]" = torch.ops.aten.mul.Tensor(sum_28, 0.00015943877551020407)
    unsqueeze_215: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_265, 0);  mul_265 = None
    unsqueeze_216: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 2);  unsqueeze_215 = None
    unsqueeze_217: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, 3);  unsqueeze_216 = None
    mul_266: "f32[512]" = torch.ops.aten.mul.Tensor(sum_29, 0.00015943877551020407)
    mul_267: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_268: "f32[512]" = torch.ops.aten.mul.Tensor(mul_266, mul_267);  mul_266 = mul_267 = None
    unsqueeze_218: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_268, 0);  mul_268 = None
    unsqueeze_219: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 2);  unsqueeze_218 = None
    unsqueeze_220: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_219, 3);  unsqueeze_219 = None
    mul_269: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_25);  primals_25 = None
    unsqueeze_221: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_269, 0);  mul_269 = None
    unsqueeze_222: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 2);  unsqueeze_221 = None
    unsqueeze_223: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, 3);  unsqueeze_222 = None
    mul_270: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_220);  sub_63 = unsqueeze_220 = None
    sub_65: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(where_13, mul_270);  where_13 = mul_270 = None
    sub_66: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(sub_65, unsqueeze_217);  sub_65 = unsqueeze_217 = None
    mul_271: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_223);  sub_66 = unsqueeze_223 = None
    mul_272: "f32[512]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_37);  sum_29 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_271, cat_1, primals_69, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_271 = cat_1 = primals_69 = None
    getitem_109: "f32[8, 736, 28, 28]" = convolution_backward_19[0]
    getitem_110: "f32[512, 736, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    slice_9: "f32[8, 256, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_109, 1, 0, 256)
    slice_10: "f32[8, 160, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_109, 1, 256, 416)
    slice_11: "f32[8, 160, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_109, 1, 416, 576)
    slice_12: "f32[8, 160, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_109, 1, 576, 736);  getitem_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_14: "f32[8, 160, 28, 28]" = torch.ops.aten.where.self(le_11, full_default, slice_12);  le_11 = slice_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_30: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_67: "f32[8, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_226);  convolution_20 = unsqueeze_226 = None
    mul_273: "f32[8, 160, 28, 28]" = torch.ops.aten.mul.Tensor(where_14, sub_67)
    sum_31: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_273, [0, 2, 3]);  mul_273 = None
    mul_274: "f32[160]" = torch.ops.aten.mul.Tensor(sum_30, 0.00015943877551020407)
    unsqueeze_227: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_274, 0);  mul_274 = None
    unsqueeze_228: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 2);  unsqueeze_227 = None
    unsqueeze_229: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, 3);  unsqueeze_228 = None
    mul_275: "f32[160]" = torch.ops.aten.mul.Tensor(sum_31, 0.00015943877551020407)
    mul_276: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_277: "f32[160]" = torch.ops.aten.mul.Tensor(mul_275, mul_276);  mul_275 = mul_276 = None
    unsqueeze_230: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_277, 0);  mul_277 = None
    unsqueeze_231: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 2);  unsqueeze_230 = None
    unsqueeze_232: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_231, 3);  unsqueeze_231 = None
    mul_278: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_23);  primals_23 = None
    unsqueeze_233: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_278, 0);  mul_278 = None
    unsqueeze_234: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 2);  unsqueeze_233 = None
    unsqueeze_235: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, 3);  unsqueeze_234 = None
    mul_279: "f32[8, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_232);  sub_67 = unsqueeze_232 = None
    sub_69: "f32[8, 160, 28, 28]" = torch.ops.aten.sub.Tensor(where_14, mul_279);  where_14 = mul_279 = None
    sub_70: "f32[8, 160, 28, 28]" = torch.ops.aten.sub.Tensor(sub_69, unsqueeze_229);  sub_69 = unsqueeze_229 = None
    mul_280: "f32[8, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_235);  sub_70 = unsqueeze_235 = None
    mul_281: "f32[160]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_34);  sum_31 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_280, convolution_19, primals_68, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_280 = convolution_19 = primals_68 = None
    getitem_112: "f32[8, 160, 28, 28]" = convolution_backward_20[0]
    getitem_113: "f32[160, 160, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(getitem_112, relu_10, primals_67, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 160, [True, True, False]);  getitem_112 = primals_67 = None
    getitem_115: "f32[8, 160, 28, 28]" = convolution_backward_21[0]
    getitem_116: "f32[160, 1, 3, 3]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    add_128: "f32[8, 160, 28, 28]" = torch.ops.aten.add.Tensor(slice_11, getitem_115);  slice_11 = getitem_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_12: "b8[8, 160, 28, 28]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
    where_15: "f32[8, 160, 28, 28]" = torch.ops.aten.where.self(le_12, full_default, add_128);  le_12 = add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_32: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_71: "f32[8, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_238);  convolution_18 = unsqueeze_238 = None
    mul_282: "f32[8, 160, 28, 28]" = torch.ops.aten.mul.Tensor(where_15, sub_71)
    sum_33: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_282, [0, 2, 3]);  mul_282 = None
    mul_283: "f32[160]" = torch.ops.aten.mul.Tensor(sum_32, 0.00015943877551020407)
    unsqueeze_239: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_283, 0);  mul_283 = None
    unsqueeze_240: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 2);  unsqueeze_239 = None
    unsqueeze_241: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, 3);  unsqueeze_240 = None
    mul_284: "f32[160]" = torch.ops.aten.mul.Tensor(sum_33, 0.00015943877551020407)
    mul_285: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_286: "f32[160]" = torch.ops.aten.mul.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None
    unsqueeze_242: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_286, 0);  mul_286 = None
    unsqueeze_243: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 2);  unsqueeze_242 = None
    unsqueeze_244: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_243, 3);  unsqueeze_243 = None
    mul_287: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_21);  primals_21 = None
    unsqueeze_245: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_287, 0);  mul_287 = None
    unsqueeze_246: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 2);  unsqueeze_245 = None
    unsqueeze_247: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, 3);  unsqueeze_246 = None
    mul_288: "f32[8, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_244);  sub_71 = unsqueeze_244 = None
    sub_73: "f32[8, 160, 28, 28]" = torch.ops.aten.sub.Tensor(where_15, mul_288);  where_15 = mul_288 = None
    sub_74: "f32[8, 160, 28, 28]" = torch.ops.aten.sub.Tensor(sub_73, unsqueeze_241);  sub_73 = unsqueeze_241 = None
    mul_289: "f32[8, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_247);  sub_74 = unsqueeze_247 = None
    mul_290: "f32[160]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_31);  sum_33 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_289, convolution_17, primals_66, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_289 = convolution_17 = primals_66 = None
    getitem_118: "f32[8, 160, 28, 28]" = convolution_backward_22[0]
    getitem_119: "f32[160, 160, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(getitem_118, relu_9, primals_65, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 160, [True, True, False]);  getitem_118 = primals_65 = None
    getitem_121: "f32[8, 160, 28, 28]" = convolution_backward_23[0]
    getitem_122: "f32[160, 1, 3, 3]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    add_129: "f32[8, 160, 28, 28]" = torch.ops.aten.add.Tensor(slice_10, getitem_121);  slice_10 = getitem_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_13: "b8[8, 160, 28, 28]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
    where_16: "f32[8, 160, 28, 28]" = torch.ops.aten.where.self(le_13, full_default, add_129);  le_13 = add_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_34: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_75: "f32[8, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_250);  convolution_16 = unsqueeze_250 = None
    mul_291: "f32[8, 160, 28, 28]" = torch.ops.aten.mul.Tensor(where_16, sub_75)
    sum_35: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_291, [0, 2, 3]);  mul_291 = None
    mul_292: "f32[160]" = torch.ops.aten.mul.Tensor(sum_34, 0.00015943877551020407)
    unsqueeze_251: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_292, 0);  mul_292 = None
    unsqueeze_252: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 2);  unsqueeze_251 = None
    unsqueeze_253: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, 3);  unsqueeze_252 = None
    mul_293: "f32[160]" = torch.ops.aten.mul.Tensor(sum_35, 0.00015943877551020407)
    mul_294: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_295: "f32[160]" = torch.ops.aten.mul.Tensor(mul_293, mul_294);  mul_293 = mul_294 = None
    unsqueeze_254: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_295, 0);  mul_295 = None
    unsqueeze_255: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 2);  unsqueeze_254 = None
    unsqueeze_256: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 3);  unsqueeze_255 = None
    mul_296: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_19);  primals_19 = None
    unsqueeze_257: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_296, 0);  mul_296 = None
    unsqueeze_258: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 2);  unsqueeze_257 = None
    unsqueeze_259: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, 3);  unsqueeze_258 = None
    mul_297: "f32[8, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_256);  sub_75 = unsqueeze_256 = None
    sub_77: "f32[8, 160, 28, 28]" = torch.ops.aten.sub.Tensor(where_16, mul_297);  where_16 = mul_297 = None
    sub_78: "f32[8, 160, 28, 28]" = torch.ops.aten.sub.Tensor(sub_77, unsqueeze_253);  sub_77 = unsqueeze_253 = None
    mul_298: "f32[8, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_259);  sub_78 = unsqueeze_259 = None
    mul_299: "f32[160]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_28);  sum_35 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_298, convolution_15, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_298 = convolution_15 = primals_64 = None
    getitem_124: "f32[8, 160, 28, 28]" = convolution_backward_24[0]
    getitem_125: "f32[160, 160, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(getitem_124, relu_8, primals_63, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 160, [True, True, False]);  getitem_124 = primals_63 = None
    getitem_127: "f32[8, 160, 28, 28]" = convolution_backward_25[0]
    getitem_128: "f32[160, 1, 3, 3]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_14: "b8[8, 160, 28, 28]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
    where_17: "f32[8, 160, 28, 28]" = torch.ops.aten.where.self(le_14, full_default, getitem_127);  le_14 = getitem_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_36: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_79: "f32[8, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_262);  convolution_14 = unsqueeze_262 = None
    mul_300: "f32[8, 160, 28, 28]" = torch.ops.aten.mul.Tensor(where_17, sub_79)
    sum_37: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_300, [0, 2, 3]);  mul_300 = None
    mul_301: "f32[160]" = torch.ops.aten.mul.Tensor(sum_36, 0.00015943877551020407)
    unsqueeze_263: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_301, 0);  mul_301 = None
    unsqueeze_264: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 2);  unsqueeze_263 = None
    unsqueeze_265: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, 3);  unsqueeze_264 = None
    mul_302: "f32[160]" = torch.ops.aten.mul.Tensor(sum_37, 0.00015943877551020407)
    mul_303: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_304: "f32[160]" = torch.ops.aten.mul.Tensor(mul_302, mul_303);  mul_302 = mul_303 = None
    unsqueeze_266: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_304, 0);  mul_304 = None
    unsqueeze_267: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 2);  unsqueeze_266 = None
    unsqueeze_268: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 3);  unsqueeze_267 = None
    mul_305: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_17);  primals_17 = None
    unsqueeze_269: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_305, 0);  mul_305 = None
    unsqueeze_270: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 2);  unsqueeze_269 = None
    unsqueeze_271: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 3);  unsqueeze_270 = None
    mul_306: "f32[8, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_268);  sub_79 = unsqueeze_268 = None
    sub_81: "f32[8, 160, 28, 28]" = torch.ops.aten.sub.Tensor(where_17, mul_306);  where_17 = mul_306 = None
    sub_82: "f32[8, 160, 28, 28]" = torch.ops.aten.sub.Tensor(sub_81, unsqueeze_265);  sub_81 = unsqueeze_265 = None
    mul_307: "f32[8, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_271);  sub_82 = unsqueeze_271 = None
    mul_308: "f32[160]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_25);  sum_37 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_307, getitem_16, primals_62, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_307 = getitem_16 = primals_62 = None
    getitem_130: "f32[8, 256, 28, 28]" = convolution_backward_26[0]
    getitem_131: "f32[160, 256, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_130: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(slice_9, getitem_130);  slice_9 = getitem_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:145, code: x = self.pool(x)
    max_pool2d_with_indices_backward_2: "f32[8, 256, 56, 56]" = torch.ops.aten.max_pool2d_with_indices_backward.default(add_130, mul_56, [3, 3], [2, 2], [0, 0], [1, 1], True, getitem_17);  add_130 = mul_56 = getitem_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:71, code: return x * self.gate(x_se)
    mul_309: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(max_pool2d_with_indices_backward_2, relu_7)
    mul_310: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(max_pool2d_with_indices_backward_2, div);  max_pool2d_with_indices_backward_2 = div = None
    sum_38: "f32[8, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_309, [2, 3], True);  mul_309 = None
    mul_311: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(sum_38, 0.16666666666666666);  sum_38 = None
    where_18: "f32[8, 256, 1, 1]" = torch.ops.aten.where.self(bitwise_and_3, mul_311, full_default);  bitwise_and_3 = mul_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:70, code: x_se = self.fc(x_se)
    sum_39: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(where_18, mean, primals_60, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_18 = mean = primals_60 = None
    getitem_133: "f32[8, 256, 1, 1]" = convolution_backward_27[0]
    getitem_134: "f32[256, 256, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:66, code: x_se = x.mean((2, 3), keepdim=True)
    expand_4: "f32[8, 256, 56, 56]" = torch.ops.aten.expand.default(getitem_133, [8, 256, 56, 56]);  getitem_133 = None
    div_8: "f32[8, 256, 56, 56]" = torch.ops.aten.div.Scalar(expand_4, 3136);  expand_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:66, code: x_se = x.mean((2, 3), keepdim=True)
    add_131: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_310, div_8);  mul_310 = div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_15: "b8[8, 256, 56, 56]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
    where_19: "f32[8, 256, 56, 56]" = torch.ops.aten.where.self(le_15, full_default, add_131);  le_15 = add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_40: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_83: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_274);  convolution_12 = unsqueeze_274 = None
    mul_312: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_19, sub_83)
    sum_41: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_312, [0, 2, 3]);  mul_312 = None
    mul_313: "f32[256]" = torch.ops.aten.mul.Tensor(sum_40, 3.985969387755102e-05)
    unsqueeze_275: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_313, 0);  mul_313 = None
    unsqueeze_276: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 2);  unsqueeze_275 = None
    unsqueeze_277: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 3);  unsqueeze_276 = None
    mul_314: "f32[256]" = torch.ops.aten.mul.Tensor(sum_41, 3.985969387755102e-05)
    mul_315: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_316: "f32[256]" = torch.ops.aten.mul.Tensor(mul_314, mul_315);  mul_314 = mul_315 = None
    unsqueeze_278: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_316, 0);  mul_316 = None
    unsqueeze_279: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 2);  unsqueeze_278 = None
    unsqueeze_280: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 3);  unsqueeze_279 = None
    mul_317: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_15);  primals_15 = None
    unsqueeze_281: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_317, 0);  mul_317 = None
    unsqueeze_282: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 2);  unsqueeze_281 = None
    unsqueeze_283: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, 3);  unsqueeze_282 = None
    mul_318: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_280);  sub_83 = unsqueeze_280 = None
    sub_85: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(where_19, mul_318);  where_19 = mul_318 = None
    sub_86: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(sub_85, unsqueeze_277);  sub_85 = unsqueeze_277 = None
    mul_319: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_283);  sub_86 = unsqueeze_283 = None
    mul_320: "f32[256]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_22);  sum_41 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_319, cat, primals_59, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_319 = cat = primals_59 = None
    getitem_136: "f32[8, 448, 56, 56]" = convolution_backward_28[0]
    getitem_137: "f32[256, 448, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    slice_13: "f32[8, 64, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_136, 1, 0, 64)
    slice_14: "f32[8, 128, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_136, 1, 64, 192)
    slice_15: "f32[8, 128, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_136, 1, 192, 320)
    slice_16: "f32[8, 128, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_136, 1, 320, 448);  getitem_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_20: "f32[8, 128, 56, 56]" = torch.ops.aten.where.self(le_16, full_default, slice_16);  le_16 = slice_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_42: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    sub_87: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_286);  convolution_11 = unsqueeze_286 = None
    mul_321: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_20, sub_87)
    sum_43: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_321, [0, 2, 3]);  mul_321 = None
    mul_322: "f32[128]" = torch.ops.aten.mul.Tensor(sum_42, 3.985969387755102e-05)
    unsqueeze_287: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_322, 0);  mul_322 = None
    unsqueeze_288: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 2);  unsqueeze_287 = None
    unsqueeze_289: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 3);  unsqueeze_288 = None
    mul_323: "f32[128]" = torch.ops.aten.mul.Tensor(sum_43, 3.985969387755102e-05)
    mul_324: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_325: "f32[128]" = torch.ops.aten.mul.Tensor(mul_323, mul_324);  mul_323 = mul_324 = None
    unsqueeze_290: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_325, 0);  mul_325 = None
    unsqueeze_291: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 2);  unsqueeze_290 = None
    unsqueeze_292: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 3);  unsqueeze_291 = None
    mul_326: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_13);  primals_13 = None
    unsqueeze_293: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_326, 0);  mul_326 = None
    unsqueeze_294: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 2);  unsqueeze_293 = None
    unsqueeze_295: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, 3);  unsqueeze_294 = None
    mul_327: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_292);  sub_87 = unsqueeze_292 = None
    sub_89: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(where_20, mul_327);  where_20 = mul_327 = None
    sub_90: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(sub_89, unsqueeze_289);  sub_89 = unsqueeze_289 = None
    mul_328: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_295);  sub_90 = unsqueeze_295 = None
    mul_329: "f32[128]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_19);  sum_43 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_328, convolution_10, primals_58, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_328 = convolution_10 = primals_58 = None
    getitem_139: "f32[8, 128, 56, 56]" = convolution_backward_29[0]
    getitem_140: "f32[128, 128, 1, 1]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(getitem_139, relu_5, primals_57, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 128, [True, True, False]);  getitem_139 = primals_57 = None
    getitem_142: "f32[8, 128, 56, 56]" = convolution_backward_30[0]
    getitem_143: "f32[128, 1, 3, 3]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    add_132: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(slice_15, getitem_142);  slice_15 = getitem_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_17: "b8[8, 128, 56, 56]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
    where_21: "f32[8, 128, 56, 56]" = torch.ops.aten.where.self(le_17, full_default, add_132);  le_17 = add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_44: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_91: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_298);  convolution_9 = unsqueeze_298 = None
    mul_330: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_21, sub_91)
    sum_45: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_330, [0, 2, 3]);  mul_330 = None
    mul_331: "f32[128]" = torch.ops.aten.mul.Tensor(sum_44, 3.985969387755102e-05)
    unsqueeze_299: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_331, 0);  mul_331 = None
    unsqueeze_300: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 2);  unsqueeze_299 = None
    unsqueeze_301: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 3);  unsqueeze_300 = None
    mul_332: "f32[128]" = torch.ops.aten.mul.Tensor(sum_45, 3.985969387755102e-05)
    mul_333: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_334: "f32[128]" = torch.ops.aten.mul.Tensor(mul_332, mul_333);  mul_332 = mul_333 = None
    unsqueeze_302: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_334, 0);  mul_334 = None
    unsqueeze_303: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 2);  unsqueeze_302 = None
    unsqueeze_304: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 3);  unsqueeze_303 = None
    mul_335: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_11);  primals_11 = None
    unsqueeze_305: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_335, 0);  mul_335 = None
    unsqueeze_306: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 2);  unsqueeze_305 = None
    unsqueeze_307: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, 3);  unsqueeze_306 = None
    mul_336: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_304);  sub_91 = unsqueeze_304 = None
    sub_93: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(where_21, mul_336);  where_21 = mul_336 = None
    sub_94: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(sub_93, unsqueeze_301);  sub_93 = unsqueeze_301 = None
    mul_337: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_307);  sub_94 = unsqueeze_307 = None
    mul_338: "f32[128]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_16);  sum_45 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_337, convolution_8, primals_56, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_337 = convolution_8 = primals_56 = None
    getitem_145: "f32[8, 128, 56, 56]" = convolution_backward_31[0]
    getitem_146: "f32[128, 128, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(getitem_145, relu_4, primals_55, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 128, [True, True, False]);  getitem_145 = primals_55 = None
    getitem_148: "f32[8, 128, 56, 56]" = convolution_backward_32[0]
    getitem_149: "f32[128, 1, 3, 3]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    add_133: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(slice_14, getitem_148);  slice_14 = getitem_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_18: "b8[8, 128, 56, 56]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
    where_22: "f32[8, 128, 56, 56]" = torch.ops.aten.where.self(le_18, full_default, add_133);  le_18 = add_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_46: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    sub_95: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_310);  convolution_7 = unsqueeze_310 = None
    mul_339: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_22, sub_95)
    sum_47: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_339, [0, 2, 3]);  mul_339 = None
    mul_340: "f32[128]" = torch.ops.aten.mul.Tensor(sum_46, 3.985969387755102e-05)
    unsqueeze_311: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_340, 0);  mul_340 = None
    unsqueeze_312: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 2);  unsqueeze_311 = None
    unsqueeze_313: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, 3);  unsqueeze_312 = None
    mul_341: "f32[128]" = torch.ops.aten.mul.Tensor(sum_47, 3.985969387755102e-05)
    mul_342: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_343: "f32[128]" = torch.ops.aten.mul.Tensor(mul_341, mul_342);  mul_341 = mul_342 = None
    unsqueeze_314: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_343, 0);  mul_343 = None
    unsqueeze_315: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 2);  unsqueeze_314 = None
    unsqueeze_316: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 3);  unsqueeze_315 = None
    mul_344: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_9);  primals_9 = None
    unsqueeze_317: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_344, 0);  mul_344 = None
    unsqueeze_318: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 2);  unsqueeze_317 = None
    unsqueeze_319: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, 3);  unsqueeze_318 = None
    mul_345: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_316);  sub_95 = unsqueeze_316 = None
    sub_97: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(where_22, mul_345);  where_22 = mul_345 = None
    sub_98: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(sub_97, unsqueeze_313);  sub_97 = unsqueeze_313 = None
    mul_346: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_319);  sub_98 = unsqueeze_319 = None
    mul_347: "f32[128]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_13);  sum_47 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_346, convolution_6, primals_54, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_346 = convolution_6 = primals_54 = None
    getitem_151: "f32[8, 128, 56, 56]" = convolution_backward_33[0]
    getitem_152: "f32[128, 128, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(getitem_151, relu_3, primals_53, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 128, [True, True, False]);  getitem_151 = primals_53 = None
    getitem_154: "f32[8, 128, 56, 56]" = convolution_backward_34[0]
    getitem_155: "f32[128, 1, 3, 3]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_19: "b8[8, 128, 56, 56]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
    where_23: "f32[8, 128, 56, 56]" = torch.ops.aten.where.self(le_19, full_default, getitem_154);  le_19 = getitem_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_48: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_99: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_322);  convolution_5 = unsqueeze_322 = None
    mul_348: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_23, sub_99)
    sum_49: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_348, [0, 2, 3]);  mul_348 = None
    mul_349: "f32[128]" = torch.ops.aten.mul.Tensor(sum_48, 3.985969387755102e-05)
    unsqueeze_323: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_349, 0);  mul_349 = None
    unsqueeze_324: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 2);  unsqueeze_323 = None
    unsqueeze_325: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, 3);  unsqueeze_324 = None
    mul_350: "f32[128]" = torch.ops.aten.mul.Tensor(sum_49, 3.985969387755102e-05)
    mul_351: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_352: "f32[128]" = torch.ops.aten.mul.Tensor(mul_350, mul_351);  mul_350 = mul_351 = None
    unsqueeze_326: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_352, 0);  mul_352 = None
    unsqueeze_327: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 2);  unsqueeze_326 = None
    unsqueeze_328: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 3);  unsqueeze_327 = None
    mul_353: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_7);  primals_7 = None
    unsqueeze_329: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_353, 0);  mul_353 = None
    unsqueeze_330: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 2);  unsqueeze_329 = None
    unsqueeze_331: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, 3);  unsqueeze_330 = None
    mul_354: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_328);  sub_99 = unsqueeze_328 = None
    sub_101: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(where_23, mul_354);  where_23 = mul_354 = None
    sub_102: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(sub_101, unsqueeze_325);  sub_101 = unsqueeze_325 = None
    mul_355: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_331);  sub_102 = unsqueeze_331 = None
    mul_356: "f32[128]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_10);  sum_49 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_355, relu_2, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_355 = primals_52 = None
    getitem_157: "f32[8, 64, 56, 56]" = convolution_backward_35[0]
    getitem_158: "f32[128, 64, 1, 1]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_134: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(slice_13, getitem_157);  slice_13 = getitem_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_20: "b8[8, 64, 56, 56]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
    where_24: "f32[8, 64, 56, 56]" = torch.ops.aten.where.self(le_20, full_default, add_134);  le_20 = add_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_50: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_103: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_334);  convolution_4 = unsqueeze_334 = None
    mul_357: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_24, sub_103)
    sum_51: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_357, [0, 2, 3]);  mul_357 = None
    mul_358: "f32[64]" = torch.ops.aten.mul.Tensor(sum_50, 3.985969387755102e-05)
    unsqueeze_335: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_358, 0);  mul_358 = None
    unsqueeze_336: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 2);  unsqueeze_335 = None
    unsqueeze_337: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, 3);  unsqueeze_336 = None
    mul_359: "f32[64]" = torch.ops.aten.mul.Tensor(sum_51, 3.985969387755102e-05)
    mul_360: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_361: "f32[64]" = torch.ops.aten.mul.Tensor(mul_359, mul_360);  mul_359 = mul_360 = None
    unsqueeze_338: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_361, 0);  mul_361 = None
    unsqueeze_339: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 2);  unsqueeze_338 = None
    unsqueeze_340: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 3);  unsqueeze_339 = None
    mul_362: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_5);  primals_5 = None
    unsqueeze_341: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_362, 0);  mul_362 = None
    unsqueeze_342: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 2);  unsqueeze_341 = None
    unsqueeze_343: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 3);  unsqueeze_342 = None
    mul_363: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_340);  sub_103 = unsqueeze_340 = None
    sub_105: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(where_24, mul_363);  where_24 = mul_363 = None
    sub_106: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(sub_105, unsqueeze_337);  sub_105 = unsqueeze_337 = None
    mul_364: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_343);  sub_106 = unsqueeze_343 = None
    mul_365: "f32[64]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_7);  sum_51 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_364, convolution_3, primals_51, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_364 = convolution_3 = primals_51 = None
    getitem_160: "f32[8, 64, 56, 56]" = convolution_backward_36[0]
    getitem_161: "f32[64, 64, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(getitem_160, relu_1, primals_50, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False]);  getitem_160 = primals_50 = None
    getitem_163: "f32[8, 64, 112, 112]" = convolution_backward_37[0]
    getitem_164: "f32[64, 1, 3, 3]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_21: "b8[8, 64, 112, 112]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
    where_25: "f32[8, 64, 112, 112]" = torch.ops.aten.where.self(le_21, full_default, getitem_163);  le_21 = getitem_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_52: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_107: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_346);  convolution_2 = unsqueeze_346 = None
    mul_366: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_25, sub_107)
    sum_53: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_366, [0, 2, 3]);  mul_366 = None
    mul_367: "f32[64]" = torch.ops.aten.mul.Tensor(sum_52, 9.964923469387754e-06)
    unsqueeze_347: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_367, 0);  mul_367 = None
    unsqueeze_348: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 2);  unsqueeze_347 = None
    unsqueeze_349: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, 3);  unsqueeze_348 = None
    mul_368: "f32[64]" = torch.ops.aten.mul.Tensor(sum_53, 9.964923469387754e-06)
    mul_369: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_370: "f32[64]" = torch.ops.aten.mul.Tensor(mul_368, mul_369);  mul_368 = mul_369 = None
    unsqueeze_350: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_370, 0);  mul_370 = None
    unsqueeze_351: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 2);  unsqueeze_350 = None
    unsqueeze_352: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 3);  unsqueeze_351 = None
    mul_371: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_3);  primals_3 = None
    unsqueeze_353: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_371, 0);  mul_371 = None
    unsqueeze_354: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 2);  unsqueeze_353 = None
    unsqueeze_355: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, 3);  unsqueeze_354 = None
    mul_372: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_352);  sub_107 = unsqueeze_352 = None
    sub_109: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(where_25, mul_372);  where_25 = mul_372 = None
    sub_110: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(sub_109, unsqueeze_349);  sub_109 = unsqueeze_349 = None
    mul_373: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_355);  sub_110 = unsqueeze_355 = None
    mul_374: "f32[64]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_4);  sum_53 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_373, convolution_1, primals_49, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_373 = convolution_1 = primals_49 = None
    getitem_166: "f32[8, 64, 112, 112]" = convolution_backward_38[0]
    getitem_167: "f32[64, 64, 1, 1]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(getitem_166, relu, primals_48, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False]);  getitem_166 = primals_48 = None
    getitem_169: "f32[8, 64, 112, 112]" = convolution_backward_39[0]
    getitem_170: "f32[64, 1, 3, 3]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_22: "b8[8, 64, 112, 112]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
    where_26: "f32[8, 64, 112, 112]" = torch.ops.aten.where.self(le_22, full_default, getitem_169);  le_22 = full_default = getitem_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_54: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    sub_111: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_358);  convolution = unsqueeze_358 = None
    mul_375: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_26, sub_111)
    sum_55: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_375, [0, 2, 3]);  mul_375 = None
    mul_376: "f32[64]" = torch.ops.aten.mul.Tensor(sum_54, 9.964923469387754e-06)
    unsqueeze_359: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_376, 0);  mul_376 = None
    unsqueeze_360: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 2);  unsqueeze_359 = None
    unsqueeze_361: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, 3);  unsqueeze_360 = None
    mul_377: "f32[64]" = torch.ops.aten.mul.Tensor(sum_55, 9.964923469387754e-06)
    mul_378: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_379: "f32[64]" = torch.ops.aten.mul.Tensor(mul_377, mul_378);  mul_377 = mul_378 = None
    unsqueeze_362: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_379, 0);  mul_379 = None
    unsqueeze_363: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 2);  unsqueeze_362 = None
    unsqueeze_364: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 3);  unsqueeze_363 = None
    mul_380: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_1);  primals_1 = None
    unsqueeze_365: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_380, 0);  mul_380 = None
    unsqueeze_366: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 2);  unsqueeze_365 = None
    unsqueeze_367: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, 3);  unsqueeze_366 = None
    mul_381: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_364);  sub_111 = unsqueeze_364 = None
    sub_113: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(where_26, mul_381);  where_26 = mul_381 = None
    sub_114: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(sub_113, unsqueeze_361);  sub_113 = unsqueeze_361 = None
    mul_382: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_367);  sub_114 = unsqueeze_367 = None
    mul_383: "f32[64]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_1);  sum_55 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_382, primals_163, primals_47, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_382 = primals_163 = primals_47 = None
    getitem_173: "f32[64, 3, 3, 3]" = convolution_backward_40[1];  convolution_backward_40 = None
    return [mul_383, sum_54, mul_374, sum_52, mul_365, sum_50, mul_356, sum_48, mul_347, sum_46, mul_338, sum_44, mul_329, sum_42, mul_320, sum_40, mul_308, sum_36, mul_299, sum_34, mul_290, sum_32, mul_281, sum_30, mul_272, sum_28, mul_260, sum_24, mul_251, sum_22, mul_242, sum_20, mul_233, sum_18, mul_224, sum_16, mul_212, sum_12, mul_203, sum_10, mul_194, sum_8, mul_185, sum_6, mul_176, sum_4, getitem_173, getitem_170, getitem_167, getitem_164, getitem_161, getitem_158, getitem_155, getitem_152, getitem_149, getitem_146, getitem_143, getitem_140, getitem_137, getitem_134, sum_39, getitem_131, getitem_128, getitem_125, getitem_122, getitem_119, getitem_116, getitem_113, getitem_110, getitem_107, sum_27, getitem_104, getitem_101, getitem_98, getitem_95, getitem_92, getitem_89, getitem_86, getitem_83, getitem_80, sum_15, getitem_77, getitem_74, getitem_71, getitem_68, getitem_65, getitem_62, getitem_59, getitem_56, getitem_53, sum_3, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    