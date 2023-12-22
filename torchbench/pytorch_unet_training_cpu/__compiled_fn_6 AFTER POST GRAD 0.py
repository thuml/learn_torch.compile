from __future__ import annotations



def forward(self, primals_1: "f32[64, 3, 3, 3]", primals_3: "f32[64]", primals_5: "f32[64, 64, 3, 3]", primals_7: "f32[64]", primals_9: "f32[128, 64, 3, 3]", primals_11: "f32[128]", primals_13: "f32[128, 128, 3, 3]", primals_15: "f32[128]", primals_17: "f32[256, 128, 3, 3]", primals_19: "f32[256]", primals_21: "f32[256, 256, 3, 3]", primals_23: "f32[256]", primals_25: "f32[512, 256, 3, 3]", primals_27: "f32[512]", primals_29: "f32[512, 512, 3, 3]", primals_31: "f32[512]", primals_33: "f32[512, 512, 3, 3]", primals_35: "f32[512]", primals_37: "f32[512, 512, 3, 3]", primals_39: "f32[512]", primals_41: "f32[512, 1024, 3, 3]", primals_43: "f32[512]", primals_45: "f32[256, 512, 3, 3]", primals_47: "f32[256]", primals_49: "f32[256, 512, 3, 3]", primals_51: "f32[256]", primals_53: "f32[128, 256, 3, 3]", primals_55: "f32[128]", primals_57: "f32[128, 256, 3, 3]", primals_59: "f32[128]", primals_61: "f32[64, 128, 3, 3]", primals_63: "f32[64]", primals_65: "f32[64, 128, 3, 3]", primals_67: "f32[64]", primals_69: "f32[64, 64, 3, 3]", primals_71: "f32[64]", primals_73: "f32[2, 64, 1, 1]", primals_75: "f32[64]", primals_76: "f32[64]", primals_78: "f32[64]", primals_79: "f32[64]", primals_81: "f32[128]", primals_82: "f32[128]", primals_84: "f32[128]", primals_85: "f32[128]", primals_87: "f32[256]", primals_88: "f32[256]", primals_90: "f32[256]", primals_91: "f32[256]", primals_93: "f32[512]", primals_94: "f32[512]", primals_96: "f32[512]", primals_97: "f32[512]", primals_99: "f32[512]", primals_100: "f32[512]", primals_102: "f32[512]", primals_103: "f32[512]", primals_105: "f32[512]", primals_106: "f32[512]", primals_108: "f32[256]", primals_109: "f32[256]", primals_111: "f32[256]", primals_112: "f32[256]", primals_114: "f32[128]", primals_115: "f32[128]", primals_117: "f32[128]", primals_118: "f32[128]", primals_120: "f32[64]", primals_121: "f32[64]", primals_123: "f32[64]", primals_124: "f32[64]", primals_126: "f32[64]", primals_127: "f32[64]", primals_129: "f32[2, 3, 640, 959]", convolution: "f32[2, 64, 640, 959]", relu: "f32[2, 64, 640, 959]", convolution_1: "f32[2, 64, 640, 959]", relu_1: "f32[2, 64, 640, 959]", getitem: "f32[2, 64, 320, 479]", getitem_1: "i64[2, 64, 320, 479]", convolution_2: "f32[2, 128, 320, 479]", relu_2: "f32[2, 128, 320, 479]", convolution_3: "f32[2, 128, 320, 479]", relu_3: "f32[2, 128, 320, 479]", getitem_2: "f32[2, 128, 160, 239]", getitem_3: "i64[2, 128, 160, 239]", convolution_4: "f32[2, 256, 160, 239]", relu_4: "f32[2, 256, 160, 239]", convolution_5: "f32[2, 256, 160, 239]", relu_5: "f32[2, 256, 160, 239]", getitem_4: "f32[2, 256, 80, 119]", getitem_5: "i64[2, 256, 80, 119]", convolution_6: "f32[2, 512, 80, 119]", relu_6: "f32[2, 512, 80, 119]", convolution_7: "f32[2, 512, 80, 119]", relu_7: "f32[2, 512, 80, 119]", getitem_6: "f32[2, 512, 40, 59]", getitem_7: "i64[2, 512, 40, 59]", convolution_8: "f32[2, 512, 40, 59]", relu_8: "f32[2, 512, 40, 59]", convolution_9: "f32[2, 512, 40, 59]", convert_element_type_26: "i64[118]", convert_element_type_27: "i64[118]", unsqueeze_81: "i64[80, 1]", unsqueeze_82: "i64[80, 1]", sub_10: "f32[80, 1]", sub_12: "f32[118]", cat: "f32[2, 1024, 80, 119]", convolution_10: "f32[2, 512, 80, 119]", relu_10: "f32[2, 512, 80, 119]", convolution_11: "f32[2, 256, 80, 119]", convert_element_type_38: "i64[238]", convert_element_type_39: "i64[238]", unsqueeze_100: "i64[160, 1]", unsqueeze_101: "i64[160, 1]", sub_16: "f32[160, 1]", sub_18: "f32[238]", cat_1: "f32[2, 512, 160, 239]", convolution_12: "f32[2, 256, 160, 239]", relu_12: "f32[2, 256, 160, 239]", convolution_13: "f32[2, 128, 160, 239]", convert_element_type_50: "i64[478]", convert_element_type_51: "i64[478]", unsqueeze_119: "i64[320, 1]", unsqueeze_120: "i64[320, 1]", sub_22: "f32[320, 1]", sub_24: "f32[478]", cat_2: "f32[2, 256, 320, 479]", convolution_14: "f32[2, 128, 320, 479]", relu_14: "f32[2, 128, 320, 479]", convolution_15: "f32[2, 64, 320, 479]", convert_element_type_62: "i64[958]", convert_element_type_63: "i64[958]", unsqueeze_138: "i64[640, 1]", unsqueeze_139: "i64[640, 1]", sub_28: "f32[640, 1]", sub_30: "f32[958]", cat_3: "f32[2, 128, 640, 959]", convolution_16: "f32[2, 64, 640, 959]", relu_16: "f32[2, 64, 640, 959]", convolution_17: "f32[2, 64, 640, 959]", relu_17: "f32[2, 64, 640, 959]", le_2: "b8[2, 64, 320, 479]", le_4: "b8[2, 128, 160, 239]", le_6: "b8[2, 256, 80, 119]", le_8: "b8[2, 512, 40, 59]", tangents_1: "f32[2, 2, 640, 959]"):
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:57, code: x1 = self.up(x1)
    sub_11: "f32[80, 1]" = torch.ops.aten.sub.Tensor(1.0, sub_10)
    sub_13: "f32[118]" = torch.ops.aten.sub.Tensor(1.0, sub_12)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:57, code: x1 = self.up(x1)
    sub_17: "f32[160, 1]" = torch.ops.aten.sub.Tensor(1.0, sub_16)
    sub_19: "f32[238]" = torch.ops.aten.sub.Tensor(1.0, sub_18)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:57, code: x1 = self.up(x1)
    sub_23: "f32[320, 1]" = torch.ops.aten.sub.Tensor(1.0, sub_22)
    sub_25: "f32[478]" = torch.ops.aten.sub.Tensor(1.0, sub_24)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:57, code: x1 = self.up(x1)
    sub_29: "f32[640, 1]" = torch.ops.aten.sub.Tensor(1.0, sub_28)
    sub_31: "f32[958]" = torch.ops.aten.sub.Tensor(1.0, sub_30)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:77, code: return self.conv(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(tangents_1, relu_17, primals_73, [2], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  tangents_1 = primals_73 = None
    getitem_8: "f32[2, 64, 640, 959]" = convolution_backward[0]
    getitem_9: "f32[2, 64, 1, 1]" = convolution_backward[1]
    getitem_10: "f32[2]" = convolution_backward[2];  convolution_backward = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25, code: return self.double_conv(x)
    le: "b8[2, 64, 640, 959]" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where: "f32[2, 64, 640, 959]" = torch.ops.aten.where.self(le, full_default, getitem_8);  le = getitem_8 = None
    add_56: "f32[64]" = torch.ops.aten.add.Tensor(primals_127, 1e-05);  primals_127 = None
    rsqrt: "f32[64]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    unsqueeze_156: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_126, 0);  primals_126 = None
    unsqueeze_157: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, 2);  unsqueeze_156 = None
    unsqueeze_158: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_157, 3);  unsqueeze_157 = None
    sum_1: "f32[64]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_34: "f32[2, 64, 640, 959]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_158);  convolution_17 = unsqueeze_158 = None
    mul_94: "f32[2, 64, 640, 959]" = torch.ops.aten.mul.Tensor(where, sub_34);  sub_34 = None
    sum_2: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_94, [0, 2, 3]);  mul_94 = None
    mul_99: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt, primals_71);  primals_71 = None
    unsqueeze_165: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_99, 0);  mul_99 = None
    unsqueeze_166: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_165, 2);  unsqueeze_165 = None
    unsqueeze_167: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, 3);  unsqueeze_166 = None
    mul_100: "f32[2, 64, 640, 959]" = torch.ops.aten.mul.Tensor(where, unsqueeze_167);  where = unsqueeze_167 = None
    mul_101: "f32[64]" = torch.ops.aten.mul.Tensor(sum_2, rsqrt);  sum_2 = rsqrt = None
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_100, relu_16, primals_69, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_100 = primals_69 = None
    getitem_11: "f32[2, 64, 640, 959]" = convolution_backward_1[0]
    getitem_12: "f32[64, 64, 3, 3]" = convolution_backward_1[1]
    getitem_13: "f32[64]" = convolution_backward_1[2];  convolution_backward_1 = None
    le_1: "b8[2, 64, 640, 959]" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
    where_1: "f32[2, 64, 640, 959]" = torch.ops.aten.where.self(le_1, full_default, getitem_11);  le_1 = getitem_11 = None
    add_57: "f32[64]" = torch.ops.aten.add.Tensor(primals_124, 1e-05);  primals_124 = None
    rsqrt_1: "f32[64]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
    unsqueeze_168: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_123, 0);  primals_123 = None
    unsqueeze_169: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, 2);  unsqueeze_168 = None
    unsqueeze_170: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_169, 3);  unsqueeze_169 = None
    sum_3: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_35: "f32[2, 64, 640, 959]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_170);  convolution_16 = unsqueeze_170 = None
    mul_102: "f32[2, 64, 640, 959]" = torch.ops.aten.mul.Tensor(where_1, sub_35);  sub_35 = None
    sum_4: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_102, [0, 2, 3]);  mul_102 = None
    mul_107: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_1, primals_67);  primals_67 = None
    unsqueeze_177: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_107, 0);  mul_107 = None
    unsqueeze_178: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_177, 2);  unsqueeze_177 = None
    unsqueeze_179: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, 3);  unsqueeze_178 = None
    mul_108: "f32[2, 64, 640, 959]" = torch.ops.aten.mul.Tensor(where_1, unsqueeze_179);  where_1 = unsqueeze_179 = None
    mul_109: "f32[64]" = torch.ops.aten.mul.Tensor(sum_4, rsqrt_1);  sum_4 = rsqrt_1 = None
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_108, cat_3, primals_65, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_108 = cat_3 = primals_65 = None
    getitem_14: "f32[2, 128, 640, 959]" = convolution_backward_2[0]
    getitem_15: "f32[64, 128, 3, 3]" = convolution_backward_2[1]
    getitem_16: "f32[64]" = convolution_backward_2[2];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:67, code: x = torch.cat([x2, x1], dim=1)
    slice_1: "f32[2, 64, 640, 959]" = torch.ops.aten.slice.Tensor(getitem_14, 1, 0, 64)
    slice_2: "f32[2, 64, 640, 959]" = torch.ops.aten.slice.Tensor(getitem_14, 1, 64, 128);  getitem_14 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:62, code: x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
    constant_pad_nd_4: "f32[2, 64, 640, 958]" = torch.ops.aten.constant_pad_nd.default(slice_2, [0, -1, 0, 0]);  slice_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:57, code: x1 = self.up(x1)
    mul_110: "f32[2, 64, 640, 958]" = torch.ops.aten.mul.Tensor(constant_pad_nd_4, sub_30);  sub_30 = None
    mul_111: "f32[2, 64, 640, 958]" = torch.ops.aten.mul.Tensor(constant_pad_nd_4, sub_31);  constant_pad_nd_4 = sub_31 = None
    mul_112: "f32[2, 64, 640, 958]" = torch.ops.aten.mul.Tensor(mul_110, sub_28)
    mul_113: "f32[2, 64, 640, 958]" = torch.ops.aten.mul.Tensor(mul_110, sub_29);  mul_110 = None
    mul_114: "f32[2, 64, 640, 958]" = torch.ops.aten.mul.Tensor(mul_111, sub_28);  sub_28 = None
    mul_115: "f32[2, 64, 640, 958]" = torch.ops.aten.mul.Tensor(mul_111, sub_29);  mul_111 = sub_29 = None
    full_default_2: "f32[2, 64, 320, 479]" = torch.ops.aten.full.default([2, 64, 320, 479], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put: "f32[2, 64, 320, 479]" = torch.ops.aten._unsafe_index_put.default(full_default_2, [None, None, unsqueeze_139, convert_element_type_63], mul_112, True);  mul_112 = None
    _unsafe_index_put_1: "f32[2, 64, 320, 479]" = torch.ops.aten._unsafe_index_put.default(full_default_2, [None, None, unsqueeze_138, convert_element_type_63], mul_113, True);  convert_element_type_63 = mul_113 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:57, code: x1 = self.up(x1)
    add_58: "f32[2, 64, 320, 479]" = torch.ops.aten.add.Tensor(_unsafe_index_put, _unsafe_index_put_1);  _unsafe_index_put = _unsafe_index_put_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:57, code: x1 = self.up(x1)
    _unsafe_index_put_2: "f32[2, 64, 320, 479]" = torch.ops.aten._unsafe_index_put.default(full_default_2, [None, None, unsqueeze_139, convert_element_type_62], mul_114, True);  unsqueeze_139 = mul_114 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:57, code: x1 = self.up(x1)
    add_59: "f32[2, 64, 320, 479]" = torch.ops.aten.add.Tensor(add_58, _unsafe_index_put_2);  add_58 = _unsafe_index_put_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:57, code: x1 = self.up(x1)
    _unsafe_index_put_3: "f32[2, 64, 320, 479]" = torch.ops.prims._unsafe_index_put_.default(full_default_2, [None, None, unsqueeze_138, convert_element_type_62], mul_115, True);  full_default_2 = unsqueeze_138 = convert_element_type_62 = mul_115 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:57, code: x1 = self.up(x1)
    add_60: "f32[2, 64, 320, 479]" = torch.ops.aten.add.Tensor(add_59, _unsafe_index_put_3);  add_59 = _unsafe_index_put_3 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25, code: return self.double_conv(x)
    where_2: "f32[2, 64, 320, 479]" = torch.ops.aten.where.self(le_2, full_default, add_60);  le_2 = add_60 = None
    add_61: "f32[64]" = torch.ops.aten.add.Tensor(primals_121, 1e-05);  primals_121 = None
    rsqrt_2: "f32[64]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
    unsqueeze_180: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_120, 0);  primals_120 = None
    unsqueeze_181: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, 2);  unsqueeze_180 = None
    unsqueeze_182: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_181, 3);  unsqueeze_181 = None
    sum_5: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_36: "f32[2, 64, 320, 479]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_182);  convolution_15 = unsqueeze_182 = None
    mul_116: "f32[2, 64, 320, 479]" = torch.ops.aten.mul.Tensor(where_2, sub_36);  sub_36 = None
    sum_6: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_116, [0, 2, 3]);  mul_116 = None
    mul_121: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_2, primals_63);  primals_63 = None
    unsqueeze_189: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_121, 0);  mul_121 = None
    unsqueeze_190: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_189, 2);  unsqueeze_189 = None
    unsqueeze_191: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, 3);  unsqueeze_190 = None
    mul_122: "f32[2, 64, 320, 479]" = torch.ops.aten.mul.Tensor(where_2, unsqueeze_191);  where_2 = unsqueeze_191 = None
    mul_123: "f32[64]" = torch.ops.aten.mul.Tensor(sum_6, rsqrt_2);  sum_6 = rsqrt_2 = None
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_122, relu_14, primals_61, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_122 = primals_61 = None
    getitem_17: "f32[2, 128, 320, 479]" = convolution_backward_3[0]
    getitem_18: "f32[64, 128, 3, 3]" = convolution_backward_3[1]
    getitem_19: "f32[64]" = convolution_backward_3[2];  convolution_backward_3 = None
    le_3: "b8[2, 128, 320, 479]" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
    where_3: "f32[2, 128, 320, 479]" = torch.ops.aten.where.self(le_3, full_default, getitem_17);  le_3 = getitem_17 = None
    add_62: "f32[128]" = torch.ops.aten.add.Tensor(primals_118, 1e-05);  primals_118 = None
    rsqrt_3: "f32[128]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    unsqueeze_192: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_117, 0);  primals_117 = None
    unsqueeze_193: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, 2);  unsqueeze_192 = None
    unsqueeze_194: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_193, 3);  unsqueeze_193 = None
    sum_7: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_37: "f32[2, 128, 320, 479]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_194);  convolution_14 = unsqueeze_194 = None
    mul_124: "f32[2, 128, 320, 479]" = torch.ops.aten.mul.Tensor(where_3, sub_37);  sub_37 = None
    sum_8: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_124, [0, 2, 3]);  mul_124 = None
    mul_129: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_3, primals_59);  primals_59 = None
    unsqueeze_201: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_129, 0);  mul_129 = None
    unsqueeze_202: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_201, 2);  unsqueeze_201 = None
    unsqueeze_203: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, 3);  unsqueeze_202 = None
    mul_130: "f32[2, 128, 320, 479]" = torch.ops.aten.mul.Tensor(where_3, unsqueeze_203);  where_3 = unsqueeze_203 = None
    mul_131: "f32[128]" = torch.ops.aten.mul.Tensor(sum_8, rsqrt_3);  sum_8 = rsqrt_3 = None
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_130, cat_2, primals_57, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_130 = cat_2 = primals_57 = None
    getitem_20: "f32[2, 256, 320, 479]" = convolution_backward_4[0]
    getitem_21: "f32[128, 256, 3, 3]" = convolution_backward_4[1]
    getitem_22: "f32[128]" = convolution_backward_4[2];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:67, code: x = torch.cat([x2, x1], dim=1)
    slice_3: "f32[2, 128, 320, 479]" = torch.ops.aten.slice.Tensor(getitem_20, 1, 0, 128)
    slice_4: "f32[2, 128, 320, 479]" = torch.ops.aten.slice.Tensor(getitem_20, 1, 128, 256);  getitem_20 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:62, code: x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
    constant_pad_nd_5: "f32[2, 128, 320, 478]" = torch.ops.aten.constant_pad_nd.default(slice_4, [0, -1, 0, 0]);  slice_4 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:57, code: x1 = self.up(x1)
    mul_132: "f32[2, 128, 320, 478]" = torch.ops.aten.mul.Tensor(constant_pad_nd_5, sub_24);  sub_24 = None
    mul_133: "f32[2, 128, 320, 478]" = torch.ops.aten.mul.Tensor(constant_pad_nd_5, sub_25);  constant_pad_nd_5 = sub_25 = None
    mul_134: "f32[2, 128, 320, 478]" = torch.ops.aten.mul.Tensor(mul_132, sub_22)
    mul_135: "f32[2, 128, 320, 478]" = torch.ops.aten.mul.Tensor(mul_132, sub_23);  mul_132 = None
    mul_136: "f32[2, 128, 320, 478]" = torch.ops.aten.mul.Tensor(mul_133, sub_22);  sub_22 = None
    mul_137: "f32[2, 128, 320, 478]" = torch.ops.aten.mul.Tensor(mul_133, sub_23);  mul_133 = sub_23 = None
    full_default_8: "f32[2, 128, 160, 239]" = torch.ops.aten.full.default([2, 128, 160, 239], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_4: "f32[2, 128, 160, 239]" = torch.ops.aten._unsafe_index_put.default(full_default_8, [None, None, unsqueeze_120, convert_element_type_51], mul_134, True);  mul_134 = None
    _unsafe_index_put_5: "f32[2, 128, 160, 239]" = torch.ops.aten._unsafe_index_put.default(full_default_8, [None, None, unsqueeze_119, convert_element_type_51], mul_135, True);  convert_element_type_51 = mul_135 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:57, code: x1 = self.up(x1)
    add_63: "f32[2, 128, 160, 239]" = torch.ops.aten.add.Tensor(_unsafe_index_put_4, _unsafe_index_put_5);  _unsafe_index_put_4 = _unsafe_index_put_5 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:57, code: x1 = self.up(x1)
    _unsafe_index_put_6: "f32[2, 128, 160, 239]" = torch.ops.aten._unsafe_index_put.default(full_default_8, [None, None, unsqueeze_120, convert_element_type_50], mul_136, True);  unsqueeze_120 = mul_136 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:57, code: x1 = self.up(x1)
    add_64: "f32[2, 128, 160, 239]" = torch.ops.aten.add.Tensor(add_63, _unsafe_index_put_6);  add_63 = _unsafe_index_put_6 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:57, code: x1 = self.up(x1)
    _unsafe_index_put_7: "f32[2, 128, 160, 239]" = torch.ops.prims._unsafe_index_put_.default(full_default_8, [None, None, unsqueeze_119, convert_element_type_50], mul_137, True);  full_default_8 = unsqueeze_119 = convert_element_type_50 = mul_137 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:57, code: x1 = self.up(x1)
    add_65: "f32[2, 128, 160, 239]" = torch.ops.aten.add.Tensor(add_64, _unsafe_index_put_7);  add_64 = _unsafe_index_put_7 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25, code: return self.double_conv(x)
    where_4: "f32[2, 128, 160, 239]" = torch.ops.aten.where.self(le_4, full_default, add_65);  le_4 = add_65 = None
    add_66: "f32[128]" = torch.ops.aten.add.Tensor(primals_115, 1e-05);  primals_115 = None
    rsqrt_4: "f32[128]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    unsqueeze_204: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_114, 0);  primals_114 = None
    unsqueeze_205: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, 2);  unsqueeze_204 = None
    unsqueeze_206: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 3);  unsqueeze_205 = None
    sum_9: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_38: "f32[2, 128, 160, 239]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_206);  convolution_13 = unsqueeze_206 = None
    mul_138: "f32[2, 128, 160, 239]" = torch.ops.aten.mul.Tensor(where_4, sub_38);  sub_38 = None
    sum_10: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_138, [0, 2, 3]);  mul_138 = None
    mul_143: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_4, primals_55);  primals_55 = None
    unsqueeze_213: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_143, 0);  mul_143 = None
    unsqueeze_214: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 2);  unsqueeze_213 = None
    unsqueeze_215: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, 3);  unsqueeze_214 = None
    mul_144: "f32[2, 128, 160, 239]" = torch.ops.aten.mul.Tensor(where_4, unsqueeze_215);  where_4 = unsqueeze_215 = None
    mul_145: "f32[128]" = torch.ops.aten.mul.Tensor(sum_10, rsqrt_4);  sum_10 = rsqrt_4 = None
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_144, relu_12, primals_53, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_144 = primals_53 = None
    getitem_23: "f32[2, 256, 160, 239]" = convolution_backward_5[0]
    getitem_24: "f32[128, 256, 3, 3]" = convolution_backward_5[1]
    getitem_25: "f32[128]" = convolution_backward_5[2];  convolution_backward_5 = None
    le_5: "b8[2, 256, 160, 239]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
    where_5: "f32[2, 256, 160, 239]" = torch.ops.aten.where.self(le_5, full_default, getitem_23);  le_5 = getitem_23 = None
    add_67: "f32[256]" = torch.ops.aten.add.Tensor(primals_112, 1e-05);  primals_112 = None
    rsqrt_5: "f32[256]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    unsqueeze_216: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_111, 0);  primals_111 = None
    unsqueeze_217: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, 2);  unsqueeze_216 = None
    unsqueeze_218: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 3);  unsqueeze_217 = None
    sum_11: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_39: "f32[2, 256, 160, 239]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_218);  convolution_12 = unsqueeze_218 = None
    mul_146: "f32[2, 256, 160, 239]" = torch.ops.aten.mul.Tensor(where_5, sub_39);  sub_39 = None
    sum_12: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_146, [0, 2, 3]);  mul_146 = None
    mul_151: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_5, primals_51);  primals_51 = None
    unsqueeze_225: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_151, 0);  mul_151 = None
    unsqueeze_226: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_225, 2);  unsqueeze_225 = None
    unsqueeze_227: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 3);  unsqueeze_226 = None
    mul_152: "f32[2, 256, 160, 239]" = torch.ops.aten.mul.Tensor(where_5, unsqueeze_227);  where_5 = unsqueeze_227 = None
    mul_153: "f32[256]" = torch.ops.aten.mul.Tensor(sum_12, rsqrt_5);  sum_12 = rsqrt_5 = None
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_152, cat_1, primals_49, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_152 = cat_1 = primals_49 = None
    getitem_26: "f32[2, 512, 160, 239]" = convolution_backward_6[0]
    getitem_27: "f32[256, 512, 3, 3]" = convolution_backward_6[1]
    getitem_28: "f32[256]" = convolution_backward_6[2];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:67, code: x = torch.cat([x2, x1], dim=1)
    slice_5: "f32[2, 256, 160, 239]" = torch.ops.aten.slice.Tensor(getitem_26, 1, 0, 256)
    slice_6: "f32[2, 256, 160, 239]" = torch.ops.aten.slice.Tensor(getitem_26, 1, 256, 512);  getitem_26 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:62, code: x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
    constant_pad_nd_6: "f32[2, 256, 160, 238]" = torch.ops.aten.constant_pad_nd.default(slice_6, [0, -1, 0, 0]);  slice_6 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:57, code: x1 = self.up(x1)
    mul_154: "f32[2, 256, 160, 238]" = torch.ops.aten.mul.Tensor(constant_pad_nd_6, sub_18);  sub_18 = None
    mul_155: "f32[2, 256, 160, 238]" = torch.ops.aten.mul.Tensor(constant_pad_nd_6, sub_19);  constant_pad_nd_6 = sub_19 = None
    mul_156: "f32[2, 256, 160, 238]" = torch.ops.aten.mul.Tensor(mul_154, sub_16)
    mul_157: "f32[2, 256, 160, 238]" = torch.ops.aten.mul.Tensor(mul_154, sub_17);  mul_154 = None
    mul_158: "f32[2, 256, 160, 238]" = torch.ops.aten.mul.Tensor(mul_155, sub_16);  sub_16 = None
    mul_159: "f32[2, 256, 160, 238]" = torch.ops.aten.mul.Tensor(mul_155, sub_17);  mul_155 = sub_17 = None
    full_default_14: "f32[2, 256, 80, 119]" = torch.ops.aten.full.default([2, 256, 80, 119], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_8: "f32[2, 256, 80, 119]" = torch.ops.aten._unsafe_index_put.default(full_default_14, [None, None, unsqueeze_101, convert_element_type_39], mul_156, True);  mul_156 = None
    _unsafe_index_put_9: "f32[2, 256, 80, 119]" = torch.ops.aten._unsafe_index_put.default(full_default_14, [None, None, unsqueeze_100, convert_element_type_39], mul_157, True);  convert_element_type_39 = mul_157 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:57, code: x1 = self.up(x1)
    add_68: "f32[2, 256, 80, 119]" = torch.ops.aten.add.Tensor(_unsafe_index_put_8, _unsafe_index_put_9);  _unsafe_index_put_8 = _unsafe_index_put_9 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:57, code: x1 = self.up(x1)
    _unsafe_index_put_10: "f32[2, 256, 80, 119]" = torch.ops.aten._unsafe_index_put.default(full_default_14, [None, None, unsqueeze_101, convert_element_type_38], mul_158, True);  unsqueeze_101 = mul_158 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:57, code: x1 = self.up(x1)
    add_69: "f32[2, 256, 80, 119]" = torch.ops.aten.add.Tensor(add_68, _unsafe_index_put_10);  add_68 = _unsafe_index_put_10 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:57, code: x1 = self.up(x1)
    _unsafe_index_put_11: "f32[2, 256, 80, 119]" = torch.ops.prims._unsafe_index_put_.default(full_default_14, [None, None, unsqueeze_100, convert_element_type_38], mul_159, True);  full_default_14 = unsqueeze_100 = convert_element_type_38 = mul_159 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:57, code: x1 = self.up(x1)
    add_70: "f32[2, 256, 80, 119]" = torch.ops.aten.add.Tensor(add_69, _unsafe_index_put_11);  add_69 = _unsafe_index_put_11 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25, code: return self.double_conv(x)
    where_6: "f32[2, 256, 80, 119]" = torch.ops.aten.where.self(le_6, full_default, add_70);  le_6 = add_70 = None
    add_71: "f32[256]" = torch.ops.aten.add.Tensor(primals_109, 1e-05);  primals_109 = None
    rsqrt_6: "f32[256]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
    unsqueeze_228: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_108, 0);  primals_108 = None
    unsqueeze_229: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, 2);  unsqueeze_228 = None
    unsqueeze_230: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 3);  unsqueeze_229 = None
    sum_13: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_40: "f32[2, 256, 80, 119]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_230);  convolution_11 = unsqueeze_230 = None
    mul_160: "f32[2, 256, 80, 119]" = torch.ops.aten.mul.Tensor(where_6, sub_40);  sub_40 = None
    sum_14: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_160, [0, 2, 3]);  mul_160 = None
    mul_165: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_6, primals_47);  primals_47 = None
    unsqueeze_237: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_165, 0);  mul_165 = None
    unsqueeze_238: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_237, 2);  unsqueeze_237 = None
    unsqueeze_239: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 3);  unsqueeze_238 = None
    mul_166: "f32[2, 256, 80, 119]" = torch.ops.aten.mul.Tensor(where_6, unsqueeze_239);  where_6 = unsqueeze_239 = None
    mul_167: "f32[256]" = torch.ops.aten.mul.Tensor(sum_14, rsqrt_6);  sum_14 = rsqrt_6 = None
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_166, relu_10, primals_45, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_166 = primals_45 = None
    getitem_29: "f32[2, 512, 80, 119]" = convolution_backward_7[0]
    getitem_30: "f32[256, 512, 3, 3]" = convolution_backward_7[1]
    getitem_31: "f32[256]" = convolution_backward_7[2];  convolution_backward_7 = None
    le_7: "b8[2, 512, 80, 119]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
    where_7: "f32[2, 512, 80, 119]" = torch.ops.aten.where.self(le_7, full_default, getitem_29);  le_7 = getitem_29 = None
    add_72: "f32[512]" = torch.ops.aten.add.Tensor(primals_106, 1e-05);  primals_106 = None
    rsqrt_7: "f32[512]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
    unsqueeze_240: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_105, 0);  primals_105 = None
    unsqueeze_241: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, 2);  unsqueeze_240 = None
    unsqueeze_242: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 3);  unsqueeze_241 = None
    sum_15: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_41: "f32[2, 512, 80, 119]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_242);  convolution_10 = unsqueeze_242 = None
    mul_168: "f32[2, 512, 80, 119]" = torch.ops.aten.mul.Tensor(where_7, sub_41);  sub_41 = None
    sum_16: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_168, [0, 2, 3]);  mul_168 = None
    mul_173: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_7, primals_43);  primals_43 = None
    unsqueeze_249: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_173, 0);  mul_173 = None
    unsqueeze_250: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_249, 2);  unsqueeze_249 = None
    unsqueeze_251: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 3);  unsqueeze_250 = None
    mul_174: "f32[2, 512, 80, 119]" = torch.ops.aten.mul.Tensor(where_7, unsqueeze_251);  where_7 = unsqueeze_251 = None
    mul_175: "f32[512]" = torch.ops.aten.mul.Tensor(sum_16, rsqrt_7);  sum_16 = rsqrt_7 = None
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_174, cat, primals_41, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_174 = cat = primals_41 = None
    getitem_32: "f32[2, 1024, 80, 119]" = convolution_backward_8[0]
    getitem_33: "f32[512, 1024, 3, 3]" = convolution_backward_8[1]
    getitem_34: "f32[512]" = convolution_backward_8[2];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:67, code: x = torch.cat([x2, x1], dim=1)
    slice_7: "f32[2, 512, 80, 119]" = torch.ops.aten.slice.Tensor(getitem_32, 1, 0, 512)
    slice_8: "f32[2, 512, 80, 119]" = torch.ops.aten.slice.Tensor(getitem_32, 1, 512, 1024);  getitem_32 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:62, code: x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
    constant_pad_nd_7: "f32[2, 512, 80, 118]" = torch.ops.aten.constant_pad_nd.default(slice_8, [0, -1, 0, 0]);  slice_8 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:57, code: x1 = self.up(x1)
    mul_176: "f32[2, 512, 80, 118]" = torch.ops.aten.mul.Tensor(constant_pad_nd_7, sub_12);  sub_12 = None
    mul_177: "f32[2, 512, 80, 118]" = torch.ops.aten.mul.Tensor(constant_pad_nd_7, sub_13);  constant_pad_nd_7 = sub_13 = None
    mul_178: "f32[2, 512, 80, 118]" = torch.ops.aten.mul.Tensor(mul_176, sub_10)
    mul_179: "f32[2, 512, 80, 118]" = torch.ops.aten.mul.Tensor(mul_176, sub_11);  mul_176 = None
    mul_180: "f32[2, 512, 80, 118]" = torch.ops.aten.mul.Tensor(mul_177, sub_10);  sub_10 = None
    mul_181: "f32[2, 512, 80, 118]" = torch.ops.aten.mul.Tensor(mul_177, sub_11);  mul_177 = sub_11 = None
    full_default_20: "f32[2, 512, 40, 59]" = torch.ops.aten.full.default([2, 512, 40, 59], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_12: "f32[2, 512, 40, 59]" = torch.ops.aten._unsafe_index_put.default(full_default_20, [None, None, unsqueeze_82, convert_element_type_27], mul_178, True);  mul_178 = None
    _unsafe_index_put_13: "f32[2, 512, 40, 59]" = torch.ops.aten._unsafe_index_put.default(full_default_20, [None, None, unsqueeze_81, convert_element_type_27], mul_179, True);  convert_element_type_27 = mul_179 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:57, code: x1 = self.up(x1)
    add_73: "f32[2, 512, 40, 59]" = torch.ops.aten.add.Tensor(_unsafe_index_put_12, _unsafe_index_put_13);  _unsafe_index_put_12 = _unsafe_index_put_13 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:57, code: x1 = self.up(x1)
    _unsafe_index_put_14: "f32[2, 512, 40, 59]" = torch.ops.aten._unsafe_index_put.default(full_default_20, [None, None, unsqueeze_82, convert_element_type_26], mul_180, True);  unsqueeze_82 = mul_180 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:57, code: x1 = self.up(x1)
    add_74: "f32[2, 512, 40, 59]" = torch.ops.aten.add.Tensor(add_73, _unsafe_index_put_14);  add_73 = _unsafe_index_put_14 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:57, code: x1 = self.up(x1)
    _unsafe_index_put_15: "f32[2, 512, 40, 59]" = torch.ops.prims._unsafe_index_put_.default(full_default_20, [None, None, unsqueeze_81, convert_element_type_26], mul_181, True);  full_default_20 = unsqueeze_81 = convert_element_type_26 = mul_181 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:57, code: x1 = self.up(x1)
    add_75: "f32[2, 512, 40, 59]" = torch.ops.aten.add.Tensor(add_74, _unsafe_index_put_15);  add_74 = _unsafe_index_put_15 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25, code: return self.double_conv(x)
    where_8: "f32[2, 512, 40, 59]" = torch.ops.aten.where.self(le_8, full_default, add_75);  le_8 = add_75 = None
    add_76: "f32[512]" = torch.ops.aten.add.Tensor(primals_103, 1e-05);  primals_103 = None
    rsqrt_8: "f32[512]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
    unsqueeze_252: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_102, 0);  primals_102 = None
    unsqueeze_253: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, 2);  unsqueeze_252 = None
    unsqueeze_254: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 3);  unsqueeze_253 = None
    sum_17: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_42: "f32[2, 512, 40, 59]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_254);  convolution_9 = unsqueeze_254 = None
    mul_182: "f32[2, 512, 40, 59]" = torch.ops.aten.mul.Tensor(where_8, sub_42);  sub_42 = None
    sum_18: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_182, [0, 2, 3]);  mul_182 = None
    mul_187: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_8, primals_39);  primals_39 = None
    unsqueeze_261: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_187, 0);  mul_187 = None
    unsqueeze_262: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 2);  unsqueeze_261 = None
    unsqueeze_263: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, 3);  unsqueeze_262 = None
    mul_188: "f32[2, 512, 40, 59]" = torch.ops.aten.mul.Tensor(where_8, unsqueeze_263);  where_8 = unsqueeze_263 = None
    mul_189: "f32[512]" = torch.ops.aten.mul.Tensor(sum_18, rsqrt_8);  sum_18 = rsqrt_8 = None
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_188, relu_8, primals_37, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_188 = primals_37 = None
    getitem_35: "f32[2, 512, 40, 59]" = convolution_backward_9[0]
    getitem_36: "f32[512, 512, 3, 3]" = convolution_backward_9[1]
    getitem_37: "f32[512]" = convolution_backward_9[2];  convolution_backward_9 = None
    le_9: "b8[2, 512, 40, 59]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
    where_9: "f32[2, 512, 40, 59]" = torch.ops.aten.where.self(le_9, full_default, getitem_35);  le_9 = getitem_35 = None
    add_77: "f32[512]" = torch.ops.aten.add.Tensor(primals_100, 1e-05);  primals_100 = None
    rsqrt_9: "f32[512]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
    unsqueeze_264: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_99, 0);  primals_99 = None
    unsqueeze_265: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, 2);  unsqueeze_264 = None
    unsqueeze_266: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 3);  unsqueeze_265 = None
    sum_19: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_43: "f32[2, 512, 40, 59]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_266);  convolution_8 = unsqueeze_266 = None
    mul_190: "f32[2, 512, 40, 59]" = torch.ops.aten.mul.Tensor(where_9, sub_43);  sub_43 = None
    sum_20: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_190, [0, 2, 3]);  mul_190 = None
    mul_195: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_9, primals_35);  primals_35 = None
    unsqueeze_273: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_195, 0);  mul_195 = None
    unsqueeze_274: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 2);  unsqueeze_273 = None
    unsqueeze_275: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 3);  unsqueeze_274 = None
    mul_196: "f32[2, 512, 40, 59]" = torch.ops.aten.mul.Tensor(where_9, unsqueeze_275);  where_9 = unsqueeze_275 = None
    mul_197: "f32[512]" = torch.ops.aten.mul.Tensor(sum_20, rsqrt_9);  sum_20 = rsqrt_9 = None
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_196, getitem_6, primals_33, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_196 = getitem_6 = primals_33 = None
    getitem_38: "f32[2, 512, 40, 59]" = convolution_backward_10[0]
    getitem_39: "f32[512, 512, 3, 3]" = convolution_backward_10[1]
    getitem_40: "f32[512]" = convolution_backward_10[2];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:39, code: return self.maxpool_conv(x)
    max_pool2d_with_indices_backward: "f32[2, 512, 80, 119]" = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_38, relu_7, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_7);  getitem_38 = getitem_7 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:39, code: return self.maxpool_conv(x)
    add_78: "f32[2, 512, 80, 119]" = torch.ops.aten.add.Tensor(slice_7, max_pool2d_with_indices_backward);  slice_7 = max_pool2d_with_indices_backward = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25, code: return self.double_conv(x)
    le_10: "b8[2, 512, 80, 119]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
    where_10: "f32[2, 512, 80, 119]" = torch.ops.aten.where.self(le_10, full_default, add_78);  le_10 = add_78 = None
    add_79: "f32[512]" = torch.ops.aten.add.Tensor(primals_97, 1e-05);  primals_97 = None
    rsqrt_10: "f32[512]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
    unsqueeze_276: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_96, 0);  primals_96 = None
    unsqueeze_277: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 2);  unsqueeze_276 = None
    unsqueeze_278: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 3);  unsqueeze_277 = None
    sum_21: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_44: "f32[2, 512, 80, 119]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_278);  convolution_7 = unsqueeze_278 = None
    mul_198: "f32[2, 512, 80, 119]" = torch.ops.aten.mul.Tensor(where_10, sub_44);  sub_44 = None
    sum_22: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_198, [0, 2, 3]);  mul_198 = None
    mul_203: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_10, primals_31);  primals_31 = None
    unsqueeze_285: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_203, 0);  mul_203 = None
    unsqueeze_286: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 2);  unsqueeze_285 = None
    unsqueeze_287: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 3);  unsqueeze_286 = None
    mul_204: "f32[2, 512, 80, 119]" = torch.ops.aten.mul.Tensor(where_10, unsqueeze_287);  where_10 = unsqueeze_287 = None
    mul_205: "f32[512]" = torch.ops.aten.mul.Tensor(sum_22, rsqrt_10);  sum_22 = rsqrt_10 = None
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_204, relu_6, primals_29, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_204 = primals_29 = None
    getitem_41: "f32[2, 512, 80, 119]" = convolution_backward_11[0]
    getitem_42: "f32[512, 512, 3, 3]" = convolution_backward_11[1]
    getitem_43: "f32[512]" = convolution_backward_11[2];  convolution_backward_11 = None
    le_11: "b8[2, 512, 80, 119]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
    where_11: "f32[2, 512, 80, 119]" = torch.ops.aten.where.self(le_11, full_default, getitem_41);  le_11 = getitem_41 = None
    add_80: "f32[512]" = torch.ops.aten.add.Tensor(primals_94, 1e-05);  primals_94 = None
    rsqrt_11: "f32[512]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
    unsqueeze_288: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_93, 0);  primals_93 = None
    unsqueeze_289: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 2);  unsqueeze_288 = None
    unsqueeze_290: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 3);  unsqueeze_289 = None
    sum_23: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_45: "f32[2, 512, 80, 119]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_290);  convolution_6 = unsqueeze_290 = None
    mul_206: "f32[2, 512, 80, 119]" = torch.ops.aten.mul.Tensor(where_11, sub_45);  sub_45 = None
    sum_24: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_206, [0, 2, 3]);  mul_206 = None
    mul_211: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_11, primals_27);  primals_27 = None
    unsqueeze_297: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_211, 0);  mul_211 = None
    unsqueeze_298: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 2);  unsqueeze_297 = None
    unsqueeze_299: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 3);  unsqueeze_298 = None
    mul_212: "f32[2, 512, 80, 119]" = torch.ops.aten.mul.Tensor(where_11, unsqueeze_299);  where_11 = unsqueeze_299 = None
    mul_213: "f32[512]" = torch.ops.aten.mul.Tensor(sum_24, rsqrt_11);  sum_24 = rsqrt_11 = None
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_212, getitem_4, primals_25, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_212 = getitem_4 = primals_25 = None
    getitem_44: "f32[2, 256, 80, 119]" = convolution_backward_12[0]
    getitem_45: "f32[512, 256, 3, 3]" = convolution_backward_12[1]
    getitem_46: "f32[512]" = convolution_backward_12[2];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:39, code: return self.maxpool_conv(x)
    max_pool2d_with_indices_backward_1: "f32[2, 256, 160, 239]" = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_44, relu_5, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_5);  getitem_44 = getitem_5 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:39, code: return self.maxpool_conv(x)
    add_81: "f32[2, 256, 160, 239]" = torch.ops.aten.add.Tensor(slice_5, max_pool2d_with_indices_backward_1);  slice_5 = max_pool2d_with_indices_backward_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25, code: return self.double_conv(x)
    le_12: "b8[2, 256, 160, 239]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
    where_12: "f32[2, 256, 160, 239]" = torch.ops.aten.where.self(le_12, full_default, add_81);  le_12 = add_81 = None
    add_82: "f32[256]" = torch.ops.aten.add.Tensor(primals_91, 1e-05);  primals_91 = None
    rsqrt_12: "f32[256]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
    unsqueeze_300: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_90, 0);  primals_90 = None
    unsqueeze_301: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 2);  unsqueeze_300 = None
    unsqueeze_302: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 3);  unsqueeze_301 = None
    sum_25: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_46: "f32[2, 256, 160, 239]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_302);  convolution_5 = unsqueeze_302 = None
    mul_214: "f32[2, 256, 160, 239]" = torch.ops.aten.mul.Tensor(where_12, sub_46);  sub_46 = None
    sum_26: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_214, [0, 2, 3]);  mul_214 = None
    mul_219: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_12, primals_23);  primals_23 = None
    unsqueeze_309: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_219, 0);  mul_219 = None
    unsqueeze_310: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 2);  unsqueeze_309 = None
    unsqueeze_311: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 3);  unsqueeze_310 = None
    mul_220: "f32[2, 256, 160, 239]" = torch.ops.aten.mul.Tensor(where_12, unsqueeze_311);  where_12 = unsqueeze_311 = None
    mul_221: "f32[256]" = torch.ops.aten.mul.Tensor(sum_26, rsqrt_12);  sum_26 = rsqrt_12 = None
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_220, relu_4, primals_21, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_220 = primals_21 = None
    getitem_47: "f32[2, 256, 160, 239]" = convolution_backward_13[0]
    getitem_48: "f32[256, 256, 3, 3]" = convolution_backward_13[1]
    getitem_49: "f32[256]" = convolution_backward_13[2];  convolution_backward_13 = None
    le_13: "b8[2, 256, 160, 239]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
    where_13: "f32[2, 256, 160, 239]" = torch.ops.aten.where.self(le_13, full_default, getitem_47);  le_13 = getitem_47 = None
    add_83: "f32[256]" = torch.ops.aten.add.Tensor(primals_88, 1e-05);  primals_88 = None
    rsqrt_13: "f32[256]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
    unsqueeze_312: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_87, 0);  primals_87 = None
    unsqueeze_313: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, 2);  unsqueeze_312 = None
    unsqueeze_314: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 3);  unsqueeze_313 = None
    sum_27: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_47: "f32[2, 256, 160, 239]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_314);  convolution_4 = unsqueeze_314 = None
    mul_222: "f32[2, 256, 160, 239]" = torch.ops.aten.mul.Tensor(where_13, sub_47);  sub_47 = None
    sum_28: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_222, [0, 2, 3]);  mul_222 = None
    mul_227: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_13, primals_19);  primals_19 = None
    unsqueeze_321: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_227, 0);  mul_227 = None
    unsqueeze_322: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 2);  unsqueeze_321 = None
    unsqueeze_323: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 3);  unsqueeze_322 = None
    mul_228: "f32[2, 256, 160, 239]" = torch.ops.aten.mul.Tensor(where_13, unsqueeze_323);  where_13 = unsqueeze_323 = None
    mul_229: "f32[256]" = torch.ops.aten.mul.Tensor(sum_28, rsqrt_13);  sum_28 = rsqrt_13 = None
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_228, getitem_2, primals_17, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_228 = getitem_2 = primals_17 = None
    getitem_50: "f32[2, 128, 160, 239]" = convolution_backward_14[0]
    getitem_51: "f32[256, 128, 3, 3]" = convolution_backward_14[1]
    getitem_52: "f32[256]" = convolution_backward_14[2];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:39, code: return self.maxpool_conv(x)
    max_pool2d_with_indices_backward_2: "f32[2, 128, 320, 479]" = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_50, relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_3);  getitem_50 = getitem_3 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:39, code: return self.maxpool_conv(x)
    add_84: "f32[2, 128, 320, 479]" = torch.ops.aten.add.Tensor(slice_3, max_pool2d_with_indices_backward_2);  slice_3 = max_pool2d_with_indices_backward_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25, code: return self.double_conv(x)
    le_14: "b8[2, 128, 320, 479]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
    where_14: "f32[2, 128, 320, 479]" = torch.ops.aten.where.self(le_14, full_default, add_84);  le_14 = add_84 = None
    add_85: "f32[128]" = torch.ops.aten.add.Tensor(primals_85, 1e-05);  primals_85 = None
    rsqrt_14: "f32[128]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
    unsqueeze_324: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_84, 0);  primals_84 = None
    unsqueeze_325: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, 2);  unsqueeze_324 = None
    unsqueeze_326: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 3);  unsqueeze_325 = None
    sum_29: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_48: "f32[2, 128, 320, 479]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_326);  convolution_3 = unsqueeze_326 = None
    mul_230: "f32[2, 128, 320, 479]" = torch.ops.aten.mul.Tensor(where_14, sub_48);  sub_48 = None
    sum_30: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_230, [0, 2, 3]);  mul_230 = None
    mul_235: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_14, primals_15);  primals_15 = None
    unsqueeze_333: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_235, 0);  mul_235 = None
    unsqueeze_334: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 2);  unsqueeze_333 = None
    unsqueeze_335: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 3);  unsqueeze_334 = None
    mul_236: "f32[2, 128, 320, 479]" = torch.ops.aten.mul.Tensor(where_14, unsqueeze_335);  where_14 = unsqueeze_335 = None
    mul_237: "f32[128]" = torch.ops.aten.mul.Tensor(sum_30, rsqrt_14);  sum_30 = rsqrt_14 = None
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_236, relu_2, primals_13, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_236 = primals_13 = None
    getitem_53: "f32[2, 128, 320, 479]" = convolution_backward_15[0]
    getitem_54: "f32[128, 128, 3, 3]" = convolution_backward_15[1]
    getitem_55: "f32[128]" = convolution_backward_15[2];  convolution_backward_15 = None
    le_15: "b8[2, 128, 320, 479]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
    where_15: "f32[2, 128, 320, 479]" = torch.ops.aten.where.self(le_15, full_default, getitem_53);  le_15 = getitem_53 = None
    add_86: "f32[128]" = torch.ops.aten.add.Tensor(primals_82, 1e-05);  primals_82 = None
    rsqrt_15: "f32[128]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
    unsqueeze_336: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_81, 0);  primals_81 = None
    unsqueeze_337: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, 2);  unsqueeze_336 = None
    unsqueeze_338: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 3);  unsqueeze_337 = None
    sum_31: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_49: "f32[2, 128, 320, 479]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_338);  convolution_2 = unsqueeze_338 = None
    mul_238: "f32[2, 128, 320, 479]" = torch.ops.aten.mul.Tensor(where_15, sub_49);  sub_49 = None
    sum_32: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_238, [0, 2, 3]);  mul_238 = None
    mul_243: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_15, primals_11);  primals_11 = None
    unsqueeze_345: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_243, 0);  mul_243 = None
    unsqueeze_346: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_345, 2);  unsqueeze_345 = None
    unsqueeze_347: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 3);  unsqueeze_346 = None
    mul_244: "f32[2, 128, 320, 479]" = torch.ops.aten.mul.Tensor(where_15, unsqueeze_347);  where_15 = unsqueeze_347 = None
    mul_245: "f32[128]" = torch.ops.aten.mul.Tensor(sum_32, rsqrt_15);  sum_32 = rsqrt_15 = None
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_244, getitem, primals_9, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_244 = getitem = primals_9 = None
    getitem_56: "f32[2, 64, 320, 479]" = convolution_backward_16[0]
    getitem_57: "f32[128, 64, 3, 3]" = convolution_backward_16[1]
    getitem_58: "f32[128]" = convolution_backward_16[2];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:39, code: return self.maxpool_conv(x)
    max_pool2d_with_indices_backward_3: "f32[2, 64, 640, 959]" = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_56, relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_1);  getitem_56 = getitem_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:39, code: return self.maxpool_conv(x)
    add_87: "f32[2, 64, 640, 959]" = torch.ops.aten.add.Tensor(slice_1, max_pool2d_with_indices_backward_3);  slice_1 = max_pool2d_with_indices_backward_3 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25, code: return self.double_conv(x)
    le_16: "b8[2, 64, 640, 959]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
    where_16: "f32[2, 64, 640, 959]" = torch.ops.aten.where.self(le_16, full_default, add_87);  le_16 = add_87 = None
    add_88: "f32[64]" = torch.ops.aten.add.Tensor(primals_79, 1e-05);  primals_79 = None
    rsqrt_16: "f32[64]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
    unsqueeze_348: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_78, 0);  primals_78 = None
    unsqueeze_349: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, 2);  unsqueeze_348 = None
    unsqueeze_350: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 3);  unsqueeze_349 = None
    sum_33: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_50: "f32[2, 64, 640, 959]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_350);  convolution_1 = unsqueeze_350 = None
    mul_246: "f32[2, 64, 640, 959]" = torch.ops.aten.mul.Tensor(where_16, sub_50);  sub_50 = None
    sum_34: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_246, [0, 2, 3]);  mul_246 = None
    mul_251: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_16, primals_7);  primals_7 = None
    unsqueeze_357: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_251, 0);  mul_251 = None
    unsqueeze_358: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 2);  unsqueeze_357 = None
    unsqueeze_359: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 3);  unsqueeze_358 = None
    mul_252: "f32[2, 64, 640, 959]" = torch.ops.aten.mul.Tensor(where_16, unsqueeze_359);  where_16 = unsqueeze_359 = None
    mul_253: "f32[64]" = torch.ops.aten.mul.Tensor(sum_34, rsqrt_16);  sum_34 = rsqrt_16 = None
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_252, relu, primals_5, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_252 = primals_5 = None
    getitem_59: "f32[2, 64, 640, 959]" = convolution_backward_17[0]
    getitem_60: "f32[64, 64, 3, 3]" = convolution_backward_17[1]
    getitem_61: "f32[64]" = convolution_backward_17[2];  convolution_backward_17 = None
    le_17: "b8[2, 64, 640, 959]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
    where_17: "f32[2, 64, 640, 959]" = torch.ops.aten.where.self(le_17, full_default, getitem_59);  le_17 = full_default = getitem_59 = None
    add_89: "f32[64]" = torch.ops.aten.add.Tensor(primals_76, 1e-05);  primals_76 = None
    rsqrt_17: "f32[64]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
    unsqueeze_360: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_75, 0);  primals_75 = None
    unsqueeze_361: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, 2);  unsqueeze_360 = None
    unsqueeze_362: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 3);  unsqueeze_361 = None
    sum_35: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_51: "f32[2, 64, 640, 959]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_362);  convolution = unsqueeze_362 = None
    mul_254: "f32[2, 64, 640, 959]" = torch.ops.aten.mul.Tensor(where_17, sub_51);  sub_51 = None
    sum_36: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_254, [0, 2, 3]);  mul_254 = None
    mul_259: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_17, primals_3);  primals_3 = None
    unsqueeze_369: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_259, 0);  mul_259 = None
    unsqueeze_370: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_369, 2);  unsqueeze_369 = None
    unsqueeze_371: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 3);  unsqueeze_370 = None
    mul_260: "f32[2, 64, 640, 959]" = torch.ops.aten.mul.Tensor(where_17, unsqueeze_371);  where_17 = unsqueeze_371 = None
    mul_261: "f32[64]" = torch.ops.aten.mul.Tensor(sum_36, rsqrt_17);  sum_36 = rsqrt_17 = None
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_260, primals_129, primals_1, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [False, True, True]);  mul_260 = primals_129 = primals_1 = None
    getitem_63: "f32[64, 3, 3, 3]" = convolution_backward_18[1]
    getitem_64: "f32[64]" = convolution_backward_18[2];  convolution_backward_18 = None
    return [getitem_63, getitem_64, mul_261, sum_35, getitem_60, getitem_61, mul_253, sum_33, getitem_57, getitem_58, mul_245, sum_31, getitem_54, getitem_55, mul_237, sum_29, getitem_51, getitem_52, mul_229, sum_27, getitem_48, getitem_49, mul_221, sum_25, getitem_45, getitem_46, mul_213, sum_23, getitem_42, getitem_43, mul_205, sum_21, getitem_39, getitem_40, mul_197, sum_19, getitem_36, getitem_37, mul_189, sum_17, getitem_33, getitem_34, mul_175, sum_15, getitem_30, getitem_31, mul_167, sum_13, getitem_27, getitem_28, mul_153, sum_11, getitem_24, getitem_25, mul_145, sum_9, getitem_21, getitem_22, mul_131, sum_7, getitem_18, getitem_19, mul_123, sum_5, getitem_15, getitem_16, mul_109, sum_3, getitem_12, getitem_13, mul_101, sum_1, getitem_9, getitem_10, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    