from __future__ import annotations



def forward(self, primals_1: "f32[64, 3, 7, 7]", primals_2: "f32[64]", primals_4: "f32[64, 64, 3, 3]", primals_5: "f32[64]", primals_7: "f32[64, 64, 3, 3]", primals_8: "f32[64]", primals_10: "f32[64, 64, 3, 3]", primals_11: "f32[64]", primals_13: "f32[64, 64, 3, 3]", primals_14: "f32[64]", primals_16: "f32[128, 64, 3, 3]", primals_17: "f32[128]", primals_19: "f32[128, 128, 3, 3]", primals_20: "f32[128]", primals_22: "f32[128, 64, 1, 1]", primals_23: "f32[128]", primals_25: "f32[128, 128, 3, 3]", primals_26: "f32[128]", primals_28: "f32[128, 128, 3, 3]", primals_29: "f32[128]", primals_31: "f32[256, 128, 3, 3]", primals_32: "f32[256]", primals_34: "f32[256, 256, 3, 3]", primals_35: "f32[256]", primals_37: "f32[256, 128, 1, 1]", primals_38: "f32[256]", primals_40: "f32[256, 256, 3, 3]", primals_41: "f32[256]", primals_43: "f32[256, 256, 3, 3]", primals_44: "f32[256]", primals_46: "f32[512, 256, 3, 3]", primals_47: "f32[512]", primals_49: "f32[512, 512, 3, 3]", primals_50: "f32[512]", primals_52: "f32[512, 256, 1, 1]", primals_53: "f32[512]", primals_55: "f32[512, 512, 3, 3]", primals_56: "f32[512]", primals_58: "f32[512, 512, 3, 3]", primals_59: "f32[512]", primals_63: "f32[64]", primals_64: "f32[64]", primals_66: "f32[64]", primals_67: "f32[64]", primals_69: "f32[64]", primals_70: "f32[64]", primals_72: "f32[64]", primals_73: "f32[64]", primals_75: "f32[64]", primals_76: "f32[64]", primals_78: "f32[128]", primals_79: "f32[128]", primals_81: "f32[128]", primals_82: "f32[128]", primals_84: "f32[128]", primals_85: "f32[128]", primals_87: "f32[128]", primals_88: "f32[128]", primals_90: "f32[128]", primals_91: "f32[128]", primals_93: "f32[256]", primals_94: "f32[256]", primals_96: "f32[256]", primals_97: "f32[256]", primals_99: "f32[256]", primals_100: "f32[256]", primals_102: "f32[256]", primals_103: "f32[256]", primals_105: "f32[256]", primals_106: "f32[256]", primals_108: "f32[512]", primals_109: "f32[512]", primals_111: "f32[512]", primals_112: "f32[512]", primals_114: "f32[512]", primals_115: "f32[512]", primals_117: "f32[512]", primals_118: "f32[512]", primals_120: "f32[512]", primals_121: "f32[512]", primals_123: "f32[4, 3, 224, 224]", convolution: "f32[4, 64, 112, 112]", relu: "f32[4, 64, 112, 112]", getitem: "f32[4, 64, 56, 56]", getitem_1: "i64[4, 64, 56, 56]", convolution_1: "f32[4, 64, 56, 56]", relu_1: "f32[4, 64, 56, 56]", convolution_2: "f32[4, 64, 56, 56]", relu_2: "f32[4, 64, 56, 56]", convolution_3: "f32[4, 64, 56, 56]", relu_3: "f32[4, 64, 56, 56]", convolution_4: "f32[4, 64, 56, 56]", relu_4: "f32[4, 64, 56, 56]", convolution_5: "f32[4, 128, 28, 28]", relu_5: "f32[4, 128, 28, 28]", convolution_6: "f32[4, 128, 28, 28]", convolution_7: "f32[4, 128, 28, 28]", relu_6: "f32[4, 128, 28, 28]", convolution_8: "f32[4, 128, 28, 28]", relu_7: "f32[4, 128, 28, 28]", convolution_9: "f32[4, 128, 28, 28]", relu_8: "f32[4, 128, 28, 28]", convolution_10: "f32[4, 256, 14, 14]", relu_9: "f32[4, 256, 14, 14]", convolution_11: "f32[4, 256, 14, 14]", convolution_12: "f32[4, 256, 14, 14]", relu_10: "f32[4, 256, 14, 14]", convolution_13: "f32[4, 256, 14, 14]", relu_11: "f32[4, 256, 14, 14]", convolution_14: "f32[4, 256, 14, 14]", relu_12: "f32[4, 256, 14, 14]", convolution_15: "f32[4, 512, 7, 7]", relu_13: "f32[4, 512, 7, 7]", convolution_16: "f32[4, 512, 7, 7]", convolution_17: "f32[4, 512, 7, 7]", relu_14: "f32[4, 512, 7, 7]", convolution_18: "f32[4, 512, 7, 7]", relu_15: "f32[4, 512, 7, 7]", convolution_19: "f32[4, 512, 7, 7]", view: "f32[4, 512]", permute_1: "f32[1000, 512]", le: "b8[4, 512, 7, 7]", tangents_1: "f32[4, 1000]"):
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:280, code: x = self.fc(x)
    mm: "f32[4, 512]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 4]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 512]" = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
    permute_3: "f32[512, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 512]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:279, code: x = torch.flatten(x, 1)
    view_2: "f32[4, 512, 1, 1]" = torch.ops.aten.reshape.default(mm, [4, 512, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:278, code: x = self.avgpool(x)
    expand: "f32[4, 512, 7, 7]" = torch.ops.aten.expand.default(view_2, [4, 512, 7, 7]);  view_2 = None
    div: "f32[4, 512, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where: "f32[4, 512, 7, 7]" = torch.ops.aten.where.self(le, full_default, div);  le = div = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    add_48: "f32[512]" = torch.ops.aten.add.Tensor(primals_121, 1e-05);  primals_121 = None
    rsqrt: "f32[512]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
    unsqueeze_160: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_120, 0);  primals_120 = None
    unsqueeze_161: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, 2);  unsqueeze_160 = None
    unsqueeze_162: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_161, 3);  unsqueeze_161 = None
    sum_2: "f32[512]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_20: "f32[4, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_162);  convolution_19 = unsqueeze_162 = None
    mul_60: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_20);  sub_20 = None
    sum_3: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_60, [0, 2, 3]);  mul_60 = None
    mul_65: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt, primals_59);  primals_59 = None
    unsqueeze_169: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_65, 0);  mul_65 = None
    unsqueeze_170: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_169, 2);  unsqueeze_169 = None
    unsqueeze_171: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, 3);  unsqueeze_170 = None
    mul_66: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where, unsqueeze_171);  unsqueeze_171 = None
    mul_67: "f32[512]" = torch.ops.aten.mul.Tensor(sum_3, rsqrt);  sum_3 = rsqrt = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_66, relu_15, primals_58, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_66 = primals_58 = None
    getitem_2: "f32[4, 512, 7, 7]" = convolution_backward[0]
    getitem_3: "f32[512, 512, 3, 3]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    le_1: "b8[4, 512, 7, 7]" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
    where_1: "f32[4, 512, 7, 7]" = torch.ops.aten.where.self(le_1, full_default, getitem_2);  le_1 = getitem_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    add_49: "f32[512]" = torch.ops.aten.add.Tensor(primals_118, 1e-05);  primals_118 = None
    rsqrt_1: "f32[512]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    unsqueeze_172: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_117, 0);  primals_117 = None
    unsqueeze_173: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, 2);  unsqueeze_172 = None
    unsqueeze_174: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_173, 3);  unsqueeze_173 = None
    sum_4: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_21: "f32[4, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_174);  convolution_18 = unsqueeze_174 = None
    mul_68: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, sub_21);  sub_21 = None
    sum_5: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_68, [0, 2, 3]);  mul_68 = None
    mul_73: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_1, primals_56);  primals_56 = None
    unsqueeze_181: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_73, 0);  mul_73 = None
    unsqueeze_182: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_181, 2);  unsqueeze_181 = None
    unsqueeze_183: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, 3);  unsqueeze_182 = None
    mul_74: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, unsqueeze_183);  where_1 = unsqueeze_183 = None
    mul_75: "f32[512]" = torch.ops.aten.mul.Tensor(sum_5, rsqrt_1);  sum_5 = rsqrt_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_74, relu_14, primals_55, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_74 = primals_55 = None
    getitem_5: "f32[4, 512, 7, 7]" = convolution_backward_1[0]
    getitem_6: "f32[512, 512, 3, 3]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    add_50: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(where, getitem_5);  where = getitem_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    le_2: "b8[4, 512, 7, 7]" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
    where_2: "f32[4, 512, 7, 7]" = torch.ops.aten.where.self(le_2, full_default, add_50);  le_2 = add_50 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:100, code: identity = self.downsample(x)
    add_51: "f32[512]" = torch.ops.aten.add.Tensor(primals_115, 1e-05);  primals_115 = None
    rsqrt_2: "f32[512]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
    unsqueeze_184: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_114, 0);  primals_114 = None
    unsqueeze_185: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, 2);  unsqueeze_184 = None
    unsqueeze_186: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_185, 3);  unsqueeze_185 = None
    sum_6: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_22: "f32[4, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_186);  convolution_17 = unsqueeze_186 = None
    mul_76: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_22);  sub_22 = None
    sum_7: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_76, [0, 2, 3]);  mul_76 = None
    mul_81: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_2, primals_53);  primals_53 = None
    unsqueeze_193: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_81, 0);  mul_81 = None
    unsqueeze_194: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_193, 2);  unsqueeze_193 = None
    unsqueeze_195: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, 3);  unsqueeze_194 = None
    mul_82: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, unsqueeze_195);  unsqueeze_195 = None
    mul_83: "f32[512]" = torch.ops.aten.mul.Tensor(sum_7, rsqrt_2);  sum_7 = rsqrt_2 = None
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_82, relu_12, primals_52, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_82 = primals_52 = None
    getitem_8: "f32[4, 256, 14, 14]" = convolution_backward_2[0]
    getitem_9: "f32[512, 256, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    add_52: "f32[512]" = torch.ops.aten.add.Tensor(primals_112, 1e-05);  primals_112 = None
    rsqrt_3: "f32[512]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    unsqueeze_196: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_111, 0);  primals_111 = None
    unsqueeze_197: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, 2);  unsqueeze_196 = None
    unsqueeze_198: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 3);  unsqueeze_197 = None
    sub_23: "f32[4, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_198);  convolution_16 = unsqueeze_198 = None
    mul_84: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_23);  sub_23 = None
    sum_9: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_84, [0, 2, 3]);  mul_84 = None
    mul_89: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_3, primals_50);  primals_50 = None
    unsqueeze_205: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_89, 0);  mul_89 = None
    unsqueeze_206: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 2);  unsqueeze_205 = None
    unsqueeze_207: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, 3);  unsqueeze_206 = None
    mul_90: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, unsqueeze_207);  where_2 = unsqueeze_207 = None
    mul_91: "f32[512]" = torch.ops.aten.mul.Tensor(sum_9, rsqrt_3);  sum_9 = rsqrt_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_90, relu_13, primals_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_90 = primals_49 = None
    getitem_11: "f32[4, 512, 7, 7]" = convolution_backward_3[0]
    getitem_12: "f32[512, 512, 3, 3]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    le_3: "b8[4, 512, 7, 7]" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
    where_3: "f32[4, 512, 7, 7]" = torch.ops.aten.where.self(le_3, full_default, getitem_11);  le_3 = getitem_11 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    add_53: "f32[512]" = torch.ops.aten.add.Tensor(primals_109, 1e-05);  primals_109 = None
    rsqrt_4: "f32[512]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    unsqueeze_208: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_108, 0);  primals_108 = None
    unsqueeze_209: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, 2);  unsqueeze_208 = None
    unsqueeze_210: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 3);  unsqueeze_209 = None
    sum_10: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_24: "f32[4, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_210);  convolution_15 = unsqueeze_210 = None
    mul_92: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_24);  sub_24 = None
    sum_11: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_92, [0, 2, 3]);  mul_92 = None
    mul_97: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_4, primals_47);  primals_47 = None
    unsqueeze_217: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_97, 0);  mul_97 = None
    unsqueeze_218: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 2);  unsqueeze_217 = None
    unsqueeze_219: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 3);  unsqueeze_218 = None
    mul_98: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, unsqueeze_219);  where_3 = unsqueeze_219 = None
    mul_99: "f32[512]" = torch.ops.aten.mul.Tensor(sum_11, rsqrt_4);  sum_11 = rsqrt_4 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_98, relu_12, primals_46, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_98 = primals_46 = None
    getitem_14: "f32[4, 256, 14, 14]" = convolution_backward_4[0]
    getitem_15: "f32[512, 256, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    add_54: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(getitem_8, getitem_14);  getitem_8 = getitem_14 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    le_4: "b8[4, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
    where_4: "f32[4, 256, 14, 14]" = torch.ops.aten.where.self(le_4, full_default, add_54);  le_4 = add_54 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    add_55: "f32[256]" = torch.ops.aten.add.Tensor(primals_106, 1e-05);  primals_106 = None
    rsqrt_5: "f32[256]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
    unsqueeze_220: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_105, 0);  primals_105 = None
    unsqueeze_221: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, 2);  unsqueeze_220 = None
    unsqueeze_222: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 3);  unsqueeze_221 = None
    sum_12: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_25: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_222);  convolution_14 = unsqueeze_222 = None
    mul_100: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_4, sub_25);  sub_25 = None
    sum_13: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_100, [0, 2, 3]);  mul_100 = None
    mul_105: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_5, primals_44);  primals_44 = None
    unsqueeze_229: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_105, 0);  mul_105 = None
    unsqueeze_230: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 2);  unsqueeze_229 = None
    unsqueeze_231: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 3);  unsqueeze_230 = None
    mul_106: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_4, unsqueeze_231);  unsqueeze_231 = None
    mul_107: "f32[256]" = torch.ops.aten.mul.Tensor(sum_13, rsqrt_5);  sum_13 = rsqrt_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_106, relu_11, primals_43, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_106 = primals_43 = None
    getitem_17: "f32[4, 256, 14, 14]" = convolution_backward_5[0]
    getitem_18: "f32[256, 256, 3, 3]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    le_5: "b8[4, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
    where_5: "f32[4, 256, 14, 14]" = torch.ops.aten.where.self(le_5, full_default, getitem_17);  le_5 = getitem_17 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    add_56: "f32[256]" = torch.ops.aten.add.Tensor(primals_103, 1e-05);  primals_103 = None
    rsqrt_6: "f32[256]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    unsqueeze_232: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_102, 0);  primals_102 = None
    unsqueeze_233: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 2);  unsqueeze_232 = None
    unsqueeze_234: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 3);  unsqueeze_233 = None
    sum_14: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_26: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_234);  convolution_13 = unsqueeze_234 = None
    mul_108: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_5, sub_26);  sub_26 = None
    sum_15: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_108, [0, 2, 3]);  mul_108 = None
    mul_113: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_6, primals_41);  primals_41 = None
    unsqueeze_241: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_113, 0);  mul_113 = None
    unsqueeze_242: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 2);  unsqueeze_241 = None
    unsqueeze_243: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 3);  unsqueeze_242 = None
    mul_114: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_5, unsqueeze_243);  where_5 = unsqueeze_243 = None
    mul_115: "f32[256]" = torch.ops.aten.mul.Tensor(sum_15, rsqrt_6);  sum_15 = rsqrt_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_114, relu_10, primals_40, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_114 = primals_40 = None
    getitem_20: "f32[4, 256, 14, 14]" = convolution_backward_6[0]
    getitem_21: "f32[256, 256, 3, 3]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    add_57: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(where_4, getitem_20);  where_4 = getitem_20 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    le_6: "b8[4, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
    where_6: "f32[4, 256, 14, 14]" = torch.ops.aten.where.self(le_6, full_default, add_57);  le_6 = add_57 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:100, code: identity = self.downsample(x)
    add_58: "f32[256]" = torch.ops.aten.add.Tensor(primals_100, 1e-05);  primals_100 = None
    rsqrt_7: "f32[256]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    unsqueeze_244: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_99, 0);  primals_99 = None
    unsqueeze_245: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 2);  unsqueeze_244 = None
    unsqueeze_246: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 3);  unsqueeze_245 = None
    sum_16: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_27: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_246);  convolution_12 = unsqueeze_246 = None
    mul_116: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_6, sub_27);  sub_27 = None
    sum_17: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_116, [0, 2, 3]);  mul_116 = None
    mul_121: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_7, primals_38);  primals_38 = None
    unsqueeze_253: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_121, 0);  mul_121 = None
    unsqueeze_254: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 2);  unsqueeze_253 = None
    unsqueeze_255: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 3);  unsqueeze_254 = None
    mul_122: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_6, unsqueeze_255);  unsqueeze_255 = None
    mul_123: "f32[256]" = torch.ops.aten.mul.Tensor(sum_17, rsqrt_7);  sum_17 = rsqrt_7 = None
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_122, relu_8, primals_37, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_122 = primals_37 = None
    getitem_23: "f32[4, 128, 28, 28]" = convolution_backward_7[0]
    getitem_24: "f32[256, 128, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    add_59: "f32[256]" = torch.ops.aten.add.Tensor(primals_97, 1e-05);  primals_97 = None
    rsqrt_8: "f32[256]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
    unsqueeze_256: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_96, 0);  primals_96 = None
    unsqueeze_257: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, 2);  unsqueeze_256 = None
    unsqueeze_258: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 3);  unsqueeze_257 = None
    sub_28: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_258);  convolution_11 = unsqueeze_258 = None
    mul_124: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_6, sub_28);  sub_28 = None
    sum_19: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_124, [0, 2, 3]);  mul_124 = None
    mul_129: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_8, primals_35);  primals_35 = None
    unsqueeze_265: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_129, 0);  mul_129 = None
    unsqueeze_266: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 2);  unsqueeze_265 = None
    unsqueeze_267: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 3);  unsqueeze_266 = None
    mul_130: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_6, unsqueeze_267);  where_6 = unsqueeze_267 = None
    mul_131: "f32[256]" = torch.ops.aten.mul.Tensor(sum_19, rsqrt_8);  sum_19 = rsqrt_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_130, relu_9, primals_34, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_130 = primals_34 = None
    getitem_26: "f32[4, 256, 14, 14]" = convolution_backward_8[0]
    getitem_27: "f32[256, 256, 3, 3]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    le_7: "b8[4, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
    where_7: "f32[4, 256, 14, 14]" = torch.ops.aten.where.self(le_7, full_default, getitem_26);  le_7 = getitem_26 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    add_60: "f32[256]" = torch.ops.aten.add.Tensor(primals_94, 1e-05);  primals_94 = None
    rsqrt_9: "f32[256]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    unsqueeze_268: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_93, 0);  primals_93 = None
    unsqueeze_269: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, 2);  unsqueeze_268 = None
    unsqueeze_270: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 3);  unsqueeze_269 = None
    sum_20: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_29: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_270);  convolution_10 = unsqueeze_270 = None
    mul_132: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_7, sub_29);  sub_29 = None
    sum_21: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_132, [0, 2, 3]);  mul_132 = None
    mul_137: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_9, primals_32);  primals_32 = None
    unsqueeze_277: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_137, 0);  mul_137 = None
    unsqueeze_278: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 2);  unsqueeze_277 = None
    unsqueeze_279: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 3);  unsqueeze_278 = None
    mul_138: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_7, unsqueeze_279);  where_7 = unsqueeze_279 = None
    mul_139: "f32[256]" = torch.ops.aten.mul.Tensor(sum_21, rsqrt_9);  sum_21 = rsqrt_9 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_138, relu_8, primals_31, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_138 = primals_31 = None
    getitem_29: "f32[4, 128, 28, 28]" = convolution_backward_9[0]
    getitem_30: "f32[256, 128, 3, 3]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    add_61: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(getitem_23, getitem_29);  getitem_23 = getitem_29 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    le_8: "b8[4, 128, 28, 28]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
    where_8: "f32[4, 128, 28, 28]" = torch.ops.aten.where.self(le_8, full_default, add_61);  le_8 = add_61 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    add_62: "f32[128]" = torch.ops.aten.add.Tensor(primals_91, 1e-05);  primals_91 = None
    rsqrt_10: "f32[128]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    unsqueeze_280: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_90, 0);  primals_90 = None
    unsqueeze_281: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 2);  unsqueeze_280 = None
    unsqueeze_282: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 3);  unsqueeze_281 = None
    sum_22: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_30: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_282);  convolution_9 = unsqueeze_282 = None
    mul_140: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_8, sub_30);  sub_30 = None
    sum_23: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_140, [0, 2, 3]);  mul_140 = None
    mul_145: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_10, primals_29);  primals_29 = None
    unsqueeze_289: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_145, 0);  mul_145 = None
    unsqueeze_290: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 2);  unsqueeze_289 = None
    unsqueeze_291: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 3);  unsqueeze_290 = None
    mul_146: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_8, unsqueeze_291);  unsqueeze_291 = None
    mul_147: "f32[128]" = torch.ops.aten.mul.Tensor(sum_23, rsqrt_10);  sum_23 = rsqrt_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_146, relu_7, primals_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_146 = primals_28 = None
    getitem_32: "f32[4, 128, 28, 28]" = convolution_backward_10[0]
    getitem_33: "f32[128, 128, 3, 3]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    le_9: "b8[4, 128, 28, 28]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
    where_9: "f32[4, 128, 28, 28]" = torch.ops.aten.where.self(le_9, full_default, getitem_32);  le_9 = getitem_32 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    add_63: "f32[128]" = torch.ops.aten.add.Tensor(primals_88, 1e-05);  primals_88 = None
    rsqrt_11: "f32[128]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    unsqueeze_292: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_87, 0);  primals_87 = None
    unsqueeze_293: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 2);  unsqueeze_292 = None
    unsqueeze_294: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 3);  unsqueeze_293 = None
    sum_24: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_31: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_294);  convolution_8 = unsqueeze_294 = None
    mul_148: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_9, sub_31);  sub_31 = None
    sum_25: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_148, [0, 2, 3]);  mul_148 = None
    mul_153: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_11, primals_26);  primals_26 = None
    unsqueeze_301: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_153, 0);  mul_153 = None
    unsqueeze_302: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 2);  unsqueeze_301 = None
    unsqueeze_303: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 3);  unsqueeze_302 = None
    mul_154: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_9, unsqueeze_303);  where_9 = unsqueeze_303 = None
    mul_155: "f32[128]" = torch.ops.aten.mul.Tensor(sum_25, rsqrt_11);  sum_25 = rsqrt_11 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_154, relu_6, primals_25, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_154 = primals_25 = None
    getitem_35: "f32[4, 128, 28, 28]" = convolution_backward_11[0]
    getitem_36: "f32[128, 128, 3, 3]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    add_64: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(where_8, getitem_35);  where_8 = getitem_35 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    le_10: "b8[4, 128, 28, 28]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
    where_10: "f32[4, 128, 28, 28]" = torch.ops.aten.where.self(le_10, full_default, add_64);  le_10 = add_64 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:100, code: identity = self.downsample(x)
    add_65: "f32[128]" = torch.ops.aten.add.Tensor(primals_85, 1e-05);  primals_85 = None
    rsqrt_12: "f32[128]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
    unsqueeze_304: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_84, 0);  primals_84 = None
    unsqueeze_305: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, 2);  unsqueeze_304 = None
    unsqueeze_306: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 3);  unsqueeze_305 = None
    sum_26: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_32: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_306);  convolution_7 = unsqueeze_306 = None
    mul_156: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_10, sub_32);  sub_32 = None
    sum_27: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_156, [0, 2, 3]);  mul_156 = None
    mul_161: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_12, primals_23);  primals_23 = None
    unsqueeze_313: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_161, 0);  mul_161 = None
    unsqueeze_314: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 2);  unsqueeze_313 = None
    unsqueeze_315: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 3);  unsqueeze_314 = None
    mul_162: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_10, unsqueeze_315);  unsqueeze_315 = None
    mul_163: "f32[128]" = torch.ops.aten.mul.Tensor(sum_27, rsqrt_12);  sum_27 = rsqrt_12 = None
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_162, relu_4, primals_22, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_162 = primals_22 = None
    getitem_38: "f32[4, 64, 56, 56]" = convolution_backward_12[0]
    getitem_39: "f32[128, 64, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    add_66: "f32[128]" = torch.ops.aten.add.Tensor(primals_82, 1e-05);  primals_82 = None
    rsqrt_13: "f32[128]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    unsqueeze_316: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_81, 0);  primals_81 = None
    unsqueeze_317: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, 2);  unsqueeze_316 = None
    unsqueeze_318: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 3);  unsqueeze_317 = None
    sub_33: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_318);  convolution_6 = unsqueeze_318 = None
    mul_164: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_10, sub_33);  sub_33 = None
    sum_29: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_164, [0, 2, 3]);  mul_164 = None
    mul_169: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_13, primals_20);  primals_20 = None
    unsqueeze_325: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_169, 0);  mul_169 = None
    unsqueeze_326: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 2);  unsqueeze_325 = None
    unsqueeze_327: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 3);  unsqueeze_326 = None
    mul_170: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_10, unsqueeze_327);  where_10 = unsqueeze_327 = None
    mul_171: "f32[128]" = torch.ops.aten.mul.Tensor(sum_29, rsqrt_13);  sum_29 = rsqrt_13 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_170, relu_5, primals_19, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_170 = primals_19 = None
    getitem_41: "f32[4, 128, 28, 28]" = convolution_backward_13[0]
    getitem_42: "f32[128, 128, 3, 3]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    le_11: "b8[4, 128, 28, 28]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
    where_11: "f32[4, 128, 28, 28]" = torch.ops.aten.where.self(le_11, full_default, getitem_41);  le_11 = getitem_41 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    add_67: "f32[128]" = torch.ops.aten.add.Tensor(primals_79, 1e-05);  primals_79 = None
    rsqrt_14: "f32[128]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    unsqueeze_328: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_78, 0);  primals_78 = None
    unsqueeze_329: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 2);  unsqueeze_328 = None
    unsqueeze_330: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 3);  unsqueeze_329 = None
    sum_30: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_34: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_330);  convolution_5 = unsqueeze_330 = None
    mul_172: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_11, sub_34);  sub_34 = None
    sum_31: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_172, [0, 2, 3]);  mul_172 = None
    mul_177: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_14, primals_17);  primals_17 = None
    unsqueeze_337: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_177, 0);  mul_177 = None
    unsqueeze_338: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 2);  unsqueeze_337 = None
    unsqueeze_339: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 3);  unsqueeze_338 = None
    mul_178: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_11, unsqueeze_339);  where_11 = unsqueeze_339 = None
    mul_179: "f32[128]" = torch.ops.aten.mul.Tensor(sum_31, rsqrt_14);  sum_31 = rsqrt_14 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_178, relu_4, primals_16, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_178 = primals_16 = None
    getitem_44: "f32[4, 64, 56, 56]" = convolution_backward_14[0]
    getitem_45: "f32[128, 64, 3, 3]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    add_68: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(getitem_38, getitem_44);  getitem_38 = getitem_44 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    le_12: "b8[4, 64, 56, 56]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
    where_12: "f32[4, 64, 56, 56]" = torch.ops.aten.where.self(le_12, full_default, add_68);  le_12 = add_68 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    add_69: "f32[64]" = torch.ops.aten.add.Tensor(primals_76, 1e-05);  primals_76 = None
    rsqrt_15: "f32[64]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
    unsqueeze_340: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_75, 0);  primals_75 = None
    unsqueeze_341: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, 2);  unsqueeze_340 = None
    unsqueeze_342: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 3);  unsqueeze_341 = None
    sum_32: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_35: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_342);  convolution_4 = unsqueeze_342 = None
    mul_180: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_12, sub_35);  sub_35 = None
    sum_33: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_180, [0, 2, 3]);  mul_180 = None
    mul_185: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_15, primals_14);  primals_14 = None
    unsqueeze_349: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_185, 0);  mul_185 = None
    unsqueeze_350: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 2);  unsqueeze_349 = None
    unsqueeze_351: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 3);  unsqueeze_350 = None
    mul_186: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_12, unsqueeze_351);  unsqueeze_351 = None
    mul_187: "f32[64]" = torch.ops.aten.mul.Tensor(sum_33, rsqrt_15);  sum_33 = rsqrt_15 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_186, relu_3, primals_13, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_186 = primals_13 = None
    getitem_47: "f32[4, 64, 56, 56]" = convolution_backward_15[0]
    getitem_48: "f32[64, 64, 3, 3]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    le_13: "b8[4, 64, 56, 56]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
    where_13: "f32[4, 64, 56, 56]" = torch.ops.aten.where.self(le_13, full_default, getitem_47);  le_13 = getitem_47 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    add_70: "f32[64]" = torch.ops.aten.add.Tensor(primals_73, 1e-05);  primals_73 = None
    rsqrt_16: "f32[64]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    unsqueeze_352: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_72, 0);  primals_72 = None
    unsqueeze_353: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, 2);  unsqueeze_352 = None
    unsqueeze_354: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 3);  unsqueeze_353 = None
    sum_34: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_36: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_354);  convolution_3 = unsqueeze_354 = None
    mul_188: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_13, sub_36);  sub_36 = None
    sum_35: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_188, [0, 2, 3]);  mul_188 = None
    mul_193: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_16, primals_11);  primals_11 = None
    unsqueeze_361: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_193, 0);  mul_193 = None
    unsqueeze_362: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
    unsqueeze_363: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 3);  unsqueeze_362 = None
    mul_194: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_13, unsqueeze_363);  where_13 = unsqueeze_363 = None
    mul_195: "f32[64]" = torch.ops.aten.mul.Tensor(sum_35, rsqrt_16);  sum_35 = rsqrt_16 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_194, relu_2, primals_10, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_194 = primals_10 = None
    getitem_50: "f32[4, 64, 56, 56]" = convolution_backward_16[0]
    getitem_51: "f32[64, 64, 3, 3]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    add_71: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(where_12, getitem_50);  where_12 = getitem_50 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    le_14: "b8[4, 64, 56, 56]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
    where_14: "f32[4, 64, 56, 56]" = torch.ops.aten.where.self(le_14, full_default, add_71);  le_14 = add_71 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    add_72: "f32[64]" = torch.ops.aten.add.Tensor(primals_70, 1e-05);  primals_70 = None
    rsqrt_17: "f32[64]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
    unsqueeze_364: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_69, 0);  primals_69 = None
    unsqueeze_365: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, 2);  unsqueeze_364 = None
    unsqueeze_366: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 3);  unsqueeze_365 = None
    sum_36: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_37: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_366);  convolution_2 = unsqueeze_366 = None
    mul_196: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_14, sub_37);  sub_37 = None
    sum_37: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_196, [0, 2, 3]);  mul_196 = None
    mul_201: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_17, primals_8);  primals_8 = None
    unsqueeze_373: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_201, 0);  mul_201 = None
    unsqueeze_374: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
    unsqueeze_375: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 3);  unsqueeze_374 = None
    mul_202: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_14, unsqueeze_375);  unsqueeze_375 = None
    mul_203: "f32[64]" = torch.ops.aten.mul.Tensor(sum_37, rsqrt_17);  sum_37 = rsqrt_17 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_202, relu_1, primals_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_202 = primals_7 = None
    getitem_53: "f32[4, 64, 56, 56]" = convolution_backward_17[0]
    getitem_54: "f32[64, 64, 3, 3]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    le_15: "b8[4, 64, 56, 56]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
    where_15: "f32[4, 64, 56, 56]" = torch.ops.aten.where.self(le_15, full_default, getitem_53);  le_15 = getitem_53 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    add_73: "f32[64]" = torch.ops.aten.add.Tensor(primals_67, 1e-05);  primals_67 = None
    rsqrt_18: "f32[64]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    unsqueeze_376: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_66, 0);  primals_66 = None
    unsqueeze_377: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 2);  unsqueeze_376 = None
    unsqueeze_378: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 3);  unsqueeze_377 = None
    sum_38: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_38: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_378);  convolution_1 = unsqueeze_378 = None
    mul_204: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_15, sub_38);  sub_38 = None
    sum_39: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_204, [0, 2, 3]);  mul_204 = None
    mul_209: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_18, primals_5);  primals_5 = None
    unsqueeze_385: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_209, 0);  mul_209 = None
    unsqueeze_386: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
    unsqueeze_387: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 3);  unsqueeze_386 = None
    mul_210: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_15, unsqueeze_387);  where_15 = unsqueeze_387 = None
    mul_211: "f32[64]" = torch.ops.aten.mul.Tensor(sum_39, rsqrt_18);  sum_39 = rsqrt_18 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_210, getitem, primals_4, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_210 = getitem = primals_4 = None
    getitem_56: "f32[4, 64, 56, 56]" = convolution_backward_18[0]
    getitem_57: "f32[64, 64, 3, 3]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    add_74: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(where_14, getitem_56);  where_14 = getitem_56 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:271, code: x = self.maxpool(x)
    max_pool2d_with_indices_backward: "f32[4, 64, 112, 112]" = torch.ops.aten.max_pool2d_with_indices_backward.default(add_74, relu, [3, 3], [2, 2], [1, 1], [1, 1], False, getitem_1);  add_74 = getitem_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:270, code: x = self.relu(x)
    le_16: "b8[4, 64, 112, 112]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
    where_16: "f32[4, 64, 112, 112]" = torch.ops.aten.where.self(le_16, full_default, max_pool2d_with_indices_backward);  le_16 = full_default = max_pool2d_with_indices_backward = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:269, code: x = self.bn1(x)
    add_75: "f32[64]" = torch.ops.aten.add.Tensor(primals_64, 1e-05);  primals_64 = None
    rsqrt_19: "f32[64]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
    unsqueeze_388: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_63, 0);  primals_63 = None
    unsqueeze_389: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 2);  unsqueeze_388 = None
    unsqueeze_390: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 3);  unsqueeze_389 = None
    sum_40: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_39: "f32[4, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_390);  convolution = unsqueeze_390 = None
    mul_212: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_16, sub_39);  sub_39 = None
    sum_41: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_212, [0, 2, 3]);  mul_212 = None
    mul_217: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_19, primals_2);  primals_2 = None
    unsqueeze_397: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_217, 0);  mul_217 = None
    unsqueeze_398: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None
    unsqueeze_399: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 3);  unsqueeze_398 = None
    mul_218: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_16, unsqueeze_399);  where_16 = unsqueeze_399 = None
    mul_219: "f32[64]" = torch.ops.aten.mul.Tensor(sum_41, rsqrt_19);  sum_41 = rsqrt_19 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:268, code: x = self.conv1(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_218, primals_123, primals_1, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_218 = primals_123 = primals_1 = None
    getitem_60: "f32[64, 3, 7, 7]" = convolution_backward_19[1];  convolution_backward_19 = None
    return [getitem_60, mul_219, sum_40, getitem_57, mul_211, sum_38, getitem_54, mul_203, sum_36, getitem_51, mul_195, sum_34, getitem_48, mul_187, sum_32, getitem_45, mul_179, sum_30, getitem_42, mul_171, sum_26, getitem_39, mul_163, sum_26, getitem_36, mul_155, sum_24, getitem_33, mul_147, sum_22, getitem_30, mul_139, sum_20, getitem_27, mul_131, sum_16, getitem_24, mul_123, sum_16, getitem_21, mul_115, sum_14, getitem_18, mul_107, sum_12, getitem_15, mul_99, sum_10, getitem_12, mul_91, sum_6, getitem_9, mul_83, sum_6, getitem_6, mul_75, sum_4, getitem_3, mul_67, sum_2, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    