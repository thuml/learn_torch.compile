from __future__ import annotations



def forward(self, primals_1: "f32[64, 3, 7, 7]", primals_2: "f32[64]", primals_4: "f32[128, 64, 1, 1]", primals_5: "f32[128]", primals_7: "f32[128, 4, 3, 3]", primals_8: "f32[128]", primals_10: "f32[256, 128, 1, 1]", primals_11: "f32[256]", primals_13: "f32[256, 64, 1, 1]", primals_14: "f32[256]", primals_16: "f32[128, 256, 1, 1]", primals_17: "f32[128]", primals_19: "f32[128, 4, 3, 3]", primals_20: "f32[128]", primals_22: "f32[256, 128, 1, 1]", primals_23: "f32[256]", primals_25: "f32[128, 256, 1, 1]", primals_26: "f32[128]", primals_28: "f32[128, 4, 3, 3]", primals_29: "f32[128]", primals_31: "f32[256, 128, 1, 1]", primals_32: "f32[256]", primals_34: "f32[256, 256, 1, 1]", primals_35: "f32[256]", primals_37: "f32[256, 8, 3, 3]", primals_38: "f32[256]", primals_40: "f32[512, 256, 1, 1]", primals_41: "f32[512]", primals_43: "f32[512, 256, 1, 1]", primals_44: "f32[512]", primals_46: "f32[256, 512, 1, 1]", primals_47: "f32[256]", primals_49: "f32[256, 8, 3, 3]", primals_50: "f32[256]", primals_52: "f32[512, 256, 1, 1]", primals_53: "f32[512]", primals_55: "f32[256, 512, 1, 1]", primals_56: "f32[256]", primals_58: "f32[256, 8, 3, 3]", primals_59: "f32[256]", primals_61: "f32[512, 256, 1, 1]", primals_62: "f32[512]", primals_64: "f32[256, 512, 1, 1]", primals_65: "f32[256]", primals_67: "f32[256, 8, 3, 3]", primals_68: "f32[256]", primals_70: "f32[512, 256, 1, 1]", primals_71: "f32[512]", primals_73: "f32[512, 512, 1, 1]", primals_74: "f32[512]", primals_76: "f32[512, 16, 3, 3]", primals_77: "f32[512]", primals_79: "f32[1024, 512, 1, 1]", primals_80: "f32[1024]", primals_82: "f32[1024, 512, 1, 1]", primals_83: "f32[1024]", primals_85: "f32[512, 1024, 1, 1]", primals_86: "f32[512]", primals_88: "f32[512, 16, 3, 3]", primals_89: "f32[512]", primals_91: "f32[1024, 512, 1, 1]", primals_92: "f32[1024]", primals_94: "f32[512, 1024, 1, 1]", primals_95: "f32[512]", primals_97: "f32[512, 16, 3, 3]", primals_98: "f32[512]", primals_100: "f32[1024, 512, 1, 1]", primals_101: "f32[1024]", primals_103: "f32[512, 1024, 1, 1]", primals_104: "f32[512]", primals_106: "f32[512, 16, 3, 3]", primals_107: "f32[512]", primals_109: "f32[1024, 512, 1, 1]", primals_110: "f32[1024]", primals_112: "f32[512, 1024, 1, 1]", primals_113: "f32[512]", primals_115: "f32[512, 16, 3, 3]", primals_116: "f32[512]", primals_118: "f32[1024, 512, 1, 1]", primals_119: "f32[1024]", primals_121: "f32[512, 1024, 1, 1]", primals_122: "f32[512]", primals_124: "f32[512, 16, 3, 3]", primals_125: "f32[512]", primals_127: "f32[1024, 512, 1, 1]", primals_128: "f32[1024]", primals_130: "f32[1024, 1024, 1, 1]", primals_131: "f32[1024]", primals_133: "f32[1024, 32, 3, 3]", primals_134: "f32[1024]", primals_136: "f32[2048, 1024, 1, 1]", primals_137: "f32[2048]", primals_139: "f32[2048, 1024, 1, 1]", primals_140: "f32[2048]", primals_142: "f32[1024, 2048, 1, 1]", primals_143: "f32[1024]", primals_145: "f32[1024, 32, 3, 3]", primals_146: "f32[1024]", primals_148: "f32[2048, 1024, 1, 1]", primals_149: "f32[2048]", primals_151: "f32[1024, 2048, 1, 1]", primals_152: "f32[1024]", primals_154: "f32[1024, 32, 3, 3]", primals_155: "f32[1024]", primals_157: "f32[2048, 1024, 1, 1]", primals_158: "f32[2048]", primals_162: "f32[64]", primals_163: "f32[64]", primals_165: "f32[128]", primals_166: "f32[128]", primals_168: "f32[128]", primals_169: "f32[128]", primals_171: "f32[256]", primals_172: "f32[256]", primals_174: "f32[256]", primals_175: "f32[256]", primals_177: "f32[128]", primals_178: "f32[128]", primals_180: "f32[128]", primals_181: "f32[128]", primals_183: "f32[256]", primals_184: "f32[256]", primals_186: "f32[128]", primals_187: "f32[128]", primals_189: "f32[128]", primals_190: "f32[128]", primals_192: "f32[256]", primals_193: "f32[256]", primals_195: "f32[256]", primals_196: "f32[256]", primals_198: "f32[256]", primals_199: "f32[256]", primals_201: "f32[512]", primals_202: "f32[512]", primals_204: "f32[512]", primals_205: "f32[512]", primals_207: "f32[256]", primals_208: "f32[256]", primals_210: "f32[256]", primals_211: "f32[256]", primals_213: "f32[512]", primals_214: "f32[512]", primals_216: "f32[256]", primals_217: "f32[256]", primals_219: "f32[256]", primals_220: "f32[256]", primals_222: "f32[512]", primals_223: "f32[512]", primals_225: "f32[256]", primals_226: "f32[256]", primals_228: "f32[256]", primals_229: "f32[256]", primals_231: "f32[512]", primals_232: "f32[512]", primals_234: "f32[512]", primals_235: "f32[512]", primals_237: "f32[512]", primals_238: "f32[512]", primals_240: "f32[1024]", primals_241: "f32[1024]", primals_243: "f32[1024]", primals_244: "f32[1024]", primals_246: "f32[512]", primals_247: "f32[512]", primals_249: "f32[512]", primals_250: "f32[512]", primals_252: "f32[1024]", primals_253: "f32[1024]", primals_255: "f32[512]", primals_256: "f32[512]", primals_258: "f32[512]", primals_259: "f32[512]", primals_261: "f32[1024]", primals_262: "f32[1024]", primals_264: "f32[512]", primals_265: "f32[512]", primals_267: "f32[512]", primals_268: "f32[512]", primals_270: "f32[1024]", primals_271: "f32[1024]", primals_273: "f32[512]", primals_274: "f32[512]", primals_276: "f32[512]", primals_277: "f32[512]", primals_279: "f32[1024]", primals_280: "f32[1024]", primals_282: "f32[512]", primals_283: "f32[512]", primals_285: "f32[512]", primals_286: "f32[512]", primals_288: "f32[1024]", primals_289: "f32[1024]", primals_291: "f32[1024]", primals_292: "f32[1024]", primals_294: "f32[1024]", primals_295: "f32[1024]", primals_297: "f32[2048]", primals_298: "f32[2048]", primals_300: "f32[2048]", primals_301: "f32[2048]", primals_303: "f32[1024]", primals_304: "f32[1024]", primals_306: "f32[1024]", primals_307: "f32[1024]", primals_309: "f32[2048]", primals_310: "f32[2048]", primals_312: "f32[1024]", primals_313: "f32[1024]", primals_315: "f32[1024]", primals_316: "f32[1024]", primals_318: "f32[2048]", primals_319: "f32[2048]", primals_321: "f32[4, 3, 224, 224]", convolution: "f32[4, 64, 112, 112]", relu: "f32[4, 64, 112, 112]", getitem: "f32[4, 64, 56, 56]", getitem_1: "i64[4, 64, 56, 56]", convolution_1: "f32[4, 128, 56, 56]", relu_1: "f32[4, 128, 56, 56]", convolution_2: "f32[4, 128, 56, 56]", relu_2: "f32[4, 128, 56, 56]", convolution_3: "f32[4, 256, 56, 56]", convolution_4: "f32[4, 256, 56, 56]", relu_3: "f32[4, 256, 56, 56]", convolution_5: "f32[4, 128, 56, 56]", relu_4: "f32[4, 128, 56, 56]", convolution_6: "f32[4, 128, 56, 56]", relu_5: "f32[4, 128, 56, 56]", convolution_7: "f32[4, 256, 56, 56]", relu_6: "f32[4, 256, 56, 56]", convolution_8: "f32[4, 128, 56, 56]", relu_7: "f32[4, 128, 56, 56]", convolution_9: "f32[4, 128, 56, 56]", relu_8: "f32[4, 128, 56, 56]", convolution_10: "f32[4, 256, 56, 56]", relu_9: "f32[4, 256, 56, 56]", convolution_11: "f32[4, 256, 56, 56]", relu_10: "f32[4, 256, 56, 56]", convolution_12: "f32[4, 256, 28, 28]", relu_11: "f32[4, 256, 28, 28]", convolution_13: "f32[4, 512, 28, 28]", convolution_14: "f32[4, 512, 28, 28]", relu_12: "f32[4, 512, 28, 28]", convolution_15: "f32[4, 256, 28, 28]", relu_13: "f32[4, 256, 28, 28]", convolution_16: "f32[4, 256, 28, 28]", relu_14: "f32[4, 256, 28, 28]", convolution_17: "f32[4, 512, 28, 28]", relu_15: "f32[4, 512, 28, 28]", convolution_18: "f32[4, 256, 28, 28]", relu_16: "f32[4, 256, 28, 28]", convolution_19: "f32[4, 256, 28, 28]", relu_17: "f32[4, 256, 28, 28]", convolution_20: "f32[4, 512, 28, 28]", relu_18: "f32[4, 512, 28, 28]", convolution_21: "f32[4, 256, 28, 28]", relu_19: "f32[4, 256, 28, 28]", convolution_22: "f32[4, 256, 28, 28]", relu_20: "f32[4, 256, 28, 28]", convolution_23: "f32[4, 512, 28, 28]", relu_21: "f32[4, 512, 28, 28]", convolution_24: "f32[4, 512, 28, 28]", relu_22: "f32[4, 512, 28, 28]", convolution_25: "f32[4, 512, 14, 14]", relu_23: "f32[4, 512, 14, 14]", convolution_26: "f32[4, 1024, 14, 14]", convolution_27: "f32[4, 1024, 14, 14]", relu_24: "f32[4, 1024, 14, 14]", convolution_28: "f32[4, 512, 14, 14]", relu_25: "f32[4, 512, 14, 14]", convolution_29: "f32[4, 512, 14, 14]", relu_26: "f32[4, 512, 14, 14]", convolution_30: "f32[4, 1024, 14, 14]", relu_27: "f32[4, 1024, 14, 14]", convolution_31: "f32[4, 512, 14, 14]", relu_28: "f32[4, 512, 14, 14]", convolution_32: "f32[4, 512, 14, 14]", relu_29: "f32[4, 512, 14, 14]", convolution_33: "f32[4, 1024, 14, 14]", relu_30: "f32[4, 1024, 14, 14]", convolution_34: "f32[4, 512, 14, 14]", relu_31: "f32[4, 512, 14, 14]", convolution_35: "f32[4, 512, 14, 14]", relu_32: "f32[4, 512, 14, 14]", convolution_36: "f32[4, 1024, 14, 14]", relu_33: "f32[4, 1024, 14, 14]", convolution_37: "f32[4, 512, 14, 14]", relu_34: "f32[4, 512, 14, 14]", convolution_38: "f32[4, 512, 14, 14]", relu_35: "f32[4, 512, 14, 14]", convolution_39: "f32[4, 1024, 14, 14]", relu_36: "f32[4, 1024, 14, 14]", convolution_40: "f32[4, 512, 14, 14]", relu_37: "f32[4, 512, 14, 14]", convolution_41: "f32[4, 512, 14, 14]", relu_38: "f32[4, 512, 14, 14]", convolution_42: "f32[4, 1024, 14, 14]", relu_39: "f32[4, 1024, 14, 14]", convolution_43: "f32[4, 1024, 14, 14]", relu_40: "f32[4, 1024, 14, 14]", convolution_44: "f32[4, 1024, 7, 7]", relu_41: "f32[4, 1024, 7, 7]", convolution_45: "f32[4, 2048, 7, 7]", convolution_46: "f32[4, 2048, 7, 7]", relu_42: "f32[4, 2048, 7, 7]", convolution_47: "f32[4, 1024, 7, 7]", relu_43: "f32[4, 1024, 7, 7]", convolution_48: "f32[4, 1024, 7, 7]", relu_44: "f32[4, 1024, 7, 7]", convolution_49: "f32[4, 2048, 7, 7]", relu_45: "f32[4, 2048, 7, 7]", convolution_50: "f32[4, 1024, 7, 7]", relu_46: "f32[4, 1024, 7, 7]", convolution_51: "f32[4, 1024, 7, 7]", relu_47: "f32[4, 1024, 7, 7]", convolution_52: "f32[4, 2048, 7, 7]", view: "f32[4, 2048]", permute_1: "f32[1000, 2048]", le: "b8[4, 2048, 7, 7]", tangents_1: "f32[4, 1000]"):
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:280, code: x = self.fc(x)
    mm: "f32[4, 2048]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 4]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 2048]" = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
    permute_3: "f32[2048, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 2048]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:279, code: x = torch.flatten(x, 1)
    view_2: "f32[4, 2048, 1, 1]" = torch.ops.aten.reshape.default(mm, [4, 2048, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:278, code: x = self.avgpool(x)
    expand: "f32[4, 2048, 7, 7]" = torch.ops.aten.expand.default(view_2, [4, 2048, 7, 7]);  view_2 = None
    div: "f32[4, 2048, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where: "f32[4, 2048, 7, 7]" = torch.ops.aten.where.self(le, full_default, div);  le = div = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    add_122: "f32[2048]" = torch.ops.aten.add.Tensor(primals_319, 1e-05);  primals_319 = None
    rsqrt: "f32[2048]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    unsqueeze_424: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(primals_318, 0);  primals_318 = None
    unsqueeze_425: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 2);  unsqueeze_424 = None
    unsqueeze_426: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 3);  unsqueeze_425 = None
    sum_2: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_53: "f32[4, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_426);  convolution_52 = unsqueeze_426 = None
    mul_159: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_53);  sub_53 = None
    sum_3: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_159, [0, 2, 3]);  mul_159 = None
    mul_164: "f32[2048]" = torch.ops.aten.mul.Tensor(rsqrt, primals_158);  primals_158 = None
    unsqueeze_433: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_164, 0);  mul_164 = None
    unsqueeze_434: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 2);  unsqueeze_433 = None
    unsqueeze_435: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 3);  unsqueeze_434 = None
    mul_165: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where, unsqueeze_435);  unsqueeze_435 = None
    mul_166: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_3, rsqrt);  sum_3 = rsqrt = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_165, relu_47, primals_157, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_165 = primals_157 = None
    getitem_2: "f32[4, 1024, 7, 7]" = convolution_backward[0]
    getitem_3: "f32[2048, 1024, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    le_1: "b8[4, 1024, 7, 7]" = torch.ops.aten.le.Scalar(relu_47, 0);  relu_47 = None
    where_1: "f32[4, 1024, 7, 7]" = torch.ops.aten.where.self(le_1, full_default, getitem_2);  le_1 = getitem_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    add_123: "f32[1024]" = torch.ops.aten.add.Tensor(primals_316, 1e-05);  primals_316 = None
    rsqrt_1: "f32[1024]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
    unsqueeze_436: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(primals_315, 0);  primals_315 = None
    unsqueeze_437: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 2);  unsqueeze_436 = None
    unsqueeze_438: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 3);  unsqueeze_437 = None
    sum_4: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_54: "f32[4, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_438);  convolution_51 = unsqueeze_438 = None
    mul_167: "f32[4, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, sub_54);  sub_54 = None
    sum_5: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_167, [0, 2, 3]);  mul_167 = None
    mul_172: "f32[1024]" = torch.ops.aten.mul.Tensor(rsqrt_1, primals_155);  primals_155 = None
    unsqueeze_445: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_172, 0);  mul_172 = None
    unsqueeze_446: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 2);  unsqueeze_445 = None
    unsqueeze_447: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 3);  unsqueeze_446 = None
    mul_173: "f32[4, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, unsqueeze_447);  where_1 = unsqueeze_447 = None
    mul_174: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_5, rsqrt_1);  sum_5 = rsqrt_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_173, relu_46, primals_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_173 = primals_154 = None
    getitem_5: "f32[4, 1024, 7, 7]" = convolution_backward_1[0]
    getitem_6: "f32[1024, 32, 3, 3]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    le_2: "b8[4, 1024, 7, 7]" = torch.ops.aten.le.Scalar(relu_46, 0);  relu_46 = None
    where_2: "f32[4, 1024, 7, 7]" = torch.ops.aten.where.self(le_2, full_default, getitem_5);  le_2 = getitem_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    add_124: "f32[1024]" = torch.ops.aten.add.Tensor(primals_313, 1e-05);  primals_313 = None
    rsqrt_2: "f32[1024]" = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
    unsqueeze_448: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(primals_312, 0);  primals_312 = None
    unsqueeze_449: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, 2);  unsqueeze_448 = None
    unsqueeze_450: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 3);  unsqueeze_449 = None
    sum_6: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_55: "f32[4, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_450);  convolution_50 = unsqueeze_450 = None
    mul_175: "f32[4, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_55);  sub_55 = None
    sum_7: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_175, [0, 2, 3]);  mul_175 = None
    mul_180: "f32[1024]" = torch.ops.aten.mul.Tensor(rsqrt_2, primals_152);  primals_152 = None
    unsqueeze_457: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_180, 0);  mul_180 = None
    unsqueeze_458: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 2);  unsqueeze_457 = None
    unsqueeze_459: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 3);  unsqueeze_458 = None
    mul_181: "f32[4, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, unsqueeze_459);  where_2 = unsqueeze_459 = None
    mul_182: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_7, rsqrt_2);  sum_7 = rsqrt_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_181, relu_45, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_181 = primals_151 = None
    getitem_8: "f32[4, 2048, 7, 7]" = convolution_backward_2[0]
    getitem_9: "f32[1024, 2048, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    add_125: "f32[4, 2048, 7, 7]" = torch.ops.aten.add.Tensor(where, getitem_8);  where = getitem_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    le_3: "b8[4, 2048, 7, 7]" = torch.ops.aten.le.Scalar(relu_45, 0);  relu_45 = None
    where_3: "f32[4, 2048, 7, 7]" = torch.ops.aten.where.self(le_3, full_default, add_125);  le_3 = add_125 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    add_126: "f32[2048]" = torch.ops.aten.add.Tensor(primals_310, 1e-05);  primals_310 = None
    rsqrt_3: "f32[2048]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
    unsqueeze_460: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(primals_309, 0);  primals_309 = None
    unsqueeze_461: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, 2);  unsqueeze_460 = None
    unsqueeze_462: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 3);  unsqueeze_461 = None
    sum_8: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_56: "f32[4, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_462);  convolution_49 = unsqueeze_462 = None
    mul_183: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_56);  sub_56 = None
    sum_9: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_183, [0, 2, 3]);  mul_183 = None
    mul_188: "f32[2048]" = torch.ops.aten.mul.Tensor(rsqrt_3, primals_149);  primals_149 = None
    unsqueeze_469: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_188, 0);  mul_188 = None
    unsqueeze_470: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 2);  unsqueeze_469 = None
    unsqueeze_471: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 3);  unsqueeze_470 = None
    mul_189: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, unsqueeze_471);  unsqueeze_471 = None
    mul_190: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_9, rsqrt_3);  sum_9 = rsqrt_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_189, relu_44, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_189 = primals_148 = None
    getitem_11: "f32[4, 1024, 7, 7]" = convolution_backward_3[0]
    getitem_12: "f32[2048, 1024, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    le_4: "b8[4, 1024, 7, 7]" = torch.ops.aten.le.Scalar(relu_44, 0);  relu_44 = None
    where_4: "f32[4, 1024, 7, 7]" = torch.ops.aten.where.self(le_4, full_default, getitem_11);  le_4 = getitem_11 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    add_127: "f32[1024]" = torch.ops.aten.add.Tensor(primals_307, 1e-05);  primals_307 = None
    rsqrt_4: "f32[1024]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
    unsqueeze_472: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(primals_306, 0);  primals_306 = None
    unsqueeze_473: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, 2);  unsqueeze_472 = None
    unsqueeze_474: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 3);  unsqueeze_473 = None
    sum_10: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_57: "f32[4, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_474);  convolution_48 = unsqueeze_474 = None
    mul_191: "f32[4, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, sub_57);  sub_57 = None
    sum_11: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_191, [0, 2, 3]);  mul_191 = None
    mul_196: "f32[1024]" = torch.ops.aten.mul.Tensor(rsqrt_4, primals_146);  primals_146 = None
    unsqueeze_481: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_196, 0);  mul_196 = None
    unsqueeze_482: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 2);  unsqueeze_481 = None
    unsqueeze_483: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 3);  unsqueeze_482 = None
    mul_197: "f32[4, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, unsqueeze_483);  where_4 = unsqueeze_483 = None
    mul_198: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_11, rsqrt_4);  sum_11 = rsqrt_4 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_197, relu_43, primals_145, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_197 = primals_145 = None
    getitem_14: "f32[4, 1024, 7, 7]" = convolution_backward_4[0]
    getitem_15: "f32[1024, 32, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    le_5: "b8[4, 1024, 7, 7]" = torch.ops.aten.le.Scalar(relu_43, 0);  relu_43 = None
    where_5: "f32[4, 1024, 7, 7]" = torch.ops.aten.where.self(le_5, full_default, getitem_14);  le_5 = getitem_14 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    add_128: "f32[1024]" = torch.ops.aten.add.Tensor(primals_304, 1e-05);  primals_304 = None
    rsqrt_5: "f32[1024]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
    unsqueeze_484: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(primals_303, 0);  primals_303 = None
    unsqueeze_485: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 2);  unsqueeze_484 = None
    unsqueeze_486: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 3);  unsqueeze_485 = None
    sum_12: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_58: "f32[4, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_486);  convolution_47 = unsqueeze_486 = None
    mul_199: "f32[4, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where_5, sub_58);  sub_58 = None
    sum_13: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_199, [0, 2, 3]);  mul_199 = None
    mul_204: "f32[1024]" = torch.ops.aten.mul.Tensor(rsqrt_5, primals_143);  primals_143 = None
    unsqueeze_493: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_204, 0);  mul_204 = None
    unsqueeze_494: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 2);  unsqueeze_493 = None
    unsqueeze_495: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 3);  unsqueeze_494 = None
    mul_205: "f32[4, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where_5, unsqueeze_495);  where_5 = unsqueeze_495 = None
    mul_206: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_13, rsqrt_5);  sum_13 = rsqrt_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_205, relu_42, primals_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_205 = primals_142 = None
    getitem_17: "f32[4, 2048, 7, 7]" = convolution_backward_5[0]
    getitem_18: "f32[1024, 2048, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    add_129: "f32[4, 2048, 7, 7]" = torch.ops.aten.add.Tensor(where_3, getitem_17);  where_3 = getitem_17 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    le_6: "b8[4, 2048, 7, 7]" = torch.ops.aten.le.Scalar(relu_42, 0);  relu_42 = None
    where_6: "f32[4, 2048, 7, 7]" = torch.ops.aten.where.self(le_6, full_default, add_129);  le_6 = add_129 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:158, code: identity = self.downsample(x)
    add_130: "f32[2048]" = torch.ops.aten.add.Tensor(primals_301, 1e-05);  primals_301 = None
    rsqrt_6: "f32[2048]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
    unsqueeze_496: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(primals_300, 0);  primals_300 = None
    unsqueeze_497: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, 2);  unsqueeze_496 = None
    unsqueeze_498: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 3);  unsqueeze_497 = None
    sum_14: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_59: "f32[4, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_498);  convolution_46 = unsqueeze_498 = None
    mul_207: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, sub_59);  sub_59 = None
    sum_15: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_207, [0, 2, 3]);  mul_207 = None
    mul_212: "f32[2048]" = torch.ops.aten.mul.Tensor(rsqrt_6, primals_140);  primals_140 = None
    unsqueeze_505: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_212, 0);  mul_212 = None
    unsqueeze_506: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 2);  unsqueeze_505 = None
    unsqueeze_507: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 3);  unsqueeze_506 = None
    mul_213: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, unsqueeze_507);  unsqueeze_507 = None
    mul_214: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_15, rsqrt_6);  sum_15 = rsqrt_6 = None
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_213, relu_39, primals_139, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_213 = primals_139 = None
    getitem_20: "f32[4, 1024, 14, 14]" = convolution_backward_6[0]
    getitem_21: "f32[2048, 1024, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    add_131: "f32[2048]" = torch.ops.aten.add.Tensor(primals_298, 1e-05);  primals_298 = None
    rsqrt_7: "f32[2048]" = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
    unsqueeze_508: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(primals_297, 0);  primals_297 = None
    unsqueeze_509: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, 2);  unsqueeze_508 = None
    unsqueeze_510: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 3);  unsqueeze_509 = None
    sub_60: "f32[4, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_510);  convolution_45 = unsqueeze_510 = None
    mul_215: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, sub_60);  sub_60 = None
    sum_17: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_215, [0, 2, 3]);  mul_215 = None
    mul_220: "f32[2048]" = torch.ops.aten.mul.Tensor(rsqrt_7, primals_137);  primals_137 = None
    unsqueeze_517: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_220, 0);  mul_220 = None
    unsqueeze_518: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 2);  unsqueeze_517 = None
    unsqueeze_519: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 3);  unsqueeze_518 = None
    mul_221: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, unsqueeze_519);  where_6 = unsqueeze_519 = None
    mul_222: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_17, rsqrt_7);  sum_17 = rsqrt_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_221, relu_41, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_221 = primals_136 = None
    getitem_23: "f32[4, 1024, 7, 7]" = convolution_backward_7[0]
    getitem_24: "f32[2048, 1024, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    le_7: "b8[4, 1024, 7, 7]" = torch.ops.aten.le.Scalar(relu_41, 0);  relu_41 = None
    where_7: "f32[4, 1024, 7, 7]" = torch.ops.aten.where.self(le_7, full_default, getitem_23);  le_7 = getitem_23 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    add_132: "f32[1024]" = torch.ops.aten.add.Tensor(primals_295, 1e-05);  primals_295 = None
    rsqrt_8: "f32[1024]" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
    unsqueeze_520: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(primals_294, 0);  primals_294 = None
    unsqueeze_521: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, 2);  unsqueeze_520 = None
    unsqueeze_522: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 3);  unsqueeze_521 = None
    sum_18: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_61: "f32[4, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_522);  convolution_44 = unsqueeze_522 = None
    mul_223: "f32[4, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, sub_61);  sub_61 = None
    sum_19: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_223, [0, 2, 3]);  mul_223 = None
    mul_228: "f32[1024]" = torch.ops.aten.mul.Tensor(rsqrt_8, primals_134);  primals_134 = None
    unsqueeze_529: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_228, 0);  mul_228 = None
    unsqueeze_530: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_529, 2);  unsqueeze_529 = None
    unsqueeze_531: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 3);  unsqueeze_530 = None
    mul_229: "f32[4, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, unsqueeze_531);  where_7 = unsqueeze_531 = None
    mul_230: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_19, rsqrt_8);  sum_19 = rsqrt_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_229, relu_40, primals_133, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_229 = primals_133 = None
    getitem_26: "f32[4, 1024, 14, 14]" = convolution_backward_8[0]
    getitem_27: "f32[1024, 32, 3, 3]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    le_8: "b8[4, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_40, 0);  relu_40 = None
    where_8: "f32[4, 1024, 14, 14]" = torch.ops.aten.where.self(le_8, full_default, getitem_26);  le_8 = getitem_26 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    add_133: "f32[1024]" = torch.ops.aten.add.Tensor(primals_292, 1e-05);  primals_292 = None
    rsqrt_9: "f32[1024]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
    unsqueeze_532: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(primals_291, 0);  primals_291 = None
    unsqueeze_533: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, 2);  unsqueeze_532 = None
    unsqueeze_534: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 3);  unsqueeze_533 = None
    sum_20: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_62: "f32[4, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_534);  convolution_43 = unsqueeze_534 = None
    mul_231: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_8, sub_62);  sub_62 = None
    sum_21: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_231, [0, 2, 3]);  mul_231 = None
    mul_236: "f32[1024]" = torch.ops.aten.mul.Tensor(rsqrt_9, primals_131);  primals_131 = None
    unsqueeze_541: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_236, 0);  mul_236 = None
    unsqueeze_542: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 2);  unsqueeze_541 = None
    unsqueeze_543: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 3);  unsqueeze_542 = None
    mul_237: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_8, unsqueeze_543);  where_8 = unsqueeze_543 = None
    mul_238: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_21, rsqrt_9);  sum_21 = rsqrt_9 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_237, relu_39, primals_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_237 = primals_130 = None
    getitem_29: "f32[4, 1024, 14, 14]" = convolution_backward_9[0]
    getitem_30: "f32[1024, 1024, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    add_134: "f32[4, 1024, 14, 14]" = torch.ops.aten.add.Tensor(getitem_20, getitem_29);  getitem_20 = getitem_29 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    le_9: "b8[4, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_39, 0);  relu_39 = None
    where_9: "f32[4, 1024, 14, 14]" = torch.ops.aten.where.self(le_9, full_default, add_134);  le_9 = add_134 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    add_135: "f32[1024]" = torch.ops.aten.add.Tensor(primals_289, 1e-05);  primals_289 = None
    rsqrt_10: "f32[1024]" = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
    unsqueeze_544: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(primals_288, 0);  primals_288 = None
    unsqueeze_545: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, 2);  unsqueeze_544 = None
    unsqueeze_546: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 3);  unsqueeze_545 = None
    sum_22: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_63: "f32[4, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_546);  convolution_42 = unsqueeze_546 = None
    mul_239: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_9, sub_63);  sub_63 = None
    sum_23: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_239, [0, 2, 3]);  mul_239 = None
    mul_244: "f32[1024]" = torch.ops.aten.mul.Tensor(rsqrt_10, primals_128);  primals_128 = None
    unsqueeze_553: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_244, 0);  mul_244 = None
    unsqueeze_554: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 2);  unsqueeze_553 = None
    unsqueeze_555: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 3);  unsqueeze_554 = None
    mul_245: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_9, unsqueeze_555);  unsqueeze_555 = None
    mul_246: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_23, rsqrt_10);  sum_23 = rsqrt_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_245, relu_38, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_245 = primals_127 = None
    getitem_32: "f32[4, 512, 14, 14]" = convolution_backward_10[0]
    getitem_33: "f32[1024, 512, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    le_10: "b8[4, 512, 14, 14]" = torch.ops.aten.le.Scalar(relu_38, 0);  relu_38 = None
    where_10: "f32[4, 512, 14, 14]" = torch.ops.aten.where.self(le_10, full_default, getitem_32);  le_10 = getitem_32 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    add_136: "f32[512]" = torch.ops.aten.add.Tensor(primals_286, 1e-05);  primals_286 = None
    rsqrt_11: "f32[512]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
    unsqueeze_556: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_285, 0);  primals_285 = None
    unsqueeze_557: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, 2);  unsqueeze_556 = None
    unsqueeze_558: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 3);  unsqueeze_557 = None
    sum_24: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_64: "f32[4, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_558);  convolution_41 = unsqueeze_558 = None
    mul_247: "f32[4, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_10, sub_64);  sub_64 = None
    sum_25: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_247, [0, 2, 3]);  mul_247 = None
    mul_252: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_11, primals_125);  primals_125 = None
    unsqueeze_565: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_252, 0);  mul_252 = None
    unsqueeze_566: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 2);  unsqueeze_565 = None
    unsqueeze_567: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 3);  unsqueeze_566 = None
    mul_253: "f32[4, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_10, unsqueeze_567);  where_10 = unsqueeze_567 = None
    mul_254: "f32[512]" = torch.ops.aten.mul.Tensor(sum_25, rsqrt_11);  sum_25 = rsqrt_11 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_253, relu_37, primals_124, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_253 = primals_124 = None
    getitem_35: "f32[4, 512, 14, 14]" = convolution_backward_11[0]
    getitem_36: "f32[512, 16, 3, 3]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    le_11: "b8[4, 512, 14, 14]" = torch.ops.aten.le.Scalar(relu_37, 0);  relu_37 = None
    where_11: "f32[4, 512, 14, 14]" = torch.ops.aten.where.self(le_11, full_default, getitem_35);  le_11 = getitem_35 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    add_137: "f32[512]" = torch.ops.aten.add.Tensor(primals_283, 1e-05);  primals_283 = None
    rsqrt_12: "f32[512]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
    unsqueeze_568: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_282, 0);  primals_282 = None
    unsqueeze_569: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, 2);  unsqueeze_568 = None
    unsqueeze_570: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 3);  unsqueeze_569 = None
    sum_26: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_65: "f32[4, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_570);  convolution_40 = unsqueeze_570 = None
    mul_255: "f32[4, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_11, sub_65);  sub_65 = None
    sum_27: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_255, [0, 2, 3]);  mul_255 = None
    mul_260: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_12, primals_122);  primals_122 = None
    unsqueeze_577: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_260, 0);  mul_260 = None
    unsqueeze_578: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_577, 2);  unsqueeze_577 = None
    unsqueeze_579: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 3);  unsqueeze_578 = None
    mul_261: "f32[4, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_11, unsqueeze_579);  where_11 = unsqueeze_579 = None
    mul_262: "f32[512]" = torch.ops.aten.mul.Tensor(sum_27, rsqrt_12);  sum_27 = rsqrt_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_261, relu_36, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_261 = primals_121 = None
    getitem_38: "f32[4, 1024, 14, 14]" = convolution_backward_12[0]
    getitem_39: "f32[512, 1024, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    add_138: "f32[4, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_9, getitem_38);  where_9 = getitem_38 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    le_12: "b8[4, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_36, 0);  relu_36 = None
    where_12: "f32[4, 1024, 14, 14]" = torch.ops.aten.where.self(le_12, full_default, add_138);  le_12 = add_138 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    add_139: "f32[1024]" = torch.ops.aten.add.Tensor(primals_280, 1e-05);  primals_280 = None
    rsqrt_13: "f32[1024]" = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
    unsqueeze_580: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(primals_279, 0);  primals_279 = None
    unsqueeze_581: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, 2);  unsqueeze_580 = None
    unsqueeze_582: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 3);  unsqueeze_581 = None
    sum_28: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_66: "f32[4, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_582);  convolution_39 = unsqueeze_582 = None
    mul_263: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, sub_66);  sub_66 = None
    sum_29: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_263, [0, 2, 3]);  mul_263 = None
    mul_268: "f32[1024]" = torch.ops.aten.mul.Tensor(rsqrt_13, primals_119);  primals_119 = None
    unsqueeze_589: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_268, 0);  mul_268 = None
    unsqueeze_590: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_589, 2);  unsqueeze_589 = None
    unsqueeze_591: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 3);  unsqueeze_590 = None
    mul_269: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, unsqueeze_591);  unsqueeze_591 = None
    mul_270: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_29, rsqrt_13);  sum_29 = rsqrt_13 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_269, relu_35, primals_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_269 = primals_118 = None
    getitem_41: "f32[4, 512, 14, 14]" = convolution_backward_13[0]
    getitem_42: "f32[1024, 512, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    le_13: "b8[4, 512, 14, 14]" = torch.ops.aten.le.Scalar(relu_35, 0);  relu_35 = None
    where_13: "f32[4, 512, 14, 14]" = torch.ops.aten.where.self(le_13, full_default, getitem_41);  le_13 = getitem_41 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    add_140: "f32[512]" = torch.ops.aten.add.Tensor(primals_277, 1e-05);  primals_277 = None
    rsqrt_14: "f32[512]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    unsqueeze_592: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_276, 0);  primals_276 = None
    unsqueeze_593: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 2);  unsqueeze_592 = None
    unsqueeze_594: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 3);  unsqueeze_593 = None
    sum_30: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_67: "f32[4, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_594);  convolution_38 = unsqueeze_594 = None
    mul_271: "f32[4, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_13, sub_67);  sub_67 = None
    sum_31: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_271, [0, 2, 3]);  mul_271 = None
    mul_276: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_14, primals_116);  primals_116 = None
    unsqueeze_601: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_276, 0);  mul_276 = None
    unsqueeze_602: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 2);  unsqueeze_601 = None
    unsqueeze_603: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 3);  unsqueeze_602 = None
    mul_277: "f32[4, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_13, unsqueeze_603);  where_13 = unsqueeze_603 = None
    mul_278: "f32[512]" = torch.ops.aten.mul.Tensor(sum_31, rsqrt_14);  sum_31 = rsqrt_14 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_277, relu_34, primals_115, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_277 = primals_115 = None
    getitem_44: "f32[4, 512, 14, 14]" = convolution_backward_14[0]
    getitem_45: "f32[512, 16, 3, 3]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    le_14: "b8[4, 512, 14, 14]" = torch.ops.aten.le.Scalar(relu_34, 0);  relu_34 = None
    where_14: "f32[4, 512, 14, 14]" = torch.ops.aten.where.self(le_14, full_default, getitem_44);  le_14 = getitem_44 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    add_141: "f32[512]" = torch.ops.aten.add.Tensor(primals_274, 1e-05);  primals_274 = None
    rsqrt_15: "f32[512]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
    unsqueeze_604: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_273, 0);  primals_273 = None
    unsqueeze_605: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, 2);  unsqueeze_604 = None
    unsqueeze_606: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 3);  unsqueeze_605 = None
    sum_32: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_68: "f32[4, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_606);  convolution_37 = unsqueeze_606 = None
    mul_279: "f32[4, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, sub_68);  sub_68 = None
    sum_33: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_279, [0, 2, 3]);  mul_279 = None
    mul_284: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_15, primals_113);  primals_113 = None
    unsqueeze_613: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_284, 0);  mul_284 = None
    unsqueeze_614: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_613, 2);  unsqueeze_613 = None
    unsqueeze_615: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 3);  unsqueeze_614 = None
    mul_285: "f32[4, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, unsqueeze_615);  where_14 = unsqueeze_615 = None
    mul_286: "f32[512]" = torch.ops.aten.mul.Tensor(sum_33, rsqrt_15);  sum_33 = rsqrt_15 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_285, relu_33, primals_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_285 = primals_112 = None
    getitem_47: "f32[4, 1024, 14, 14]" = convolution_backward_15[0]
    getitem_48: "f32[512, 1024, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    add_142: "f32[4, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_12, getitem_47);  where_12 = getitem_47 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    le_15: "b8[4, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_33, 0);  relu_33 = None
    where_15: "f32[4, 1024, 14, 14]" = torch.ops.aten.where.self(le_15, full_default, add_142);  le_15 = add_142 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    add_143: "f32[1024]" = torch.ops.aten.add.Tensor(primals_271, 1e-05);  primals_271 = None
    rsqrt_16: "f32[1024]" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
    unsqueeze_616: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(primals_270, 0);  primals_270 = None
    unsqueeze_617: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, 2);  unsqueeze_616 = None
    unsqueeze_618: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 3);  unsqueeze_617 = None
    sum_34: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_69: "f32[4, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_618);  convolution_36 = unsqueeze_618 = None
    mul_287: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, sub_69);  sub_69 = None
    sum_35: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_287, [0, 2, 3]);  mul_287 = None
    mul_292: "f32[1024]" = torch.ops.aten.mul.Tensor(rsqrt_16, primals_110);  primals_110 = None
    unsqueeze_625: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_292, 0);  mul_292 = None
    unsqueeze_626: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_625, 2);  unsqueeze_625 = None
    unsqueeze_627: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 3);  unsqueeze_626 = None
    mul_293: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, unsqueeze_627);  unsqueeze_627 = None
    mul_294: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_35, rsqrt_16);  sum_35 = rsqrt_16 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_293, relu_32, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_293 = primals_109 = None
    getitem_50: "f32[4, 512, 14, 14]" = convolution_backward_16[0]
    getitem_51: "f32[1024, 512, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    le_16: "b8[4, 512, 14, 14]" = torch.ops.aten.le.Scalar(relu_32, 0);  relu_32 = None
    where_16: "f32[4, 512, 14, 14]" = torch.ops.aten.where.self(le_16, full_default, getitem_50);  le_16 = getitem_50 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    add_144: "f32[512]" = torch.ops.aten.add.Tensor(primals_268, 1e-05);  primals_268 = None
    rsqrt_17: "f32[512]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    unsqueeze_628: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_267, 0);  primals_267 = None
    unsqueeze_629: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, 2);  unsqueeze_628 = None
    unsqueeze_630: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 3);  unsqueeze_629 = None
    sum_36: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_70: "f32[4, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_630);  convolution_35 = unsqueeze_630 = None
    mul_295: "f32[4, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_16, sub_70);  sub_70 = None
    sum_37: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_295, [0, 2, 3]);  mul_295 = None
    mul_300: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_17, primals_107);  primals_107 = None
    unsqueeze_637: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_300, 0);  mul_300 = None
    unsqueeze_638: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_637, 2);  unsqueeze_637 = None
    unsqueeze_639: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 3);  unsqueeze_638 = None
    mul_301: "f32[4, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_16, unsqueeze_639);  where_16 = unsqueeze_639 = None
    mul_302: "f32[512]" = torch.ops.aten.mul.Tensor(sum_37, rsqrt_17);  sum_37 = rsqrt_17 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_301, relu_31, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_301 = primals_106 = None
    getitem_53: "f32[4, 512, 14, 14]" = convolution_backward_17[0]
    getitem_54: "f32[512, 16, 3, 3]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    le_17: "b8[4, 512, 14, 14]" = torch.ops.aten.le.Scalar(relu_31, 0);  relu_31 = None
    where_17: "f32[4, 512, 14, 14]" = torch.ops.aten.where.self(le_17, full_default, getitem_53);  le_17 = getitem_53 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    add_145: "f32[512]" = torch.ops.aten.add.Tensor(primals_265, 1e-05);  primals_265 = None
    rsqrt_18: "f32[512]" = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
    unsqueeze_640: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_264, 0);  primals_264 = None
    unsqueeze_641: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, 2);  unsqueeze_640 = None
    unsqueeze_642: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 3);  unsqueeze_641 = None
    sum_38: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_71: "f32[4, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_642);  convolution_34 = unsqueeze_642 = None
    mul_303: "f32[4, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_17, sub_71);  sub_71 = None
    sum_39: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_303, [0, 2, 3]);  mul_303 = None
    mul_308: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_18, primals_104);  primals_104 = None
    unsqueeze_649: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_308, 0);  mul_308 = None
    unsqueeze_650: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_649, 2);  unsqueeze_649 = None
    unsqueeze_651: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 3);  unsqueeze_650 = None
    mul_309: "f32[4, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_17, unsqueeze_651);  where_17 = unsqueeze_651 = None
    mul_310: "f32[512]" = torch.ops.aten.mul.Tensor(sum_39, rsqrt_18);  sum_39 = rsqrt_18 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_309, relu_30, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_309 = primals_103 = None
    getitem_56: "f32[4, 1024, 14, 14]" = convolution_backward_18[0]
    getitem_57: "f32[512, 1024, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    add_146: "f32[4, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_15, getitem_56);  where_15 = getitem_56 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    le_18: "b8[4, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_30, 0);  relu_30 = None
    where_18: "f32[4, 1024, 14, 14]" = torch.ops.aten.where.self(le_18, full_default, add_146);  le_18 = add_146 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    add_147: "f32[1024]" = torch.ops.aten.add.Tensor(primals_262, 1e-05);  primals_262 = None
    rsqrt_19: "f32[1024]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
    unsqueeze_652: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(primals_261, 0);  primals_261 = None
    unsqueeze_653: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, 2);  unsqueeze_652 = None
    unsqueeze_654: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 3);  unsqueeze_653 = None
    sum_40: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    sub_72: "f32[4, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_654);  convolution_33 = unsqueeze_654 = None
    mul_311: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_18, sub_72);  sub_72 = None
    sum_41: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_311, [0, 2, 3]);  mul_311 = None
    mul_316: "f32[1024]" = torch.ops.aten.mul.Tensor(rsqrt_19, primals_101);  primals_101 = None
    unsqueeze_661: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_316, 0);  mul_316 = None
    unsqueeze_662: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_661, 2);  unsqueeze_661 = None
    unsqueeze_663: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 3);  unsqueeze_662 = None
    mul_317: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_18, unsqueeze_663);  unsqueeze_663 = None
    mul_318: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_41, rsqrt_19);  sum_41 = rsqrt_19 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_317, relu_29, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_317 = primals_100 = None
    getitem_59: "f32[4, 512, 14, 14]" = convolution_backward_19[0]
    getitem_60: "f32[1024, 512, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    le_19: "b8[4, 512, 14, 14]" = torch.ops.aten.le.Scalar(relu_29, 0);  relu_29 = None
    where_19: "f32[4, 512, 14, 14]" = torch.ops.aten.where.self(le_19, full_default, getitem_59);  le_19 = getitem_59 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    add_148: "f32[512]" = torch.ops.aten.add.Tensor(primals_259, 1e-05);  primals_259 = None
    rsqrt_20: "f32[512]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
    unsqueeze_664: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_258, 0);  primals_258 = None
    unsqueeze_665: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, 2);  unsqueeze_664 = None
    unsqueeze_666: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 3);  unsqueeze_665 = None
    sum_42: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_73: "f32[4, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_666);  convolution_32 = unsqueeze_666 = None
    mul_319: "f32[4, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_19, sub_73);  sub_73 = None
    sum_43: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_319, [0, 2, 3]);  mul_319 = None
    mul_324: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_20, primals_98);  primals_98 = None
    unsqueeze_673: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_324, 0);  mul_324 = None
    unsqueeze_674: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_673, 2);  unsqueeze_673 = None
    unsqueeze_675: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 3);  unsqueeze_674 = None
    mul_325: "f32[4, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_19, unsqueeze_675);  where_19 = unsqueeze_675 = None
    mul_326: "f32[512]" = torch.ops.aten.mul.Tensor(sum_43, rsqrt_20);  sum_43 = rsqrt_20 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_325, relu_28, primals_97, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_325 = primals_97 = None
    getitem_62: "f32[4, 512, 14, 14]" = convolution_backward_20[0]
    getitem_63: "f32[512, 16, 3, 3]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    le_20: "b8[4, 512, 14, 14]" = torch.ops.aten.le.Scalar(relu_28, 0);  relu_28 = None
    where_20: "f32[4, 512, 14, 14]" = torch.ops.aten.where.self(le_20, full_default, getitem_62);  le_20 = getitem_62 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    add_149: "f32[512]" = torch.ops.aten.add.Tensor(primals_256, 1e-05);  primals_256 = None
    rsqrt_21: "f32[512]" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
    unsqueeze_676: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_255, 0);  primals_255 = None
    unsqueeze_677: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, 2);  unsqueeze_676 = None
    unsqueeze_678: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 3);  unsqueeze_677 = None
    sum_44: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    sub_74: "f32[4, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_678);  convolution_31 = unsqueeze_678 = None
    mul_327: "f32[4, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_20, sub_74);  sub_74 = None
    sum_45: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_327, [0, 2, 3]);  mul_327 = None
    mul_332: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_21, primals_95);  primals_95 = None
    unsqueeze_685: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_332, 0);  mul_332 = None
    unsqueeze_686: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_685, 2);  unsqueeze_685 = None
    unsqueeze_687: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 3);  unsqueeze_686 = None
    mul_333: "f32[4, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_20, unsqueeze_687);  where_20 = unsqueeze_687 = None
    mul_334: "f32[512]" = torch.ops.aten.mul.Tensor(sum_45, rsqrt_21);  sum_45 = rsqrt_21 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_333, relu_27, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_333 = primals_94 = None
    getitem_65: "f32[4, 1024, 14, 14]" = convolution_backward_21[0]
    getitem_66: "f32[512, 1024, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    add_150: "f32[4, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_18, getitem_65);  where_18 = getitem_65 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    le_21: "b8[4, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_27, 0);  relu_27 = None
    where_21: "f32[4, 1024, 14, 14]" = torch.ops.aten.where.self(le_21, full_default, add_150);  le_21 = add_150 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    add_151: "f32[1024]" = torch.ops.aten.add.Tensor(primals_253, 1e-05);  primals_253 = None
    rsqrt_22: "f32[1024]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
    unsqueeze_688: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(primals_252, 0);  primals_252 = None
    unsqueeze_689: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, 2);  unsqueeze_688 = None
    unsqueeze_690: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 3);  unsqueeze_689 = None
    sum_46: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_75: "f32[4, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_690);  convolution_30 = unsqueeze_690 = None
    mul_335: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_21, sub_75);  sub_75 = None
    sum_47: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_335, [0, 2, 3]);  mul_335 = None
    mul_340: "f32[1024]" = torch.ops.aten.mul.Tensor(rsqrt_22, primals_92);  primals_92 = None
    unsqueeze_697: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_340, 0);  mul_340 = None
    unsqueeze_698: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_697, 2);  unsqueeze_697 = None
    unsqueeze_699: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 3);  unsqueeze_698 = None
    mul_341: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_21, unsqueeze_699);  unsqueeze_699 = None
    mul_342: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_47, rsqrt_22);  sum_47 = rsqrt_22 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_341, relu_26, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_341 = primals_91 = None
    getitem_68: "f32[4, 512, 14, 14]" = convolution_backward_22[0]
    getitem_69: "f32[1024, 512, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    le_22: "b8[4, 512, 14, 14]" = torch.ops.aten.le.Scalar(relu_26, 0);  relu_26 = None
    where_22: "f32[4, 512, 14, 14]" = torch.ops.aten.where.self(le_22, full_default, getitem_68);  le_22 = getitem_68 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    add_152: "f32[512]" = torch.ops.aten.add.Tensor(primals_250, 1e-05);  primals_250 = None
    rsqrt_23: "f32[512]" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
    unsqueeze_700: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_249, 0);  primals_249 = None
    unsqueeze_701: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, 2);  unsqueeze_700 = None
    unsqueeze_702: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 3);  unsqueeze_701 = None
    sum_48: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    sub_76: "f32[4, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_702);  convolution_29 = unsqueeze_702 = None
    mul_343: "f32[4, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_22, sub_76);  sub_76 = None
    sum_49: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_343, [0, 2, 3]);  mul_343 = None
    mul_348: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_23, primals_89);  primals_89 = None
    unsqueeze_709: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_348, 0);  mul_348 = None
    unsqueeze_710: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_709, 2);  unsqueeze_709 = None
    unsqueeze_711: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 3);  unsqueeze_710 = None
    mul_349: "f32[4, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_22, unsqueeze_711);  where_22 = unsqueeze_711 = None
    mul_350: "f32[512]" = torch.ops.aten.mul.Tensor(sum_49, rsqrt_23);  sum_49 = rsqrt_23 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_349, relu_25, primals_88, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_349 = primals_88 = None
    getitem_71: "f32[4, 512, 14, 14]" = convolution_backward_23[0]
    getitem_72: "f32[512, 16, 3, 3]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    le_23: "b8[4, 512, 14, 14]" = torch.ops.aten.le.Scalar(relu_25, 0);  relu_25 = None
    where_23: "f32[4, 512, 14, 14]" = torch.ops.aten.where.self(le_23, full_default, getitem_71);  le_23 = getitem_71 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    add_153: "f32[512]" = torch.ops.aten.add.Tensor(primals_247, 1e-05);  primals_247 = None
    rsqrt_24: "f32[512]" = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
    unsqueeze_712: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_246, 0);  primals_246 = None
    unsqueeze_713: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, 2);  unsqueeze_712 = None
    unsqueeze_714: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 3);  unsqueeze_713 = None
    sum_50: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_77: "f32[4, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_714);  convolution_28 = unsqueeze_714 = None
    mul_351: "f32[4, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_23, sub_77);  sub_77 = None
    sum_51: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_351, [0, 2, 3]);  mul_351 = None
    mul_356: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_24, primals_86);  primals_86 = None
    unsqueeze_721: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_356, 0);  mul_356 = None
    unsqueeze_722: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_721, 2);  unsqueeze_721 = None
    unsqueeze_723: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 3);  unsqueeze_722 = None
    mul_357: "f32[4, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_23, unsqueeze_723);  where_23 = unsqueeze_723 = None
    mul_358: "f32[512]" = torch.ops.aten.mul.Tensor(sum_51, rsqrt_24);  sum_51 = rsqrt_24 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_357, relu_24, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_357 = primals_85 = None
    getitem_74: "f32[4, 1024, 14, 14]" = convolution_backward_24[0]
    getitem_75: "f32[512, 1024, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    add_154: "f32[4, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_21, getitem_74);  where_21 = getitem_74 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    le_24: "b8[4, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_24, 0);  relu_24 = None
    where_24: "f32[4, 1024, 14, 14]" = torch.ops.aten.where.self(le_24, full_default, add_154);  le_24 = add_154 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:158, code: identity = self.downsample(x)
    add_155: "f32[1024]" = torch.ops.aten.add.Tensor(primals_244, 1e-05);  primals_244 = None
    rsqrt_25: "f32[1024]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
    unsqueeze_724: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(primals_243, 0);  primals_243 = None
    unsqueeze_725: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, 2);  unsqueeze_724 = None
    unsqueeze_726: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 3);  unsqueeze_725 = None
    sum_52: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_78: "f32[4, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_726);  convolution_27 = unsqueeze_726 = None
    mul_359: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_24, sub_78);  sub_78 = None
    sum_53: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_359, [0, 2, 3]);  mul_359 = None
    mul_364: "f32[1024]" = torch.ops.aten.mul.Tensor(rsqrt_25, primals_83);  primals_83 = None
    unsqueeze_733: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_364, 0);  mul_364 = None
    unsqueeze_734: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_733, 2);  unsqueeze_733 = None
    unsqueeze_735: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 3);  unsqueeze_734 = None
    mul_365: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_24, unsqueeze_735);  unsqueeze_735 = None
    mul_366: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_53, rsqrt_25);  sum_53 = rsqrt_25 = None
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_365, relu_21, primals_82, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_365 = primals_82 = None
    getitem_77: "f32[4, 512, 28, 28]" = convolution_backward_25[0]
    getitem_78: "f32[1024, 512, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    add_156: "f32[1024]" = torch.ops.aten.add.Tensor(primals_241, 1e-05);  primals_241 = None
    rsqrt_26: "f32[1024]" = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
    unsqueeze_736: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(primals_240, 0);  primals_240 = None
    unsqueeze_737: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, 2);  unsqueeze_736 = None
    unsqueeze_738: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 3);  unsqueeze_737 = None
    sub_79: "f32[4, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_738);  convolution_26 = unsqueeze_738 = None
    mul_367: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_24, sub_79);  sub_79 = None
    sum_55: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_367, [0, 2, 3]);  mul_367 = None
    mul_372: "f32[1024]" = torch.ops.aten.mul.Tensor(rsqrt_26, primals_80);  primals_80 = None
    unsqueeze_745: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_372, 0);  mul_372 = None
    unsqueeze_746: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_745, 2);  unsqueeze_745 = None
    unsqueeze_747: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 3);  unsqueeze_746 = None
    mul_373: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_24, unsqueeze_747);  where_24 = unsqueeze_747 = None
    mul_374: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_55, rsqrt_26);  sum_55 = rsqrt_26 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_373, relu_23, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_373 = primals_79 = None
    getitem_80: "f32[4, 512, 14, 14]" = convolution_backward_26[0]
    getitem_81: "f32[1024, 512, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    le_25: "b8[4, 512, 14, 14]" = torch.ops.aten.le.Scalar(relu_23, 0);  relu_23 = None
    where_25: "f32[4, 512, 14, 14]" = torch.ops.aten.where.self(le_25, full_default, getitem_80);  le_25 = getitem_80 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    add_157: "f32[512]" = torch.ops.aten.add.Tensor(primals_238, 1e-05);  primals_238 = None
    rsqrt_27: "f32[512]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
    unsqueeze_748: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_237, 0);  primals_237 = None
    unsqueeze_749: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, 2);  unsqueeze_748 = None
    unsqueeze_750: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 3);  unsqueeze_749 = None
    sum_56: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_80: "f32[4, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_750);  convolution_25 = unsqueeze_750 = None
    mul_375: "f32[4, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_25, sub_80);  sub_80 = None
    sum_57: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_375, [0, 2, 3]);  mul_375 = None
    mul_380: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_27, primals_77);  primals_77 = None
    unsqueeze_757: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_380, 0);  mul_380 = None
    unsqueeze_758: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_757, 2);  unsqueeze_757 = None
    unsqueeze_759: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 3);  unsqueeze_758 = None
    mul_381: "f32[4, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_25, unsqueeze_759);  where_25 = unsqueeze_759 = None
    mul_382: "f32[512]" = torch.ops.aten.mul.Tensor(sum_57, rsqrt_27);  sum_57 = rsqrt_27 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_381, relu_22, primals_76, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_381 = primals_76 = None
    getitem_83: "f32[4, 512, 28, 28]" = convolution_backward_27[0]
    getitem_84: "f32[512, 16, 3, 3]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    le_26: "b8[4, 512, 28, 28]" = torch.ops.aten.le.Scalar(relu_22, 0);  relu_22 = None
    where_26: "f32[4, 512, 28, 28]" = torch.ops.aten.where.self(le_26, full_default, getitem_83);  le_26 = getitem_83 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    add_158: "f32[512]" = torch.ops.aten.add.Tensor(primals_235, 1e-05);  primals_235 = None
    rsqrt_28: "f32[512]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
    unsqueeze_760: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_234, 0);  primals_234 = None
    unsqueeze_761: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, 2);  unsqueeze_760 = None
    unsqueeze_762: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 3);  unsqueeze_761 = None
    sum_58: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    sub_81: "f32[4, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_762);  convolution_24 = unsqueeze_762 = None
    mul_383: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_26, sub_81);  sub_81 = None
    sum_59: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_383, [0, 2, 3]);  mul_383 = None
    mul_388: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_28, primals_74);  primals_74 = None
    unsqueeze_769: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_388, 0);  mul_388 = None
    unsqueeze_770: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_769, 2);  unsqueeze_769 = None
    unsqueeze_771: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 3);  unsqueeze_770 = None
    mul_389: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_26, unsqueeze_771);  where_26 = unsqueeze_771 = None
    mul_390: "f32[512]" = torch.ops.aten.mul.Tensor(sum_59, rsqrt_28);  sum_59 = rsqrt_28 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_389, relu_21, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_389 = primals_73 = None
    getitem_86: "f32[4, 512, 28, 28]" = convolution_backward_28[0]
    getitem_87: "f32[512, 512, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    add_159: "f32[4, 512, 28, 28]" = torch.ops.aten.add.Tensor(getitem_77, getitem_86);  getitem_77 = getitem_86 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    le_27: "b8[4, 512, 28, 28]" = torch.ops.aten.le.Scalar(relu_21, 0);  relu_21 = None
    where_27: "f32[4, 512, 28, 28]" = torch.ops.aten.where.self(le_27, full_default, add_159);  le_27 = add_159 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    add_160: "f32[512]" = torch.ops.aten.add.Tensor(primals_232, 1e-05);  primals_232 = None
    rsqrt_29: "f32[512]" = torch.ops.aten.rsqrt.default(add_160);  add_160 = None
    unsqueeze_772: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_231, 0);  primals_231 = None
    unsqueeze_773: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, 2);  unsqueeze_772 = None
    unsqueeze_774: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 3);  unsqueeze_773 = None
    sum_60: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_82: "f32[4, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_774);  convolution_23 = unsqueeze_774 = None
    mul_391: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_27, sub_82);  sub_82 = None
    sum_61: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_391, [0, 2, 3]);  mul_391 = None
    mul_396: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_29, primals_71);  primals_71 = None
    unsqueeze_781: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_396, 0);  mul_396 = None
    unsqueeze_782: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_781, 2);  unsqueeze_781 = None
    unsqueeze_783: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 3);  unsqueeze_782 = None
    mul_397: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_27, unsqueeze_783);  unsqueeze_783 = None
    mul_398: "f32[512]" = torch.ops.aten.mul.Tensor(sum_61, rsqrt_29);  sum_61 = rsqrt_29 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_397, relu_20, primals_70, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_397 = primals_70 = None
    getitem_89: "f32[4, 256, 28, 28]" = convolution_backward_29[0]
    getitem_90: "f32[512, 256, 1, 1]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    le_28: "b8[4, 256, 28, 28]" = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None
    where_28: "f32[4, 256, 28, 28]" = torch.ops.aten.where.self(le_28, full_default, getitem_89);  le_28 = getitem_89 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    add_161: "f32[256]" = torch.ops.aten.add.Tensor(primals_229, 1e-05);  primals_229 = None
    rsqrt_30: "f32[256]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    unsqueeze_784: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_228, 0);  primals_228 = None
    unsqueeze_785: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_784, 2);  unsqueeze_784 = None
    unsqueeze_786: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 3);  unsqueeze_785 = None
    sum_62: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
    sub_83: "f32[4, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_786);  convolution_22 = unsqueeze_786 = None
    mul_399: "f32[4, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_28, sub_83);  sub_83 = None
    sum_63: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_399, [0, 2, 3]);  mul_399 = None
    mul_404: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_30, primals_68);  primals_68 = None
    unsqueeze_793: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_404, 0);  mul_404 = None
    unsqueeze_794: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_793, 2);  unsqueeze_793 = None
    unsqueeze_795: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 3);  unsqueeze_794 = None
    mul_405: "f32[4, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_28, unsqueeze_795);  where_28 = unsqueeze_795 = None
    mul_406: "f32[256]" = torch.ops.aten.mul.Tensor(sum_63, rsqrt_30);  sum_63 = rsqrt_30 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_405, relu_19, primals_67, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_405 = primals_67 = None
    getitem_92: "f32[4, 256, 28, 28]" = convolution_backward_30[0]
    getitem_93: "f32[256, 8, 3, 3]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    le_29: "b8[4, 256, 28, 28]" = torch.ops.aten.le.Scalar(relu_19, 0);  relu_19 = None
    where_29: "f32[4, 256, 28, 28]" = torch.ops.aten.where.self(le_29, full_default, getitem_92);  le_29 = getitem_92 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    add_162: "f32[256]" = torch.ops.aten.add.Tensor(primals_226, 1e-05);  primals_226 = None
    rsqrt_31: "f32[256]" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
    unsqueeze_796: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_225, 0);  primals_225 = None
    unsqueeze_797: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_796, 2);  unsqueeze_796 = None
    unsqueeze_798: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_797, 3);  unsqueeze_797 = None
    sum_64: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    sub_84: "f32[4, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_798);  convolution_21 = unsqueeze_798 = None
    mul_407: "f32[4, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_29, sub_84);  sub_84 = None
    sum_65: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_407, [0, 2, 3]);  mul_407 = None
    mul_412: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_31, primals_65);  primals_65 = None
    unsqueeze_805: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_412, 0);  mul_412 = None
    unsqueeze_806: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_805, 2);  unsqueeze_805 = None
    unsqueeze_807: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 3);  unsqueeze_806 = None
    mul_413: "f32[4, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_29, unsqueeze_807);  where_29 = unsqueeze_807 = None
    mul_414: "f32[256]" = torch.ops.aten.mul.Tensor(sum_65, rsqrt_31);  sum_65 = rsqrt_31 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_413, relu_18, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_413 = primals_64 = None
    getitem_95: "f32[4, 512, 28, 28]" = convolution_backward_31[0]
    getitem_96: "f32[256, 512, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    add_163: "f32[4, 512, 28, 28]" = torch.ops.aten.add.Tensor(where_27, getitem_95);  where_27 = getitem_95 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    le_30: "b8[4, 512, 28, 28]" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
    where_30: "f32[4, 512, 28, 28]" = torch.ops.aten.where.self(le_30, full_default, add_163);  le_30 = add_163 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    add_164: "f32[512]" = torch.ops.aten.add.Tensor(primals_223, 1e-05);  primals_223 = None
    rsqrt_32: "f32[512]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
    unsqueeze_808: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_222, 0);  primals_222 = None
    unsqueeze_809: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_808, 2);  unsqueeze_808 = None
    unsqueeze_810: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 3);  unsqueeze_809 = None
    sum_66: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
    sub_85: "f32[4, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_810);  convolution_20 = unsqueeze_810 = None
    mul_415: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_30, sub_85);  sub_85 = None
    sum_67: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_415, [0, 2, 3]);  mul_415 = None
    mul_420: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_32, primals_62);  primals_62 = None
    unsqueeze_817: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_420, 0);  mul_420 = None
    unsqueeze_818: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_817, 2);  unsqueeze_817 = None
    unsqueeze_819: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 3);  unsqueeze_818 = None
    mul_421: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_30, unsqueeze_819);  unsqueeze_819 = None
    mul_422: "f32[512]" = torch.ops.aten.mul.Tensor(sum_67, rsqrt_32);  sum_67 = rsqrt_32 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_421, relu_17, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_421 = primals_61 = None
    getitem_98: "f32[4, 256, 28, 28]" = convolution_backward_32[0]
    getitem_99: "f32[512, 256, 1, 1]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    le_31: "b8[4, 256, 28, 28]" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
    where_31: "f32[4, 256, 28, 28]" = torch.ops.aten.where.self(le_31, full_default, getitem_98);  le_31 = getitem_98 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    add_165: "f32[256]" = torch.ops.aten.add.Tensor(primals_220, 1e-05);  primals_220 = None
    rsqrt_33: "f32[256]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
    unsqueeze_820: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_219, 0);  primals_219 = None
    unsqueeze_821: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_820, 2);  unsqueeze_820 = None
    unsqueeze_822: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 3);  unsqueeze_821 = None
    sum_68: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_86: "f32[4, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_822);  convolution_19 = unsqueeze_822 = None
    mul_423: "f32[4, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_31, sub_86);  sub_86 = None
    sum_69: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_423, [0, 2, 3]);  mul_423 = None
    mul_428: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_33, primals_59);  primals_59 = None
    unsqueeze_829: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_428, 0);  mul_428 = None
    unsqueeze_830: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_829, 2);  unsqueeze_829 = None
    unsqueeze_831: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, 3);  unsqueeze_830 = None
    mul_429: "f32[4, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_31, unsqueeze_831);  where_31 = unsqueeze_831 = None
    mul_430: "f32[256]" = torch.ops.aten.mul.Tensor(sum_69, rsqrt_33);  sum_69 = rsqrt_33 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_429, relu_16, primals_58, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_429 = primals_58 = None
    getitem_101: "f32[4, 256, 28, 28]" = convolution_backward_33[0]
    getitem_102: "f32[256, 8, 3, 3]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    le_32: "b8[4, 256, 28, 28]" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
    where_32: "f32[4, 256, 28, 28]" = torch.ops.aten.where.self(le_32, full_default, getitem_101);  le_32 = getitem_101 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    add_166: "f32[256]" = torch.ops.aten.add.Tensor(primals_217, 1e-05);  primals_217 = None
    rsqrt_34: "f32[256]" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
    unsqueeze_832: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_216, 0);  primals_216 = None
    unsqueeze_833: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_832, 2);  unsqueeze_832 = None
    unsqueeze_834: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_833, 3);  unsqueeze_833 = None
    sum_70: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
    sub_87: "f32[4, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_834);  convolution_18 = unsqueeze_834 = None
    mul_431: "f32[4, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_32, sub_87);  sub_87 = None
    sum_71: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_431, [0, 2, 3]);  mul_431 = None
    mul_436: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_34, primals_56);  primals_56 = None
    unsqueeze_841: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_436, 0);  mul_436 = None
    unsqueeze_842: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_841, 2);  unsqueeze_841 = None
    unsqueeze_843: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_842, 3);  unsqueeze_842 = None
    mul_437: "f32[4, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_32, unsqueeze_843);  where_32 = unsqueeze_843 = None
    mul_438: "f32[256]" = torch.ops.aten.mul.Tensor(sum_71, rsqrt_34);  sum_71 = rsqrt_34 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_437, relu_15, primals_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_437 = primals_55 = None
    getitem_104: "f32[4, 512, 28, 28]" = convolution_backward_34[0]
    getitem_105: "f32[256, 512, 1, 1]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    add_167: "f32[4, 512, 28, 28]" = torch.ops.aten.add.Tensor(where_30, getitem_104);  where_30 = getitem_104 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    le_33: "b8[4, 512, 28, 28]" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
    where_33: "f32[4, 512, 28, 28]" = torch.ops.aten.where.self(le_33, full_default, add_167);  le_33 = add_167 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    add_168: "f32[512]" = torch.ops.aten.add.Tensor(primals_214, 1e-05);  primals_214 = None
    rsqrt_35: "f32[512]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
    unsqueeze_844: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_213, 0);  primals_213 = None
    unsqueeze_845: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_844, 2);  unsqueeze_844 = None
    unsqueeze_846: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_845, 3);  unsqueeze_845 = None
    sum_72: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
    sub_88: "f32[4, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_846);  convolution_17 = unsqueeze_846 = None
    mul_439: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_33, sub_88);  sub_88 = None
    sum_73: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_439, [0, 2, 3]);  mul_439 = None
    mul_444: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_35, primals_53);  primals_53 = None
    unsqueeze_853: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_444, 0);  mul_444 = None
    unsqueeze_854: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_853, 2);  unsqueeze_853 = None
    unsqueeze_855: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, 3);  unsqueeze_854 = None
    mul_445: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_33, unsqueeze_855);  unsqueeze_855 = None
    mul_446: "f32[512]" = torch.ops.aten.mul.Tensor(sum_73, rsqrt_35);  sum_73 = rsqrt_35 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_445, relu_14, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_445 = primals_52 = None
    getitem_107: "f32[4, 256, 28, 28]" = convolution_backward_35[0]
    getitem_108: "f32[512, 256, 1, 1]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    le_34: "b8[4, 256, 28, 28]" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
    where_34: "f32[4, 256, 28, 28]" = torch.ops.aten.where.self(le_34, full_default, getitem_107);  le_34 = getitem_107 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    add_169: "f32[256]" = torch.ops.aten.add.Tensor(primals_211, 1e-05);  primals_211 = None
    rsqrt_36: "f32[256]" = torch.ops.aten.rsqrt.default(add_169);  add_169 = None
    unsqueeze_856: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_210, 0);  primals_210 = None
    unsqueeze_857: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_856, 2);  unsqueeze_856 = None
    unsqueeze_858: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_857, 3);  unsqueeze_857 = None
    sum_74: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
    sub_89: "f32[4, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_858);  convolution_16 = unsqueeze_858 = None
    mul_447: "f32[4, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_34, sub_89);  sub_89 = None
    sum_75: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_447, [0, 2, 3]);  mul_447 = None
    mul_452: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_36, primals_50);  primals_50 = None
    unsqueeze_865: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_452, 0);  mul_452 = None
    unsqueeze_866: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_865, 2);  unsqueeze_865 = None
    unsqueeze_867: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, 3);  unsqueeze_866 = None
    mul_453: "f32[4, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_34, unsqueeze_867);  where_34 = unsqueeze_867 = None
    mul_454: "f32[256]" = torch.ops.aten.mul.Tensor(sum_75, rsqrt_36);  sum_75 = rsqrt_36 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_453, relu_13, primals_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_453 = primals_49 = None
    getitem_110: "f32[4, 256, 28, 28]" = convolution_backward_36[0]
    getitem_111: "f32[256, 8, 3, 3]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    le_35: "b8[4, 256, 28, 28]" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
    where_35: "f32[4, 256, 28, 28]" = torch.ops.aten.where.self(le_35, full_default, getitem_110);  le_35 = getitem_110 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    add_170: "f32[256]" = torch.ops.aten.add.Tensor(primals_208, 1e-05);  primals_208 = None
    rsqrt_37: "f32[256]" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
    unsqueeze_868: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_207, 0);  primals_207 = None
    unsqueeze_869: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_868, 2);  unsqueeze_868 = None
    unsqueeze_870: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_869, 3);  unsqueeze_869 = None
    sum_76: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
    sub_90: "f32[4, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_870);  convolution_15 = unsqueeze_870 = None
    mul_455: "f32[4, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_35, sub_90);  sub_90 = None
    sum_77: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_455, [0, 2, 3]);  mul_455 = None
    mul_460: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_37, primals_47);  primals_47 = None
    unsqueeze_877: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_460, 0);  mul_460 = None
    unsqueeze_878: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_877, 2);  unsqueeze_877 = None
    unsqueeze_879: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, 3);  unsqueeze_878 = None
    mul_461: "f32[4, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_35, unsqueeze_879);  where_35 = unsqueeze_879 = None
    mul_462: "f32[256]" = torch.ops.aten.mul.Tensor(sum_77, rsqrt_37);  sum_77 = rsqrt_37 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_461, relu_12, primals_46, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_461 = primals_46 = None
    getitem_113: "f32[4, 512, 28, 28]" = convolution_backward_37[0]
    getitem_114: "f32[256, 512, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    add_171: "f32[4, 512, 28, 28]" = torch.ops.aten.add.Tensor(where_33, getitem_113);  where_33 = getitem_113 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    le_36: "b8[4, 512, 28, 28]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
    where_36: "f32[4, 512, 28, 28]" = torch.ops.aten.where.self(le_36, full_default, add_171);  le_36 = add_171 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:158, code: identity = self.downsample(x)
    add_172: "f32[512]" = torch.ops.aten.add.Tensor(primals_205, 1e-05);  primals_205 = None
    rsqrt_38: "f32[512]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
    unsqueeze_880: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_204, 0);  primals_204 = None
    unsqueeze_881: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_880, 2);  unsqueeze_880 = None
    unsqueeze_882: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_881, 3);  unsqueeze_881 = None
    sum_78: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
    sub_91: "f32[4, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_882);  convolution_14 = unsqueeze_882 = None
    mul_463: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_36, sub_91);  sub_91 = None
    sum_79: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_463, [0, 2, 3]);  mul_463 = None
    mul_468: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_38, primals_44);  primals_44 = None
    unsqueeze_889: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_468, 0);  mul_468 = None
    unsqueeze_890: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_889, 2);  unsqueeze_889 = None
    unsqueeze_891: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, 3);  unsqueeze_890 = None
    mul_469: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_36, unsqueeze_891);  unsqueeze_891 = None
    mul_470: "f32[512]" = torch.ops.aten.mul.Tensor(sum_79, rsqrt_38);  sum_79 = rsqrt_38 = None
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_469, relu_9, primals_43, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_469 = primals_43 = None
    getitem_116: "f32[4, 256, 56, 56]" = convolution_backward_38[0]
    getitem_117: "f32[512, 256, 1, 1]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    add_173: "f32[512]" = torch.ops.aten.add.Tensor(primals_202, 1e-05);  primals_202 = None
    rsqrt_39: "f32[512]" = torch.ops.aten.rsqrt.default(add_173);  add_173 = None
    unsqueeze_892: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_201, 0);  primals_201 = None
    unsqueeze_893: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_892, 2);  unsqueeze_892 = None
    unsqueeze_894: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_893, 3);  unsqueeze_893 = None
    sub_92: "f32[4, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_894);  convolution_13 = unsqueeze_894 = None
    mul_471: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_36, sub_92);  sub_92 = None
    sum_81: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_471, [0, 2, 3]);  mul_471 = None
    mul_476: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_39, primals_41);  primals_41 = None
    unsqueeze_901: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_476, 0);  mul_476 = None
    unsqueeze_902: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_901, 2);  unsqueeze_901 = None
    unsqueeze_903: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_902, 3);  unsqueeze_902 = None
    mul_477: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_36, unsqueeze_903);  where_36 = unsqueeze_903 = None
    mul_478: "f32[512]" = torch.ops.aten.mul.Tensor(sum_81, rsqrt_39);  sum_81 = rsqrt_39 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_477, relu_11, primals_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_477 = primals_40 = None
    getitem_119: "f32[4, 256, 28, 28]" = convolution_backward_39[0]
    getitem_120: "f32[512, 256, 1, 1]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    le_37: "b8[4, 256, 28, 28]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
    where_37: "f32[4, 256, 28, 28]" = torch.ops.aten.where.self(le_37, full_default, getitem_119);  le_37 = getitem_119 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    add_174: "f32[256]" = torch.ops.aten.add.Tensor(primals_199, 1e-05);  primals_199 = None
    rsqrt_40: "f32[256]" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
    unsqueeze_904: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_198, 0);  primals_198 = None
    unsqueeze_905: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_904, 2);  unsqueeze_904 = None
    unsqueeze_906: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_905, 3);  unsqueeze_905 = None
    sum_82: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
    sub_93: "f32[4, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_906);  convolution_12 = unsqueeze_906 = None
    mul_479: "f32[4, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_37, sub_93);  sub_93 = None
    sum_83: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_479, [0, 2, 3]);  mul_479 = None
    mul_484: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_40, primals_38);  primals_38 = None
    unsqueeze_913: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_484, 0);  mul_484 = None
    unsqueeze_914: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_913, 2);  unsqueeze_913 = None
    unsqueeze_915: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_914, 3);  unsqueeze_914 = None
    mul_485: "f32[4, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_37, unsqueeze_915);  where_37 = unsqueeze_915 = None
    mul_486: "f32[256]" = torch.ops.aten.mul.Tensor(sum_83, rsqrt_40);  sum_83 = rsqrt_40 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_485, relu_10, primals_37, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_485 = primals_37 = None
    getitem_122: "f32[4, 256, 56, 56]" = convolution_backward_40[0]
    getitem_123: "f32[256, 8, 3, 3]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    le_38: "b8[4, 256, 56, 56]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
    where_38: "f32[4, 256, 56, 56]" = torch.ops.aten.where.self(le_38, full_default, getitem_122);  le_38 = getitem_122 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    add_175: "f32[256]" = torch.ops.aten.add.Tensor(primals_196, 1e-05);  primals_196 = None
    rsqrt_41: "f32[256]" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
    unsqueeze_916: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_195, 0);  primals_195 = None
    unsqueeze_917: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_916, 2);  unsqueeze_916 = None
    unsqueeze_918: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_917, 3);  unsqueeze_917 = None
    sum_84: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3])
    sub_94: "f32[4, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_918);  convolution_11 = unsqueeze_918 = None
    mul_487: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_38, sub_94);  sub_94 = None
    sum_85: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_487, [0, 2, 3]);  mul_487 = None
    mul_492: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_41, primals_35);  primals_35 = None
    unsqueeze_925: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_492, 0);  mul_492 = None
    unsqueeze_926: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_925, 2);  unsqueeze_925 = None
    unsqueeze_927: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_926, 3);  unsqueeze_926 = None
    mul_493: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_38, unsqueeze_927);  where_38 = unsqueeze_927 = None
    mul_494: "f32[256]" = torch.ops.aten.mul.Tensor(sum_85, rsqrt_41);  sum_85 = rsqrt_41 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_493, relu_9, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_493 = primals_34 = None
    getitem_125: "f32[4, 256, 56, 56]" = convolution_backward_41[0]
    getitem_126: "f32[256, 256, 1, 1]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    add_176: "f32[4, 256, 56, 56]" = torch.ops.aten.add.Tensor(getitem_116, getitem_125);  getitem_116 = getitem_125 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    le_39: "b8[4, 256, 56, 56]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
    where_39: "f32[4, 256, 56, 56]" = torch.ops.aten.where.self(le_39, full_default, add_176);  le_39 = add_176 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    add_177: "f32[256]" = torch.ops.aten.add.Tensor(primals_193, 1e-05);  primals_193 = None
    rsqrt_42: "f32[256]" = torch.ops.aten.rsqrt.default(add_177);  add_177 = None
    unsqueeze_928: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_192, 0);  primals_192 = None
    unsqueeze_929: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_928, 2);  unsqueeze_928 = None
    unsqueeze_930: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_929, 3);  unsqueeze_929 = None
    sum_86: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
    sub_95: "f32[4, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_930);  convolution_10 = unsqueeze_930 = None
    mul_495: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_39, sub_95);  sub_95 = None
    sum_87: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_495, [0, 2, 3]);  mul_495 = None
    mul_500: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_42, primals_32);  primals_32 = None
    unsqueeze_937: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_500, 0);  mul_500 = None
    unsqueeze_938: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_937, 2);  unsqueeze_937 = None
    unsqueeze_939: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_938, 3);  unsqueeze_938 = None
    mul_501: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_39, unsqueeze_939);  unsqueeze_939 = None
    mul_502: "f32[256]" = torch.ops.aten.mul.Tensor(sum_87, rsqrt_42);  sum_87 = rsqrt_42 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_501, relu_8, primals_31, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_501 = primals_31 = None
    getitem_128: "f32[4, 128, 56, 56]" = convolution_backward_42[0]
    getitem_129: "f32[256, 128, 1, 1]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    le_40: "b8[4, 128, 56, 56]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
    where_40: "f32[4, 128, 56, 56]" = torch.ops.aten.where.self(le_40, full_default, getitem_128);  le_40 = getitem_128 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    add_178: "f32[128]" = torch.ops.aten.add.Tensor(primals_190, 1e-05);  primals_190 = None
    rsqrt_43: "f32[128]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
    unsqueeze_940: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_189, 0);  primals_189 = None
    unsqueeze_941: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_940, 2);  unsqueeze_940 = None
    unsqueeze_942: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_941, 3);  unsqueeze_941 = None
    sum_88: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_40, [0, 2, 3])
    sub_96: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_942);  convolution_9 = unsqueeze_942 = None
    mul_503: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_40, sub_96);  sub_96 = None
    sum_89: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_503, [0, 2, 3]);  mul_503 = None
    mul_508: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_43, primals_29);  primals_29 = None
    unsqueeze_949: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_508, 0);  mul_508 = None
    unsqueeze_950: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_949, 2);  unsqueeze_949 = None
    unsqueeze_951: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_950, 3);  unsqueeze_950 = None
    mul_509: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_40, unsqueeze_951);  where_40 = unsqueeze_951 = None
    mul_510: "f32[128]" = torch.ops.aten.mul.Tensor(sum_89, rsqrt_43);  sum_89 = rsqrt_43 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_509, relu_7, primals_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_509 = primals_28 = None
    getitem_131: "f32[4, 128, 56, 56]" = convolution_backward_43[0]
    getitem_132: "f32[128, 4, 3, 3]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    le_41: "b8[4, 128, 56, 56]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
    where_41: "f32[4, 128, 56, 56]" = torch.ops.aten.where.self(le_41, full_default, getitem_131);  le_41 = getitem_131 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    add_179: "f32[128]" = torch.ops.aten.add.Tensor(primals_187, 1e-05);  primals_187 = None
    rsqrt_44: "f32[128]" = torch.ops.aten.rsqrt.default(add_179);  add_179 = None
    unsqueeze_952: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_186, 0);  primals_186 = None
    unsqueeze_953: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_952, 2);  unsqueeze_952 = None
    unsqueeze_954: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_953, 3);  unsqueeze_953 = None
    sum_90: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_41, [0, 2, 3])
    sub_97: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_954);  convolution_8 = unsqueeze_954 = None
    mul_511: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_41, sub_97);  sub_97 = None
    sum_91: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_511, [0, 2, 3]);  mul_511 = None
    mul_516: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_44, primals_26);  primals_26 = None
    unsqueeze_961: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_516, 0);  mul_516 = None
    unsqueeze_962: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_961, 2);  unsqueeze_961 = None
    unsqueeze_963: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_962, 3);  unsqueeze_962 = None
    mul_517: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_41, unsqueeze_963);  where_41 = unsqueeze_963 = None
    mul_518: "f32[128]" = torch.ops.aten.mul.Tensor(sum_91, rsqrt_44);  sum_91 = rsqrt_44 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_517, relu_6, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_517 = primals_25 = None
    getitem_134: "f32[4, 256, 56, 56]" = convolution_backward_44[0]
    getitem_135: "f32[128, 256, 1, 1]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    add_180: "f32[4, 256, 56, 56]" = torch.ops.aten.add.Tensor(where_39, getitem_134);  where_39 = getitem_134 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    le_42: "b8[4, 256, 56, 56]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
    where_42: "f32[4, 256, 56, 56]" = torch.ops.aten.where.self(le_42, full_default, add_180);  le_42 = add_180 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    add_181: "f32[256]" = torch.ops.aten.add.Tensor(primals_184, 1e-05);  primals_184 = None
    rsqrt_45: "f32[256]" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
    unsqueeze_964: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_183, 0);  primals_183 = None
    unsqueeze_965: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_964, 2);  unsqueeze_964 = None
    unsqueeze_966: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_965, 3);  unsqueeze_965 = None
    sum_92: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_42, [0, 2, 3])
    sub_98: "f32[4, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_966);  convolution_7 = unsqueeze_966 = None
    mul_519: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_42, sub_98);  sub_98 = None
    sum_93: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_519, [0, 2, 3]);  mul_519 = None
    mul_524: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_45, primals_23);  primals_23 = None
    unsqueeze_973: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_524, 0);  mul_524 = None
    unsqueeze_974: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_973, 2);  unsqueeze_973 = None
    unsqueeze_975: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_974, 3);  unsqueeze_974 = None
    mul_525: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_42, unsqueeze_975);  unsqueeze_975 = None
    mul_526: "f32[256]" = torch.ops.aten.mul.Tensor(sum_93, rsqrt_45);  sum_93 = rsqrt_45 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_525, relu_5, primals_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_525 = primals_22 = None
    getitem_137: "f32[4, 128, 56, 56]" = convolution_backward_45[0]
    getitem_138: "f32[256, 128, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    le_43: "b8[4, 128, 56, 56]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
    where_43: "f32[4, 128, 56, 56]" = torch.ops.aten.where.self(le_43, full_default, getitem_137);  le_43 = getitem_137 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    add_182: "f32[128]" = torch.ops.aten.add.Tensor(primals_181, 1e-05);  primals_181 = None
    rsqrt_46: "f32[128]" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
    unsqueeze_976: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_180, 0);  primals_180 = None
    unsqueeze_977: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_976, 2);  unsqueeze_976 = None
    unsqueeze_978: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_977, 3);  unsqueeze_977 = None
    sum_94: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
    sub_99: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_978);  convolution_6 = unsqueeze_978 = None
    mul_527: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_43, sub_99);  sub_99 = None
    sum_95: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_527, [0, 2, 3]);  mul_527 = None
    mul_532: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_46, primals_20);  primals_20 = None
    unsqueeze_985: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_532, 0);  mul_532 = None
    unsqueeze_986: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_985, 2);  unsqueeze_985 = None
    unsqueeze_987: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_986, 3);  unsqueeze_986 = None
    mul_533: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_43, unsqueeze_987);  where_43 = unsqueeze_987 = None
    mul_534: "f32[128]" = torch.ops.aten.mul.Tensor(sum_95, rsqrt_46);  sum_95 = rsqrt_46 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_533, relu_4, primals_19, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_533 = primals_19 = None
    getitem_140: "f32[4, 128, 56, 56]" = convolution_backward_46[0]
    getitem_141: "f32[128, 4, 3, 3]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    le_44: "b8[4, 128, 56, 56]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
    where_44: "f32[4, 128, 56, 56]" = torch.ops.aten.where.self(le_44, full_default, getitem_140);  le_44 = getitem_140 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    add_183: "f32[128]" = torch.ops.aten.add.Tensor(primals_178, 1e-05);  primals_178 = None
    rsqrt_47: "f32[128]" = torch.ops.aten.rsqrt.default(add_183);  add_183 = None
    unsqueeze_988: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_177, 0);  primals_177 = None
    unsqueeze_989: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_988, 2);  unsqueeze_988 = None
    unsqueeze_990: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_989, 3);  unsqueeze_989 = None
    sum_96: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_44, [0, 2, 3])
    sub_100: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_990);  convolution_5 = unsqueeze_990 = None
    mul_535: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_44, sub_100);  sub_100 = None
    sum_97: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_535, [0, 2, 3]);  mul_535 = None
    mul_540: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_47, primals_17);  primals_17 = None
    unsqueeze_997: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_540, 0);  mul_540 = None
    unsqueeze_998: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_997, 2);  unsqueeze_997 = None
    unsqueeze_999: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_998, 3);  unsqueeze_998 = None
    mul_541: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_44, unsqueeze_999);  where_44 = unsqueeze_999 = None
    mul_542: "f32[128]" = torch.ops.aten.mul.Tensor(sum_97, rsqrt_47);  sum_97 = rsqrt_47 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_541, relu_3, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_541 = primals_16 = None
    getitem_143: "f32[4, 256, 56, 56]" = convolution_backward_47[0]
    getitem_144: "f32[128, 256, 1, 1]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    add_184: "f32[4, 256, 56, 56]" = torch.ops.aten.add.Tensor(where_42, getitem_143);  where_42 = getitem_143 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    le_45: "b8[4, 256, 56, 56]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
    where_45: "f32[4, 256, 56, 56]" = torch.ops.aten.where.self(le_45, full_default, add_184);  le_45 = add_184 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:158, code: identity = self.downsample(x)
    add_185: "f32[256]" = torch.ops.aten.add.Tensor(primals_175, 1e-05);  primals_175 = None
    rsqrt_48: "f32[256]" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
    unsqueeze_1000: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_174, 0);  primals_174 = None
    unsqueeze_1001: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1000, 2);  unsqueeze_1000 = None
    unsqueeze_1002: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1001, 3);  unsqueeze_1001 = None
    sum_98: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_45, [0, 2, 3])
    sub_101: "f32[4, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_1002);  convolution_4 = unsqueeze_1002 = None
    mul_543: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_45, sub_101);  sub_101 = None
    sum_99: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_543, [0, 2, 3]);  mul_543 = None
    mul_548: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_48, primals_14);  primals_14 = None
    unsqueeze_1009: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_548, 0);  mul_548 = None
    unsqueeze_1010: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1009, 2);  unsqueeze_1009 = None
    unsqueeze_1011: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1010, 3);  unsqueeze_1010 = None
    mul_549: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_45, unsqueeze_1011);  unsqueeze_1011 = None
    mul_550: "f32[256]" = torch.ops.aten.mul.Tensor(sum_99, rsqrt_48);  sum_99 = rsqrt_48 = None
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_549, getitem, primals_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_549 = primals_13 = None
    getitem_146: "f32[4, 64, 56, 56]" = convolution_backward_48[0]
    getitem_147: "f32[256, 64, 1, 1]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    add_186: "f32[256]" = torch.ops.aten.add.Tensor(primals_172, 1e-05);  primals_172 = None
    rsqrt_49: "f32[256]" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
    unsqueeze_1012: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_171, 0);  primals_171 = None
    unsqueeze_1013: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1012, 2);  unsqueeze_1012 = None
    unsqueeze_1014: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1013, 3);  unsqueeze_1013 = None
    sub_102: "f32[4, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_1014);  convolution_3 = unsqueeze_1014 = None
    mul_551: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_45, sub_102);  sub_102 = None
    sum_101: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_551, [0, 2, 3]);  mul_551 = None
    mul_556: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_49, primals_11);  primals_11 = None
    unsqueeze_1021: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_556, 0);  mul_556 = None
    unsqueeze_1022: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1021, 2);  unsqueeze_1021 = None
    unsqueeze_1023: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1022, 3);  unsqueeze_1022 = None
    mul_557: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_45, unsqueeze_1023);  where_45 = unsqueeze_1023 = None
    mul_558: "f32[256]" = torch.ops.aten.mul.Tensor(sum_101, rsqrt_49);  sum_101 = rsqrt_49 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_557, relu_2, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_557 = primals_10 = None
    getitem_149: "f32[4, 128, 56, 56]" = convolution_backward_49[0]
    getitem_150: "f32[256, 128, 1, 1]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    le_46: "b8[4, 128, 56, 56]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
    where_46: "f32[4, 128, 56, 56]" = torch.ops.aten.where.self(le_46, full_default, getitem_149);  le_46 = getitem_149 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    add_187: "f32[128]" = torch.ops.aten.add.Tensor(primals_169, 1e-05);  primals_169 = None
    rsqrt_50: "f32[128]" = torch.ops.aten.rsqrt.default(add_187);  add_187 = None
    unsqueeze_1024: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_168, 0);  primals_168 = None
    unsqueeze_1025: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1024, 2);  unsqueeze_1024 = None
    unsqueeze_1026: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1025, 3);  unsqueeze_1025 = None
    sum_102: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_46, [0, 2, 3])
    sub_103: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_1026);  convolution_2 = unsqueeze_1026 = None
    mul_559: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_46, sub_103);  sub_103 = None
    sum_103: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_559, [0, 2, 3]);  mul_559 = None
    mul_564: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_50, primals_8);  primals_8 = None
    unsqueeze_1033: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_564, 0);  mul_564 = None
    unsqueeze_1034: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1033, 2);  unsqueeze_1033 = None
    unsqueeze_1035: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1034, 3);  unsqueeze_1034 = None
    mul_565: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_46, unsqueeze_1035);  where_46 = unsqueeze_1035 = None
    mul_566: "f32[128]" = torch.ops.aten.mul.Tensor(sum_103, rsqrt_50);  sum_103 = rsqrt_50 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_565, relu_1, primals_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_565 = primals_7 = None
    getitem_152: "f32[4, 128, 56, 56]" = convolution_backward_50[0]
    getitem_153: "f32[128, 4, 3, 3]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    le_47: "b8[4, 128, 56, 56]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
    where_47: "f32[4, 128, 56, 56]" = torch.ops.aten.where.self(le_47, full_default, getitem_152);  le_47 = getitem_152 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    add_188: "f32[128]" = torch.ops.aten.add.Tensor(primals_166, 1e-05);  primals_166 = None
    rsqrt_51: "f32[128]" = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
    unsqueeze_1036: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_165, 0);  primals_165 = None
    unsqueeze_1037: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1036, 2);  unsqueeze_1036 = None
    unsqueeze_1038: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1037, 3);  unsqueeze_1037 = None
    sum_104: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_47, [0, 2, 3])
    sub_104: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_1038);  convolution_1 = unsqueeze_1038 = None
    mul_567: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_47, sub_104);  sub_104 = None
    sum_105: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_567, [0, 2, 3]);  mul_567 = None
    mul_572: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_51, primals_5);  primals_5 = None
    unsqueeze_1045: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_572, 0);  mul_572 = None
    unsqueeze_1046: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1045, 2);  unsqueeze_1045 = None
    unsqueeze_1047: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1046, 3);  unsqueeze_1046 = None
    mul_573: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_47, unsqueeze_1047);  where_47 = unsqueeze_1047 = None
    mul_574: "f32[128]" = torch.ops.aten.mul.Tensor(sum_105, rsqrt_51);  sum_105 = rsqrt_51 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_573, getitem, primals_4, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_573 = getitem = primals_4 = None
    getitem_155: "f32[4, 64, 56, 56]" = convolution_backward_51[0]
    getitem_156: "f32[128, 64, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    add_189: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(getitem_146, getitem_155);  getitem_146 = getitem_155 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:271, code: x = self.maxpool(x)
    max_pool2d_with_indices_backward: "f32[4, 64, 112, 112]" = torch.ops.aten.max_pool2d_with_indices_backward.default(add_189, relu, [3, 3], [2, 2], [1, 1], [1, 1], False, getitem_1);  add_189 = getitem_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:270, code: x = self.relu(x)
    le_48: "b8[4, 64, 112, 112]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
    where_48: "f32[4, 64, 112, 112]" = torch.ops.aten.where.self(le_48, full_default, max_pool2d_with_indices_backward);  le_48 = full_default = max_pool2d_with_indices_backward = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:269, code: x = self.bn1(x)
    add_190: "f32[64]" = torch.ops.aten.add.Tensor(primals_163, 1e-05);  primals_163 = None
    rsqrt_52: "f32[64]" = torch.ops.aten.rsqrt.default(add_190);  add_190 = None
    unsqueeze_1048: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_162, 0);  primals_162 = None
    unsqueeze_1049: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1048, 2);  unsqueeze_1048 = None
    unsqueeze_1050: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1049, 3);  unsqueeze_1049 = None
    sum_106: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_48, [0, 2, 3])
    sub_105: "f32[4, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1050);  convolution = unsqueeze_1050 = None
    mul_575: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_48, sub_105);  sub_105 = None
    sum_107: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_575, [0, 2, 3]);  mul_575 = None
    mul_580: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_52, primals_2);  primals_2 = None
    unsqueeze_1057: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_580, 0);  mul_580 = None
    unsqueeze_1058: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1057, 2);  unsqueeze_1057 = None
    unsqueeze_1059: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1058, 3);  unsqueeze_1058 = None
    mul_581: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_48, unsqueeze_1059);  where_48 = unsqueeze_1059 = None
    mul_582: "f32[64]" = torch.ops.aten.mul.Tensor(sum_107, rsqrt_52);  sum_107 = rsqrt_52 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:268, code: x = self.conv1(x)
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_581, primals_321, primals_1, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_581 = primals_321 = primals_1 = None
    getitem_159: "f32[64, 3, 7, 7]" = convolution_backward_52[1];  convolution_backward_52 = None
    return [getitem_159, mul_582, sum_106, getitem_156, mul_574, sum_104, getitem_153, mul_566, sum_102, getitem_150, mul_558, sum_98, getitem_147, mul_550, sum_98, getitem_144, mul_542, sum_96, getitem_141, mul_534, sum_94, getitem_138, mul_526, sum_92, getitem_135, mul_518, sum_90, getitem_132, mul_510, sum_88, getitem_129, mul_502, sum_86, getitem_126, mul_494, sum_84, getitem_123, mul_486, sum_82, getitem_120, mul_478, sum_78, getitem_117, mul_470, sum_78, getitem_114, mul_462, sum_76, getitem_111, mul_454, sum_74, getitem_108, mul_446, sum_72, getitem_105, mul_438, sum_70, getitem_102, mul_430, sum_68, getitem_99, mul_422, sum_66, getitem_96, mul_414, sum_64, getitem_93, mul_406, sum_62, getitem_90, mul_398, sum_60, getitem_87, mul_390, sum_58, getitem_84, mul_382, sum_56, getitem_81, mul_374, sum_52, getitem_78, mul_366, sum_52, getitem_75, mul_358, sum_50, getitem_72, mul_350, sum_48, getitem_69, mul_342, sum_46, getitem_66, mul_334, sum_44, getitem_63, mul_326, sum_42, getitem_60, mul_318, sum_40, getitem_57, mul_310, sum_38, getitem_54, mul_302, sum_36, getitem_51, mul_294, sum_34, getitem_48, mul_286, sum_32, getitem_45, mul_278, sum_30, getitem_42, mul_270, sum_28, getitem_39, mul_262, sum_26, getitem_36, mul_254, sum_24, getitem_33, mul_246, sum_22, getitem_30, mul_238, sum_20, getitem_27, mul_230, sum_18, getitem_24, mul_222, sum_14, getitem_21, mul_214, sum_14, getitem_18, mul_206, sum_12, getitem_15, mul_198, sum_10, getitem_12, mul_190, sum_8, getitem_9, mul_182, sum_6, getitem_6, mul_174, sum_4, getitem_3, mul_166, sum_2, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    