from __future__ import annotations



def forward(self, primals_1: "f32[32, 3, 3, 3]", primals_2: "f32[32]", primals_4: "f32[32, 1, 3, 3]", primals_5: "f32[32]", primals_7: "f32[16, 32, 1, 1]", primals_8: "f32[16]", primals_10: "f32[96, 16, 1, 1]", primals_11: "f32[96]", primals_13: "f32[96, 1, 3, 3]", primals_14: "f32[96]", primals_16: "f32[24, 96, 1, 1]", primals_17: "f32[24]", primals_19: "f32[144, 24, 1, 1]", primals_20: "f32[144]", primals_22: "f32[144, 1, 3, 3]", primals_23: "f32[144]", primals_25: "f32[24, 144, 1, 1]", primals_26: "f32[24]", primals_28: "f32[144, 24, 1, 1]", primals_29: "f32[144]", primals_31: "f32[144, 1, 3, 3]", primals_32: "f32[144]", primals_34: "f32[32, 144, 1, 1]", primals_35: "f32[32]", primals_37: "f32[192, 32, 1, 1]", primals_38: "f32[192]", primals_40: "f32[192, 1, 3, 3]", primals_41: "f32[192]", primals_43: "f32[32, 192, 1, 1]", primals_44: "f32[32]", primals_46: "f32[192, 32, 1, 1]", primals_47: "f32[192]", primals_49: "f32[192, 1, 3, 3]", primals_50: "f32[192]", primals_52: "f32[32, 192, 1, 1]", primals_53: "f32[32]", primals_55: "f32[192, 32, 1, 1]", primals_56: "f32[192]", primals_58: "f32[192, 1, 3, 3]", primals_59: "f32[192]", primals_61: "f32[64, 192, 1, 1]", primals_62: "f32[64]", primals_64: "f32[384, 64, 1, 1]", primals_65: "f32[384]", primals_67: "f32[384, 1, 3, 3]", primals_68: "f32[384]", primals_70: "f32[64, 384, 1, 1]", primals_71: "f32[64]", primals_73: "f32[384, 64, 1, 1]", primals_74: "f32[384]", primals_76: "f32[384, 1, 3, 3]", primals_77: "f32[384]", primals_79: "f32[64, 384, 1, 1]", primals_80: "f32[64]", primals_82: "f32[384, 64, 1, 1]", primals_83: "f32[384]", primals_85: "f32[384, 1, 3, 3]", primals_86: "f32[384]", primals_88: "f32[64, 384, 1, 1]", primals_89: "f32[64]", primals_91: "f32[384, 64, 1, 1]", primals_92: "f32[384]", primals_94: "f32[384, 1, 3, 3]", primals_95: "f32[384]", primals_97: "f32[96, 384, 1, 1]", primals_98: "f32[96]", primals_100: "f32[576, 96, 1, 1]", primals_101: "f32[576]", primals_103: "f32[576, 1, 3, 3]", primals_104: "f32[576]", primals_106: "f32[96, 576, 1, 1]", primals_107: "f32[96]", primals_109: "f32[576, 96, 1, 1]", primals_110: "f32[576]", primals_112: "f32[576, 1, 3, 3]", primals_113: "f32[576]", primals_115: "f32[96, 576, 1, 1]", primals_116: "f32[96]", primals_118: "f32[576, 96, 1, 1]", primals_119: "f32[576]", primals_121: "f32[576, 1, 3, 3]", primals_122: "f32[576]", primals_124: "f32[160, 576, 1, 1]", primals_125: "f32[160]", primals_127: "f32[960, 160, 1, 1]", primals_128: "f32[960]", primals_130: "f32[960, 1, 3, 3]", primals_131: "f32[960]", primals_133: "f32[160, 960, 1, 1]", primals_134: "f32[160]", primals_136: "f32[960, 160, 1, 1]", primals_137: "f32[960]", primals_139: "f32[960, 1, 3, 3]", primals_140: "f32[960]", primals_142: "f32[160, 960, 1, 1]", primals_143: "f32[160]", primals_145: "f32[960, 160, 1, 1]", primals_146: "f32[960]", primals_148: "f32[960, 1, 3, 3]", primals_149: "f32[960]", primals_151: "f32[320, 960, 1, 1]", primals_152: "f32[320]", primals_154: "f32[1280, 320, 1, 1]", primals_155: "f32[1280]", primals_159: "f32[32]", primals_160: "f32[32]", primals_162: "f32[32]", primals_163: "f32[32]", primals_165: "f32[16]", primals_166: "f32[16]", primals_168: "f32[96]", primals_169: "f32[96]", primals_171: "f32[96]", primals_172: "f32[96]", primals_174: "f32[24]", primals_175: "f32[24]", primals_177: "f32[144]", primals_178: "f32[144]", primals_180: "f32[144]", primals_181: "f32[144]", primals_183: "f32[24]", primals_184: "f32[24]", primals_186: "f32[144]", primals_187: "f32[144]", primals_189: "f32[144]", primals_190: "f32[144]", primals_192: "f32[32]", primals_193: "f32[32]", primals_195: "f32[192]", primals_196: "f32[192]", primals_198: "f32[192]", primals_199: "f32[192]", primals_201: "f32[32]", primals_202: "f32[32]", primals_204: "f32[192]", primals_205: "f32[192]", primals_207: "f32[192]", primals_208: "f32[192]", primals_210: "f32[32]", primals_211: "f32[32]", primals_213: "f32[192]", primals_214: "f32[192]", primals_216: "f32[192]", primals_217: "f32[192]", primals_219: "f32[64]", primals_220: "f32[64]", primals_222: "f32[384]", primals_223: "f32[384]", primals_225: "f32[384]", primals_226: "f32[384]", primals_228: "f32[64]", primals_229: "f32[64]", primals_231: "f32[384]", primals_232: "f32[384]", primals_234: "f32[384]", primals_235: "f32[384]", primals_237: "f32[64]", primals_238: "f32[64]", primals_240: "f32[384]", primals_241: "f32[384]", primals_243: "f32[384]", primals_244: "f32[384]", primals_246: "f32[64]", primals_247: "f32[64]", primals_249: "f32[384]", primals_250: "f32[384]", primals_252: "f32[384]", primals_253: "f32[384]", primals_255: "f32[96]", primals_256: "f32[96]", primals_258: "f32[576]", primals_259: "f32[576]", primals_261: "f32[576]", primals_262: "f32[576]", primals_264: "f32[96]", primals_265: "f32[96]", primals_267: "f32[576]", primals_268: "f32[576]", primals_270: "f32[576]", primals_271: "f32[576]", primals_273: "f32[96]", primals_274: "f32[96]", primals_276: "f32[576]", primals_277: "f32[576]", primals_279: "f32[576]", primals_280: "f32[576]", primals_282: "f32[160]", primals_283: "f32[160]", primals_285: "f32[960]", primals_286: "f32[960]", primals_288: "f32[960]", primals_289: "f32[960]", primals_291: "f32[160]", primals_292: "f32[160]", primals_294: "f32[960]", primals_295: "f32[960]", primals_297: "f32[960]", primals_298: "f32[960]", primals_300: "f32[160]", primals_301: "f32[160]", primals_303: "f32[960]", primals_304: "f32[960]", primals_306: "f32[960]", primals_307: "f32[960]", primals_309: "f32[320]", primals_310: "f32[320]", primals_312: "f32[1280]", primals_313: "f32[1280]", primals_315: "f32[4, 3, 224, 224]", convolution: "f32[4, 32, 112, 112]", clamp_max: "f32[4, 32, 112, 112]", convolution_1: "f32[4, 32, 112, 112]", clamp_max_1: "f32[4, 32, 112, 112]", convolution_2: "f32[4, 16, 112, 112]", add_5: "f32[4, 16, 112, 112]", convolution_3: "f32[4, 96, 112, 112]", clamp_max_2: "f32[4, 96, 112, 112]", convolution_4: "f32[4, 96, 56, 56]", clamp_max_3: "f32[4, 96, 56, 56]", convolution_5: "f32[4, 24, 56, 56]", add_11: "f32[4, 24, 56, 56]", convolution_6: "f32[4, 144, 56, 56]", clamp_max_4: "f32[4, 144, 56, 56]", convolution_7: "f32[4, 144, 56, 56]", clamp_max_5: "f32[4, 144, 56, 56]", convolution_8: "f32[4, 24, 56, 56]", add_18: "f32[4, 24, 56, 56]", convolution_9: "f32[4, 144, 56, 56]", clamp_max_6: "f32[4, 144, 56, 56]", convolution_10: "f32[4, 144, 28, 28]", clamp_max_7: "f32[4, 144, 28, 28]", convolution_11: "f32[4, 32, 28, 28]", add_24: "f32[4, 32, 28, 28]", convolution_12: "f32[4, 192, 28, 28]", clamp_max_8: "f32[4, 192, 28, 28]", convolution_13: "f32[4, 192, 28, 28]", clamp_max_9: "f32[4, 192, 28, 28]", convolution_14: "f32[4, 32, 28, 28]", add_31: "f32[4, 32, 28, 28]", convolution_15: "f32[4, 192, 28, 28]", clamp_max_10: "f32[4, 192, 28, 28]", convolution_16: "f32[4, 192, 28, 28]", clamp_max_11: "f32[4, 192, 28, 28]", convolution_17: "f32[4, 32, 28, 28]", add_38: "f32[4, 32, 28, 28]", convolution_18: "f32[4, 192, 28, 28]", clamp_max_12: "f32[4, 192, 28, 28]", convolution_19: "f32[4, 192, 14, 14]", clamp_max_13: "f32[4, 192, 14, 14]", convolution_20: "f32[4, 64, 14, 14]", add_44: "f32[4, 64, 14, 14]", convolution_21: "f32[4, 384, 14, 14]", clamp_max_14: "f32[4, 384, 14, 14]", convolution_22: "f32[4, 384, 14, 14]", clamp_max_15: "f32[4, 384, 14, 14]", convolution_23: "f32[4, 64, 14, 14]", add_51: "f32[4, 64, 14, 14]", convolution_24: "f32[4, 384, 14, 14]", clamp_max_16: "f32[4, 384, 14, 14]", convolution_25: "f32[4, 384, 14, 14]", clamp_max_17: "f32[4, 384, 14, 14]", convolution_26: "f32[4, 64, 14, 14]", add_58: "f32[4, 64, 14, 14]", convolution_27: "f32[4, 384, 14, 14]", clamp_max_18: "f32[4, 384, 14, 14]", convolution_28: "f32[4, 384, 14, 14]", clamp_max_19: "f32[4, 384, 14, 14]", convolution_29: "f32[4, 64, 14, 14]", add_65: "f32[4, 64, 14, 14]", convolution_30: "f32[4, 384, 14, 14]", clamp_max_20: "f32[4, 384, 14, 14]", convolution_31: "f32[4, 384, 14, 14]", clamp_max_21: "f32[4, 384, 14, 14]", convolution_32: "f32[4, 96, 14, 14]", add_71: "f32[4, 96, 14, 14]", convolution_33: "f32[4, 576, 14, 14]", clamp_max_22: "f32[4, 576, 14, 14]", convolution_34: "f32[4, 576, 14, 14]", clamp_max_23: "f32[4, 576, 14, 14]", convolution_35: "f32[4, 96, 14, 14]", add_78: "f32[4, 96, 14, 14]", convolution_36: "f32[4, 576, 14, 14]", clamp_max_24: "f32[4, 576, 14, 14]", convolution_37: "f32[4, 576, 14, 14]", clamp_max_25: "f32[4, 576, 14, 14]", convolution_38: "f32[4, 96, 14, 14]", add_85: "f32[4, 96, 14, 14]", convolution_39: "f32[4, 576, 14, 14]", clamp_max_26: "f32[4, 576, 14, 14]", convolution_40: "f32[4, 576, 7, 7]", clamp_max_27: "f32[4, 576, 7, 7]", convolution_41: "f32[4, 160, 7, 7]", add_91: "f32[4, 160, 7, 7]", convolution_42: "f32[4, 960, 7, 7]", clamp_max_28: "f32[4, 960, 7, 7]", convolution_43: "f32[4, 960, 7, 7]", clamp_max_29: "f32[4, 960, 7, 7]", convolution_44: "f32[4, 160, 7, 7]", add_98: "f32[4, 160, 7, 7]", convolution_45: "f32[4, 960, 7, 7]", clamp_max_30: "f32[4, 960, 7, 7]", convolution_46: "f32[4, 960, 7, 7]", clamp_max_31: "f32[4, 960, 7, 7]", convolution_47: "f32[4, 160, 7, 7]", add_105: "f32[4, 160, 7, 7]", convolution_48: "f32[4, 960, 7, 7]", clamp_max_32: "f32[4, 960, 7, 7]", convolution_49: "f32[4, 960, 7, 7]", clamp_max_33: "f32[4, 960, 7, 7]", convolution_50: "f32[4, 320, 7, 7]", add_111: "f32[4, 320, 7, 7]", convolution_51: "f32[4, 1280, 7, 7]", clone_35: "f32[4, 1280]", permute_1: "f32[1000, 1280]", bitwise_or: "b8[4, 1280, 7, 7]", bitwise_or_1: "b8[4, 960, 7, 7]", bitwise_or_2: "b8[4, 960, 7, 7]", bitwise_or_3: "b8[4, 960, 7, 7]", bitwise_or_4: "b8[4, 960, 7, 7]", bitwise_or_5: "b8[4, 960, 7, 7]", bitwise_or_6: "b8[4, 960, 7, 7]", bitwise_or_7: "b8[4, 576, 7, 7]", bitwise_or_8: "b8[4, 576, 14, 14]", bitwise_or_9: "b8[4, 576, 14, 14]", bitwise_or_10: "b8[4, 576, 14, 14]", bitwise_or_11: "b8[4, 576, 14, 14]", bitwise_or_12: "b8[4, 576, 14, 14]", bitwise_or_13: "b8[4, 384, 14, 14]", bitwise_or_14: "b8[4, 384, 14, 14]", bitwise_or_15: "b8[4, 384, 14, 14]", bitwise_or_16: "b8[4, 384, 14, 14]", bitwise_or_17: "b8[4, 384, 14, 14]", bitwise_or_18: "b8[4, 384, 14, 14]", bitwise_or_19: "b8[4, 384, 14, 14]", bitwise_or_20: "b8[4, 384, 14, 14]", bitwise_or_21: "b8[4, 192, 14, 14]", bitwise_or_22: "b8[4, 192, 28, 28]", bitwise_or_23: "b8[4, 192, 28, 28]", bitwise_or_24: "b8[4, 192, 28, 28]", bitwise_or_25: "b8[4, 192, 28, 28]", bitwise_or_26: "b8[4, 192, 28, 28]", bitwise_or_27: "b8[4, 144, 28, 28]", bitwise_or_28: "b8[4, 144, 56, 56]", bitwise_or_29: "b8[4, 144, 56, 56]", bitwise_or_30: "b8[4, 144, 56, 56]", bitwise_or_31: "b8[4, 96, 56, 56]", bitwise_or_32: "b8[4, 96, 112, 112]", bitwise_or_33: "b8[4, 32, 112, 112]", bitwise_or_34: "b8[4, 32, 112, 112]", tangents_1: "f32[4, 1000]"):
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:170, code: x = self.classifier(x)
    mm: "f32[4, 1280]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 4]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 1280]" = torch.ops.aten.mm.default(permute_2, clone_35);  permute_2 = clone_35 = None
    permute_3: "f32[1280, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 1280]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:169, code: x = torch.flatten(x, 1)
    view_2: "f32[4, 1280, 1, 1]" = torch.ops.aten.reshape.default(mm, [4, 1280, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:168, code: x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
    expand: "f32[4, 1280, 7, 7]" = torch.ops.aten.expand.default(view_2, [4, 1280, 7, 7]);  view_2 = None
    div: "f32[4, 1280, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:166, code: x = self.features(x)
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where: "f32[4, 1280, 7, 7]" = torch.ops.aten.where.self(bitwise_or, full_default, div);  bitwise_or = div = None
    add_114: "f32[1280]" = torch.ops.aten.add.Tensor(primals_313, 1e-05);  primals_313 = None
    rsqrt: "f32[1280]" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
    unsqueeze_416: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_312, 0);  primals_312 = None
    unsqueeze_417: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 2);  unsqueeze_416 = None
    unsqueeze_418: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 3);  unsqueeze_417 = None
    sum_2: "f32[1280]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_52: "f32[4, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_418);  convolution_51 = unsqueeze_418 = None
    mul_156: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_52);  sub_52 = None
    sum_3: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_156, [0, 2, 3]);  mul_156 = None
    mul_161: "f32[1280]" = torch.ops.aten.mul.Tensor(rsqrt, primals_155);  primals_155 = None
    unsqueeze_425: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_161, 0);  mul_161 = None
    unsqueeze_426: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 2);  unsqueeze_425 = None
    unsqueeze_427: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 3);  unsqueeze_426 = None
    mul_162: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(where, unsqueeze_427);  where = unsqueeze_427 = None
    mul_163: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_3, rsqrt);  sum_3 = rsqrt = None
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_162, add_111, primals_154, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_162 = add_111 = primals_154 = None
    getitem: "f32[4, 320, 7, 7]" = convolution_backward[0]
    getitem_1: "f32[1280, 320, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:64, code: return self.conv(x)
    add_115: "f32[320]" = torch.ops.aten.add.Tensor(primals_310, 1e-05);  primals_310 = None
    rsqrt_1: "f32[320]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
    unsqueeze_428: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(primals_309, 0);  primals_309 = None
    unsqueeze_429: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 2);  unsqueeze_428 = None
    unsqueeze_430: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 3);  unsqueeze_429 = None
    sum_4: "f32[320]" = torch.ops.aten.sum.dim_IntList(getitem, [0, 2, 3])
    sub_53: "f32[4, 320, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_430);  convolution_50 = unsqueeze_430 = None
    mul_164: "f32[4, 320, 7, 7]" = torch.ops.aten.mul.Tensor(getitem, sub_53);  sub_53 = None
    sum_5: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_164, [0, 2, 3]);  mul_164 = None
    mul_169: "f32[320]" = torch.ops.aten.mul.Tensor(rsqrt_1, primals_152);  primals_152 = None
    unsqueeze_437: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_169, 0);  mul_169 = None
    unsqueeze_438: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 2);  unsqueeze_437 = None
    unsqueeze_439: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 3);  unsqueeze_438 = None
    mul_170: "f32[4, 320, 7, 7]" = torch.ops.aten.mul.Tensor(getitem, unsqueeze_439);  getitem = unsqueeze_439 = None
    mul_171: "f32[320]" = torch.ops.aten.mul.Tensor(sum_5, rsqrt_1);  sum_5 = rsqrt_1 = None
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_170, clamp_max_33, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_170 = clamp_max_33 = primals_151 = None
    getitem_3: "f32[4, 960, 7, 7]" = convolution_backward_1[0]
    getitem_4: "f32[320, 960, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    where_1: "f32[4, 960, 7, 7]" = torch.ops.aten.where.self(bitwise_or_1, full_default, getitem_3);  bitwise_or_1 = getitem_3 = None
    add_116: "f32[960]" = torch.ops.aten.add.Tensor(primals_307, 1e-05);  primals_307 = None
    rsqrt_2: "f32[960]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
    unsqueeze_440: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(primals_306, 0);  primals_306 = None
    unsqueeze_441: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 2);  unsqueeze_440 = None
    unsqueeze_442: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 3);  unsqueeze_441 = None
    sum_6: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_54: "f32[4, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_442);  convolution_49 = unsqueeze_442 = None
    mul_172: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, sub_54);  sub_54 = None
    sum_7: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_172, [0, 2, 3]);  mul_172 = None
    mul_177: "f32[960]" = torch.ops.aten.mul.Tensor(rsqrt_2, primals_149);  primals_149 = None
    unsqueeze_449: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_177, 0);  mul_177 = None
    unsqueeze_450: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 2);  unsqueeze_449 = None
    unsqueeze_451: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 3);  unsqueeze_450 = None
    mul_178: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, unsqueeze_451);  where_1 = unsqueeze_451 = None
    mul_179: "f32[960]" = torch.ops.aten.mul.Tensor(sum_7, rsqrt_2);  sum_7 = rsqrt_2 = None
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_178, clamp_max_32, primals_148, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 960, [True, True, False]);  mul_178 = clamp_max_32 = primals_148 = None
    getitem_6: "f32[4, 960, 7, 7]" = convolution_backward_2[0]
    getitem_7: "f32[960, 1, 3, 3]" = convolution_backward_2[1];  convolution_backward_2 = None
    where_2: "f32[4, 960, 7, 7]" = torch.ops.aten.where.self(bitwise_or_2, full_default, getitem_6);  bitwise_or_2 = getitem_6 = None
    add_117: "f32[960]" = torch.ops.aten.add.Tensor(primals_304, 1e-05);  primals_304 = None
    rsqrt_3: "f32[960]" = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
    unsqueeze_452: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(primals_303, 0);  primals_303 = None
    unsqueeze_453: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 2);  unsqueeze_452 = None
    unsqueeze_454: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 3);  unsqueeze_453 = None
    sum_8: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_55: "f32[4, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_454);  convolution_48 = unsqueeze_454 = None
    mul_180: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_55);  sub_55 = None
    sum_9: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_180, [0, 2, 3]);  mul_180 = None
    mul_185: "f32[960]" = torch.ops.aten.mul.Tensor(rsqrt_3, primals_146);  primals_146 = None
    unsqueeze_461: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_185, 0);  mul_185 = None
    unsqueeze_462: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None
    unsqueeze_463: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 3);  unsqueeze_462 = None
    mul_186: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, unsqueeze_463);  where_2 = unsqueeze_463 = None
    mul_187: "f32[960]" = torch.ops.aten.mul.Tensor(sum_9, rsqrt_3);  sum_9 = rsqrt_3 = None
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_186, add_105, primals_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_186 = add_105 = primals_145 = None
    getitem_9: "f32[4, 160, 7, 7]" = convolution_backward_3[0]
    getitem_10: "f32[960, 160, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:62, code: return x + self.conv(x)
    add_118: "f32[160]" = torch.ops.aten.add.Tensor(primals_301, 1e-05);  primals_301 = None
    rsqrt_4: "f32[160]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
    unsqueeze_464: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(primals_300, 0);  primals_300 = None
    unsqueeze_465: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 2);  unsqueeze_464 = None
    unsqueeze_466: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 3);  unsqueeze_465 = None
    sum_10: "f32[160]" = torch.ops.aten.sum.dim_IntList(getitem_9, [0, 2, 3])
    sub_56: "f32[4, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_466);  convolution_47 = unsqueeze_466 = None
    mul_188: "f32[4, 160, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_9, sub_56);  sub_56 = None
    sum_11: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_188, [0, 2, 3]);  mul_188 = None
    mul_193: "f32[160]" = torch.ops.aten.mul.Tensor(rsqrt_4, primals_143);  primals_143 = None
    unsqueeze_473: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_193, 0);  mul_193 = None
    unsqueeze_474: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 2);  unsqueeze_473 = None
    unsqueeze_475: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, 3);  unsqueeze_474 = None
    mul_194: "f32[4, 160, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_9, unsqueeze_475);  unsqueeze_475 = None
    mul_195: "f32[160]" = torch.ops.aten.mul.Tensor(sum_11, rsqrt_4);  sum_11 = rsqrt_4 = None
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_194, clamp_max_31, primals_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_194 = clamp_max_31 = primals_142 = None
    getitem_12: "f32[4, 960, 7, 7]" = convolution_backward_4[0]
    getitem_13: "f32[160, 960, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    where_3: "f32[4, 960, 7, 7]" = torch.ops.aten.where.self(bitwise_or_3, full_default, getitem_12);  bitwise_or_3 = getitem_12 = None
    add_119: "f32[960]" = torch.ops.aten.add.Tensor(primals_298, 1e-05);  primals_298 = None
    rsqrt_5: "f32[960]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
    unsqueeze_476: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(primals_297, 0);  primals_297 = None
    unsqueeze_477: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 2);  unsqueeze_476 = None
    unsqueeze_478: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 3);  unsqueeze_477 = None
    sum_12: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_57: "f32[4, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_478);  convolution_46 = unsqueeze_478 = None
    mul_196: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_57);  sub_57 = None
    sum_13: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_196, [0, 2, 3]);  mul_196 = None
    mul_201: "f32[960]" = torch.ops.aten.mul.Tensor(rsqrt_5, primals_140);  primals_140 = None
    unsqueeze_485: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_201, 0);  mul_201 = None
    unsqueeze_486: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 2);  unsqueeze_485 = None
    unsqueeze_487: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 3);  unsqueeze_486 = None
    mul_202: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, unsqueeze_487);  where_3 = unsqueeze_487 = None
    mul_203: "f32[960]" = torch.ops.aten.mul.Tensor(sum_13, rsqrt_5);  sum_13 = rsqrt_5 = None
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_202, clamp_max_30, primals_139, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 960, [True, True, False]);  mul_202 = clamp_max_30 = primals_139 = None
    getitem_15: "f32[4, 960, 7, 7]" = convolution_backward_5[0]
    getitem_16: "f32[960, 1, 3, 3]" = convolution_backward_5[1];  convolution_backward_5 = None
    where_4: "f32[4, 960, 7, 7]" = torch.ops.aten.where.self(bitwise_or_4, full_default, getitem_15);  bitwise_or_4 = getitem_15 = None
    add_120: "f32[960]" = torch.ops.aten.add.Tensor(primals_295, 1e-05);  primals_295 = None
    rsqrt_6: "f32[960]" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
    unsqueeze_488: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(primals_294, 0);  primals_294 = None
    unsqueeze_489: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 2);  unsqueeze_488 = None
    unsqueeze_490: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 3);  unsqueeze_489 = None
    sum_14: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_58: "f32[4, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_490);  convolution_45 = unsqueeze_490 = None
    mul_204: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, sub_58);  sub_58 = None
    sum_15: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_204, [0, 2, 3]);  mul_204 = None
    mul_209: "f32[960]" = torch.ops.aten.mul.Tensor(rsqrt_6, primals_137);  primals_137 = None
    unsqueeze_497: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_209, 0);  mul_209 = None
    unsqueeze_498: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 2);  unsqueeze_497 = None
    unsqueeze_499: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 3);  unsqueeze_498 = None
    mul_210: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, unsqueeze_499);  where_4 = unsqueeze_499 = None
    mul_211: "f32[960]" = torch.ops.aten.mul.Tensor(sum_15, rsqrt_6);  sum_15 = rsqrt_6 = None
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_210, add_98, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_210 = add_98 = primals_136 = None
    getitem_18: "f32[4, 160, 7, 7]" = convolution_backward_6[0]
    getitem_19: "f32[960, 160, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:62, code: return x + self.conv(x)
    add_121: "f32[4, 160, 7, 7]" = torch.ops.aten.add.Tensor(getitem_9, getitem_18);  getitem_9 = getitem_18 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:62, code: return x + self.conv(x)
    add_122: "f32[160]" = torch.ops.aten.add.Tensor(primals_292, 1e-05);  primals_292 = None
    rsqrt_7: "f32[160]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    unsqueeze_500: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(primals_291, 0);  primals_291 = None
    unsqueeze_501: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 2);  unsqueeze_500 = None
    unsqueeze_502: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 3);  unsqueeze_501 = None
    sum_16: "f32[160]" = torch.ops.aten.sum.dim_IntList(add_121, [0, 2, 3])
    sub_59: "f32[4, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_502);  convolution_44 = unsqueeze_502 = None
    mul_212: "f32[4, 160, 7, 7]" = torch.ops.aten.mul.Tensor(add_121, sub_59);  sub_59 = None
    sum_17: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_212, [0, 2, 3]);  mul_212 = None
    mul_217: "f32[160]" = torch.ops.aten.mul.Tensor(rsqrt_7, primals_134);  primals_134 = None
    unsqueeze_509: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_217, 0);  mul_217 = None
    unsqueeze_510: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 2);  unsqueeze_509 = None
    unsqueeze_511: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, 3);  unsqueeze_510 = None
    mul_218: "f32[4, 160, 7, 7]" = torch.ops.aten.mul.Tensor(add_121, unsqueeze_511);  unsqueeze_511 = None
    mul_219: "f32[160]" = torch.ops.aten.mul.Tensor(sum_17, rsqrt_7);  sum_17 = rsqrt_7 = None
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_218, clamp_max_29, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_218 = clamp_max_29 = primals_133 = None
    getitem_21: "f32[4, 960, 7, 7]" = convolution_backward_7[0]
    getitem_22: "f32[160, 960, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    where_5: "f32[4, 960, 7, 7]" = torch.ops.aten.where.self(bitwise_or_5, full_default, getitem_21);  bitwise_or_5 = getitem_21 = None
    add_123: "f32[960]" = torch.ops.aten.add.Tensor(primals_289, 1e-05);  primals_289 = None
    rsqrt_8: "f32[960]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
    unsqueeze_512: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(primals_288, 0);  primals_288 = None
    unsqueeze_513: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 2);  unsqueeze_512 = None
    unsqueeze_514: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_513, 3);  unsqueeze_513 = None
    sum_18: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_60: "f32[4, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_514);  convolution_43 = unsqueeze_514 = None
    mul_220: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_5, sub_60);  sub_60 = None
    sum_19: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_220, [0, 2, 3]);  mul_220 = None
    mul_225: "f32[960]" = torch.ops.aten.mul.Tensor(rsqrt_8, primals_131);  primals_131 = None
    unsqueeze_521: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_225, 0);  mul_225 = None
    unsqueeze_522: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 2);  unsqueeze_521 = None
    unsqueeze_523: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, 3);  unsqueeze_522 = None
    mul_226: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_5, unsqueeze_523);  where_5 = unsqueeze_523 = None
    mul_227: "f32[960]" = torch.ops.aten.mul.Tensor(sum_19, rsqrt_8);  sum_19 = rsqrt_8 = None
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_226, clamp_max_28, primals_130, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 960, [True, True, False]);  mul_226 = clamp_max_28 = primals_130 = None
    getitem_24: "f32[4, 960, 7, 7]" = convolution_backward_8[0]
    getitem_25: "f32[960, 1, 3, 3]" = convolution_backward_8[1];  convolution_backward_8 = None
    where_6: "f32[4, 960, 7, 7]" = torch.ops.aten.where.self(bitwise_or_6, full_default, getitem_24);  bitwise_or_6 = getitem_24 = None
    add_124: "f32[960]" = torch.ops.aten.add.Tensor(primals_286, 1e-05);  primals_286 = None
    rsqrt_9: "f32[960]" = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
    unsqueeze_524: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(primals_285, 0);  primals_285 = None
    unsqueeze_525: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 2);  unsqueeze_524 = None
    unsqueeze_526: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_525, 3);  unsqueeze_525 = None
    sum_20: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_61: "f32[4, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_526);  convolution_42 = unsqueeze_526 = None
    mul_228: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, sub_61);  sub_61 = None
    sum_21: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_228, [0, 2, 3]);  mul_228 = None
    mul_233: "f32[960]" = torch.ops.aten.mul.Tensor(rsqrt_9, primals_128);  primals_128 = None
    unsqueeze_533: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_233, 0);  mul_233 = None
    unsqueeze_534: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 2);  unsqueeze_533 = None
    unsqueeze_535: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, 3);  unsqueeze_534 = None
    mul_234: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, unsqueeze_535);  where_6 = unsqueeze_535 = None
    mul_235: "f32[960]" = torch.ops.aten.mul.Tensor(sum_21, rsqrt_9);  sum_21 = rsqrt_9 = None
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_234, add_91, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_234 = add_91 = primals_127 = None
    getitem_27: "f32[4, 160, 7, 7]" = convolution_backward_9[0]
    getitem_28: "f32[960, 160, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:62, code: return x + self.conv(x)
    add_125: "f32[4, 160, 7, 7]" = torch.ops.aten.add.Tensor(add_121, getitem_27);  add_121 = getitem_27 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:64, code: return self.conv(x)
    add_126: "f32[160]" = torch.ops.aten.add.Tensor(primals_283, 1e-05);  primals_283 = None
    rsqrt_10: "f32[160]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
    unsqueeze_536: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(primals_282, 0);  primals_282 = None
    unsqueeze_537: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 2);  unsqueeze_536 = None
    unsqueeze_538: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_537, 3);  unsqueeze_537 = None
    sum_22: "f32[160]" = torch.ops.aten.sum.dim_IntList(add_125, [0, 2, 3])
    sub_62: "f32[4, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_538);  convolution_41 = unsqueeze_538 = None
    mul_236: "f32[4, 160, 7, 7]" = torch.ops.aten.mul.Tensor(add_125, sub_62);  sub_62 = None
    sum_23: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_236, [0, 2, 3]);  mul_236 = None
    mul_241: "f32[160]" = torch.ops.aten.mul.Tensor(rsqrt_10, primals_125);  primals_125 = None
    unsqueeze_545: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_241, 0);  mul_241 = None
    unsqueeze_546: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 2);  unsqueeze_545 = None
    unsqueeze_547: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, 3);  unsqueeze_546 = None
    mul_242: "f32[4, 160, 7, 7]" = torch.ops.aten.mul.Tensor(add_125, unsqueeze_547);  add_125 = unsqueeze_547 = None
    mul_243: "f32[160]" = torch.ops.aten.mul.Tensor(sum_23, rsqrt_10);  sum_23 = rsqrt_10 = None
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_242, clamp_max_27, primals_124, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_242 = clamp_max_27 = primals_124 = None
    getitem_30: "f32[4, 576, 7, 7]" = convolution_backward_10[0]
    getitem_31: "f32[160, 576, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    where_7: "f32[4, 576, 7, 7]" = torch.ops.aten.where.self(bitwise_or_7, full_default, getitem_30);  bitwise_or_7 = getitem_30 = None
    add_127: "f32[576]" = torch.ops.aten.add.Tensor(primals_280, 1e-05);  primals_280 = None
    rsqrt_11: "f32[576]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
    unsqueeze_548: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(primals_279, 0);  primals_279 = None
    unsqueeze_549: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 2);  unsqueeze_548 = None
    unsqueeze_550: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_549, 3);  unsqueeze_549 = None
    sum_24: "f32[576]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_63: "f32[4, 576, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_550);  convolution_40 = unsqueeze_550 = None
    mul_244: "f32[4, 576, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, sub_63);  sub_63 = None
    sum_25: "f32[576]" = torch.ops.aten.sum.dim_IntList(mul_244, [0, 2, 3]);  mul_244 = None
    mul_249: "f32[576]" = torch.ops.aten.mul.Tensor(rsqrt_11, primals_122);  primals_122 = None
    unsqueeze_557: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_249, 0);  mul_249 = None
    unsqueeze_558: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 2);  unsqueeze_557 = None
    unsqueeze_559: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, 3);  unsqueeze_558 = None
    mul_250: "f32[4, 576, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, unsqueeze_559);  where_7 = unsqueeze_559 = None
    mul_251: "f32[576]" = torch.ops.aten.mul.Tensor(sum_25, rsqrt_11);  sum_25 = rsqrt_11 = None
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_250, clamp_max_26, primals_121, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 576, [True, True, False]);  mul_250 = clamp_max_26 = primals_121 = None
    getitem_33: "f32[4, 576, 14, 14]" = convolution_backward_11[0]
    getitem_34: "f32[576, 1, 3, 3]" = convolution_backward_11[1];  convolution_backward_11 = None
    where_8: "f32[4, 576, 14, 14]" = torch.ops.aten.where.self(bitwise_or_8, full_default, getitem_33);  bitwise_or_8 = getitem_33 = None
    add_128: "f32[576]" = torch.ops.aten.add.Tensor(primals_277, 1e-05);  primals_277 = None
    rsqrt_12: "f32[576]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
    unsqueeze_560: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(primals_276, 0);  primals_276 = None
    unsqueeze_561: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 2);  unsqueeze_560 = None
    unsqueeze_562: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_561, 3);  unsqueeze_561 = None
    sum_26: "f32[576]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_64: "f32[4, 576, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_562);  convolution_39 = unsqueeze_562 = None
    mul_252: "f32[4, 576, 14, 14]" = torch.ops.aten.mul.Tensor(where_8, sub_64);  sub_64 = None
    sum_27: "f32[576]" = torch.ops.aten.sum.dim_IntList(mul_252, [0, 2, 3]);  mul_252 = None
    mul_257: "f32[576]" = torch.ops.aten.mul.Tensor(rsqrt_12, primals_119);  primals_119 = None
    unsqueeze_569: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_257, 0);  mul_257 = None
    unsqueeze_570: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 2);  unsqueeze_569 = None
    unsqueeze_571: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, 3);  unsqueeze_570 = None
    mul_258: "f32[4, 576, 14, 14]" = torch.ops.aten.mul.Tensor(where_8, unsqueeze_571);  where_8 = unsqueeze_571 = None
    mul_259: "f32[576]" = torch.ops.aten.mul.Tensor(sum_27, rsqrt_12);  sum_27 = rsqrt_12 = None
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_258, add_85, primals_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_258 = add_85 = primals_118 = None
    getitem_36: "f32[4, 96, 14, 14]" = convolution_backward_12[0]
    getitem_37: "f32[576, 96, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:62, code: return x + self.conv(x)
    add_129: "f32[96]" = torch.ops.aten.add.Tensor(primals_274, 1e-05);  primals_274 = None
    rsqrt_13: "f32[96]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
    unsqueeze_572: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_273, 0);  primals_273 = None
    unsqueeze_573: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 2);  unsqueeze_572 = None
    unsqueeze_574: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_573, 3);  unsqueeze_573 = None
    sum_28: "f32[96]" = torch.ops.aten.sum.dim_IntList(getitem_36, [0, 2, 3])
    sub_65: "f32[4, 96, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_574);  convolution_38 = unsqueeze_574 = None
    mul_260: "f32[4, 96, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_36, sub_65);  sub_65 = None
    sum_29: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_260, [0, 2, 3]);  mul_260 = None
    mul_265: "f32[96]" = torch.ops.aten.mul.Tensor(rsqrt_13, primals_116);  primals_116 = None
    unsqueeze_581: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_265, 0);  mul_265 = None
    unsqueeze_582: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 2);  unsqueeze_581 = None
    unsqueeze_583: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, 3);  unsqueeze_582 = None
    mul_266: "f32[4, 96, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_36, unsqueeze_583);  unsqueeze_583 = None
    mul_267: "f32[96]" = torch.ops.aten.mul.Tensor(sum_29, rsqrt_13);  sum_29 = rsqrt_13 = None
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_266, clamp_max_25, primals_115, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_266 = clamp_max_25 = primals_115 = None
    getitem_39: "f32[4, 576, 14, 14]" = convolution_backward_13[0]
    getitem_40: "f32[96, 576, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    where_9: "f32[4, 576, 14, 14]" = torch.ops.aten.where.self(bitwise_or_9, full_default, getitem_39);  bitwise_or_9 = getitem_39 = None
    add_130: "f32[576]" = torch.ops.aten.add.Tensor(primals_271, 1e-05);  primals_271 = None
    rsqrt_14: "f32[576]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
    unsqueeze_584: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(primals_270, 0);  primals_270 = None
    unsqueeze_585: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 2);  unsqueeze_584 = None
    unsqueeze_586: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_585, 3);  unsqueeze_585 = None
    sum_30: "f32[576]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_66: "f32[4, 576, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_586);  convolution_37 = unsqueeze_586 = None
    mul_268: "f32[4, 576, 14, 14]" = torch.ops.aten.mul.Tensor(where_9, sub_66);  sub_66 = None
    sum_31: "f32[576]" = torch.ops.aten.sum.dim_IntList(mul_268, [0, 2, 3]);  mul_268 = None
    mul_273: "f32[576]" = torch.ops.aten.mul.Tensor(rsqrt_14, primals_113);  primals_113 = None
    unsqueeze_593: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_273, 0);  mul_273 = None
    unsqueeze_594: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 2);  unsqueeze_593 = None
    unsqueeze_595: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, 3);  unsqueeze_594 = None
    mul_274: "f32[4, 576, 14, 14]" = torch.ops.aten.mul.Tensor(where_9, unsqueeze_595);  where_9 = unsqueeze_595 = None
    mul_275: "f32[576]" = torch.ops.aten.mul.Tensor(sum_31, rsqrt_14);  sum_31 = rsqrt_14 = None
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_274, clamp_max_24, primals_112, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 576, [True, True, False]);  mul_274 = clamp_max_24 = primals_112 = None
    getitem_42: "f32[4, 576, 14, 14]" = convolution_backward_14[0]
    getitem_43: "f32[576, 1, 3, 3]" = convolution_backward_14[1];  convolution_backward_14 = None
    where_10: "f32[4, 576, 14, 14]" = torch.ops.aten.where.self(bitwise_or_10, full_default, getitem_42);  bitwise_or_10 = getitem_42 = None
    add_131: "f32[576]" = torch.ops.aten.add.Tensor(primals_268, 1e-05);  primals_268 = None
    rsqrt_15: "f32[576]" = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
    unsqueeze_596: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(primals_267, 0);  primals_267 = None
    unsqueeze_597: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 2);  unsqueeze_596 = None
    unsqueeze_598: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_597, 3);  unsqueeze_597 = None
    sum_32: "f32[576]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_67: "f32[4, 576, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_598);  convolution_36 = unsqueeze_598 = None
    mul_276: "f32[4, 576, 14, 14]" = torch.ops.aten.mul.Tensor(where_10, sub_67);  sub_67 = None
    sum_33: "f32[576]" = torch.ops.aten.sum.dim_IntList(mul_276, [0, 2, 3]);  mul_276 = None
    mul_281: "f32[576]" = torch.ops.aten.mul.Tensor(rsqrt_15, primals_110);  primals_110 = None
    unsqueeze_605: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_281, 0);  mul_281 = None
    unsqueeze_606: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 2);  unsqueeze_605 = None
    unsqueeze_607: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, 3);  unsqueeze_606 = None
    mul_282: "f32[4, 576, 14, 14]" = torch.ops.aten.mul.Tensor(where_10, unsqueeze_607);  where_10 = unsqueeze_607 = None
    mul_283: "f32[576]" = torch.ops.aten.mul.Tensor(sum_33, rsqrt_15);  sum_33 = rsqrt_15 = None
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_282, add_78, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_282 = add_78 = primals_109 = None
    getitem_45: "f32[4, 96, 14, 14]" = convolution_backward_15[0]
    getitem_46: "f32[576, 96, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:62, code: return x + self.conv(x)
    add_132: "f32[4, 96, 14, 14]" = torch.ops.aten.add.Tensor(getitem_36, getitem_45);  getitem_36 = getitem_45 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:62, code: return x + self.conv(x)
    add_133: "f32[96]" = torch.ops.aten.add.Tensor(primals_265, 1e-05);  primals_265 = None
    rsqrt_16: "f32[96]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
    unsqueeze_608: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_264, 0);  primals_264 = None
    unsqueeze_609: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 2);  unsqueeze_608 = None
    unsqueeze_610: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_609, 3);  unsqueeze_609 = None
    sum_34: "f32[96]" = torch.ops.aten.sum.dim_IntList(add_132, [0, 2, 3])
    sub_68: "f32[4, 96, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_610);  convolution_35 = unsqueeze_610 = None
    mul_284: "f32[4, 96, 14, 14]" = torch.ops.aten.mul.Tensor(add_132, sub_68);  sub_68 = None
    sum_35: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_284, [0, 2, 3]);  mul_284 = None
    mul_289: "f32[96]" = torch.ops.aten.mul.Tensor(rsqrt_16, primals_107);  primals_107 = None
    unsqueeze_617: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_289, 0);  mul_289 = None
    unsqueeze_618: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 2);  unsqueeze_617 = None
    unsqueeze_619: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, 3);  unsqueeze_618 = None
    mul_290: "f32[4, 96, 14, 14]" = torch.ops.aten.mul.Tensor(add_132, unsqueeze_619);  unsqueeze_619 = None
    mul_291: "f32[96]" = torch.ops.aten.mul.Tensor(sum_35, rsqrt_16);  sum_35 = rsqrt_16 = None
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_290, clamp_max_23, primals_106, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_290 = clamp_max_23 = primals_106 = None
    getitem_48: "f32[4, 576, 14, 14]" = convolution_backward_16[0]
    getitem_49: "f32[96, 576, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    where_11: "f32[4, 576, 14, 14]" = torch.ops.aten.where.self(bitwise_or_11, full_default, getitem_48);  bitwise_or_11 = getitem_48 = None
    add_134: "f32[576]" = torch.ops.aten.add.Tensor(primals_262, 1e-05);  primals_262 = None
    rsqrt_17: "f32[576]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
    unsqueeze_620: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(primals_261, 0);  primals_261 = None
    unsqueeze_621: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 2);  unsqueeze_620 = None
    unsqueeze_622: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_621, 3);  unsqueeze_621 = None
    sum_36: "f32[576]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_69: "f32[4, 576, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_622);  convolution_34 = unsqueeze_622 = None
    mul_292: "f32[4, 576, 14, 14]" = torch.ops.aten.mul.Tensor(where_11, sub_69);  sub_69 = None
    sum_37: "f32[576]" = torch.ops.aten.sum.dim_IntList(mul_292, [0, 2, 3]);  mul_292 = None
    mul_297: "f32[576]" = torch.ops.aten.mul.Tensor(rsqrt_17, primals_104);  primals_104 = None
    unsqueeze_629: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_297, 0);  mul_297 = None
    unsqueeze_630: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 2);  unsqueeze_629 = None
    unsqueeze_631: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, 3);  unsqueeze_630 = None
    mul_298: "f32[4, 576, 14, 14]" = torch.ops.aten.mul.Tensor(where_11, unsqueeze_631);  where_11 = unsqueeze_631 = None
    mul_299: "f32[576]" = torch.ops.aten.mul.Tensor(sum_37, rsqrt_17);  sum_37 = rsqrt_17 = None
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_298, clamp_max_22, primals_103, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 576, [True, True, False]);  mul_298 = clamp_max_22 = primals_103 = None
    getitem_51: "f32[4, 576, 14, 14]" = convolution_backward_17[0]
    getitem_52: "f32[576, 1, 3, 3]" = convolution_backward_17[1];  convolution_backward_17 = None
    where_12: "f32[4, 576, 14, 14]" = torch.ops.aten.where.self(bitwise_or_12, full_default, getitem_51);  bitwise_or_12 = getitem_51 = None
    add_135: "f32[576]" = torch.ops.aten.add.Tensor(primals_259, 1e-05);  primals_259 = None
    rsqrt_18: "f32[576]" = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
    unsqueeze_632: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(primals_258, 0);  primals_258 = None
    unsqueeze_633: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 2);  unsqueeze_632 = None
    unsqueeze_634: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_633, 3);  unsqueeze_633 = None
    sum_38: "f32[576]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_70: "f32[4, 576, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_634);  convolution_33 = unsqueeze_634 = None
    mul_300: "f32[4, 576, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, sub_70);  sub_70 = None
    sum_39: "f32[576]" = torch.ops.aten.sum.dim_IntList(mul_300, [0, 2, 3]);  mul_300 = None
    mul_305: "f32[576]" = torch.ops.aten.mul.Tensor(rsqrt_18, primals_101);  primals_101 = None
    unsqueeze_641: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_305, 0);  mul_305 = None
    unsqueeze_642: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 2);  unsqueeze_641 = None
    unsqueeze_643: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, 3);  unsqueeze_642 = None
    mul_306: "f32[4, 576, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, unsqueeze_643);  where_12 = unsqueeze_643 = None
    mul_307: "f32[576]" = torch.ops.aten.mul.Tensor(sum_39, rsqrt_18);  sum_39 = rsqrt_18 = None
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_306, add_71, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_306 = add_71 = primals_100 = None
    getitem_54: "f32[4, 96, 14, 14]" = convolution_backward_18[0]
    getitem_55: "f32[576, 96, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:62, code: return x + self.conv(x)
    add_136: "f32[4, 96, 14, 14]" = torch.ops.aten.add.Tensor(add_132, getitem_54);  add_132 = getitem_54 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:64, code: return self.conv(x)
    add_137: "f32[96]" = torch.ops.aten.add.Tensor(primals_256, 1e-05);  primals_256 = None
    rsqrt_19: "f32[96]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
    unsqueeze_644: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_255, 0);  primals_255 = None
    unsqueeze_645: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 2);  unsqueeze_644 = None
    unsqueeze_646: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_645, 3);  unsqueeze_645 = None
    sum_40: "f32[96]" = torch.ops.aten.sum.dim_IntList(add_136, [0, 2, 3])
    sub_71: "f32[4, 96, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_646);  convolution_32 = unsqueeze_646 = None
    mul_308: "f32[4, 96, 14, 14]" = torch.ops.aten.mul.Tensor(add_136, sub_71);  sub_71 = None
    sum_41: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_308, [0, 2, 3]);  mul_308 = None
    mul_313: "f32[96]" = torch.ops.aten.mul.Tensor(rsqrt_19, primals_98);  primals_98 = None
    unsqueeze_653: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_313, 0);  mul_313 = None
    unsqueeze_654: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 2);  unsqueeze_653 = None
    unsqueeze_655: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, 3);  unsqueeze_654 = None
    mul_314: "f32[4, 96, 14, 14]" = torch.ops.aten.mul.Tensor(add_136, unsqueeze_655);  add_136 = unsqueeze_655 = None
    mul_315: "f32[96]" = torch.ops.aten.mul.Tensor(sum_41, rsqrt_19);  sum_41 = rsqrt_19 = None
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_314, clamp_max_21, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_314 = clamp_max_21 = primals_97 = None
    getitem_57: "f32[4, 384, 14, 14]" = convolution_backward_19[0]
    getitem_58: "f32[96, 384, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    where_13: "f32[4, 384, 14, 14]" = torch.ops.aten.where.self(bitwise_or_13, full_default, getitem_57);  bitwise_or_13 = getitem_57 = None
    add_138: "f32[384]" = torch.ops.aten.add.Tensor(primals_253, 1e-05);  primals_253 = None
    rsqrt_20: "f32[384]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
    unsqueeze_656: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_252, 0);  primals_252 = None
    unsqueeze_657: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 2);  unsqueeze_656 = None
    unsqueeze_658: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_657, 3);  unsqueeze_657 = None
    sum_42: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_72: "f32[4, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_658);  convolution_31 = unsqueeze_658 = None
    mul_316: "f32[4, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_13, sub_72);  sub_72 = None
    sum_43: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_316, [0, 2, 3]);  mul_316 = None
    mul_321: "f32[384]" = torch.ops.aten.mul.Tensor(rsqrt_20, primals_95);  primals_95 = None
    unsqueeze_665: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_321, 0);  mul_321 = None
    unsqueeze_666: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 2);  unsqueeze_665 = None
    unsqueeze_667: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, 3);  unsqueeze_666 = None
    mul_322: "f32[4, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_13, unsqueeze_667);  where_13 = unsqueeze_667 = None
    mul_323: "f32[384]" = torch.ops.aten.mul.Tensor(sum_43, rsqrt_20);  sum_43 = rsqrt_20 = None
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_322, clamp_max_20, primals_94, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 384, [True, True, False]);  mul_322 = clamp_max_20 = primals_94 = None
    getitem_60: "f32[4, 384, 14, 14]" = convolution_backward_20[0]
    getitem_61: "f32[384, 1, 3, 3]" = convolution_backward_20[1];  convolution_backward_20 = None
    where_14: "f32[4, 384, 14, 14]" = torch.ops.aten.where.self(bitwise_or_14, full_default, getitem_60);  bitwise_or_14 = getitem_60 = None
    add_139: "f32[384]" = torch.ops.aten.add.Tensor(primals_250, 1e-05);  primals_250 = None
    rsqrt_21: "f32[384]" = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
    unsqueeze_668: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_249, 0);  primals_249 = None
    unsqueeze_669: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 2);  unsqueeze_668 = None
    unsqueeze_670: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_669, 3);  unsqueeze_669 = None
    sum_44: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_73: "f32[4, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_670);  convolution_30 = unsqueeze_670 = None
    mul_324: "f32[4, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, sub_73);  sub_73 = None
    sum_45: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_324, [0, 2, 3]);  mul_324 = None
    mul_329: "f32[384]" = torch.ops.aten.mul.Tensor(rsqrt_21, primals_92);  primals_92 = None
    unsqueeze_677: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_329, 0);  mul_329 = None
    unsqueeze_678: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 2);  unsqueeze_677 = None
    unsqueeze_679: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, 3);  unsqueeze_678 = None
    mul_330: "f32[4, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, unsqueeze_679);  where_14 = unsqueeze_679 = None
    mul_331: "f32[384]" = torch.ops.aten.mul.Tensor(sum_45, rsqrt_21);  sum_45 = rsqrt_21 = None
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_330, add_65, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_330 = add_65 = primals_91 = None
    getitem_63: "f32[4, 64, 14, 14]" = convolution_backward_21[0]
    getitem_64: "f32[384, 64, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:62, code: return x + self.conv(x)
    add_140: "f32[64]" = torch.ops.aten.add.Tensor(primals_247, 1e-05);  primals_247 = None
    rsqrt_22: "f32[64]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    unsqueeze_680: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_246, 0);  primals_246 = None
    unsqueeze_681: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 2);  unsqueeze_680 = None
    unsqueeze_682: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_681, 3);  unsqueeze_681 = None
    sum_46: "f32[64]" = torch.ops.aten.sum.dim_IntList(getitem_63, [0, 2, 3])
    sub_74: "f32[4, 64, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_682);  convolution_29 = unsqueeze_682 = None
    mul_332: "f32[4, 64, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_63, sub_74);  sub_74 = None
    sum_47: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_332, [0, 2, 3]);  mul_332 = None
    mul_337: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_22, primals_89);  primals_89 = None
    unsqueeze_689: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_337, 0);  mul_337 = None
    unsqueeze_690: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 2);  unsqueeze_689 = None
    unsqueeze_691: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, 3);  unsqueeze_690 = None
    mul_338: "f32[4, 64, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_63, unsqueeze_691);  unsqueeze_691 = None
    mul_339: "f32[64]" = torch.ops.aten.mul.Tensor(sum_47, rsqrt_22);  sum_47 = rsqrt_22 = None
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_338, clamp_max_19, primals_88, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_338 = clamp_max_19 = primals_88 = None
    getitem_66: "f32[4, 384, 14, 14]" = convolution_backward_22[0]
    getitem_67: "f32[64, 384, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    where_15: "f32[4, 384, 14, 14]" = torch.ops.aten.where.self(bitwise_or_15, full_default, getitem_66);  bitwise_or_15 = getitem_66 = None
    add_141: "f32[384]" = torch.ops.aten.add.Tensor(primals_244, 1e-05);  primals_244 = None
    rsqrt_23: "f32[384]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
    unsqueeze_692: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_243, 0);  primals_243 = None
    unsqueeze_693: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 2);  unsqueeze_692 = None
    unsqueeze_694: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_693, 3);  unsqueeze_693 = None
    sum_48: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_75: "f32[4, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_694);  convolution_28 = unsqueeze_694 = None
    mul_340: "f32[4, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, sub_75);  sub_75 = None
    sum_49: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_340, [0, 2, 3]);  mul_340 = None
    mul_345: "f32[384]" = torch.ops.aten.mul.Tensor(rsqrt_23, primals_86);  primals_86 = None
    unsqueeze_701: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_345, 0);  mul_345 = None
    unsqueeze_702: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 2);  unsqueeze_701 = None
    unsqueeze_703: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, 3);  unsqueeze_702 = None
    mul_346: "f32[4, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, unsqueeze_703);  where_15 = unsqueeze_703 = None
    mul_347: "f32[384]" = torch.ops.aten.mul.Tensor(sum_49, rsqrt_23);  sum_49 = rsqrt_23 = None
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_346, clamp_max_18, primals_85, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 384, [True, True, False]);  mul_346 = clamp_max_18 = primals_85 = None
    getitem_69: "f32[4, 384, 14, 14]" = convolution_backward_23[0]
    getitem_70: "f32[384, 1, 3, 3]" = convolution_backward_23[1];  convolution_backward_23 = None
    where_16: "f32[4, 384, 14, 14]" = torch.ops.aten.where.self(bitwise_or_16, full_default, getitem_69);  bitwise_or_16 = getitem_69 = None
    add_142: "f32[384]" = torch.ops.aten.add.Tensor(primals_241, 1e-05);  primals_241 = None
    rsqrt_24: "f32[384]" = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
    unsqueeze_704: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_240, 0);  primals_240 = None
    unsqueeze_705: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 2);  unsqueeze_704 = None
    unsqueeze_706: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_705, 3);  unsqueeze_705 = None
    sum_50: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_76: "f32[4, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_706);  convolution_27 = unsqueeze_706 = None
    mul_348: "f32[4, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_16, sub_76);  sub_76 = None
    sum_51: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_348, [0, 2, 3]);  mul_348 = None
    mul_353: "f32[384]" = torch.ops.aten.mul.Tensor(rsqrt_24, primals_83);  primals_83 = None
    unsqueeze_713: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_353, 0);  mul_353 = None
    unsqueeze_714: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 2);  unsqueeze_713 = None
    unsqueeze_715: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, 3);  unsqueeze_714 = None
    mul_354: "f32[4, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_16, unsqueeze_715);  where_16 = unsqueeze_715 = None
    mul_355: "f32[384]" = torch.ops.aten.mul.Tensor(sum_51, rsqrt_24);  sum_51 = rsqrt_24 = None
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_354, add_58, primals_82, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_354 = add_58 = primals_82 = None
    getitem_72: "f32[4, 64, 14, 14]" = convolution_backward_24[0]
    getitem_73: "f32[384, 64, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:62, code: return x + self.conv(x)
    add_143: "f32[4, 64, 14, 14]" = torch.ops.aten.add.Tensor(getitem_63, getitem_72);  getitem_63 = getitem_72 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:62, code: return x + self.conv(x)
    add_144: "f32[64]" = torch.ops.aten.add.Tensor(primals_238, 1e-05);  primals_238 = None
    rsqrt_25: "f32[64]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    unsqueeze_716: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_237, 0);  primals_237 = None
    unsqueeze_717: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 2);  unsqueeze_716 = None
    unsqueeze_718: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_717, 3);  unsqueeze_717 = None
    sum_52: "f32[64]" = torch.ops.aten.sum.dim_IntList(add_143, [0, 2, 3])
    sub_77: "f32[4, 64, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_718);  convolution_26 = unsqueeze_718 = None
    mul_356: "f32[4, 64, 14, 14]" = torch.ops.aten.mul.Tensor(add_143, sub_77);  sub_77 = None
    sum_53: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_356, [0, 2, 3]);  mul_356 = None
    mul_361: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_25, primals_80);  primals_80 = None
    unsqueeze_725: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_361, 0);  mul_361 = None
    unsqueeze_726: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 2);  unsqueeze_725 = None
    unsqueeze_727: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, 3);  unsqueeze_726 = None
    mul_362: "f32[4, 64, 14, 14]" = torch.ops.aten.mul.Tensor(add_143, unsqueeze_727);  unsqueeze_727 = None
    mul_363: "f32[64]" = torch.ops.aten.mul.Tensor(sum_53, rsqrt_25);  sum_53 = rsqrt_25 = None
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_362, clamp_max_17, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_362 = clamp_max_17 = primals_79 = None
    getitem_75: "f32[4, 384, 14, 14]" = convolution_backward_25[0]
    getitem_76: "f32[64, 384, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    where_17: "f32[4, 384, 14, 14]" = torch.ops.aten.where.self(bitwise_or_17, full_default, getitem_75);  bitwise_or_17 = getitem_75 = None
    add_145: "f32[384]" = torch.ops.aten.add.Tensor(primals_235, 1e-05);  primals_235 = None
    rsqrt_26: "f32[384]" = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
    unsqueeze_728: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_234, 0);  primals_234 = None
    unsqueeze_729: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 2);  unsqueeze_728 = None
    unsqueeze_730: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_729, 3);  unsqueeze_729 = None
    sum_54: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_78: "f32[4, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_730);  convolution_25 = unsqueeze_730 = None
    mul_364: "f32[4, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_17, sub_78);  sub_78 = None
    sum_55: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_364, [0, 2, 3]);  mul_364 = None
    mul_369: "f32[384]" = torch.ops.aten.mul.Tensor(rsqrt_26, primals_77);  primals_77 = None
    unsqueeze_737: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_369, 0);  mul_369 = None
    unsqueeze_738: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 2);  unsqueeze_737 = None
    unsqueeze_739: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, 3);  unsqueeze_738 = None
    mul_370: "f32[4, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_17, unsqueeze_739);  where_17 = unsqueeze_739 = None
    mul_371: "f32[384]" = torch.ops.aten.mul.Tensor(sum_55, rsqrt_26);  sum_55 = rsqrt_26 = None
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_370, clamp_max_16, primals_76, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 384, [True, True, False]);  mul_370 = clamp_max_16 = primals_76 = None
    getitem_78: "f32[4, 384, 14, 14]" = convolution_backward_26[0]
    getitem_79: "f32[384, 1, 3, 3]" = convolution_backward_26[1];  convolution_backward_26 = None
    where_18: "f32[4, 384, 14, 14]" = torch.ops.aten.where.self(bitwise_or_18, full_default, getitem_78);  bitwise_or_18 = getitem_78 = None
    add_146: "f32[384]" = torch.ops.aten.add.Tensor(primals_232, 1e-05);  primals_232 = None
    rsqrt_27: "f32[384]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
    unsqueeze_740: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_231, 0);  primals_231 = None
    unsqueeze_741: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 2);  unsqueeze_740 = None
    unsqueeze_742: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_741, 3);  unsqueeze_741 = None
    sum_56: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    sub_79: "f32[4, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_742);  convolution_24 = unsqueeze_742 = None
    mul_372: "f32[4, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_18, sub_79);  sub_79 = None
    sum_57: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_372, [0, 2, 3]);  mul_372 = None
    mul_377: "f32[384]" = torch.ops.aten.mul.Tensor(rsqrt_27, primals_74);  primals_74 = None
    unsqueeze_749: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_377, 0);  mul_377 = None
    unsqueeze_750: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 2);  unsqueeze_749 = None
    unsqueeze_751: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, 3);  unsqueeze_750 = None
    mul_378: "f32[4, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_18, unsqueeze_751);  where_18 = unsqueeze_751 = None
    mul_379: "f32[384]" = torch.ops.aten.mul.Tensor(sum_57, rsqrt_27);  sum_57 = rsqrt_27 = None
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_378, add_51, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_378 = add_51 = primals_73 = None
    getitem_81: "f32[4, 64, 14, 14]" = convolution_backward_27[0]
    getitem_82: "f32[384, 64, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:62, code: return x + self.conv(x)
    add_147: "f32[4, 64, 14, 14]" = torch.ops.aten.add.Tensor(add_143, getitem_81);  add_143 = getitem_81 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:62, code: return x + self.conv(x)
    add_148: "f32[64]" = torch.ops.aten.add.Tensor(primals_229, 1e-05);  primals_229 = None
    rsqrt_28: "f32[64]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
    unsqueeze_752: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_228, 0);  primals_228 = None
    unsqueeze_753: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 2);  unsqueeze_752 = None
    unsqueeze_754: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_753, 3);  unsqueeze_753 = None
    sum_58: "f32[64]" = torch.ops.aten.sum.dim_IntList(add_147, [0, 2, 3])
    sub_80: "f32[4, 64, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_754);  convolution_23 = unsqueeze_754 = None
    mul_380: "f32[4, 64, 14, 14]" = torch.ops.aten.mul.Tensor(add_147, sub_80);  sub_80 = None
    sum_59: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_380, [0, 2, 3]);  mul_380 = None
    mul_385: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_28, primals_71);  primals_71 = None
    unsqueeze_761: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_385, 0);  mul_385 = None
    unsqueeze_762: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 2);  unsqueeze_761 = None
    unsqueeze_763: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, 3);  unsqueeze_762 = None
    mul_386: "f32[4, 64, 14, 14]" = torch.ops.aten.mul.Tensor(add_147, unsqueeze_763);  unsqueeze_763 = None
    mul_387: "f32[64]" = torch.ops.aten.mul.Tensor(sum_59, rsqrt_28);  sum_59 = rsqrt_28 = None
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_386, clamp_max_15, primals_70, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_386 = clamp_max_15 = primals_70 = None
    getitem_84: "f32[4, 384, 14, 14]" = convolution_backward_28[0]
    getitem_85: "f32[64, 384, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
    where_19: "f32[4, 384, 14, 14]" = torch.ops.aten.where.self(bitwise_or_19, full_default, getitem_84);  bitwise_or_19 = getitem_84 = None
    add_149: "f32[384]" = torch.ops.aten.add.Tensor(primals_226, 1e-05);  primals_226 = None
    rsqrt_29: "f32[384]" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
    unsqueeze_764: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_225, 0);  primals_225 = None
    unsqueeze_765: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 2);  unsqueeze_764 = None
    unsqueeze_766: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_765, 3);  unsqueeze_765 = None
    sum_60: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_81: "f32[4, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_766);  convolution_22 = unsqueeze_766 = None
    mul_388: "f32[4, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_19, sub_81);  sub_81 = None
    sum_61: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_388, [0, 2, 3]);  mul_388 = None
    mul_393: "f32[384]" = torch.ops.aten.mul.Tensor(rsqrt_29, primals_68);  primals_68 = None
    unsqueeze_773: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_393, 0);  mul_393 = None
    unsqueeze_774: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 2);  unsqueeze_773 = None
    unsqueeze_775: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, 3);  unsqueeze_774 = None
    mul_394: "f32[4, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_19, unsqueeze_775);  where_19 = unsqueeze_775 = None
    mul_395: "f32[384]" = torch.ops.aten.mul.Tensor(sum_61, rsqrt_29);  sum_61 = rsqrt_29 = None
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_394, clamp_max_14, primals_67, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 384, [True, True, False]);  mul_394 = clamp_max_14 = primals_67 = None
    getitem_87: "f32[4, 384, 14, 14]" = convolution_backward_29[0]
    getitem_88: "f32[384, 1, 3, 3]" = convolution_backward_29[1];  convolution_backward_29 = None
    where_20: "f32[4, 384, 14, 14]" = torch.ops.aten.where.self(bitwise_or_20, full_default, getitem_87);  bitwise_or_20 = getitem_87 = None
    add_150: "f32[384]" = torch.ops.aten.add.Tensor(primals_223, 1e-05);  primals_223 = None
    rsqrt_30: "f32[384]" = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
    unsqueeze_776: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_222, 0);  primals_222 = None
    unsqueeze_777: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 2);  unsqueeze_776 = None
    unsqueeze_778: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_777, 3);  unsqueeze_777 = None
    sum_62: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    sub_82: "f32[4, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_778);  convolution_21 = unsqueeze_778 = None
    mul_396: "f32[4, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_20, sub_82);  sub_82 = None
    sum_63: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_396, [0, 2, 3]);  mul_396 = None
    mul_401: "f32[384]" = torch.ops.aten.mul.Tensor(rsqrt_30, primals_65);  primals_65 = None
    unsqueeze_785: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_401, 0);  mul_401 = None
    unsqueeze_786: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 2);  unsqueeze_785 = None
    unsqueeze_787: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, 3);  unsqueeze_786 = None
    mul_402: "f32[4, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_20, unsqueeze_787);  where_20 = unsqueeze_787 = None
    mul_403: "f32[384]" = torch.ops.aten.mul.Tensor(sum_63, rsqrt_30);  sum_63 = rsqrt_30 = None
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_402, add_44, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_402 = add_44 = primals_64 = None
    getitem_90: "f32[4, 64, 14, 14]" = convolution_backward_30[0]
    getitem_91: "f32[384, 64, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:62, code: return x + self.conv(x)
    add_151: "f32[4, 64, 14, 14]" = torch.ops.aten.add.Tensor(add_147, getitem_90);  add_147 = getitem_90 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:64, code: return self.conv(x)
    add_152: "f32[64]" = torch.ops.aten.add.Tensor(primals_220, 1e-05);  primals_220 = None
    rsqrt_31: "f32[64]" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
    unsqueeze_788: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_219, 0);  primals_219 = None
    unsqueeze_789: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, 2);  unsqueeze_788 = None
    unsqueeze_790: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_789, 3);  unsqueeze_789 = None
    sum_64: "f32[64]" = torch.ops.aten.sum.dim_IntList(add_151, [0, 2, 3])
    sub_83: "f32[4, 64, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_790);  convolution_20 = unsqueeze_790 = None
    mul_404: "f32[4, 64, 14, 14]" = torch.ops.aten.mul.Tensor(add_151, sub_83);  sub_83 = None
    sum_65: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_404, [0, 2, 3]);  mul_404 = None
    mul_409: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_31, primals_62);  primals_62 = None
    unsqueeze_797: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_409, 0);  mul_409 = None
    unsqueeze_798: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_797, 2);  unsqueeze_797 = None
    unsqueeze_799: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, 3);  unsqueeze_798 = None
    mul_410: "f32[4, 64, 14, 14]" = torch.ops.aten.mul.Tensor(add_151, unsqueeze_799);  add_151 = unsqueeze_799 = None
    mul_411: "f32[64]" = torch.ops.aten.mul.Tensor(sum_65, rsqrt_31);  sum_65 = rsqrt_31 = None
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_410, clamp_max_13, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_410 = clamp_max_13 = primals_61 = None
    getitem_93: "f32[4, 192, 14, 14]" = convolution_backward_31[0]
    getitem_94: "f32[64, 192, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    where_21: "f32[4, 192, 14, 14]" = torch.ops.aten.where.self(bitwise_or_21, full_default, getitem_93);  bitwise_or_21 = getitem_93 = None
    add_153: "f32[192]" = torch.ops.aten.add.Tensor(primals_217, 1e-05);  primals_217 = None
    rsqrt_32: "f32[192]" = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
    unsqueeze_800: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_216, 0);  primals_216 = None
    unsqueeze_801: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 2);  unsqueeze_800 = None
    unsqueeze_802: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_801, 3);  unsqueeze_801 = None
    sum_66: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_84: "f32[4, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_802);  convolution_19 = unsqueeze_802 = None
    mul_412: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_21, sub_84);  sub_84 = None
    sum_67: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_412, [0, 2, 3]);  mul_412 = None
    mul_417: "f32[192]" = torch.ops.aten.mul.Tensor(rsqrt_32, primals_59);  primals_59 = None
    unsqueeze_809: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_417, 0);  mul_417 = None
    unsqueeze_810: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 2);  unsqueeze_809 = None
    unsqueeze_811: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, 3);  unsqueeze_810 = None
    mul_418: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_21, unsqueeze_811);  where_21 = unsqueeze_811 = None
    mul_419: "f32[192]" = torch.ops.aten.mul.Tensor(sum_67, rsqrt_32);  sum_67 = rsqrt_32 = None
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_418, clamp_max_12, primals_58, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False]);  mul_418 = clamp_max_12 = primals_58 = None
    getitem_96: "f32[4, 192, 28, 28]" = convolution_backward_32[0]
    getitem_97: "f32[192, 1, 3, 3]" = convolution_backward_32[1];  convolution_backward_32 = None
    where_22: "f32[4, 192, 28, 28]" = torch.ops.aten.where.self(bitwise_or_22, full_default, getitem_96);  bitwise_or_22 = getitem_96 = None
    add_154: "f32[192]" = torch.ops.aten.add.Tensor(primals_214, 1e-05);  primals_214 = None
    rsqrt_33: "f32[192]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
    unsqueeze_812: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_213, 0);  primals_213 = None
    unsqueeze_813: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, 2);  unsqueeze_812 = None
    unsqueeze_814: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_813, 3);  unsqueeze_813 = None
    sum_68: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    sub_85: "f32[4, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_814);  convolution_18 = unsqueeze_814 = None
    mul_420: "f32[4, 192, 28, 28]" = torch.ops.aten.mul.Tensor(where_22, sub_85);  sub_85 = None
    sum_69: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_420, [0, 2, 3]);  mul_420 = None
    mul_425: "f32[192]" = torch.ops.aten.mul.Tensor(rsqrt_33, primals_56);  primals_56 = None
    unsqueeze_821: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_425, 0);  mul_425 = None
    unsqueeze_822: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 2);  unsqueeze_821 = None
    unsqueeze_823: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, 3);  unsqueeze_822 = None
    mul_426: "f32[4, 192, 28, 28]" = torch.ops.aten.mul.Tensor(where_22, unsqueeze_823);  where_22 = unsqueeze_823 = None
    mul_427: "f32[192]" = torch.ops.aten.mul.Tensor(sum_69, rsqrt_33);  sum_69 = rsqrt_33 = None
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_426, add_38, primals_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_426 = add_38 = primals_55 = None
    getitem_99: "f32[4, 32, 28, 28]" = convolution_backward_33[0]
    getitem_100: "f32[192, 32, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:62, code: return x + self.conv(x)
    add_155: "f32[32]" = torch.ops.aten.add.Tensor(primals_211, 1e-05);  primals_211 = None
    rsqrt_34: "f32[32]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
    unsqueeze_824: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(primals_210, 0);  primals_210 = None
    unsqueeze_825: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, 2);  unsqueeze_824 = None
    unsqueeze_826: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_825, 3);  unsqueeze_825 = None
    sum_70: "f32[32]" = torch.ops.aten.sum.dim_IntList(getitem_99, [0, 2, 3])
    sub_86: "f32[4, 32, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_826);  convolution_17 = unsqueeze_826 = None
    mul_428: "f32[4, 32, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_99, sub_86);  sub_86 = None
    sum_71: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_428, [0, 2, 3]);  mul_428 = None
    mul_433: "f32[32]" = torch.ops.aten.mul.Tensor(rsqrt_34, primals_53);  primals_53 = None
    unsqueeze_833: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_433, 0);  mul_433 = None
    unsqueeze_834: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_833, 2);  unsqueeze_833 = None
    unsqueeze_835: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, 3);  unsqueeze_834 = None
    mul_434: "f32[4, 32, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_99, unsqueeze_835);  unsqueeze_835 = None
    mul_435: "f32[32]" = torch.ops.aten.mul.Tensor(sum_71, rsqrt_34);  sum_71 = rsqrt_34 = None
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_434, clamp_max_11, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_434 = clamp_max_11 = primals_52 = None
    getitem_102: "f32[4, 192, 28, 28]" = convolution_backward_34[0]
    getitem_103: "f32[32, 192, 1, 1]" = convolution_backward_34[1];  convolution_backward_34 = None
    where_23: "f32[4, 192, 28, 28]" = torch.ops.aten.where.self(bitwise_or_23, full_default, getitem_102);  bitwise_or_23 = getitem_102 = None
    add_156: "f32[192]" = torch.ops.aten.add.Tensor(primals_208, 1e-05);  primals_208 = None
    rsqrt_35: "f32[192]" = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
    unsqueeze_836: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_207, 0);  primals_207 = None
    unsqueeze_837: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, 2);  unsqueeze_836 = None
    unsqueeze_838: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_837, 3);  unsqueeze_837 = None
    sum_72: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_87: "f32[4, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_838);  convolution_16 = unsqueeze_838 = None
    mul_436: "f32[4, 192, 28, 28]" = torch.ops.aten.mul.Tensor(where_23, sub_87);  sub_87 = None
    sum_73: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_436, [0, 2, 3]);  mul_436 = None
    mul_441: "f32[192]" = torch.ops.aten.mul.Tensor(rsqrt_35, primals_50);  primals_50 = None
    unsqueeze_845: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_441, 0);  mul_441 = None
    unsqueeze_846: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_845, 2);  unsqueeze_845 = None
    unsqueeze_847: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, 3);  unsqueeze_846 = None
    mul_442: "f32[4, 192, 28, 28]" = torch.ops.aten.mul.Tensor(where_23, unsqueeze_847);  where_23 = unsqueeze_847 = None
    mul_443: "f32[192]" = torch.ops.aten.mul.Tensor(sum_73, rsqrt_35);  sum_73 = rsqrt_35 = None
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_442, clamp_max_10, primals_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False]);  mul_442 = clamp_max_10 = primals_49 = None
    getitem_105: "f32[4, 192, 28, 28]" = convolution_backward_35[0]
    getitem_106: "f32[192, 1, 3, 3]" = convolution_backward_35[1];  convolution_backward_35 = None
    where_24: "f32[4, 192, 28, 28]" = torch.ops.aten.where.self(bitwise_or_24, full_default, getitem_105);  bitwise_or_24 = getitem_105 = None
    add_157: "f32[192]" = torch.ops.aten.add.Tensor(primals_205, 1e-05);  primals_205 = None
    rsqrt_36: "f32[192]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
    unsqueeze_848: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_204, 0);  primals_204 = None
    unsqueeze_849: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, 2);  unsqueeze_848 = None
    unsqueeze_850: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_849, 3);  unsqueeze_849 = None
    sum_74: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_88: "f32[4, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_850);  convolution_15 = unsqueeze_850 = None
    mul_444: "f32[4, 192, 28, 28]" = torch.ops.aten.mul.Tensor(where_24, sub_88);  sub_88 = None
    sum_75: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_444, [0, 2, 3]);  mul_444 = None
    mul_449: "f32[192]" = torch.ops.aten.mul.Tensor(rsqrt_36, primals_47);  primals_47 = None
    unsqueeze_857: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_449, 0);  mul_449 = None
    unsqueeze_858: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_857, 2);  unsqueeze_857 = None
    unsqueeze_859: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, 3);  unsqueeze_858 = None
    mul_450: "f32[4, 192, 28, 28]" = torch.ops.aten.mul.Tensor(where_24, unsqueeze_859);  where_24 = unsqueeze_859 = None
    mul_451: "f32[192]" = torch.ops.aten.mul.Tensor(sum_75, rsqrt_36);  sum_75 = rsqrt_36 = None
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_450, add_31, primals_46, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_450 = add_31 = primals_46 = None
    getitem_108: "f32[4, 32, 28, 28]" = convolution_backward_36[0]
    getitem_109: "f32[192, 32, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:62, code: return x + self.conv(x)
    add_158: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(getitem_99, getitem_108);  getitem_99 = getitem_108 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:62, code: return x + self.conv(x)
    add_159: "f32[32]" = torch.ops.aten.add.Tensor(primals_202, 1e-05);  primals_202 = None
    rsqrt_37: "f32[32]" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
    unsqueeze_860: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(primals_201, 0);  primals_201 = None
    unsqueeze_861: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_860, 2);  unsqueeze_860 = None
    unsqueeze_862: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_861, 3);  unsqueeze_861 = None
    sum_76: "f32[32]" = torch.ops.aten.sum.dim_IntList(add_158, [0, 2, 3])
    sub_89: "f32[4, 32, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_862);  convolution_14 = unsqueeze_862 = None
    mul_452: "f32[4, 32, 28, 28]" = torch.ops.aten.mul.Tensor(add_158, sub_89);  sub_89 = None
    sum_77: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_452, [0, 2, 3]);  mul_452 = None
    mul_457: "f32[32]" = torch.ops.aten.mul.Tensor(rsqrt_37, primals_44);  primals_44 = None
    unsqueeze_869: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_457, 0);  mul_457 = None
    unsqueeze_870: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_869, 2);  unsqueeze_869 = None
    unsqueeze_871: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_870, 3);  unsqueeze_870 = None
    mul_458: "f32[4, 32, 28, 28]" = torch.ops.aten.mul.Tensor(add_158, unsqueeze_871);  unsqueeze_871 = None
    mul_459: "f32[32]" = torch.ops.aten.mul.Tensor(sum_77, rsqrt_37);  sum_77 = rsqrt_37 = None
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_458, clamp_max_9, primals_43, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_458 = clamp_max_9 = primals_43 = None
    getitem_111: "f32[4, 192, 28, 28]" = convolution_backward_37[0]
    getitem_112: "f32[32, 192, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
    where_25: "f32[4, 192, 28, 28]" = torch.ops.aten.where.self(bitwise_or_25, full_default, getitem_111);  bitwise_or_25 = getitem_111 = None
    add_160: "f32[192]" = torch.ops.aten.add.Tensor(primals_199, 1e-05);  primals_199 = None
    rsqrt_38: "f32[192]" = torch.ops.aten.rsqrt.default(add_160);  add_160 = None
    unsqueeze_872: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_198, 0);  primals_198 = None
    unsqueeze_873: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, 2);  unsqueeze_872 = None
    unsqueeze_874: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_873, 3);  unsqueeze_873 = None
    sum_78: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_90: "f32[4, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_874);  convolution_13 = unsqueeze_874 = None
    mul_460: "f32[4, 192, 28, 28]" = torch.ops.aten.mul.Tensor(where_25, sub_90);  sub_90 = None
    sum_79: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_460, [0, 2, 3]);  mul_460 = None
    mul_465: "f32[192]" = torch.ops.aten.mul.Tensor(rsqrt_38, primals_41);  primals_41 = None
    unsqueeze_881: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_465, 0);  mul_465 = None
    unsqueeze_882: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_881, 2);  unsqueeze_881 = None
    unsqueeze_883: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, 3);  unsqueeze_882 = None
    mul_466: "f32[4, 192, 28, 28]" = torch.ops.aten.mul.Tensor(where_25, unsqueeze_883);  where_25 = unsqueeze_883 = None
    mul_467: "f32[192]" = torch.ops.aten.mul.Tensor(sum_79, rsqrt_38);  sum_79 = rsqrt_38 = None
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_466, clamp_max_8, primals_40, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False]);  mul_466 = clamp_max_8 = primals_40 = None
    getitem_114: "f32[4, 192, 28, 28]" = convolution_backward_38[0]
    getitem_115: "f32[192, 1, 3, 3]" = convolution_backward_38[1];  convolution_backward_38 = None
    where_26: "f32[4, 192, 28, 28]" = torch.ops.aten.where.self(bitwise_or_26, full_default, getitem_114);  bitwise_or_26 = getitem_114 = None
    add_161: "f32[192]" = torch.ops.aten.add.Tensor(primals_196, 1e-05);  primals_196 = None
    rsqrt_39: "f32[192]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    unsqueeze_884: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_195, 0);  primals_195 = None
    unsqueeze_885: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_884, 2);  unsqueeze_884 = None
    unsqueeze_886: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_885, 3);  unsqueeze_885 = None
    sum_80: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    sub_91: "f32[4, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_886);  convolution_12 = unsqueeze_886 = None
    mul_468: "f32[4, 192, 28, 28]" = torch.ops.aten.mul.Tensor(where_26, sub_91);  sub_91 = None
    sum_81: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_468, [0, 2, 3]);  mul_468 = None
    mul_473: "f32[192]" = torch.ops.aten.mul.Tensor(rsqrt_39, primals_38);  primals_38 = None
    unsqueeze_893: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_473, 0);  mul_473 = None
    unsqueeze_894: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_893, 2);  unsqueeze_893 = None
    unsqueeze_895: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, 3);  unsqueeze_894 = None
    mul_474: "f32[4, 192, 28, 28]" = torch.ops.aten.mul.Tensor(where_26, unsqueeze_895);  where_26 = unsqueeze_895 = None
    mul_475: "f32[192]" = torch.ops.aten.mul.Tensor(sum_81, rsqrt_39);  sum_81 = rsqrt_39 = None
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_474, add_24, primals_37, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_474 = add_24 = primals_37 = None
    getitem_117: "f32[4, 32, 28, 28]" = convolution_backward_39[0]
    getitem_118: "f32[192, 32, 1, 1]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:62, code: return x + self.conv(x)
    add_162: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_158, getitem_117);  add_158 = getitem_117 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:64, code: return self.conv(x)
    add_163: "f32[32]" = torch.ops.aten.add.Tensor(primals_193, 1e-05);  primals_193 = None
    rsqrt_40: "f32[32]" = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
    unsqueeze_896: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(primals_192, 0);  primals_192 = None
    unsqueeze_897: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, 2);  unsqueeze_896 = None
    unsqueeze_898: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_897, 3);  unsqueeze_897 = None
    sum_82: "f32[32]" = torch.ops.aten.sum.dim_IntList(add_162, [0, 2, 3])
    sub_92: "f32[4, 32, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_898);  convolution_11 = unsqueeze_898 = None
    mul_476: "f32[4, 32, 28, 28]" = torch.ops.aten.mul.Tensor(add_162, sub_92);  sub_92 = None
    sum_83: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_476, [0, 2, 3]);  mul_476 = None
    mul_481: "f32[32]" = torch.ops.aten.mul.Tensor(rsqrt_40, primals_35);  primals_35 = None
    unsqueeze_905: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_481, 0);  mul_481 = None
    unsqueeze_906: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_905, 2);  unsqueeze_905 = None
    unsqueeze_907: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, 3);  unsqueeze_906 = None
    mul_482: "f32[4, 32, 28, 28]" = torch.ops.aten.mul.Tensor(add_162, unsqueeze_907);  add_162 = unsqueeze_907 = None
    mul_483: "f32[32]" = torch.ops.aten.mul.Tensor(sum_83, rsqrt_40);  sum_83 = rsqrt_40 = None
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_482, clamp_max_7, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_482 = clamp_max_7 = primals_34 = None
    getitem_120: "f32[4, 144, 28, 28]" = convolution_backward_40[0]
    getitem_121: "f32[32, 144, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    where_27: "f32[4, 144, 28, 28]" = torch.ops.aten.where.self(bitwise_or_27, full_default, getitem_120);  bitwise_or_27 = getitem_120 = None
    add_164: "f32[144]" = torch.ops.aten.add.Tensor(primals_190, 1e-05);  primals_190 = None
    rsqrt_41: "f32[144]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
    unsqueeze_908: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(primals_189, 0);  primals_189 = None
    unsqueeze_909: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_908, 2);  unsqueeze_908 = None
    unsqueeze_910: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_909, 3);  unsqueeze_909 = None
    sum_84: "f32[144]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_93: "f32[4, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_910);  convolution_10 = unsqueeze_910 = None
    mul_484: "f32[4, 144, 28, 28]" = torch.ops.aten.mul.Tensor(where_27, sub_93);  sub_93 = None
    sum_85: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_484, [0, 2, 3]);  mul_484 = None
    mul_489: "f32[144]" = torch.ops.aten.mul.Tensor(rsqrt_41, primals_32);  primals_32 = None
    unsqueeze_917: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_489, 0);  mul_489 = None
    unsqueeze_918: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_917, 2);  unsqueeze_917 = None
    unsqueeze_919: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, 3);  unsqueeze_918 = None
    mul_490: "f32[4, 144, 28, 28]" = torch.ops.aten.mul.Tensor(where_27, unsqueeze_919);  where_27 = unsqueeze_919 = None
    mul_491: "f32[144]" = torch.ops.aten.mul.Tensor(sum_85, rsqrt_41);  sum_85 = rsqrt_41 = None
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_490, clamp_max_6, primals_31, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 144, [True, True, False]);  mul_490 = clamp_max_6 = primals_31 = None
    getitem_123: "f32[4, 144, 56, 56]" = convolution_backward_41[0]
    getitem_124: "f32[144, 1, 3, 3]" = convolution_backward_41[1];  convolution_backward_41 = None
    where_28: "f32[4, 144, 56, 56]" = torch.ops.aten.where.self(bitwise_or_28, full_default, getitem_123);  bitwise_or_28 = getitem_123 = None
    add_165: "f32[144]" = torch.ops.aten.add.Tensor(primals_187, 1e-05);  primals_187 = None
    rsqrt_42: "f32[144]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
    unsqueeze_920: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(primals_186, 0);  primals_186 = None
    unsqueeze_921: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_920, 2);  unsqueeze_920 = None
    unsqueeze_922: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_921, 3);  unsqueeze_921 = None
    sum_86: "f32[144]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
    sub_94: "f32[4, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_922);  convolution_9 = unsqueeze_922 = None
    mul_492: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(where_28, sub_94);  sub_94 = None
    sum_87: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_492, [0, 2, 3]);  mul_492 = None
    mul_497: "f32[144]" = torch.ops.aten.mul.Tensor(rsqrt_42, primals_29);  primals_29 = None
    unsqueeze_929: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_497, 0);  mul_497 = None
    unsqueeze_930: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_929, 2);  unsqueeze_929 = None
    unsqueeze_931: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_930, 3);  unsqueeze_930 = None
    mul_498: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(where_28, unsqueeze_931);  where_28 = unsqueeze_931 = None
    mul_499: "f32[144]" = torch.ops.aten.mul.Tensor(sum_87, rsqrt_42);  sum_87 = rsqrt_42 = None
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_498, add_18, primals_28, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_498 = add_18 = primals_28 = None
    getitem_126: "f32[4, 24, 56, 56]" = convolution_backward_42[0]
    getitem_127: "f32[144, 24, 1, 1]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:62, code: return x + self.conv(x)
    add_166: "f32[24]" = torch.ops.aten.add.Tensor(primals_184, 1e-05);  primals_184 = None
    rsqrt_43: "f32[24]" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
    unsqueeze_932: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(primals_183, 0);  primals_183 = None
    unsqueeze_933: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_932, 2);  unsqueeze_932 = None
    unsqueeze_934: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_933, 3);  unsqueeze_933 = None
    sum_88: "f32[24]" = torch.ops.aten.sum.dim_IntList(getitem_126, [0, 2, 3])
    sub_95: "f32[4, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_934);  convolution_8 = unsqueeze_934 = None
    mul_500: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_126, sub_95);  sub_95 = None
    sum_89: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_500, [0, 2, 3]);  mul_500 = None
    mul_505: "f32[24]" = torch.ops.aten.mul.Tensor(rsqrt_43, primals_26);  primals_26 = None
    unsqueeze_941: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_505, 0);  mul_505 = None
    unsqueeze_942: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_941, 2);  unsqueeze_941 = None
    unsqueeze_943: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_942, 3);  unsqueeze_942 = None
    mul_506: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_126, unsqueeze_943);  unsqueeze_943 = None
    mul_507: "f32[24]" = torch.ops.aten.mul.Tensor(sum_89, rsqrt_43);  sum_89 = rsqrt_43 = None
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_506, clamp_max_5, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_506 = clamp_max_5 = primals_25 = None
    getitem_129: "f32[4, 144, 56, 56]" = convolution_backward_43[0]
    getitem_130: "f32[24, 144, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None
    where_29: "f32[4, 144, 56, 56]" = torch.ops.aten.where.self(bitwise_or_29, full_default, getitem_129);  bitwise_or_29 = getitem_129 = None
    add_167: "f32[144]" = torch.ops.aten.add.Tensor(primals_181, 1e-05);  primals_181 = None
    rsqrt_44: "f32[144]" = torch.ops.aten.rsqrt.default(add_167);  add_167 = None
    unsqueeze_944: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(primals_180, 0);  primals_180 = None
    unsqueeze_945: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_944, 2);  unsqueeze_944 = None
    unsqueeze_946: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_945, 3);  unsqueeze_945 = None
    sum_90: "f32[144]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    sub_96: "f32[4, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_946);  convolution_7 = unsqueeze_946 = None
    mul_508: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(where_29, sub_96);  sub_96 = None
    sum_91: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_508, [0, 2, 3]);  mul_508 = None
    mul_513: "f32[144]" = torch.ops.aten.mul.Tensor(rsqrt_44, primals_23);  primals_23 = None
    unsqueeze_953: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_513, 0);  mul_513 = None
    unsqueeze_954: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_953, 2);  unsqueeze_953 = None
    unsqueeze_955: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_954, 3);  unsqueeze_954 = None
    mul_514: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(where_29, unsqueeze_955);  where_29 = unsqueeze_955 = None
    mul_515: "f32[144]" = torch.ops.aten.mul.Tensor(sum_91, rsqrt_44);  sum_91 = rsqrt_44 = None
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_514, clamp_max_4, primals_22, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 144, [True, True, False]);  mul_514 = clamp_max_4 = primals_22 = None
    getitem_132: "f32[4, 144, 56, 56]" = convolution_backward_44[0]
    getitem_133: "f32[144, 1, 3, 3]" = convolution_backward_44[1];  convolution_backward_44 = None
    where_30: "f32[4, 144, 56, 56]" = torch.ops.aten.where.self(bitwise_or_30, full_default, getitem_132);  bitwise_or_30 = getitem_132 = None
    add_168: "f32[144]" = torch.ops.aten.add.Tensor(primals_178, 1e-05);  primals_178 = None
    rsqrt_45: "f32[144]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
    unsqueeze_956: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(primals_177, 0);  primals_177 = None
    unsqueeze_957: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_956, 2);  unsqueeze_956 = None
    unsqueeze_958: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_957, 3);  unsqueeze_957 = None
    sum_92: "f32[144]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
    sub_97: "f32[4, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_958);  convolution_6 = unsqueeze_958 = None
    mul_516: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(where_30, sub_97);  sub_97 = None
    sum_93: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_516, [0, 2, 3]);  mul_516 = None
    mul_521: "f32[144]" = torch.ops.aten.mul.Tensor(rsqrt_45, primals_20);  primals_20 = None
    unsqueeze_965: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_521, 0);  mul_521 = None
    unsqueeze_966: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_965, 2);  unsqueeze_965 = None
    unsqueeze_967: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_966, 3);  unsqueeze_966 = None
    mul_522: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(where_30, unsqueeze_967);  where_30 = unsqueeze_967 = None
    mul_523: "f32[144]" = torch.ops.aten.mul.Tensor(sum_93, rsqrt_45);  sum_93 = rsqrt_45 = None
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_522, add_11, primals_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_522 = add_11 = primals_19 = None
    getitem_135: "f32[4, 24, 56, 56]" = convolution_backward_45[0]
    getitem_136: "f32[144, 24, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:62, code: return x + self.conv(x)
    add_169: "f32[4, 24, 56, 56]" = torch.ops.aten.add.Tensor(getitem_126, getitem_135);  getitem_126 = getitem_135 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:64, code: return self.conv(x)
    add_170: "f32[24]" = torch.ops.aten.add.Tensor(primals_175, 1e-05);  primals_175 = None
    rsqrt_46: "f32[24]" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
    unsqueeze_968: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(primals_174, 0);  primals_174 = None
    unsqueeze_969: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_968, 2);  unsqueeze_968 = None
    unsqueeze_970: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_969, 3);  unsqueeze_969 = None
    sum_94: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_169, [0, 2, 3])
    sub_98: "f32[4, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_970);  convolution_5 = unsqueeze_970 = None
    mul_524: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_169, sub_98);  sub_98 = None
    sum_95: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_524, [0, 2, 3]);  mul_524 = None
    mul_529: "f32[24]" = torch.ops.aten.mul.Tensor(rsqrt_46, primals_17);  primals_17 = None
    unsqueeze_977: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_529, 0);  mul_529 = None
    unsqueeze_978: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_977, 2);  unsqueeze_977 = None
    unsqueeze_979: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_978, 3);  unsqueeze_978 = None
    mul_530: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_169, unsqueeze_979);  add_169 = unsqueeze_979 = None
    mul_531: "f32[24]" = torch.ops.aten.mul.Tensor(sum_95, rsqrt_46);  sum_95 = rsqrt_46 = None
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_530, clamp_max_3, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_530 = clamp_max_3 = primals_16 = None
    getitem_138: "f32[4, 96, 56, 56]" = convolution_backward_46[0]
    getitem_139: "f32[24, 96, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    where_31: "f32[4, 96, 56, 56]" = torch.ops.aten.where.self(bitwise_or_31, full_default, getitem_138);  bitwise_or_31 = getitem_138 = None
    add_171: "f32[96]" = torch.ops.aten.add.Tensor(primals_172, 1e-05);  primals_172 = None
    rsqrt_47: "f32[96]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
    unsqueeze_980: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_171, 0);  primals_171 = None
    unsqueeze_981: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_980, 2);  unsqueeze_980 = None
    unsqueeze_982: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_981, 3);  unsqueeze_981 = None
    sum_96: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_99: "f32[4, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_982);  convolution_4 = unsqueeze_982 = None
    mul_532: "f32[4, 96, 56, 56]" = torch.ops.aten.mul.Tensor(where_31, sub_99);  sub_99 = None
    sum_97: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_532, [0, 2, 3]);  mul_532 = None
    mul_537: "f32[96]" = torch.ops.aten.mul.Tensor(rsqrt_47, primals_14);  primals_14 = None
    unsqueeze_989: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_537, 0);  mul_537 = None
    unsqueeze_990: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_989, 2);  unsqueeze_989 = None
    unsqueeze_991: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_990, 3);  unsqueeze_990 = None
    mul_538: "f32[4, 96, 56, 56]" = torch.ops.aten.mul.Tensor(where_31, unsqueeze_991);  where_31 = unsqueeze_991 = None
    mul_539: "f32[96]" = torch.ops.aten.mul.Tensor(sum_97, rsqrt_47);  sum_97 = rsqrt_47 = None
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_538, clamp_max_2, primals_13, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 96, [True, True, False]);  mul_538 = clamp_max_2 = primals_13 = None
    getitem_141: "f32[4, 96, 112, 112]" = convolution_backward_47[0]
    getitem_142: "f32[96, 1, 3, 3]" = convolution_backward_47[1];  convolution_backward_47 = None
    where_32: "f32[4, 96, 112, 112]" = torch.ops.aten.where.self(bitwise_or_32, full_default, getitem_141);  bitwise_or_32 = getitem_141 = None
    add_172: "f32[96]" = torch.ops.aten.add.Tensor(primals_169, 1e-05);  primals_169 = None
    rsqrt_48: "f32[96]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
    unsqueeze_992: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_168, 0);  primals_168 = None
    unsqueeze_993: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_992, 2);  unsqueeze_992 = None
    unsqueeze_994: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_993, 3);  unsqueeze_993 = None
    sum_98: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
    sub_100: "f32[4, 96, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_994);  convolution_3 = unsqueeze_994 = None
    mul_540: "f32[4, 96, 112, 112]" = torch.ops.aten.mul.Tensor(where_32, sub_100);  sub_100 = None
    sum_99: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_540, [0, 2, 3]);  mul_540 = None
    mul_545: "f32[96]" = torch.ops.aten.mul.Tensor(rsqrt_48, primals_11);  primals_11 = None
    unsqueeze_1001: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_545, 0);  mul_545 = None
    unsqueeze_1002: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1001, 2);  unsqueeze_1001 = None
    unsqueeze_1003: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1002, 3);  unsqueeze_1002 = None
    mul_546: "f32[4, 96, 112, 112]" = torch.ops.aten.mul.Tensor(where_32, unsqueeze_1003);  where_32 = unsqueeze_1003 = None
    mul_547: "f32[96]" = torch.ops.aten.mul.Tensor(sum_99, rsqrt_48);  sum_99 = rsqrt_48 = None
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_546, add_5, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_546 = add_5 = primals_10 = None
    getitem_144: "f32[4, 16, 112, 112]" = convolution_backward_48[0]
    getitem_145: "f32[96, 16, 1, 1]" = convolution_backward_48[1];  convolution_backward_48 = None
    add_173: "f32[16]" = torch.ops.aten.add.Tensor(primals_166, 1e-05);  primals_166 = None
    rsqrt_49: "f32[16]" = torch.ops.aten.rsqrt.default(add_173);  add_173 = None
    unsqueeze_1004: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(primals_165, 0);  primals_165 = None
    unsqueeze_1005: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1004, 2);  unsqueeze_1004 = None
    unsqueeze_1006: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1005, 3);  unsqueeze_1005 = None
    sum_100: "f32[16]" = torch.ops.aten.sum.dim_IntList(getitem_144, [0, 2, 3])
    sub_101: "f32[4, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_1006);  convolution_2 = unsqueeze_1006 = None
    mul_548: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_144, sub_101);  sub_101 = None
    sum_101: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_548, [0, 2, 3]);  mul_548 = None
    mul_553: "f32[16]" = torch.ops.aten.mul.Tensor(rsqrt_49, primals_8);  primals_8 = None
    unsqueeze_1013: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_553, 0);  mul_553 = None
    unsqueeze_1014: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1013, 2);  unsqueeze_1013 = None
    unsqueeze_1015: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1014, 3);  unsqueeze_1014 = None
    mul_554: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_144, unsqueeze_1015);  getitem_144 = unsqueeze_1015 = None
    mul_555: "f32[16]" = torch.ops.aten.mul.Tensor(sum_101, rsqrt_49);  sum_101 = rsqrt_49 = None
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_554, clamp_max_1, primals_7, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_554 = clamp_max_1 = primals_7 = None
    getitem_147: "f32[4, 32, 112, 112]" = convolution_backward_49[0]
    getitem_148: "f32[16, 32, 1, 1]" = convolution_backward_49[1];  convolution_backward_49 = None
    where_33: "f32[4, 32, 112, 112]" = torch.ops.aten.where.self(bitwise_or_33, full_default, getitem_147);  bitwise_or_33 = getitem_147 = None
    add_174: "f32[32]" = torch.ops.aten.add.Tensor(primals_163, 1e-05);  primals_163 = None
    rsqrt_50: "f32[32]" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
    unsqueeze_1016: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(primals_162, 0);  primals_162 = None
    unsqueeze_1017: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1016, 2);  unsqueeze_1016 = None
    unsqueeze_1018: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1017, 3);  unsqueeze_1017 = None
    sum_102: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
    sub_102: "f32[4, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_1018);  convolution_1 = unsqueeze_1018 = None
    mul_556: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_33, sub_102);  sub_102 = None
    sum_103: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_556, [0, 2, 3]);  mul_556 = None
    mul_561: "f32[32]" = torch.ops.aten.mul.Tensor(rsqrt_50, primals_5);  primals_5 = None
    unsqueeze_1025: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_561, 0);  mul_561 = None
    unsqueeze_1026: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1025, 2);  unsqueeze_1025 = None
    unsqueeze_1027: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1026, 3);  unsqueeze_1026 = None
    mul_562: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_33, unsqueeze_1027);  where_33 = unsqueeze_1027 = None
    mul_563: "f32[32]" = torch.ops.aten.mul.Tensor(sum_103, rsqrt_50);  sum_103 = rsqrt_50 = None
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_562, clamp_max, primals_4, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_562 = clamp_max = primals_4 = None
    getitem_150: "f32[4, 32, 112, 112]" = convolution_backward_50[0]
    getitem_151: "f32[32, 1, 3, 3]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:166, code: x = self.features(x)
    where_34: "f32[4, 32, 112, 112]" = torch.ops.aten.where.self(bitwise_or_34, full_default, getitem_150);  bitwise_or_34 = full_default = getitem_150 = None
    add_175: "f32[32]" = torch.ops.aten.add.Tensor(primals_160, 1e-05);  primals_160 = None
    rsqrt_51: "f32[32]" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
    unsqueeze_1028: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(primals_159, 0);  primals_159 = None
    unsqueeze_1029: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1028, 2);  unsqueeze_1028 = None
    unsqueeze_1030: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1029, 3);  unsqueeze_1029 = None
    sum_104: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
    sub_103: "f32[4, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1030);  convolution = unsqueeze_1030 = None
    mul_564: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_34, sub_103);  sub_103 = None
    sum_105: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_564, [0, 2, 3]);  mul_564 = None
    mul_569: "f32[32]" = torch.ops.aten.mul.Tensor(rsqrt_51, primals_2);  primals_2 = None
    unsqueeze_1037: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_569, 0);  mul_569 = None
    unsqueeze_1038: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1037, 2);  unsqueeze_1037 = None
    unsqueeze_1039: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1038, 3);  unsqueeze_1038 = None
    mul_570: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_34, unsqueeze_1039);  where_34 = unsqueeze_1039 = None
    mul_571: "f32[32]" = torch.ops.aten.mul.Tensor(sum_105, rsqrt_51);  sum_105 = rsqrt_51 = None
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_570, primals_315, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_570 = primals_315 = primals_1 = None
    getitem_154: "f32[32, 3, 3, 3]" = convolution_backward_51[1];  convolution_backward_51 = None
    return [getitem_154, mul_571, sum_104, getitem_151, mul_563, sum_102, getitem_148, mul_555, sum_100, getitem_145, mul_547, sum_98, getitem_142, mul_539, sum_96, getitem_139, mul_531, sum_94, getitem_136, mul_523, sum_92, getitem_133, mul_515, sum_90, getitem_130, mul_507, sum_88, getitem_127, mul_499, sum_86, getitem_124, mul_491, sum_84, getitem_121, mul_483, sum_82, getitem_118, mul_475, sum_80, getitem_115, mul_467, sum_78, getitem_112, mul_459, sum_76, getitem_109, mul_451, sum_74, getitem_106, mul_443, sum_72, getitem_103, mul_435, sum_70, getitem_100, mul_427, sum_68, getitem_97, mul_419, sum_66, getitem_94, mul_411, sum_64, getitem_91, mul_403, sum_62, getitem_88, mul_395, sum_60, getitem_85, mul_387, sum_58, getitem_82, mul_379, sum_56, getitem_79, mul_371, sum_54, getitem_76, mul_363, sum_52, getitem_73, mul_355, sum_50, getitem_70, mul_347, sum_48, getitem_67, mul_339, sum_46, getitem_64, mul_331, sum_44, getitem_61, mul_323, sum_42, getitem_58, mul_315, sum_40, getitem_55, mul_307, sum_38, getitem_52, mul_299, sum_36, getitem_49, mul_291, sum_34, getitem_46, mul_283, sum_32, getitem_43, mul_275, sum_30, getitem_40, mul_267, sum_28, getitem_37, mul_259, sum_26, getitem_34, mul_251, sum_24, getitem_31, mul_243, sum_22, getitem_28, mul_235, sum_20, getitem_25, mul_227, sum_18, getitem_22, mul_219, sum_16, getitem_19, mul_211, sum_14, getitem_16, mul_203, sum_12, getitem_13, mul_195, sum_10, getitem_10, mul_187, sum_8, getitem_7, mul_179, sum_6, getitem_4, mul_171, sum_4, getitem_1, mul_163, sum_2, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    