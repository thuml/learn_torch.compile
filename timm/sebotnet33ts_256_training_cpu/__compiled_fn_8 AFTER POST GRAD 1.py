from __future__ import annotations



def forward(self, primals_1: "f32[24]", primals_3: "f32[32]", primals_5: "f32[64]", primals_7: "f32[64]", primals_9: "f32[64]", primals_11: "f32[256]", primals_13: "f32[256]", primals_15: "f32[64]", primals_17: "f32[64]", primals_19: "f32[256]", primals_21: "f32[128]", primals_23: "f32[128]", primals_25: "f32[512]", primals_27: "f32[512]", primals_29: "f32[128]", primals_31: "f32[128]", primals_33: "f32[512]", primals_35: "f32[128]", primals_39: "f32[128]", primals_41: "f32[512]", primals_43: "f32[256]", primals_45: "f32[256]", primals_47: "f32[1024]", primals_49: "f32[1024]", primals_51: "f32[256]", primals_53: "f32[256]", primals_55: "f32[1024]", primals_57: "f32[256]", primals_61: "f32[256]", primals_63: "f32[1024]", primals_65: "f32[512]", primals_69: "f32[512]", primals_71: "f32[1536]", primals_73: "f32[1536]", primals_75: "f32[512]", primals_79: "f32[512]", primals_81: "f32[1536]", primals_83: "f32[1280]", primals_85: "f32[24, 3, 3, 3]", primals_86: "f32[32, 24, 3, 3]", primals_87: "f32[64, 32, 3, 3]", primals_88: "f32[64, 64, 1, 1]", primals_89: "f32[64, 64, 3, 3]", primals_90: "f32[8, 64, 1, 1]", primals_92: "f32[64, 8, 1, 1]", primals_94: "f32[256, 64, 1, 1]", primals_95: "f32[256, 64, 1, 1]", primals_96: "f32[64, 256, 1, 1]", primals_97: "f32[64, 64, 3, 3]", primals_98: "f32[8, 64, 1, 1]", primals_100: "f32[64, 8, 1, 1]", primals_102: "f32[256, 64, 1, 1]", primals_103: "f32[128, 256, 1, 1]", primals_104: "f32[128, 128, 3, 3]", primals_105: "f32[8, 128, 1, 1]", primals_107: "f32[128, 8, 1, 1]", primals_109: "f32[512, 128, 1, 1]", primals_110: "f32[512, 256, 1, 1]", primals_111: "f32[128, 512, 1, 1]", primals_112: "f32[128, 128, 3, 3]", primals_113: "f32[8, 128, 1, 1]", primals_115: "f32[128, 8, 1, 1]", primals_117: "f32[512, 128, 1, 1]", primals_118: "f32[128, 512, 1, 1]", primals_119: "f32[384, 128, 1, 1]", primals_120: "f32[512, 128, 1, 1]", primals_121: "f32[256, 512, 1, 1]", primals_122: "f32[256, 256, 3, 3]", primals_123: "f32[16, 256, 1, 1]", primals_125: "f32[256, 16, 1, 1]", primals_127: "f32[1024, 256, 1, 1]", primals_128: "f32[1024, 512, 1, 1]", primals_129: "f32[256, 1024, 1, 1]", primals_130: "f32[256, 256, 3, 3]", primals_131: "f32[16, 256, 1, 1]", primals_133: "f32[256, 16, 1, 1]", primals_135: "f32[1024, 256, 1, 1]", primals_136: "f32[256, 1024, 1, 1]", primals_137: "f32[768, 256, 1, 1]", primals_138: "f32[1024, 256, 1, 1]", primals_139: "f32[512, 1024, 1, 1]", primals_140: "f32[1536, 512, 1, 1]", primals_141: "f32[1536, 512, 1, 1]", primals_142: "f32[1536, 1024, 1, 1]", primals_143: "f32[512, 1536, 1, 1]", primals_144: "f32[1536, 512, 1, 1]", primals_145: "f32[1536, 512, 1, 1]", primals_146: "f32[1280, 1536, 1, 1]", primals_263: "f32[8, 3, 256, 256]", convolution: "f32[8, 24, 128, 128]", squeeze_1: "f32[24]", mul_7: "f32[8, 24, 128, 128]", convolution_1: "f32[8, 32, 128, 128]", squeeze_4: "f32[32]", mul_15: "f32[8, 32, 128, 128]", convolution_2: "f32[8, 64, 64, 64]", squeeze_7: "f32[64]", mul_23: "f32[8, 64, 64, 64]", convolution_3: "f32[8, 64, 64, 64]", squeeze_10: "f32[64]", mul_31: "f32[8, 64, 64, 64]", convolution_4: "f32[8, 64, 64, 64]", squeeze_13: "f32[64]", add_24: "f32[8, 64, 64, 64]", mean: "f32[8, 64, 1, 1]", relu: "f32[8, 8, 1, 1]", convolution_6: "f32[8, 64, 1, 1]", mul_40: "f32[8, 64, 64, 64]", convolution_7: "f32[8, 256, 64, 64]", squeeze_16: "f32[256]", convolution_8: "f32[8, 256, 64, 64]", squeeze_19: "f32[256]", mul_55: "f32[8, 256, 64, 64]", convolution_9: "f32[8, 64, 64, 64]", squeeze_22: "f32[64]", mul_63: "f32[8, 64, 64, 64]", convolution_10: "f32[8, 64, 64, 64]", squeeze_25: "f32[64]", add_45: "f32[8, 64, 64, 64]", mean_1: "f32[8, 64, 1, 1]", relu_1: "f32[8, 8, 1, 1]", convolution_12: "f32[8, 64, 1, 1]", mul_72: "f32[8, 64, 64, 64]", convolution_13: "f32[8, 256, 64, 64]", squeeze_28: "f32[256]", mul_80: "f32[8, 256, 64, 64]", convolution_14: "f32[8, 128, 64, 64]", squeeze_31: "f32[128]", mul_88: "f32[8, 128, 64, 64]", convolution_15: "f32[8, 128, 32, 32]", squeeze_34: "f32[128]", add_61: "f32[8, 128, 32, 32]", mean_2: "f32[8, 128, 1, 1]", relu_2: "f32[8, 8, 1, 1]", convolution_17: "f32[8, 128, 1, 1]", mul_97: "f32[8, 128, 32, 32]", convolution_18: "f32[8, 512, 32, 32]", squeeze_37: "f32[512]", convolution_19: "f32[8, 512, 32, 32]", squeeze_40: "f32[512]", mul_112: "f32[8, 512, 32, 32]", convolution_20: "f32[8, 128, 32, 32]", squeeze_43: "f32[128]", mul_120: "f32[8, 128, 32, 32]", convolution_21: "f32[8, 128, 32, 32]", squeeze_46: "f32[128]", add_82: "f32[8, 128, 32, 32]", mean_3: "f32[8, 128, 1, 1]", relu_3: "f32[8, 8, 1, 1]", convolution_23: "f32[8, 128, 1, 1]", mul_129: "f32[8, 128, 32, 32]", convolution_24: "f32[8, 512, 32, 32]", squeeze_49: "f32[512]", mul_137: "f32[8, 512, 32, 32]", convolution_25: "f32[8, 128, 32, 32]", squeeze_52: "f32[128]", mul_145: "f32[8, 128, 32, 32]", view_7: "f32[32768, 32]", view_13: "f32[32768, 32]", bmm_1: "f32[32, 1024, 32]", squeeze_55: "f32[128]", mul_154: "f32[8, 128, 32, 32]", convolution_27: "f32[8, 512, 32, 32]", squeeze_58: "f32[512]", mul_162: "f32[8, 512, 32, 32]", convolution_28: "f32[8, 256, 32, 32]", squeeze_61: "f32[256]", mul_170: "f32[8, 256, 32, 32]", convolution_29: "f32[8, 256, 16, 16]", squeeze_64: "f32[256]", add_116: "f32[8, 256, 16, 16]", mean_4: "f32[8, 256, 1, 1]", relu_4: "f32[8, 16, 1, 1]", convolution_31: "f32[8, 256, 1, 1]", mul_179: "f32[8, 256, 16, 16]", convolution_32: "f32[8, 1024, 16, 16]", squeeze_67: "f32[1024]", convolution_33: "f32[8, 1024, 16, 16]", squeeze_70: "f32[1024]", mul_194: "f32[8, 1024, 16, 16]", convolution_34: "f32[8, 256, 16, 16]", squeeze_73: "f32[256]", mul_202: "f32[8, 256, 16, 16]", convolution_35: "f32[8, 256, 16, 16]", squeeze_76: "f32[256]", add_137: "f32[8, 256, 16, 16]", mean_5: "f32[8, 256, 1, 1]", relu_5: "f32[8, 16, 1, 1]", convolution_37: "f32[8, 256, 1, 1]", mul_211: "f32[8, 256, 16, 16]", convolution_38: "f32[8, 1024, 16, 16]", squeeze_79: "f32[1024]", mul_219: "f32[8, 1024, 16, 16]", convolution_39: "f32[8, 256, 16, 16]", squeeze_82: "f32[256]", mul_227: "f32[8, 256, 16, 16]", view_31: "f32[8192, 64]", view_37: "f32[8192, 64]", bmm_3: "f32[32, 256, 64]", squeeze_85: "f32[256]", mul_236: "f32[8, 256, 16, 16]", convolution_41: "f32[8, 1024, 16, 16]", squeeze_88: "f32[1024]", mul_244: "f32[8, 1024, 16, 16]", convolution_42: "f32[8, 512, 16, 16]", squeeze_91: "f32[512]", mul_252: "f32[8, 512, 16, 16]", view_55: "f32[8192, 128]", view_61: "f32[8192, 128]", view_71: "f32[8, 512, 16, 16]", avg_pool2d: "f32[8, 512, 8, 8]", squeeze_94: "f32[512]", mul_261: "f32[8, 512, 8, 8]", convolution_44: "f32[8, 1536, 8, 8]", squeeze_97: "f32[1536]", convolution_45: "f32[8, 1536, 8, 8]", squeeze_100: "f32[1536]", mul_276: "f32[8, 1536, 8, 8]", convolution_46: "f32[8, 512, 8, 8]", squeeze_103: "f32[512]", mul_284: "f32[8, 512, 8, 8]", view_79: "f32[2048, 128]", view_85: "f32[2048, 128]", bmm_7: "f32[32, 64, 128]", squeeze_106: "f32[512]", mul_293: "f32[8, 512, 8, 8]", convolution_48: "f32[8, 1536, 8, 8]", squeeze_109: "f32[1536]", mul_301: "f32[8, 1536, 8, 8]", convolution_49: "f32[8, 1280, 8, 8]", squeeze_112: "f32[1280]", clone_62: "f32[8, 1280]", permute_33: "f32[1000, 1280]", mul_311: "f32[8, 1280, 8, 8]", unsqueeze_154: "f32[1, 1280, 1, 1]", mul_323: "f32[8, 1536, 8, 8]", unsqueeze_166: "f32[1, 1536, 1, 1]", mul_335: "f32[8, 512, 8, 8]", unsqueeze_178: "f32[1, 512, 1, 1]", permute_41: "f32[32, 64, 64]", permute_42: "f32[32, 128, 64]", alias_16: "f32[32, 64, 64]", permute_46: "f32[15, 128]", permute_52: "f32[15, 128]", permute_54: "f32[32, 128, 64]", permute_55: "f32[32, 64, 128]", mul_350: "f32[8, 512, 8, 8]", unsqueeze_190: "f32[1, 512, 1, 1]", mul_362: "f32[8, 1536, 8, 8]", unsqueeze_202: "f32[1, 1536, 1, 1]", unsqueeze_214: "f32[1, 1536, 1, 1]", mul_383: "f32[8, 512, 8, 8]", unsqueeze_226: "f32[1, 512, 1, 1]", permute_62: "f32[32, 256, 256]", permute_63: "f32[32, 128, 256]", alias_17: "f32[32, 256, 256]", permute_67: "f32[31, 128]", permute_73: "f32[31, 128]", permute_75: "f32[32, 128, 256]", permute_76: "f32[32, 256, 128]", mul_398: "f32[8, 512, 16, 16]", unsqueeze_238: "f32[1, 512, 1, 1]", mul_410: "f32[8, 1024, 16, 16]", unsqueeze_250: "f32[1, 1024, 1, 1]", mul_422: "f32[8, 256, 16, 16]", unsqueeze_262: "f32[1, 256, 1, 1]", permute_83: "f32[32, 256, 256]", permute_84: "f32[32, 64, 256]", alias_18: "f32[32, 256, 256]", permute_88: "f32[31, 64]", permute_94: "f32[31, 64]", permute_96: "f32[32, 64, 256]", permute_97: "f32[32, 256, 64]", mul_437: "f32[8, 256, 16, 16]", unsqueeze_274: "f32[1, 256, 1, 1]", mul_449: "f32[8, 1024, 16, 16]", unsqueeze_286: "f32[1, 1024, 1, 1]", unsqueeze_298: "f32[1, 256, 1, 1]", mul_477: "f32[8, 256, 16, 16]", unsqueeze_310: "f32[1, 256, 1, 1]", mul_489: "f32[8, 1024, 16, 16]", unsqueeze_322: "f32[1, 1024, 1, 1]", unsqueeze_334: "f32[1, 1024, 1, 1]", unsqueeze_346: "f32[1, 256, 1, 1]", mul_526: "f32[8, 256, 32, 32]", unsqueeze_358: "f32[1, 256, 1, 1]", mul_538: "f32[8, 512, 32, 32]", unsqueeze_370: "f32[1, 512, 1, 1]", mul_550: "f32[8, 128, 32, 32]", unsqueeze_382: "f32[1, 128, 1, 1]", permute_110: "f32[32, 1024, 1024]", permute_111: "f32[32, 32, 1024]", alias_27: "f32[32, 1024, 1024]", permute_115: "f32[63, 32]", permute_121: "f32[63, 32]", permute_123: "f32[32, 32, 1024]", permute_124: "f32[32, 1024, 32]", mul_565: "f32[8, 128, 32, 32]", unsqueeze_394: "f32[1, 128, 1, 1]", mul_577: "f32[8, 512, 32, 32]", unsqueeze_406: "f32[1, 512, 1, 1]", unsqueeze_418: "f32[1, 128, 1, 1]", mul_605: "f32[8, 128, 32, 32]", unsqueeze_430: "f32[1, 128, 1, 1]", mul_617: "f32[8, 512, 32, 32]", unsqueeze_442: "f32[1, 512, 1, 1]", unsqueeze_454: "f32[1, 512, 1, 1]", unsqueeze_466: "f32[1, 128, 1, 1]", mul_654: "f32[8, 128, 64, 64]", unsqueeze_478: "f32[1, 128, 1, 1]", mul_666: "f32[8, 256, 64, 64]", unsqueeze_490: "f32[1, 256, 1, 1]", unsqueeze_502: "f32[1, 64, 1, 1]", mul_694: "f32[8, 64, 64, 64]", unsqueeze_514: "f32[1, 64, 1, 1]", mul_706: "f32[8, 256, 64, 64]", unsqueeze_526: "f32[1, 256, 1, 1]", unsqueeze_538: "f32[1, 256, 1, 1]", unsqueeze_550: "f32[1, 64, 1, 1]", mul_743: "f32[8, 64, 64, 64]", unsqueeze_562: "f32[1, 64, 1, 1]", mul_755: "f32[8, 64, 64, 64]", unsqueeze_574: "f32[1, 64, 1, 1]", mul_767: "f32[8, 32, 128, 128]", unsqueeze_586: "f32[1, 32, 1, 1]", mul_779: "f32[8, 24, 128, 128]", unsqueeze_598: "f32[1, 24, 1, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_4: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_24)
    mul_39: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_24, sigmoid_4);  sigmoid_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_5: "f32[8, 64, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_6);  convolution_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_8: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_45)
    mul_71: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_45, sigmoid_8);  sigmoid_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_9: "f32[8, 64, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_12);  convolution_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_12: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(add_61)
    mul_96: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_61, sigmoid_12);  sigmoid_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_13: "f32[8, 128, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_17);  convolution_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_16: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(add_82)
    mul_128: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_82, sigmoid_16);  sigmoid_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_17: "f32[8, 128, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_23);  convolution_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    permute_7: "f32[32, 32, 1024]" = torch.ops.aten.permute.default(bmm_1, [0, 2, 1]);  bmm_1 = None
    clone_22: "f32[32, 32, 1024]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_23: "f32[8, 128, 32, 32]" = torch.ops.aten.reshape.default(clone_22, [8, 128, 32, 32]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_23: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_116)
    mul_178: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_116, sigmoid_23);  sigmoid_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_24: "f32[8, 256, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_31);  convolution_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_27: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_137)
    mul_210: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_137, sigmoid_27);  sigmoid_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_28: "f32[8, 256, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_37);  convolution_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    permute_15: "f32[32, 64, 256]" = torch.ops.aten.permute.default(bmm_3, [0, 2, 1]);  bmm_3 = None
    clone_38: "f32[32, 64, 256]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    view_47: "f32[8, 256, 16, 16]" = torch.ops.aten.reshape.default(clone_38, [8, 256, 16, 16]);  clone_38 = None
    permute_31: "f32[32, 128, 64]" = torch.ops.aten.permute.default(bmm_7, [0, 2, 1]);  bmm_7 = None
    clone_58: "f32[32, 128, 64]" = torch.ops.aten.clone.default(permute_31, memory_format = torch.contiguous_format);  permute_31 = None
    view_95: "f32[8, 512, 8, 8]" = torch.ops.aten.reshape.default(clone_58, [8, 512, 8, 8]);  clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    mm_8: "f32[8, 1280]" = torch.ops.aten.mm.default(tangents_1, permute_33);  permute_33 = None
    permute_34: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_9: "f32[1000, 1280]" = torch.ops.aten.mm.default(permute_34, clone_62);  permute_34 = clone_62 = None
    permute_35: "f32[1280, 1000]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_5: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_97: "f32[1000]" = torch.ops.aten.reshape.default(sum_5, [1000]);  sum_5 = None
    permute_36: "f32[1000, 1280]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_98: "f32[8, 1280, 1, 1]" = torch.ops.aten.reshape.default(mm_8, [8, 1280, 1, 1]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand_24: "f32[8, 1280, 8, 8]" = torch.ops.aten.expand.default(view_98, [8, 1280, 8, 8]);  view_98 = None
    div_4: "f32[8, 1280, 8, 8]" = torch.ops.aten.div.Scalar(expand_24, 64);  expand_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_312: "f32[8, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(div_4, mul_311);  div_4 = mul_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_6: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_312, [0, 2, 3])
    sub_43: "f32[8, 1280, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_154);  convolution_49 = unsqueeze_154 = None
    mul_313: "f32[8, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(mul_312, sub_43)
    sum_7: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_313, [0, 2, 3]);  mul_313 = None
    mul_314: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_6, 0.001953125)
    unsqueeze_155: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_314, 0);  mul_314 = None
    unsqueeze_156: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_155, 2);  unsqueeze_155 = None
    unsqueeze_157: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, 3);  unsqueeze_156 = None
    mul_315: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_7, 0.001953125)
    mul_316: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_317: "f32[1280]" = torch.ops.aten.mul.Tensor(mul_315, mul_316);  mul_315 = mul_316 = None
    unsqueeze_158: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_317, 0);  mul_317 = None
    unsqueeze_159: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, 2);  unsqueeze_158 = None
    unsqueeze_160: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_159, 3);  unsqueeze_159 = None
    mul_318: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_83);  primals_83 = None
    unsqueeze_161: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_318, 0);  mul_318 = None
    unsqueeze_162: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_161, 2);  unsqueeze_161 = None
    unsqueeze_163: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, 3);  unsqueeze_162 = None
    mul_319: "f32[8, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_160);  sub_43 = unsqueeze_160 = None
    sub_45: "f32[8, 1280, 8, 8]" = torch.ops.aten.sub.Tensor(mul_312, mul_319);  mul_312 = mul_319 = None
    sub_46: "f32[8, 1280, 8, 8]" = torch.ops.aten.sub.Tensor(sub_45, unsqueeze_157);  sub_45 = unsqueeze_157 = None
    mul_320: "f32[8, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_163);  sub_46 = unsqueeze_163 = None
    mul_321: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_112);  sum_7 = squeeze_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_320, mul_301, primals_146, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_320 = mul_301 = primals_146 = None
    getitem_88: "f32[8, 1536, 8, 8]" = convolution_backward[0]
    getitem_89: "f32[1280, 1536, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    mul_324: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(getitem_88, mul_323);  getitem_88 = mul_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_8: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_324, [0, 2, 3])
    sub_48: "f32[8, 1536, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_166);  convolution_48 = unsqueeze_166 = None
    mul_325: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(mul_324, sub_48)
    sum_9: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_325, [0, 2, 3]);  mul_325 = None
    mul_326: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_8, 0.001953125)
    unsqueeze_167: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_326, 0);  mul_326 = None
    unsqueeze_168: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 2);  unsqueeze_167 = None
    unsqueeze_169: "f32[1, 1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, 3);  unsqueeze_168 = None
    mul_327: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_9, 0.001953125)
    mul_328: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_329: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_327, mul_328);  mul_327 = mul_328 = None
    unsqueeze_170: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_329, 0);  mul_329 = None
    unsqueeze_171: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, 2);  unsqueeze_170 = None
    unsqueeze_172: "f32[1, 1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_171, 3);  unsqueeze_171 = None
    mul_330: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_81);  primals_81 = None
    unsqueeze_173: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_330, 0);  mul_330 = None
    unsqueeze_174: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_173, 2);  unsqueeze_173 = None
    unsqueeze_175: "f32[1, 1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, 3);  unsqueeze_174 = None
    mul_331: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_172);  sub_48 = unsqueeze_172 = None
    sub_50: "f32[8, 1536, 8, 8]" = torch.ops.aten.sub.Tensor(mul_324, mul_331);  mul_331 = None
    sub_51: "f32[8, 1536, 8, 8]" = torch.ops.aten.sub.Tensor(sub_50, unsqueeze_169);  sub_50 = unsqueeze_169 = None
    mul_332: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_175);  sub_51 = unsqueeze_175 = None
    mul_333: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_109);  sum_9 = squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_332, mul_293, primals_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_332 = mul_293 = primals_145 = None
    getitem_91: "f32[8, 512, 8, 8]" = convolution_backward_1[0]
    getitem_92: "f32[1536, 512, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_336: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(getitem_91, mul_335);  getitem_91 = mul_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_10: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_336, [0, 2, 3])
    sub_53: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(view_95, unsqueeze_178);  view_95 = unsqueeze_178 = None
    mul_337: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_336, sub_53)
    sum_11: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_337, [0, 2, 3]);  mul_337 = None
    mul_338: "f32[512]" = torch.ops.aten.mul.Tensor(sum_10, 0.001953125)
    unsqueeze_179: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_338, 0);  mul_338 = None
    unsqueeze_180: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_179, 2);  unsqueeze_179 = None
    unsqueeze_181: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, 3);  unsqueeze_180 = None
    mul_339: "f32[512]" = torch.ops.aten.mul.Tensor(sum_11, 0.001953125)
    mul_340: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_341: "f32[512]" = torch.ops.aten.mul.Tensor(mul_339, mul_340);  mul_339 = mul_340 = None
    unsqueeze_182: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_341, 0);  mul_341 = None
    unsqueeze_183: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, 2);  unsqueeze_182 = None
    unsqueeze_184: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_183, 3);  unsqueeze_183 = None
    mul_342: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_79);  primals_79 = None
    unsqueeze_185: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_342, 0);  mul_342 = None
    unsqueeze_186: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_185, 2);  unsqueeze_185 = None
    unsqueeze_187: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, 3);  unsqueeze_186 = None
    mul_343: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_184);  sub_53 = unsqueeze_184 = None
    sub_55: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(mul_336, mul_343);  mul_336 = mul_343 = None
    sub_56: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_55, unsqueeze_181);  sub_55 = unsqueeze_181 = None
    mul_344: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_187);  sub_56 = unsqueeze_187 = None
    mul_345: "f32[512]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_106);  sum_11 = squeeze_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    view_99: "f32[32, 128, 64]" = torch.ops.aten.reshape.default(mul_344, [32, 128, 64]);  mul_344 = None
    permute_40: "f32[32, 64, 128]" = torch.ops.aten.permute.default(view_99, [0, 2, 1]);  view_99 = None
    bmm_8: "f32[32, 64, 128]" = torch.ops.aten.bmm.default(permute_41, permute_40);  permute_41 = None
    bmm_9: "f32[32, 64, 64]" = torch.ops.aten.bmm.default(permute_40, permute_42);  permute_40 = permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    mul_346: "f32[32, 64, 64]" = torch.ops.aten.mul.Tensor(bmm_9, alias_16);  bmm_9 = None
    sum_12: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_346, [-1], True)
    mul_347: "f32[32, 64, 64]" = torch.ops.aten.mul.Tensor(alias_16, sum_12);  alias_16 = sum_12 = None
    sub_57: "f32[32, 64, 64]" = torch.ops.aten.sub.Tensor(mul_346, mul_347);  mul_346 = mul_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    view_103: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.reshape.default(sub_57, [32, 8, 8, 8, 8])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_43: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.permute.default(view_103, [0, 2, 4, 1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    sum_13: "f32[32, 8, 1, 8, 8]" = torch.ops.aten.sum.dim_IntList(permute_43, [2], True);  permute_43 = None
    view_104: "f32[256, 8, 8]" = torch.ops.aten.reshape.default(sum_13, [256, 8, 8]);  sum_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    full_default_3: "f32[256, 8, 15]" = torch.ops.aten.full.default([256, 8, 15], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter: "f32[256, 8, 15]" = torch.ops.aten.slice_scatter.default(full_default_3, view_104, 2, 7, 9223372036854775807);  view_104 = None
    full_default_4: "f32[256, 9, 15]" = torch.ops.aten.full.default([256, 9, 15], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_1: "f32[256, 9, 15]" = torch.ops.aten.slice_scatter.default(full_default_4, slice_scatter, 1, 0, 8);  slice_scatter = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_105: "f32[256, 135]" = torch.ops.aten.reshape.default(slice_scatter_1, [256, 135]);  slice_scatter_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_16: "f32[256, 128]" = torch.ops.aten.constant_pad_nd.default(view_105, [0, -7]);  view_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_106: "f32[256, 8, 16]" = torch.ops.aten.reshape.default(constant_pad_nd_16, [256, 8, 16]);  constant_pad_nd_16 = None
    constant_pad_nd_17: "f32[256, 8, 15]" = torch.ops.aten.constant_pad_nd.default(view_106, [0, -1]);  view_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_107: "f32[32, 8, 8, 15]" = torch.ops.aten.reshape.default(constant_pad_nd_17, [32, 8, 8, 15]);  constant_pad_nd_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    view_108: "f32[2048, 15]" = torch.ops.aten.reshape.default(view_107, [2048, 15]);  view_107 = None
    permute_44: "f32[15, 2048]" = torch.ops.aten.permute.default(view_108, [1, 0])
    mm_10: "f32[15, 128]" = torch.ops.aten.mm.default(permute_44, view_85);  permute_44 = view_85 = None
    permute_45: "f32[128, 15]" = torch.ops.aten.permute.default(mm_10, [1, 0]);  mm_10 = None
    mm_11: "f32[2048, 128]" = torch.ops.aten.mm.default(view_108, permute_46);  view_108 = permute_46 = None
    view_109: "f32[32, 8, 8, 128]" = torch.ops.aten.reshape.default(mm_11, [32, 8, 8, 128]);  mm_11 = None
    permute_47: "f32[15, 128]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    permute_48: "f32[32, 8, 8, 128]" = torch.ops.aten.permute.default(view_109, [0, 2, 1, 3]);  view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_49: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.permute.default(view_103, [0, 1, 3, 2, 4]);  view_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    sum_14: "f32[32, 8, 1, 8, 8]" = torch.ops.aten.sum.dim_IntList(permute_49, [2], True);  permute_49 = None
    view_110: "f32[256, 8, 8]" = torch.ops.aten.reshape.default(sum_14, [256, 8, 8]);  sum_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_scatter_3: "f32[256, 8, 15]" = torch.ops.aten.slice_scatter.default(full_default_3, view_110, 2, 7, 9223372036854775807);  full_default_3 = view_110 = None
    slice_scatter_4: "f32[256, 9, 15]" = torch.ops.aten.slice_scatter.default(full_default_4, slice_scatter_3, 1, 0, 8);  full_default_4 = slice_scatter_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_111: "f32[256, 135]" = torch.ops.aten.reshape.default(slice_scatter_4, [256, 135]);  slice_scatter_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_18: "f32[256, 128]" = torch.ops.aten.constant_pad_nd.default(view_111, [0, -7]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_112: "f32[256, 8, 16]" = torch.ops.aten.reshape.default(constant_pad_nd_18, [256, 8, 16]);  constant_pad_nd_18 = None
    constant_pad_nd_19: "f32[256, 8, 15]" = torch.ops.aten.constant_pad_nd.default(view_112, [0, -1]);  view_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_113: "f32[32, 8, 8, 15]" = torch.ops.aten.reshape.default(constant_pad_nd_19, [32, 8, 8, 15]);  constant_pad_nd_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    view_114: "f32[2048, 15]" = torch.ops.aten.reshape.default(view_113, [2048, 15]);  view_113 = None
    permute_50: "f32[15, 2048]" = torch.ops.aten.permute.default(view_114, [1, 0])
    mm_12: "f32[15, 128]" = torch.ops.aten.mm.default(permute_50, view_79);  permute_50 = view_79 = None
    permute_51: "f32[128, 15]" = torch.ops.aten.permute.default(mm_12, [1, 0]);  mm_12 = None
    mm_13: "f32[2048, 128]" = torch.ops.aten.mm.default(view_114, permute_52);  view_114 = permute_52 = None
    view_115: "f32[32, 8, 8, 128]" = torch.ops.aten.reshape.default(mm_13, [32, 8, 8, 128]);  mm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    add_211: "f32[32, 8, 8, 128]" = torch.ops.aten.add.Tensor(permute_48, view_115);  permute_48 = view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_53: "f32[15, 128]" = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    clone_63: "f32[32, 8, 8, 128]" = torch.ops.aten.clone.default(add_211, memory_format = torch.contiguous_format);  add_211 = None
    view_116: "f32[32, 64, 128]" = torch.ops.aten.reshape.default(clone_63, [32, 64, 128]);  clone_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    mul_348: "f32[32, 64, 64]" = torch.ops.aten.mul.Tensor(sub_57, 0.08838834764831845);  sub_57 = None
    bmm_10: "f32[32, 128, 64]" = torch.ops.aten.bmm.default(permute_54, mul_348);  permute_54 = None
    bmm_11: "f32[32, 64, 128]" = torch.ops.aten.bmm.default(mul_348, permute_55);  mul_348 = permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    add_212: "f32[32, 64, 128]" = torch.ops.aten.add.Tensor(view_116, bmm_11);  view_116 = bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    permute_56: "f32[32, 128, 64]" = torch.ops.aten.permute.default(bmm_8, [0, 2, 1]);  bmm_8 = None
    clone_64: "f32[32, 128, 64]" = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None
    view_120: "f32[8, 512, 8, 8]" = torch.ops.aten.reshape.default(clone_64, [8, 512, 8, 8]);  clone_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    view_121: "f32[8, 512, 8, 8]" = torch.ops.aten.reshape.default(bmm_10, [8, 512, 8, 8]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    permute_57: "f32[32, 128, 64]" = torch.ops.aten.permute.default(add_212, [0, 2, 1]);  add_212 = None
    clone_65: "f32[32, 128, 64]" = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
    view_122: "f32[8, 512, 8, 8]" = torch.ops.aten.reshape.default(clone_65, [8, 512, 8, 8]);  clone_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    cat: "f32[8, 1536, 8, 8]" = torch.ops.aten.cat.default([view_122, view_121, view_120], 1);  view_122 = view_121 = view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(cat, mul_284, primals_144, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  cat = mul_284 = primals_144 = None
    getitem_94: "f32[8, 512, 8, 8]" = convolution_backward_2[0]
    getitem_95: "f32[1536, 512, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_351: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(getitem_94, mul_350);  getitem_94 = mul_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_15: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_351, [0, 2, 3])
    sub_59: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_190);  convolution_46 = unsqueeze_190 = None
    mul_352: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_351, sub_59)
    sum_16: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_352, [0, 2, 3]);  mul_352 = None
    mul_353: "f32[512]" = torch.ops.aten.mul.Tensor(sum_15, 0.001953125)
    unsqueeze_191: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_353, 0);  mul_353 = None
    unsqueeze_192: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_191, 2);  unsqueeze_191 = None
    unsqueeze_193: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, 3);  unsqueeze_192 = None
    mul_354: "f32[512]" = torch.ops.aten.mul.Tensor(sum_16, 0.001953125)
    mul_355: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_356: "f32[512]" = torch.ops.aten.mul.Tensor(mul_354, mul_355);  mul_354 = mul_355 = None
    unsqueeze_194: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_356, 0);  mul_356 = None
    unsqueeze_195: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, 2);  unsqueeze_194 = None
    unsqueeze_196: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_195, 3);  unsqueeze_195 = None
    mul_357: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_75);  primals_75 = None
    unsqueeze_197: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_357, 0);  mul_357 = None
    unsqueeze_198: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 2);  unsqueeze_197 = None
    unsqueeze_199: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, 3);  unsqueeze_198 = None
    mul_358: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_196);  sub_59 = unsqueeze_196 = None
    sub_61: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(mul_351, mul_358);  mul_351 = mul_358 = None
    sub_62: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_61, unsqueeze_193);  sub_61 = unsqueeze_193 = None
    mul_359: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_199);  sub_62 = unsqueeze_199 = None
    mul_360: "f32[512]" = torch.ops.aten.mul.Tensor(sum_16, squeeze_103);  sum_16 = squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_359, mul_276, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_359 = mul_276 = primals_143 = None
    getitem_97: "f32[8, 1536, 8, 8]" = convolution_backward_3[0]
    getitem_98: "f32[512, 1536, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_214: "f32[8, 1536, 8, 8]" = torch.ops.aten.add.Tensor(mul_324, getitem_97);  mul_324 = getitem_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    mul_363: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(add_214, mul_362);  add_214 = mul_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_17: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_363, [0, 2, 3])
    sub_64: "f32[8, 1536, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_202);  convolution_45 = unsqueeze_202 = None
    mul_364: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(mul_363, sub_64)
    sum_18: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_364, [0, 2, 3]);  mul_364 = None
    mul_365: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_17, 0.001953125)
    unsqueeze_203: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_365, 0);  mul_365 = None
    unsqueeze_204: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 2);  unsqueeze_203 = None
    unsqueeze_205: "f32[1, 1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, 3);  unsqueeze_204 = None
    mul_366: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_18, 0.001953125)
    mul_367: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_368: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_366, mul_367);  mul_366 = mul_367 = None
    unsqueeze_206: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_368, 0);  mul_368 = None
    unsqueeze_207: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, 2);  unsqueeze_206 = None
    unsqueeze_208: "f32[1, 1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_207, 3);  unsqueeze_207 = None
    mul_369: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_73);  primals_73 = None
    unsqueeze_209: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_369, 0);  mul_369 = None
    unsqueeze_210: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 2);  unsqueeze_209 = None
    unsqueeze_211: "f32[1, 1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, 3);  unsqueeze_210 = None
    mul_370: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_208);  sub_64 = unsqueeze_208 = None
    sub_66: "f32[8, 1536, 8, 8]" = torch.ops.aten.sub.Tensor(mul_363, mul_370);  mul_370 = None
    sub_67: "f32[8, 1536, 8, 8]" = torch.ops.aten.sub.Tensor(sub_66, unsqueeze_205);  sub_66 = None
    mul_371: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_211);  sub_67 = unsqueeze_211 = None
    mul_372: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_18, squeeze_100);  sum_18 = squeeze_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_371, mul_244, primals_142, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_371 = primals_142 = None
    getitem_100: "f32[8, 1024, 16, 16]" = convolution_backward_4[0]
    getitem_101: "f32[1536, 1024, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_68: "f32[8, 1536, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_214);  convolution_44 = unsqueeze_214 = None
    mul_373: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(mul_363, sub_68)
    sum_20: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_373, [0, 2, 3]);  mul_373 = None
    mul_375: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_20, 0.001953125)
    mul_376: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_377: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_375, mul_376);  mul_375 = mul_376 = None
    unsqueeze_218: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_377, 0);  mul_377 = None
    unsqueeze_219: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 2);  unsqueeze_218 = None
    unsqueeze_220: "f32[1, 1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_219, 3);  unsqueeze_219 = None
    mul_378: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_71);  primals_71 = None
    unsqueeze_221: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_378, 0);  mul_378 = None
    unsqueeze_222: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 2);  unsqueeze_221 = None
    unsqueeze_223: "f32[1, 1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, 3);  unsqueeze_222 = None
    mul_379: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_220);  sub_68 = unsqueeze_220 = None
    sub_70: "f32[8, 1536, 8, 8]" = torch.ops.aten.sub.Tensor(mul_363, mul_379);  mul_363 = mul_379 = None
    sub_71: "f32[8, 1536, 8, 8]" = torch.ops.aten.sub.Tensor(sub_70, unsqueeze_205);  sub_70 = unsqueeze_205 = None
    mul_380: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_223);  sub_71 = unsqueeze_223 = None
    mul_381: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_20, squeeze_97);  sum_20 = squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_380, mul_261, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_380 = mul_261 = primals_141 = None
    getitem_103: "f32[8, 512, 8, 8]" = convolution_backward_5[0]
    getitem_104: "f32[1536, 512, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_384: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(getitem_103, mul_383);  getitem_103 = mul_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_21: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_384, [0, 2, 3])
    sub_73: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(avg_pool2d, unsqueeze_226);  avg_pool2d = unsqueeze_226 = None
    mul_385: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_384, sub_73)
    sum_22: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_385, [0, 2, 3]);  mul_385 = None
    mul_386: "f32[512]" = torch.ops.aten.mul.Tensor(sum_21, 0.001953125)
    unsqueeze_227: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_386, 0);  mul_386 = None
    unsqueeze_228: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 2);  unsqueeze_227 = None
    unsqueeze_229: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, 3);  unsqueeze_228 = None
    mul_387: "f32[512]" = torch.ops.aten.mul.Tensor(sum_22, 0.001953125)
    mul_388: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_389: "f32[512]" = torch.ops.aten.mul.Tensor(mul_387, mul_388);  mul_387 = mul_388 = None
    unsqueeze_230: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_389, 0);  mul_389 = None
    unsqueeze_231: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 2);  unsqueeze_230 = None
    unsqueeze_232: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_231, 3);  unsqueeze_231 = None
    mul_390: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_69);  primals_69 = None
    unsqueeze_233: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_390, 0);  mul_390 = None
    unsqueeze_234: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 2);  unsqueeze_233 = None
    unsqueeze_235: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, 3);  unsqueeze_234 = None
    mul_391: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_232);  sub_73 = unsqueeze_232 = None
    sub_75: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(mul_384, mul_391);  mul_384 = mul_391 = None
    sub_76: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_75, unsqueeze_229);  sub_75 = unsqueeze_229 = None
    mul_392: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_235);  sub_76 = unsqueeze_235 = None
    mul_393: "f32[512]" = torch.ops.aten.mul.Tensor(sum_22, squeeze_94);  sum_22 = squeeze_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:156, code: out = self.pool(out)
    avg_pool2d_backward: "f32[8, 512, 16, 16]" = torch.ops.aten.avg_pool2d_backward.default(mul_392, view_71, [2, 2], [2, 2], [0, 0], False, True, None);  mul_392 = view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    view_123: "f32[32, 128, 256]" = torch.ops.aten.reshape.default(avg_pool2d_backward, [32, 128, 256]);  avg_pool2d_backward = None
    permute_61: "f32[32, 256, 128]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
    bmm_12: "f32[32, 256, 128]" = torch.ops.aten.bmm.default(permute_62, permute_61);  permute_62 = None
    bmm_13: "f32[32, 256, 256]" = torch.ops.aten.bmm.default(permute_61, permute_63);  permute_61 = permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    mul_394: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(bmm_13, alias_17);  bmm_13 = None
    sum_23: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_394, [-1], True)
    mul_395: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(alias_17, sum_23);  alias_17 = sum_23 = None
    sub_77: "f32[32, 256, 256]" = torch.ops.aten.sub.Tensor(mul_394, mul_395);  mul_394 = mul_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    view_127: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.reshape.default(sub_77, [32, 16, 16, 16, 16])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_64: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(view_127, [0, 2, 4, 1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    sum_24: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.sum.dim_IntList(permute_64, [2], True);  permute_64 = None
    view_128: "f32[512, 16, 16]" = torch.ops.aten.reshape.default(sum_24, [512, 16, 16]);  sum_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    full_default_12: "f32[512, 16, 31]" = torch.ops.aten.full.default([512, 16, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_6: "f32[512, 16, 31]" = torch.ops.aten.slice_scatter.default(full_default_12, view_128, 2, 15, 9223372036854775807);  view_128 = None
    full_default_13: "f32[512, 17, 31]" = torch.ops.aten.full.default([512, 17, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_7: "f32[512, 17, 31]" = torch.ops.aten.slice_scatter.default(full_default_13, slice_scatter_6, 1, 0, 16);  slice_scatter_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_129: "f32[512, 527]" = torch.ops.aten.reshape.default(slice_scatter_7, [512, 527]);  slice_scatter_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_20: "f32[512, 512]" = torch.ops.aten.constant_pad_nd.default(view_129, [0, -15]);  view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_130: "f32[512, 16, 32]" = torch.ops.aten.reshape.default(constant_pad_nd_20, [512, 16, 32]);  constant_pad_nd_20 = None
    constant_pad_nd_21: "f32[512, 16, 31]" = torch.ops.aten.constant_pad_nd.default(view_130, [0, -1]);  view_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_131: "f32[32, 16, 16, 31]" = torch.ops.aten.reshape.default(constant_pad_nd_21, [32, 16, 16, 31]);  constant_pad_nd_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    view_132: "f32[8192, 31]" = torch.ops.aten.reshape.default(view_131, [8192, 31]);  view_131 = None
    permute_65: "f32[31, 8192]" = torch.ops.aten.permute.default(view_132, [1, 0])
    mm_14: "f32[31, 128]" = torch.ops.aten.mm.default(permute_65, view_61);  permute_65 = view_61 = None
    permute_66: "f32[128, 31]" = torch.ops.aten.permute.default(mm_14, [1, 0]);  mm_14 = None
    mm_15: "f32[8192, 128]" = torch.ops.aten.mm.default(view_132, permute_67);  view_132 = permute_67 = None
    view_133: "f32[32, 16, 16, 128]" = torch.ops.aten.reshape.default(mm_15, [32, 16, 16, 128]);  mm_15 = None
    permute_68: "f32[31, 128]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    permute_69: "f32[32, 16, 16, 128]" = torch.ops.aten.permute.default(view_133, [0, 2, 1, 3]);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_70: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(view_127, [0, 1, 3, 2, 4]);  view_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    sum_25: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.sum.dim_IntList(permute_70, [2], True);  permute_70 = None
    view_134: "f32[512, 16, 16]" = torch.ops.aten.reshape.default(sum_25, [512, 16, 16]);  sum_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_scatter_9: "f32[512, 16, 31]" = torch.ops.aten.slice_scatter.default(full_default_12, view_134, 2, 15, 9223372036854775807);  view_134 = None
    slice_scatter_10: "f32[512, 17, 31]" = torch.ops.aten.slice_scatter.default(full_default_13, slice_scatter_9, 1, 0, 16);  slice_scatter_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_135: "f32[512, 527]" = torch.ops.aten.reshape.default(slice_scatter_10, [512, 527]);  slice_scatter_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_22: "f32[512, 512]" = torch.ops.aten.constant_pad_nd.default(view_135, [0, -15]);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_136: "f32[512, 16, 32]" = torch.ops.aten.reshape.default(constant_pad_nd_22, [512, 16, 32]);  constant_pad_nd_22 = None
    constant_pad_nd_23: "f32[512, 16, 31]" = torch.ops.aten.constant_pad_nd.default(view_136, [0, -1]);  view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_137: "f32[32, 16, 16, 31]" = torch.ops.aten.reshape.default(constant_pad_nd_23, [32, 16, 16, 31]);  constant_pad_nd_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    view_138: "f32[8192, 31]" = torch.ops.aten.reshape.default(view_137, [8192, 31]);  view_137 = None
    permute_71: "f32[31, 8192]" = torch.ops.aten.permute.default(view_138, [1, 0])
    mm_16: "f32[31, 128]" = torch.ops.aten.mm.default(permute_71, view_55);  permute_71 = view_55 = None
    permute_72: "f32[128, 31]" = torch.ops.aten.permute.default(mm_16, [1, 0]);  mm_16 = None
    mm_17: "f32[8192, 128]" = torch.ops.aten.mm.default(view_138, permute_73);  view_138 = permute_73 = None
    view_139: "f32[32, 16, 16, 128]" = torch.ops.aten.reshape.default(mm_17, [32, 16, 16, 128]);  mm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    add_217: "f32[32, 16, 16, 128]" = torch.ops.aten.add.Tensor(permute_69, view_139);  permute_69 = view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_74: "f32[31, 128]" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    clone_66: "f32[32, 16, 16, 128]" = torch.ops.aten.clone.default(add_217, memory_format = torch.contiguous_format);  add_217 = None
    view_140: "f32[32, 256, 128]" = torch.ops.aten.reshape.default(clone_66, [32, 256, 128]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    mul_396: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(sub_77, 0.08838834764831845);  sub_77 = None
    bmm_14: "f32[32, 128, 256]" = torch.ops.aten.bmm.default(permute_75, mul_396);  permute_75 = None
    bmm_15: "f32[32, 256, 128]" = torch.ops.aten.bmm.default(mul_396, permute_76);  mul_396 = permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    add_218: "f32[32, 256, 128]" = torch.ops.aten.add.Tensor(view_140, bmm_15);  view_140 = bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    permute_77: "f32[32, 128, 256]" = torch.ops.aten.permute.default(bmm_12, [0, 2, 1]);  bmm_12 = None
    clone_67: "f32[32, 128, 256]" = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None
    view_144: "f32[8, 512, 16, 16]" = torch.ops.aten.reshape.default(clone_67, [8, 512, 16, 16]);  clone_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    view_145: "f32[8, 512, 16, 16]" = torch.ops.aten.reshape.default(bmm_14, [8, 512, 16, 16]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    permute_78: "f32[32, 128, 256]" = torch.ops.aten.permute.default(add_218, [0, 2, 1]);  add_218 = None
    clone_68: "f32[32, 128, 256]" = torch.ops.aten.clone.default(permute_78, memory_format = torch.contiguous_format);  permute_78 = None
    view_146: "f32[8, 512, 16, 16]" = torch.ops.aten.reshape.default(clone_68, [8, 512, 16, 16]);  clone_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    cat_1: "f32[8, 1536, 16, 16]" = torch.ops.aten.cat.default([view_146, view_145, view_144], 1);  view_146 = view_145 = view_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(cat_1, mul_252, primals_140, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  cat_1 = mul_252 = primals_140 = None
    getitem_106: "f32[8, 512, 16, 16]" = convolution_backward_6[0]
    getitem_107: "f32[1536, 512, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_399: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_106, mul_398);  getitem_106 = mul_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_26: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_399, [0, 2, 3])
    sub_79: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_238);  convolution_42 = unsqueeze_238 = None
    mul_400: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_399, sub_79)
    sum_27: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_400, [0, 2, 3]);  mul_400 = None
    mul_401: "f32[512]" = torch.ops.aten.mul.Tensor(sum_26, 0.00048828125)
    unsqueeze_239: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_401, 0);  mul_401 = None
    unsqueeze_240: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 2);  unsqueeze_239 = None
    unsqueeze_241: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, 3);  unsqueeze_240 = None
    mul_402: "f32[512]" = torch.ops.aten.mul.Tensor(sum_27, 0.00048828125)
    mul_403: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_404: "f32[512]" = torch.ops.aten.mul.Tensor(mul_402, mul_403);  mul_402 = mul_403 = None
    unsqueeze_242: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_404, 0);  mul_404 = None
    unsqueeze_243: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 2);  unsqueeze_242 = None
    unsqueeze_244: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_243, 3);  unsqueeze_243 = None
    mul_405: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_65);  primals_65 = None
    unsqueeze_245: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_405, 0);  mul_405 = None
    unsqueeze_246: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 2);  unsqueeze_245 = None
    unsqueeze_247: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, 3);  unsqueeze_246 = None
    mul_406: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_244);  sub_79 = unsqueeze_244 = None
    sub_81: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(mul_399, mul_406);  mul_399 = mul_406 = None
    sub_82: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_81, unsqueeze_241);  sub_81 = unsqueeze_241 = None
    mul_407: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_247);  sub_82 = unsqueeze_247 = None
    mul_408: "f32[512]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_91);  sum_27 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_407, mul_244, primals_139, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_407 = mul_244 = primals_139 = None
    getitem_109: "f32[8, 1024, 16, 16]" = convolution_backward_7[0]
    getitem_110: "f32[512, 1024, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_220: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(getitem_100, getitem_109);  getitem_100 = getitem_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    mul_411: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(add_220, mul_410);  add_220 = mul_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_28: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_411, [0, 2, 3])
    sub_84: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_250);  convolution_41 = unsqueeze_250 = None
    mul_412: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_411, sub_84)
    sum_29: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_412, [0, 2, 3]);  mul_412 = None
    mul_413: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_28, 0.00048828125)
    unsqueeze_251: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_413, 0);  mul_413 = None
    unsqueeze_252: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 2);  unsqueeze_251 = None
    unsqueeze_253: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, 3);  unsqueeze_252 = None
    mul_414: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_29, 0.00048828125)
    mul_415: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_416: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_414, mul_415);  mul_414 = mul_415 = None
    unsqueeze_254: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_416, 0);  mul_416 = None
    unsqueeze_255: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 2);  unsqueeze_254 = None
    unsqueeze_256: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 3);  unsqueeze_255 = None
    mul_417: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_63);  primals_63 = None
    unsqueeze_257: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_417, 0);  mul_417 = None
    unsqueeze_258: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 2);  unsqueeze_257 = None
    unsqueeze_259: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, 3);  unsqueeze_258 = None
    mul_418: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_256);  sub_84 = unsqueeze_256 = None
    sub_86: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(mul_411, mul_418);  mul_418 = None
    sub_87: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_86, unsqueeze_253);  sub_86 = unsqueeze_253 = None
    mul_419: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_259);  sub_87 = unsqueeze_259 = None
    mul_420: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_88);  sum_29 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_419, mul_236, primals_138, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_419 = mul_236 = primals_138 = None
    getitem_112: "f32[8, 256, 16, 16]" = convolution_backward_8[0]
    getitem_113: "f32[1024, 256, 1, 1]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    full_default_20: "f32[8, 256, 16, 16]" = torch.ops.aten.full.default([8, 256, 16, 16], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    mul_423: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_112, mul_422);  getitem_112 = mul_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_30: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_423, [0, 2, 3])
    sub_89: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(view_47, unsqueeze_262);  view_47 = unsqueeze_262 = None
    mul_424: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_423, sub_89)
    sum_31: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_424, [0, 2, 3]);  mul_424 = None
    mul_425: "f32[256]" = torch.ops.aten.mul.Tensor(sum_30, 0.00048828125)
    unsqueeze_263: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_425, 0);  mul_425 = None
    unsqueeze_264: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 2);  unsqueeze_263 = None
    unsqueeze_265: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, 3);  unsqueeze_264 = None
    mul_426: "f32[256]" = torch.ops.aten.mul.Tensor(sum_31, 0.00048828125)
    mul_427: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_428: "f32[256]" = torch.ops.aten.mul.Tensor(mul_426, mul_427);  mul_426 = mul_427 = None
    unsqueeze_266: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_428, 0);  mul_428 = None
    unsqueeze_267: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 2);  unsqueeze_266 = None
    unsqueeze_268: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 3);  unsqueeze_267 = None
    mul_429: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_61);  primals_61 = None
    unsqueeze_269: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_429, 0);  mul_429 = None
    unsqueeze_270: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 2);  unsqueeze_269 = None
    unsqueeze_271: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 3);  unsqueeze_270 = None
    mul_430: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_268);  sub_89 = unsqueeze_268 = None
    sub_91: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(mul_423, mul_430);  mul_423 = mul_430 = None
    sub_92: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_91, unsqueeze_265);  sub_91 = unsqueeze_265 = None
    mul_431: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_271);  sub_92 = unsqueeze_271 = None
    mul_432: "f32[256]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_85);  sum_31 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    view_147: "f32[32, 64, 256]" = torch.ops.aten.reshape.default(mul_431, [32, 64, 256]);  mul_431 = None
    permute_82: "f32[32, 256, 64]" = torch.ops.aten.permute.default(view_147, [0, 2, 1]);  view_147 = None
    bmm_16: "f32[32, 256, 64]" = torch.ops.aten.bmm.default(permute_83, permute_82);  permute_83 = None
    bmm_17: "f32[32, 256, 256]" = torch.ops.aten.bmm.default(permute_82, permute_84);  permute_82 = permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    mul_433: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(bmm_17, alias_18);  bmm_17 = None
    sum_32: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_433, [-1], True)
    mul_434: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(alias_18, sum_32);  alias_18 = sum_32 = None
    sub_93: "f32[32, 256, 256]" = torch.ops.aten.sub.Tensor(mul_433, mul_434);  mul_433 = mul_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    view_151: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.reshape.default(sub_93, [32, 16, 16, 16, 16])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_85: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(view_151, [0, 2, 4, 1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    sum_33: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.sum.dim_IntList(permute_85, [2], True);  permute_85 = None
    view_152: "f32[512, 16, 16]" = torch.ops.aten.reshape.default(sum_33, [512, 16, 16]);  sum_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_scatter_12: "f32[512, 16, 31]" = torch.ops.aten.slice_scatter.default(full_default_12, view_152, 2, 15, 9223372036854775807);  view_152 = None
    slice_scatter_13: "f32[512, 17, 31]" = torch.ops.aten.slice_scatter.default(full_default_13, slice_scatter_12, 1, 0, 16);  slice_scatter_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_153: "f32[512, 527]" = torch.ops.aten.reshape.default(slice_scatter_13, [512, 527]);  slice_scatter_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_24: "f32[512, 512]" = torch.ops.aten.constant_pad_nd.default(view_153, [0, -15]);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_154: "f32[512, 16, 32]" = torch.ops.aten.reshape.default(constant_pad_nd_24, [512, 16, 32]);  constant_pad_nd_24 = None
    constant_pad_nd_25: "f32[512, 16, 31]" = torch.ops.aten.constant_pad_nd.default(view_154, [0, -1]);  view_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_155: "f32[32, 16, 16, 31]" = torch.ops.aten.reshape.default(constant_pad_nd_25, [32, 16, 16, 31]);  constant_pad_nd_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    view_156: "f32[8192, 31]" = torch.ops.aten.reshape.default(view_155, [8192, 31]);  view_155 = None
    permute_86: "f32[31, 8192]" = torch.ops.aten.permute.default(view_156, [1, 0])
    mm_18: "f32[31, 64]" = torch.ops.aten.mm.default(permute_86, view_37);  permute_86 = view_37 = None
    permute_87: "f32[64, 31]" = torch.ops.aten.permute.default(mm_18, [1, 0]);  mm_18 = None
    mm_19: "f32[8192, 64]" = torch.ops.aten.mm.default(view_156, permute_88);  view_156 = permute_88 = None
    view_157: "f32[32, 16, 16, 64]" = torch.ops.aten.reshape.default(mm_19, [32, 16, 16, 64]);  mm_19 = None
    permute_89: "f32[31, 64]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    permute_90: "f32[32, 16, 16, 64]" = torch.ops.aten.permute.default(view_157, [0, 2, 1, 3]);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_91: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(view_151, [0, 1, 3, 2, 4]);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    sum_34: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.sum.dim_IntList(permute_91, [2], True);  permute_91 = None
    view_158: "f32[512, 16, 16]" = torch.ops.aten.reshape.default(sum_34, [512, 16, 16]);  sum_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_scatter_15: "f32[512, 16, 31]" = torch.ops.aten.slice_scatter.default(full_default_12, view_158, 2, 15, 9223372036854775807);  full_default_12 = view_158 = None
    slice_scatter_16: "f32[512, 17, 31]" = torch.ops.aten.slice_scatter.default(full_default_13, slice_scatter_15, 1, 0, 16);  full_default_13 = slice_scatter_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_159: "f32[512, 527]" = torch.ops.aten.reshape.default(slice_scatter_16, [512, 527]);  slice_scatter_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_26: "f32[512, 512]" = torch.ops.aten.constant_pad_nd.default(view_159, [0, -15]);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_160: "f32[512, 16, 32]" = torch.ops.aten.reshape.default(constant_pad_nd_26, [512, 16, 32]);  constant_pad_nd_26 = None
    constant_pad_nd_27: "f32[512, 16, 31]" = torch.ops.aten.constant_pad_nd.default(view_160, [0, -1]);  view_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_161: "f32[32, 16, 16, 31]" = torch.ops.aten.reshape.default(constant_pad_nd_27, [32, 16, 16, 31]);  constant_pad_nd_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    view_162: "f32[8192, 31]" = torch.ops.aten.reshape.default(view_161, [8192, 31]);  view_161 = None
    permute_92: "f32[31, 8192]" = torch.ops.aten.permute.default(view_162, [1, 0])
    mm_20: "f32[31, 64]" = torch.ops.aten.mm.default(permute_92, view_31);  permute_92 = view_31 = None
    permute_93: "f32[64, 31]" = torch.ops.aten.permute.default(mm_20, [1, 0]);  mm_20 = None
    mm_21: "f32[8192, 64]" = torch.ops.aten.mm.default(view_162, permute_94);  view_162 = permute_94 = None
    view_163: "f32[32, 16, 16, 64]" = torch.ops.aten.reshape.default(mm_21, [32, 16, 16, 64]);  mm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    add_223: "f32[32, 16, 16, 64]" = torch.ops.aten.add.Tensor(permute_90, view_163);  permute_90 = view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_95: "f32[31, 64]" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    clone_69: "f32[32, 16, 16, 64]" = torch.ops.aten.clone.default(add_223, memory_format = torch.contiguous_format);  add_223 = None
    view_164: "f32[32, 256, 64]" = torch.ops.aten.reshape.default(clone_69, [32, 256, 64]);  clone_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    mul_435: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(sub_93, 0.125);  sub_93 = None
    bmm_18: "f32[32, 64, 256]" = torch.ops.aten.bmm.default(permute_96, mul_435);  permute_96 = None
    bmm_19: "f32[32, 256, 64]" = torch.ops.aten.bmm.default(mul_435, permute_97);  mul_435 = permute_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    add_224: "f32[32, 256, 64]" = torch.ops.aten.add.Tensor(view_164, bmm_19);  view_164 = bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    permute_98: "f32[32, 64, 256]" = torch.ops.aten.permute.default(bmm_16, [0, 2, 1]);  bmm_16 = None
    clone_70: "f32[32, 64, 256]" = torch.ops.aten.clone.default(permute_98, memory_format = torch.contiguous_format);  permute_98 = None
    view_168: "f32[8, 256, 16, 16]" = torch.ops.aten.reshape.default(clone_70, [8, 256, 16, 16]);  clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    view_169: "f32[8, 256, 16, 16]" = torch.ops.aten.reshape.default(bmm_18, [8, 256, 16, 16]);  bmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    permute_99: "f32[32, 64, 256]" = torch.ops.aten.permute.default(add_224, [0, 2, 1]);  add_224 = None
    clone_71: "f32[32, 64, 256]" = torch.ops.aten.clone.default(permute_99, memory_format = torch.contiguous_format);  permute_99 = None
    view_170: "f32[8, 256, 16, 16]" = torch.ops.aten.reshape.default(clone_71, [8, 256, 16, 16]);  clone_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    cat_2: "f32[8, 768, 16, 16]" = torch.ops.aten.cat.default([view_170, view_169, view_168], 1);  view_170 = view_169 = view_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(cat_2, mul_227, primals_137, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  cat_2 = mul_227 = primals_137 = None
    getitem_115: "f32[8, 256, 16, 16]" = convolution_backward_9[0]
    getitem_116: "f32[768, 256, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_438: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_115, mul_437);  getitem_115 = mul_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_35: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_438, [0, 2, 3])
    sub_95: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_274);  convolution_39 = unsqueeze_274 = None
    mul_439: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_438, sub_95)
    sum_36: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_439, [0, 2, 3]);  mul_439 = None
    mul_440: "f32[256]" = torch.ops.aten.mul.Tensor(sum_35, 0.00048828125)
    unsqueeze_275: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_440, 0);  mul_440 = None
    unsqueeze_276: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 2);  unsqueeze_275 = None
    unsqueeze_277: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 3);  unsqueeze_276 = None
    mul_441: "f32[256]" = torch.ops.aten.mul.Tensor(sum_36, 0.00048828125)
    mul_442: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_443: "f32[256]" = torch.ops.aten.mul.Tensor(mul_441, mul_442);  mul_441 = mul_442 = None
    unsqueeze_278: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_443, 0);  mul_443 = None
    unsqueeze_279: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 2);  unsqueeze_278 = None
    unsqueeze_280: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 3);  unsqueeze_279 = None
    mul_444: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_57);  primals_57 = None
    unsqueeze_281: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_444, 0);  mul_444 = None
    unsqueeze_282: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 2);  unsqueeze_281 = None
    unsqueeze_283: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, 3);  unsqueeze_282 = None
    mul_445: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_280);  sub_95 = unsqueeze_280 = None
    sub_97: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(mul_438, mul_445);  mul_438 = mul_445 = None
    sub_98: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_97, unsqueeze_277);  sub_97 = unsqueeze_277 = None
    mul_446: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_283);  sub_98 = unsqueeze_283 = None
    mul_447: "f32[256]" = torch.ops.aten.mul.Tensor(sum_36, squeeze_82);  sum_36 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_446, mul_219, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_446 = mul_219 = primals_136 = None
    getitem_118: "f32[8, 1024, 16, 16]" = convolution_backward_10[0]
    getitem_119: "f32[256, 1024, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_226: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_411, getitem_118);  mul_411 = getitem_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    mul_450: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(add_226, mul_449);  add_226 = mul_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_37: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_450, [0, 2, 3])
    sub_100: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_286);  convolution_38 = unsqueeze_286 = None
    mul_451: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_450, sub_100)
    sum_38: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_451, [0, 2, 3]);  mul_451 = None
    mul_452: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_37, 0.00048828125)
    unsqueeze_287: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_452, 0);  mul_452 = None
    unsqueeze_288: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 2);  unsqueeze_287 = None
    unsqueeze_289: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 3);  unsqueeze_288 = None
    mul_453: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_38, 0.00048828125)
    mul_454: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_455: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_453, mul_454);  mul_453 = mul_454 = None
    unsqueeze_290: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_455, 0);  mul_455 = None
    unsqueeze_291: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 2);  unsqueeze_290 = None
    unsqueeze_292: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 3);  unsqueeze_291 = None
    mul_456: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_55);  primals_55 = None
    unsqueeze_293: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_456, 0);  mul_456 = None
    unsqueeze_294: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 2);  unsqueeze_293 = None
    unsqueeze_295: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, 3);  unsqueeze_294 = None
    mul_457: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_292);  sub_100 = unsqueeze_292 = None
    sub_102: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(mul_450, mul_457);  mul_457 = None
    sub_103: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_102, unsqueeze_289);  sub_102 = unsqueeze_289 = None
    mul_458: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_295);  sub_103 = unsqueeze_295 = None
    mul_459: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_38, squeeze_79);  sum_38 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_458, mul_211, primals_135, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_458 = mul_211 = primals_135 = None
    getitem_121: "f32[8, 256, 16, 16]" = convolution_backward_11[0]
    getitem_122: "f32[1024, 256, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_460: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_121, mul_210);  mul_210 = None
    mul_461: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_121, sigmoid_28);  getitem_121 = None
    sum_39: "f32[8, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_460, [2, 3], True);  mul_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_104: "f32[8, 256, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_28)
    mul_462: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_28, sub_104);  sigmoid_28 = sub_104 = None
    mul_463: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(sum_39, mul_462);  sum_39 = mul_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_463, relu_5, primals_133, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_463 = primals_133 = None
    getitem_124: "f32[8, 16, 1, 1]" = convolution_backward_12[0]
    getitem_125: "f32[256, 16, 1, 1]" = convolution_backward_12[1]
    getitem_126: "f32[256]" = convolution_backward_12[2];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le: "b8[8, 16, 1, 1]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
    full_default_29: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where: "f32[8, 16, 1, 1]" = torch.ops.aten.where.self(le, full_default_29, getitem_124);  le = getitem_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(where, mean_5, primals_131, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where = mean_5 = primals_131 = None
    getitem_127: "f32[8, 256, 1, 1]" = convolution_backward_13[0]
    getitem_128: "f32[16, 256, 1, 1]" = convolution_backward_13[1]
    getitem_129: "f32[16]" = convolution_backward_13[2];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_25: "f32[8, 256, 16, 16]" = torch.ops.aten.expand.default(getitem_127, [8, 256, 16, 16]);  getitem_127 = None
    div_5: "f32[8, 256, 16, 16]" = torch.ops.aten.div.Scalar(expand_25, 256);  expand_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_228: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_461, div_5);  mul_461 = div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_51: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_137)
    sub_105: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(full_default_20, sigmoid_51)
    mul_464: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_137, sub_105);  add_137 = sub_105 = None
    add_229: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Scalar(mul_464, 1);  mul_464 = None
    mul_465: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_51, add_229);  sigmoid_51 = add_229 = None
    mul_466: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_228, mul_465);  add_228 = mul_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_40: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_466, [0, 2, 3])
    sub_106: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_298);  convolution_35 = unsqueeze_298 = None
    mul_467: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_466, sub_106)
    sum_41: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_467, [0, 2, 3]);  mul_467 = None
    mul_468: "f32[256]" = torch.ops.aten.mul.Tensor(sum_40, 0.00048828125)
    unsqueeze_299: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_468, 0);  mul_468 = None
    unsqueeze_300: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 2);  unsqueeze_299 = None
    unsqueeze_301: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 3);  unsqueeze_300 = None
    mul_469: "f32[256]" = torch.ops.aten.mul.Tensor(sum_41, 0.00048828125)
    mul_470: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_471: "f32[256]" = torch.ops.aten.mul.Tensor(mul_469, mul_470);  mul_469 = mul_470 = None
    unsqueeze_302: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_471, 0);  mul_471 = None
    unsqueeze_303: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 2);  unsqueeze_302 = None
    unsqueeze_304: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 3);  unsqueeze_303 = None
    mul_472: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_53);  primals_53 = None
    unsqueeze_305: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_472, 0);  mul_472 = None
    unsqueeze_306: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 2);  unsqueeze_305 = None
    unsqueeze_307: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, 3);  unsqueeze_306 = None
    mul_473: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_304);  sub_106 = unsqueeze_304 = None
    sub_108: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(mul_466, mul_473);  mul_466 = mul_473 = None
    sub_109: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_108, unsqueeze_301);  sub_108 = unsqueeze_301 = None
    mul_474: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_307);  sub_109 = unsqueeze_307 = None
    mul_475: "f32[256]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_76);  sum_41 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_474, mul_202, primals_130, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_474 = mul_202 = primals_130 = None
    getitem_130: "f32[8, 256, 16, 16]" = convolution_backward_14[0]
    getitem_131: "f32[256, 256, 3, 3]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_478: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_130, mul_477);  getitem_130 = mul_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_42: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_478, [0, 2, 3])
    sub_111: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_310);  convolution_34 = unsqueeze_310 = None
    mul_479: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_478, sub_111)
    sum_43: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_479, [0, 2, 3]);  mul_479 = None
    mul_480: "f32[256]" = torch.ops.aten.mul.Tensor(sum_42, 0.00048828125)
    unsqueeze_311: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_480, 0);  mul_480 = None
    unsqueeze_312: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 2);  unsqueeze_311 = None
    unsqueeze_313: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, 3);  unsqueeze_312 = None
    mul_481: "f32[256]" = torch.ops.aten.mul.Tensor(sum_43, 0.00048828125)
    mul_482: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_483: "f32[256]" = torch.ops.aten.mul.Tensor(mul_481, mul_482);  mul_481 = mul_482 = None
    unsqueeze_314: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_483, 0);  mul_483 = None
    unsqueeze_315: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 2);  unsqueeze_314 = None
    unsqueeze_316: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 3);  unsqueeze_315 = None
    mul_484: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_51);  primals_51 = None
    unsqueeze_317: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_484, 0);  mul_484 = None
    unsqueeze_318: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 2);  unsqueeze_317 = None
    unsqueeze_319: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, 3);  unsqueeze_318 = None
    mul_485: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_316);  sub_111 = unsqueeze_316 = None
    sub_113: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(mul_478, mul_485);  mul_478 = mul_485 = None
    sub_114: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_113, unsqueeze_313);  sub_113 = unsqueeze_313 = None
    mul_486: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_319);  sub_114 = unsqueeze_319 = None
    mul_487: "f32[256]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_73);  sum_43 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_486, mul_194, primals_129, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_486 = mul_194 = primals_129 = None
    getitem_133: "f32[8, 1024, 16, 16]" = convolution_backward_15[0]
    getitem_134: "f32[256, 1024, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_231: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_450, getitem_133);  mul_450 = getitem_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    mul_490: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(add_231, mul_489);  add_231 = mul_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_44: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_490, [0, 2, 3])
    sub_116: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_322);  convolution_33 = unsqueeze_322 = None
    mul_491: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_490, sub_116)
    sum_45: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_491, [0, 2, 3]);  mul_491 = None
    mul_492: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_44, 0.00048828125)
    unsqueeze_323: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_492, 0);  mul_492 = None
    unsqueeze_324: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 2);  unsqueeze_323 = None
    unsqueeze_325: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, 3);  unsqueeze_324 = None
    mul_493: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_45, 0.00048828125)
    mul_494: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_495: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_493, mul_494);  mul_493 = mul_494 = None
    unsqueeze_326: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_495, 0);  mul_495 = None
    unsqueeze_327: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 2);  unsqueeze_326 = None
    unsqueeze_328: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 3);  unsqueeze_327 = None
    mul_496: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_49);  primals_49 = None
    unsqueeze_329: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_496, 0);  mul_496 = None
    unsqueeze_330: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 2);  unsqueeze_329 = None
    unsqueeze_331: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, 3);  unsqueeze_330 = None
    mul_497: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_328);  sub_116 = unsqueeze_328 = None
    sub_118: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(mul_490, mul_497);  mul_497 = None
    sub_119: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_118, unsqueeze_325);  sub_118 = None
    mul_498: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_331);  sub_119 = unsqueeze_331 = None
    mul_499: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_70);  sum_45 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_498, mul_162, primals_128, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_498 = primals_128 = None
    getitem_136: "f32[8, 512, 32, 32]" = convolution_backward_16[0]
    getitem_137: "f32[1024, 512, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_120: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_334);  convolution_32 = unsqueeze_334 = None
    mul_500: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_490, sub_120)
    sum_47: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_500, [0, 2, 3]);  mul_500 = None
    mul_502: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_47, 0.00048828125)
    mul_503: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_504: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_502, mul_503);  mul_502 = mul_503 = None
    unsqueeze_338: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_504, 0);  mul_504 = None
    unsqueeze_339: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 2);  unsqueeze_338 = None
    unsqueeze_340: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 3);  unsqueeze_339 = None
    mul_505: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_47);  primals_47 = None
    unsqueeze_341: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_505, 0);  mul_505 = None
    unsqueeze_342: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 2);  unsqueeze_341 = None
    unsqueeze_343: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 3);  unsqueeze_342 = None
    mul_506: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_340);  sub_120 = unsqueeze_340 = None
    sub_122: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(mul_490, mul_506);  mul_490 = mul_506 = None
    sub_123: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_122, unsqueeze_325);  sub_122 = unsqueeze_325 = None
    mul_507: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_343);  sub_123 = unsqueeze_343 = None
    mul_508: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_67);  sum_47 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_507, mul_179, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_507 = mul_179 = primals_127 = None
    getitem_139: "f32[8, 256, 16, 16]" = convolution_backward_17[0]
    getitem_140: "f32[1024, 256, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_509: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_139, mul_178);  mul_178 = None
    mul_510: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_139, sigmoid_24);  getitem_139 = None
    sum_48: "f32[8, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_509, [2, 3], True);  mul_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_124: "f32[8, 256, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_24)
    mul_511: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_24, sub_124);  sigmoid_24 = sub_124 = None
    mul_512: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(sum_48, mul_511);  sum_48 = mul_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_512, relu_4, primals_125, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_512 = primals_125 = None
    getitem_142: "f32[8, 16, 1, 1]" = convolution_backward_18[0]
    getitem_143: "f32[256, 16, 1, 1]" = convolution_backward_18[1]
    getitem_144: "f32[256]" = convolution_backward_18[2];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_1: "b8[8, 16, 1, 1]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
    where_1: "f32[8, 16, 1, 1]" = torch.ops.aten.where.self(le_1, full_default_29, getitem_142);  le_1 = getitem_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(where_1, mean_4, primals_123, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_1 = mean_4 = primals_123 = None
    getitem_145: "f32[8, 256, 1, 1]" = convolution_backward_19[0]
    getitem_146: "f32[16, 256, 1, 1]" = convolution_backward_19[1]
    getitem_147: "f32[16]" = convolution_backward_19[2];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_26: "f32[8, 256, 16, 16]" = torch.ops.aten.expand.default(getitem_145, [8, 256, 16, 16]);  getitem_145 = None
    div_6: "f32[8, 256, 16, 16]" = torch.ops.aten.div.Scalar(expand_26, 256);  expand_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_233: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_510, div_6);  mul_510 = div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_54: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_116)
    sub_125: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(full_default_20, sigmoid_54);  full_default_20 = None
    mul_513: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_116, sub_125);  add_116 = sub_125 = None
    add_234: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Scalar(mul_513, 1);  mul_513 = None
    mul_514: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_54, add_234);  sigmoid_54 = add_234 = None
    mul_515: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_233, mul_514);  add_233 = mul_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_49: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_515, [0, 2, 3])
    sub_126: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_346);  convolution_29 = unsqueeze_346 = None
    mul_516: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_515, sub_126)
    sum_50: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_516, [0, 2, 3]);  mul_516 = None
    mul_517: "f32[256]" = torch.ops.aten.mul.Tensor(sum_49, 0.00048828125)
    unsqueeze_347: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_517, 0);  mul_517 = None
    unsqueeze_348: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 2);  unsqueeze_347 = None
    unsqueeze_349: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, 3);  unsqueeze_348 = None
    mul_518: "f32[256]" = torch.ops.aten.mul.Tensor(sum_50, 0.00048828125)
    mul_519: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_520: "f32[256]" = torch.ops.aten.mul.Tensor(mul_518, mul_519);  mul_518 = mul_519 = None
    unsqueeze_350: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_520, 0);  mul_520 = None
    unsqueeze_351: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 2);  unsqueeze_350 = None
    unsqueeze_352: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 3);  unsqueeze_351 = None
    mul_521: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_45);  primals_45 = None
    unsqueeze_353: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_521, 0);  mul_521 = None
    unsqueeze_354: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 2);  unsqueeze_353 = None
    unsqueeze_355: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, 3);  unsqueeze_354 = None
    mul_522: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_352);  sub_126 = unsqueeze_352 = None
    sub_128: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(mul_515, mul_522);  mul_515 = mul_522 = None
    sub_129: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_128, unsqueeze_349);  sub_128 = unsqueeze_349 = None
    mul_523: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_355);  sub_129 = unsqueeze_355 = None
    mul_524: "f32[256]" = torch.ops.aten.mul.Tensor(sum_50, squeeze_64);  sum_50 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_523, mul_170, primals_122, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_523 = mul_170 = primals_122 = None
    getitem_148: "f32[8, 256, 32, 32]" = convolution_backward_20[0]
    getitem_149: "f32[256, 256, 3, 3]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_527: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_148, mul_526);  getitem_148 = mul_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_51: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_527, [0, 2, 3])
    sub_131: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_358);  convolution_28 = unsqueeze_358 = None
    mul_528: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_527, sub_131)
    sum_52: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_528, [0, 2, 3]);  mul_528 = None
    mul_529: "f32[256]" = torch.ops.aten.mul.Tensor(sum_51, 0.0001220703125)
    unsqueeze_359: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_529, 0);  mul_529 = None
    unsqueeze_360: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 2);  unsqueeze_359 = None
    unsqueeze_361: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, 3);  unsqueeze_360 = None
    mul_530: "f32[256]" = torch.ops.aten.mul.Tensor(sum_52, 0.0001220703125)
    mul_531: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_532: "f32[256]" = torch.ops.aten.mul.Tensor(mul_530, mul_531);  mul_530 = mul_531 = None
    unsqueeze_362: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_532, 0);  mul_532 = None
    unsqueeze_363: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 2);  unsqueeze_362 = None
    unsqueeze_364: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 3);  unsqueeze_363 = None
    mul_533: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_43);  primals_43 = None
    unsqueeze_365: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_533, 0);  mul_533 = None
    unsqueeze_366: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 2);  unsqueeze_365 = None
    unsqueeze_367: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, 3);  unsqueeze_366 = None
    mul_534: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_364);  sub_131 = unsqueeze_364 = None
    sub_133: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(mul_527, mul_534);  mul_527 = mul_534 = None
    sub_134: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(sub_133, unsqueeze_361);  sub_133 = unsqueeze_361 = None
    mul_535: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_367);  sub_134 = unsqueeze_367 = None
    mul_536: "f32[256]" = torch.ops.aten.mul.Tensor(sum_52, squeeze_61);  sum_52 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_535, mul_162, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_535 = mul_162 = primals_121 = None
    getitem_151: "f32[8, 512, 32, 32]" = convolution_backward_21[0]
    getitem_152: "f32[256, 512, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_236: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(getitem_136, getitem_151);  getitem_136 = getitem_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    mul_539: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_236, mul_538);  add_236 = mul_538 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_53: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_539, [0, 2, 3])
    sub_136: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_370);  convolution_27 = unsqueeze_370 = None
    mul_540: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_539, sub_136)
    sum_54: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_540, [0, 2, 3]);  mul_540 = None
    mul_541: "f32[512]" = torch.ops.aten.mul.Tensor(sum_53, 0.0001220703125)
    unsqueeze_371: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_541, 0);  mul_541 = None
    unsqueeze_372: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 2);  unsqueeze_371 = None
    unsqueeze_373: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, 3);  unsqueeze_372 = None
    mul_542: "f32[512]" = torch.ops.aten.mul.Tensor(sum_54, 0.0001220703125)
    mul_543: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_544: "f32[512]" = torch.ops.aten.mul.Tensor(mul_542, mul_543);  mul_542 = mul_543 = None
    unsqueeze_374: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_544, 0);  mul_544 = None
    unsqueeze_375: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 2);  unsqueeze_374 = None
    unsqueeze_376: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_375, 3);  unsqueeze_375 = None
    mul_545: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_41);  primals_41 = None
    unsqueeze_377: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_545, 0);  mul_545 = None
    unsqueeze_378: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 2);  unsqueeze_377 = None
    unsqueeze_379: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, 3);  unsqueeze_378 = None
    mul_546: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_376);  sub_136 = unsqueeze_376 = None
    sub_138: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(mul_539, mul_546);  mul_546 = None
    sub_139: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(sub_138, unsqueeze_373);  sub_138 = unsqueeze_373 = None
    mul_547: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_379);  sub_139 = unsqueeze_379 = None
    mul_548: "f32[512]" = torch.ops.aten.mul.Tensor(sum_54, squeeze_58);  sum_54 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_547, mul_154, primals_120, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_547 = mul_154 = primals_120 = None
    getitem_154: "f32[8, 128, 32, 32]" = convolution_backward_22[0]
    getitem_155: "f32[512, 128, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    full_default_37: "f32[8, 128, 32, 32]" = torch.ops.aten.full.default([8, 128, 32, 32], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    mul_551: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_154, mul_550);  getitem_154 = mul_550 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_55: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_551, [0, 2, 3])
    sub_141: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(view_23, unsqueeze_382);  view_23 = unsqueeze_382 = None
    mul_552: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_551, sub_141)
    sum_56: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_552, [0, 2, 3]);  mul_552 = None
    mul_553: "f32[128]" = torch.ops.aten.mul.Tensor(sum_55, 0.0001220703125)
    unsqueeze_383: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_553, 0);  mul_553 = None
    unsqueeze_384: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 2);  unsqueeze_383 = None
    unsqueeze_385: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, 3);  unsqueeze_384 = None
    mul_554: "f32[128]" = torch.ops.aten.mul.Tensor(sum_56, 0.0001220703125)
    mul_555: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_556: "f32[128]" = torch.ops.aten.mul.Tensor(mul_554, mul_555);  mul_554 = mul_555 = None
    unsqueeze_386: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_556, 0);  mul_556 = None
    unsqueeze_387: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 2);  unsqueeze_386 = None
    unsqueeze_388: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_387, 3);  unsqueeze_387 = None
    mul_557: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_39);  primals_39 = None
    unsqueeze_389: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_557, 0);  mul_557 = None
    unsqueeze_390: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 2);  unsqueeze_389 = None
    unsqueeze_391: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, 3);  unsqueeze_390 = None
    mul_558: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_388);  sub_141 = unsqueeze_388 = None
    sub_143: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(mul_551, mul_558);  mul_551 = mul_558 = None
    sub_144: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_143, unsqueeze_385);  sub_143 = unsqueeze_385 = None
    mul_559: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_391);  sub_144 = unsqueeze_391 = None
    mul_560: "f32[128]" = torch.ops.aten.mul.Tensor(sum_56, squeeze_55);  sum_56 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    view_171: "f32[32, 32, 1024]" = torch.ops.aten.reshape.default(mul_559, [32, 32, 1024]);  mul_559 = None
    permute_109: "f32[32, 1024, 32]" = torch.ops.aten.permute.default(view_171, [0, 2, 1]);  view_171 = None
    bmm_20: "f32[32, 1024, 32]" = torch.ops.aten.bmm.default(permute_110, permute_109);  permute_110 = None
    bmm_21: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(permute_109, permute_111);  permute_109 = permute_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    mul_561: "f32[32, 1024, 1024]" = torch.ops.aten.mul.Tensor(bmm_21, alias_27);  bmm_21 = None
    sum_57: "f32[32, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_561, [-1], True)
    mul_562: "f32[32, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_27, sum_57);  alias_27 = sum_57 = None
    sub_145: "f32[32, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_561, mul_562);  mul_561 = mul_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    view_175: "f32[32, 32, 32, 32, 32]" = torch.ops.aten.reshape.default(sub_145, [32, 32, 32, 32, 32])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_112: "f32[32, 32, 32, 32, 32]" = torch.ops.aten.permute.default(view_175, [0, 2, 4, 1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    sum_58: "f32[32, 32, 1, 32, 32]" = torch.ops.aten.sum.dim_IntList(permute_112, [2], True);  permute_112 = None
    view_176: "f32[1024, 32, 32]" = torch.ops.aten.reshape.default(sum_58, [1024, 32, 32]);  sum_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    full_default_38: "f32[1024, 32, 63]" = torch.ops.aten.full.default([1024, 32, 63], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_18: "f32[1024, 32, 63]" = torch.ops.aten.slice_scatter.default(full_default_38, view_176, 2, 31, 9223372036854775807);  view_176 = None
    full_default_39: "f32[1024, 33, 63]" = torch.ops.aten.full.default([1024, 33, 63], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_19: "f32[1024, 33, 63]" = torch.ops.aten.slice_scatter.default(full_default_39, slice_scatter_18, 1, 0, 32);  slice_scatter_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_177: "f32[1024, 2079]" = torch.ops.aten.reshape.default(slice_scatter_19, [1024, 2079]);  slice_scatter_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_28: "f32[1024, 2048]" = torch.ops.aten.constant_pad_nd.default(view_177, [0, -31]);  view_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_178: "f32[1024, 32, 64]" = torch.ops.aten.reshape.default(constant_pad_nd_28, [1024, 32, 64]);  constant_pad_nd_28 = None
    constant_pad_nd_29: "f32[1024, 32, 63]" = torch.ops.aten.constant_pad_nd.default(view_178, [0, -1]);  view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_179: "f32[32, 32, 32, 63]" = torch.ops.aten.reshape.default(constant_pad_nd_29, [32, 32, 32, 63]);  constant_pad_nd_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    view_180: "f32[32768, 63]" = torch.ops.aten.reshape.default(view_179, [32768, 63]);  view_179 = None
    permute_113: "f32[63, 32768]" = torch.ops.aten.permute.default(view_180, [1, 0])
    mm_22: "f32[63, 32]" = torch.ops.aten.mm.default(permute_113, view_13);  permute_113 = view_13 = None
    permute_114: "f32[32, 63]" = torch.ops.aten.permute.default(mm_22, [1, 0]);  mm_22 = None
    mm_23: "f32[32768, 32]" = torch.ops.aten.mm.default(view_180, permute_115);  view_180 = permute_115 = None
    view_181: "f32[32, 32, 32, 32]" = torch.ops.aten.reshape.default(mm_23, [32, 32, 32, 32]);  mm_23 = None
    permute_116: "f32[63, 32]" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    permute_117: "f32[32, 32, 32, 32]" = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_118: "f32[32, 32, 32, 32, 32]" = torch.ops.aten.permute.default(view_175, [0, 1, 3, 2, 4]);  view_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    sum_59: "f32[32, 32, 1, 32, 32]" = torch.ops.aten.sum.dim_IntList(permute_118, [2], True);  permute_118 = None
    view_182: "f32[1024, 32, 32]" = torch.ops.aten.reshape.default(sum_59, [1024, 32, 32]);  sum_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_scatter_21: "f32[1024, 32, 63]" = torch.ops.aten.slice_scatter.default(full_default_38, view_182, 2, 31, 9223372036854775807);  full_default_38 = view_182 = None
    slice_scatter_22: "f32[1024, 33, 63]" = torch.ops.aten.slice_scatter.default(full_default_39, slice_scatter_21, 1, 0, 32);  full_default_39 = slice_scatter_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_183: "f32[1024, 2079]" = torch.ops.aten.reshape.default(slice_scatter_22, [1024, 2079]);  slice_scatter_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_30: "f32[1024, 2048]" = torch.ops.aten.constant_pad_nd.default(view_183, [0, -31]);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_184: "f32[1024, 32, 64]" = torch.ops.aten.reshape.default(constant_pad_nd_30, [1024, 32, 64]);  constant_pad_nd_30 = None
    constant_pad_nd_31: "f32[1024, 32, 63]" = torch.ops.aten.constant_pad_nd.default(view_184, [0, -1]);  view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_185: "f32[32, 32, 32, 63]" = torch.ops.aten.reshape.default(constant_pad_nd_31, [32, 32, 32, 63]);  constant_pad_nd_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    view_186: "f32[32768, 63]" = torch.ops.aten.reshape.default(view_185, [32768, 63]);  view_185 = None
    permute_119: "f32[63, 32768]" = torch.ops.aten.permute.default(view_186, [1, 0])
    mm_24: "f32[63, 32]" = torch.ops.aten.mm.default(permute_119, view_7);  permute_119 = view_7 = None
    permute_120: "f32[32, 63]" = torch.ops.aten.permute.default(mm_24, [1, 0]);  mm_24 = None
    mm_25: "f32[32768, 32]" = torch.ops.aten.mm.default(view_186, permute_121);  view_186 = permute_121 = None
    view_187: "f32[32, 32, 32, 32]" = torch.ops.aten.reshape.default(mm_25, [32, 32, 32, 32]);  mm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    add_239: "f32[32, 32, 32, 32]" = torch.ops.aten.add.Tensor(permute_117, view_187);  permute_117 = view_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_122: "f32[63, 32]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    clone_72: "f32[32, 32, 32, 32]" = torch.ops.aten.clone.default(add_239, memory_format = torch.contiguous_format);  add_239 = None
    view_188: "f32[32, 1024, 32]" = torch.ops.aten.reshape.default(clone_72, [32, 1024, 32]);  clone_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    mul_563: "f32[32, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_145, 0.1767766952966369);  sub_145 = None
    bmm_22: "f32[32, 32, 1024]" = torch.ops.aten.bmm.default(permute_123, mul_563);  permute_123 = None
    bmm_23: "f32[32, 1024, 32]" = torch.ops.aten.bmm.default(mul_563, permute_124);  mul_563 = permute_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    add_240: "f32[32, 1024, 32]" = torch.ops.aten.add.Tensor(view_188, bmm_23);  view_188 = bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    permute_125: "f32[32, 32, 1024]" = torch.ops.aten.permute.default(bmm_20, [0, 2, 1]);  bmm_20 = None
    clone_73: "f32[32, 32, 1024]" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
    view_192: "f32[8, 128, 32, 32]" = torch.ops.aten.reshape.default(clone_73, [8, 128, 32, 32]);  clone_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    view_193: "f32[8, 128, 32, 32]" = torch.ops.aten.reshape.default(bmm_22, [8, 128, 32, 32]);  bmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    permute_126: "f32[32, 32, 1024]" = torch.ops.aten.permute.default(add_240, [0, 2, 1]);  add_240 = None
    clone_74: "f32[32, 32, 1024]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
    view_194: "f32[8, 128, 32, 32]" = torch.ops.aten.reshape.default(clone_74, [8, 128, 32, 32]);  clone_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    cat_3: "f32[8, 384, 32, 32]" = torch.ops.aten.cat.default([view_194, view_193, view_192], 1);  view_194 = view_193 = view_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(cat_3, mul_145, primals_119, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  cat_3 = mul_145 = primals_119 = None
    getitem_157: "f32[8, 128, 32, 32]" = convolution_backward_23[0]
    getitem_158: "f32[384, 128, 1, 1]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_566: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_157, mul_565);  getitem_157 = mul_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_60: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_566, [0, 2, 3])
    sub_147: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_394);  convolution_25 = unsqueeze_394 = None
    mul_567: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_566, sub_147)
    sum_61: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_567, [0, 2, 3]);  mul_567 = None
    mul_568: "f32[128]" = torch.ops.aten.mul.Tensor(sum_60, 0.0001220703125)
    unsqueeze_395: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_568, 0);  mul_568 = None
    unsqueeze_396: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 2);  unsqueeze_395 = None
    unsqueeze_397: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, 3);  unsqueeze_396 = None
    mul_569: "f32[128]" = torch.ops.aten.mul.Tensor(sum_61, 0.0001220703125)
    mul_570: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_571: "f32[128]" = torch.ops.aten.mul.Tensor(mul_569, mul_570);  mul_569 = mul_570 = None
    unsqueeze_398: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_571, 0);  mul_571 = None
    unsqueeze_399: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 2);  unsqueeze_398 = None
    unsqueeze_400: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 3);  unsqueeze_399 = None
    mul_572: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_35);  primals_35 = None
    unsqueeze_401: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_572, 0);  mul_572 = None
    unsqueeze_402: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 2);  unsqueeze_401 = None
    unsqueeze_403: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, 3);  unsqueeze_402 = None
    mul_573: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_400);  sub_147 = unsqueeze_400 = None
    sub_149: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(mul_566, mul_573);  mul_566 = mul_573 = None
    sub_150: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_149, unsqueeze_397);  sub_149 = unsqueeze_397 = None
    mul_574: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_150, unsqueeze_403);  sub_150 = unsqueeze_403 = None
    mul_575: "f32[128]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_52);  sum_61 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_574, mul_137, primals_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_574 = mul_137 = primals_118 = None
    getitem_160: "f32[8, 512, 32, 32]" = convolution_backward_24[0]
    getitem_161: "f32[128, 512, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_242: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_539, getitem_160);  mul_539 = getitem_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    mul_578: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_242, mul_577);  add_242 = mul_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_62: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_578, [0, 2, 3])
    sub_152: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_406);  convolution_24 = unsqueeze_406 = None
    mul_579: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_578, sub_152)
    sum_63: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_579, [0, 2, 3]);  mul_579 = None
    mul_580: "f32[512]" = torch.ops.aten.mul.Tensor(sum_62, 0.0001220703125)
    unsqueeze_407: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_580, 0);  mul_580 = None
    unsqueeze_408: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 2);  unsqueeze_407 = None
    unsqueeze_409: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, 3);  unsqueeze_408 = None
    mul_581: "f32[512]" = torch.ops.aten.mul.Tensor(sum_63, 0.0001220703125)
    mul_582: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_583: "f32[512]" = torch.ops.aten.mul.Tensor(mul_581, mul_582);  mul_581 = mul_582 = None
    unsqueeze_410: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_583, 0);  mul_583 = None
    unsqueeze_411: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 2);  unsqueeze_410 = None
    unsqueeze_412: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 3);  unsqueeze_411 = None
    mul_584: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_33);  primals_33 = None
    unsqueeze_413: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_584, 0);  mul_584 = None
    unsqueeze_414: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 2);  unsqueeze_413 = None
    unsqueeze_415: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, 3);  unsqueeze_414 = None
    mul_585: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_412);  sub_152 = unsqueeze_412 = None
    sub_154: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(mul_578, mul_585);  mul_585 = None
    sub_155: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(sub_154, unsqueeze_409);  sub_154 = unsqueeze_409 = None
    mul_586: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_415);  sub_155 = unsqueeze_415 = None
    mul_587: "f32[512]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_49);  sum_63 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_586, mul_129, primals_117, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_586 = mul_129 = primals_117 = None
    getitem_163: "f32[8, 128, 32, 32]" = convolution_backward_25[0]
    getitem_164: "f32[512, 128, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_588: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_163, mul_128);  mul_128 = None
    mul_589: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_163, sigmoid_17);  getitem_163 = None
    sum_64: "f32[8, 128, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_588, [2, 3], True);  mul_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_156: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_17)
    mul_590: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_17, sub_156);  sigmoid_17 = sub_156 = None
    mul_591: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sum_64, mul_590);  sum_64 = mul_590 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_591, relu_3, primals_115, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_591 = primals_115 = None
    getitem_166: "f32[8, 8, 1, 1]" = convolution_backward_26[0]
    getitem_167: "f32[128, 8, 1, 1]" = convolution_backward_26[1]
    getitem_168: "f32[128]" = convolution_backward_26[2];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_2: "b8[8, 8, 1, 1]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
    where_2: "f32[8, 8, 1, 1]" = torch.ops.aten.where.self(le_2, full_default_29, getitem_166);  le_2 = getitem_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(where_2, mean_3, primals_113, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_2 = mean_3 = primals_113 = None
    getitem_169: "f32[8, 128, 1, 1]" = convolution_backward_27[0]
    getitem_170: "f32[8, 128, 1, 1]" = convolution_backward_27[1]
    getitem_171: "f32[8]" = convolution_backward_27[2];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_27: "f32[8, 128, 32, 32]" = torch.ops.aten.expand.default(getitem_169, [8, 128, 32, 32]);  getitem_169 = None
    div_7: "f32[8, 128, 32, 32]" = torch.ops.aten.div.Scalar(expand_27, 1024);  expand_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_244: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_589, div_7);  mul_589 = div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_60: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(add_82)
    sub_157: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(full_default_37, sigmoid_60)
    mul_592: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_82, sub_157);  add_82 = sub_157 = None
    add_245: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Scalar(mul_592, 1);  mul_592 = None
    mul_593: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_60, add_245);  sigmoid_60 = add_245 = None
    mul_594: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_244, mul_593);  add_244 = mul_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_65: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_594, [0, 2, 3])
    sub_158: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_418);  convolution_21 = unsqueeze_418 = None
    mul_595: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_594, sub_158)
    sum_66: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_595, [0, 2, 3]);  mul_595 = None
    mul_596: "f32[128]" = torch.ops.aten.mul.Tensor(sum_65, 0.0001220703125)
    unsqueeze_419: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_596, 0);  mul_596 = None
    unsqueeze_420: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 2);  unsqueeze_419 = None
    unsqueeze_421: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, 3);  unsqueeze_420 = None
    mul_597: "f32[128]" = torch.ops.aten.mul.Tensor(sum_66, 0.0001220703125)
    mul_598: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_599: "f32[128]" = torch.ops.aten.mul.Tensor(mul_597, mul_598);  mul_597 = mul_598 = None
    unsqueeze_422: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_599, 0);  mul_599 = None
    unsqueeze_423: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 2);  unsqueeze_422 = None
    unsqueeze_424: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 3);  unsqueeze_423 = None
    mul_600: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_31);  primals_31 = None
    unsqueeze_425: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_600, 0);  mul_600 = None
    unsqueeze_426: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 2);  unsqueeze_425 = None
    unsqueeze_427: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 3);  unsqueeze_426 = None
    mul_601: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_158, unsqueeze_424);  sub_158 = unsqueeze_424 = None
    sub_160: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(mul_594, mul_601);  mul_594 = mul_601 = None
    sub_161: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_160, unsqueeze_421);  sub_160 = unsqueeze_421 = None
    mul_602: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_427);  sub_161 = unsqueeze_427 = None
    mul_603: "f32[128]" = torch.ops.aten.mul.Tensor(sum_66, squeeze_46);  sum_66 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_602, mul_120, primals_112, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_602 = mul_120 = primals_112 = None
    getitem_172: "f32[8, 128, 32, 32]" = convolution_backward_28[0]
    getitem_173: "f32[128, 128, 3, 3]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_606: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_172, mul_605);  getitem_172 = mul_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_67: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_606, [0, 2, 3])
    sub_163: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_430);  convolution_20 = unsqueeze_430 = None
    mul_607: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_606, sub_163)
    sum_68: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_607, [0, 2, 3]);  mul_607 = None
    mul_608: "f32[128]" = torch.ops.aten.mul.Tensor(sum_67, 0.0001220703125)
    unsqueeze_431: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_608, 0);  mul_608 = None
    unsqueeze_432: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 2);  unsqueeze_431 = None
    unsqueeze_433: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, 3);  unsqueeze_432 = None
    mul_609: "f32[128]" = torch.ops.aten.mul.Tensor(sum_68, 0.0001220703125)
    mul_610: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_611: "f32[128]" = torch.ops.aten.mul.Tensor(mul_609, mul_610);  mul_609 = mul_610 = None
    unsqueeze_434: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_611, 0);  mul_611 = None
    unsqueeze_435: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 2);  unsqueeze_434 = None
    unsqueeze_436: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 3);  unsqueeze_435 = None
    mul_612: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_29);  primals_29 = None
    unsqueeze_437: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_612, 0);  mul_612 = None
    unsqueeze_438: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 2);  unsqueeze_437 = None
    unsqueeze_439: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 3);  unsqueeze_438 = None
    mul_613: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_163, unsqueeze_436);  sub_163 = unsqueeze_436 = None
    sub_165: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(mul_606, mul_613);  mul_606 = mul_613 = None
    sub_166: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_165, unsqueeze_433);  sub_165 = unsqueeze_433 = None
    mul_614: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_439);  sub_166 = unsqueeze_439 = None
    mul_615: "f32[128]" = torch.ops.aten.mul.Tensor(sum_68, squeeze_43);  sum_68 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_614, mul_112, primals_111, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_614 = mul_112 = primals_111 = None
    getitem_175: "f32[8, 512, 32, 32]" = convolution_backward_29[0]
    getitem_176: "f32[128, 512, 1, 1]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_247: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_578, getitem_175);  mul_578 = getitem_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    mul_618: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_247, mul_617);  add_247 = mul_617 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_69: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_618, [0, 2, 3])
    sub_168: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_442);  convolution_19 = unsqueeze_442 = None
    mul_619: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_618, sub_168)
    sum_70: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_619, [0, 2, 3]);  mul_619 = None
    mul_620: "f32[512]" = torch.ops.aten.mul.Tensor(sum_69, 0.0001220703125)
    unsqueeze_443: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_620, 0);  mul_620 = None
    unsqueeze_444: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 2);  unsqueeze_443 = None
    unsqueeze_445: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, 3);  unsqueeze_444 = None
    mul_621: "f32[512]" = torch.ops.aten.mul.Tensor(sum_70, 0.0001220703125)
    mul_622: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_623: "f32[512]" = torch.ops.aten.mul.Tensor(mul_621, mul_622);  mul_621 = mul_622 = None
    unsqueeze_446: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_623, 0);  mul_623 = None
    unsqueeze_447: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 2);  unsqueeze_446 = None
    unsqueeze_448: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 3);  unsqueeze_447 = None
    mul_624: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_27);  primals_27 = None
    unsqueeze_449: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_624, 0);  mul_624 = None
    unsqueeze_450: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 2);  unsqueeze_449 = None
    unsqueeze_451: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 3);  unsqueeze_450 = None
    mul_625: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_448);  sub_168 = unsqueeze_448 = None
    sub_170: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(mul_618, mul_625);  mul_625 = None
    sub_171: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(sub_170, unsqueeze_445);  sub_170 = None
    mul_626: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_451);  sub_171 = unsqueeze_451 = None
    mul_627: "f32[512]" = torch.ops.aten.mul.Tensor(sum_70, squeeze_40);  sum_70 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_626, mul_80, primals_110, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_626 = primals_110 = None
    getitem_178: "f32[8, 256, 64, 64]" = convolution_backward_30[0]
    getitem_179: "f32[512, 256, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_172: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_454);  convolution_18 = unsqueeze_454 = None
    mul_628: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_618, sub_172)
    sum_72: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_628, [0, 2, 3]);  mul_628 = None
    mul_630: "f32[512]" = torch.ops.aten.mul.Tensor(sum_72, 0.0001220703125)
    mul_631: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_632: "f32[512]" = torch.ops.aten.mul.Tensor(mul_630, mul_631);  mul_630 = mul_631 = None
    unsqueeze_458: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_632, 0);  mul_632 = None
    unsqueeze_459: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 2);  unsqueeze_458 = None
    unsqueeze_460: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 3);  unsqueeze_459 = None
    mul_633: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_25);  primals_25 = None
    unsqueeze_461: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_633, 0);  mul_633 = None
    unsqueeze_462: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None
    unsqueeze_463: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 3);  unsqueeze_462 = None
    mul_634: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_460);  sub_172 = unsqueeze_460 = None
    sub_174: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(mul_618, mul_634);  mul_618 = mul_634 = None
    sub_175: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(sub_174, unsqueeze_445);  sub_174 = unsqueeze_445 = None
    mul_635: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_463);  sub_175 = unsqueeze_463 = None
    mul_636: "f32[512]" = torch.ops.aten.mul.Tensor(sum_72, squeeze_37);  sum_72 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_635, mul_97, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_635 = mul_97 = primals_109 = None
    getitem_181: "f32[8, 128, 32, 32]" = convolution_backward_31[0]
    getitem_182: "f32[512, 128, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_637: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_181, mul_96);  mul_96 = None
    mul_638: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_181, sigmoid_13);  getitem_181 = None
    sum_73: "f32[8, 128, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_637, [2, 3], True);  mul_637 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_176: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_13)
    mul_639: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_13, sub_176);  sigmoid_13 = sub_176 = None
    mul_640: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sum_73, mul_639);  sum_73 = mul_639 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_640, relu_2, primals_107, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_640 = primals_107 = None
    getitem_184: "f32[8, 8, 1, 1]" = convolution_backward_32[0]
    getitem_185: "f32[128, 8, 1, 1]" = convolution_backward_32[1]
    getitem_186: "f32[128]" = convolution_backward_32[2];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_3: "b8[8, 8, 1, 1]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
    where_3: "f32[8, 8, 1, 1]" = torch.ops.aten.where.self(le_3, full_default_29, getitem_184);  le_3 = getitem_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(where_3, mean_2, primals_105, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_3 = mean_2 = primals_105 = None
    getitem_187: "f32[8, 128, 1, 1]" = convolution_backward_33[0]
    getitem_188: "f32[8, 128, 1, 1]" = convolution_backward_33[1]
    getitem_189: "f32[8]" = convolution_backward_33[2];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_28: "f32[8, 128, 32, 32]" = torch.ops.aten.expand.default(getitem_187, [8, 128, 32, 32]);  getitem_187 = None
    div_8: "f32[8, 128, 32, 32]" = torch.ops.aten.div.Scalar(expand_28, 1024);  expand_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_249: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_638, div_8);  mul_638 = div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_63: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(add_61)
    sub_177: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(full_default_37, sigmoid_63);  full_default_37 = None
    mul_641: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_61, sub_177);  add_61 = sub_177 = None
    add_250: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Scalar(mul_641, 1);  mul_641 = None
    mul_642: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_63, add_250);  sigmoid_63 = add_250 = None
    mul_643: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_249, mul_642);  add_249 = mul_642 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_74: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_643, [0, 2, 3])
    sub_178: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_466);  convolution_15 = unsqueeze_466 = None
    mul_644: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_643, sub_178)
    sum_75: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_644, [0, 2, 3]);  mul_644 = None
    mul_645: "f32[128]" = torch.ops.aten.mul.Tensor(sum_74, 0.0001220703125)
    unsqueeze_467: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_645, 0);  mul_645 = None
    unsqueeze_468: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 2);  unsqueeze_467 = None
    unsqueeze_469: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, 3);  unsqueeze_468 = None
    mul_646: "f32[128]" = torch.ops.aten.mul.Tensor(sum_75, 0.0001220703125)
    mul_647: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_648: "f32[128]" = torch.ops.aten.mul.Tensor(mul_646, mul_647);  mul_646 = mul_647 = None
    unsqueeze_470: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_648, 0);  mul_648 = None
    unsqueeze_471: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 2);  unsqueeze_470 = None
    unsqueeze_472: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 3);  unsqueeze_471 = None
    mul_649: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_23);  primals_23 = None
    unsqueeze_473: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_649, 0);  mul_649 = None
    unsqueeze_474: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 2);  unsqueeze_473 = None
    unsqueeze_475: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, 3);  unsqueeze_474 = None
    mul_650: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_178, unsqueeze_472);  sub_178 = unsqueeze_472 = None
    sub_180: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(mul_643, mul_650);  mul_643 = mul_650 = None
    sub_181: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_180, unsqueeze_469);  sub_180 = unsqueeze_469 = None
    mul_651: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_475);  sub_181 = unsqueeze_475 = None
    mul_652: "f32[128]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_34);  sum_75 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_651, mul_88, primals_104, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_651 = mul_88 = primals_104 = None
    getitem_190: "f32[8, 128, 64, 64]" = convolution_backward_34[0]
    getitem_191: "f32[128, 128, 3, 3]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_655: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_190, mul_654);  getitem_190 = mul_654 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_76: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_655, [0, 2, 3])
    sub_183: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_478);  convolution_14 = unsqueeze_478 = None
    mul_656: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_655, sub_183)
    sum_77: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_656, [0, 2, 3]);  mul_656 = None
    mul_657: "f32[128]" = torch.ops.aten.mul.Tensor(sum_76, 3.0517578125e-05)
    unsqueeze_479: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_657, 0);  mul_657 = None
    unsqueeze_480: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 2);  unsqueeze_479 = None
    unsqueeze_481: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, 3);  unsqueeze_480 = None
    mul_658: "f32[128]" = torch.ops.aten.mul.Tensor(sum_77, 3.0517578125e-05)
    mul_659: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_660: "f32[128]" = torch.ops.aten.mul.Tensor(mul_658, mul_659);  mul_658 = mul_659 = None
    unsqueeze_482: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_660, 0);  mul_660 = None
    unsqueeze_483: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 2);  unsqueeze_482 = None
    unsqueeze_484: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 3);  unsqueeze_483 = None
    mul_661: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_21);  primals_21 = None
    unsqueeze_485: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_661, 0);  mul_661 = None
    unsqueeze_486: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 2);  unsqueeze_485 = None
    unsqueeze_487: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 3);  unsqueeze_486 = None
    mul_662: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_183, unsqueeze_484);  sub_183 = unsqueeze_484 = None
    sub_185: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(mul_655, mul_662);  mul_655 = mul_662 = None
    sub_186: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(sub_185, unsqueeze_481);  sub_185 = unsqueeze_481 = None
    mul_663: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_186, unsqueeze_487);  sub_186 = unsqueeze_487 = None
    mul_664: "f32[128]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_31);  sum_77 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_663, mul_80, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_663 = mul_80 = primals_103 = None
    getitem_193: "f32[8, 256, 64, 64]" = convolution_backward_35[0]
    getitem_194: "f32[128, 256, 1, 1]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_252: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(getitem_178, getitem_193);  getitem_178 = getitem_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    mul_667: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_252, mul_666);  add_252 = mul_666 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_78: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_667, [0, 2, 3])
    sub_188: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_490);  convolution_13 = unsqueeze_490 = None
    mul_668: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_667, sub_188)
    sum_79: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_668, [0, 2, 3]);  mul_668 = None
    mul_669: "f32[256]" = torch.ops.aten.mul.Tensor(sum_78, 3.0517578125e-05)
    unsqueeze_491: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_669, 0);  mul_669 = None
    unsqueeze_492: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 2);  unsqueeze_491 = None
    unsqueeze_493: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, 3);  unsqueeze_492 = None
    mul_670: "f32[256]" = torch.ops.aten.mul.Tensor(sum_79, 3.0517578125e-05)
    mul_671: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_672: "f32[256]" = torch.ops.aten.mul.Tensor(mul_670, mul_671);  mul_670 = mul_671 = None
    unsqueeze_494: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_672, 0);  mul_672 = None
    unsqueeze_495: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 2);  unsqueeze_494 = None
    unsqueeze_496: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 3);  unsqueeze_495 = None
    mul_673: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_19);  primals_19 = None
    unsqueeze_497: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_673, 0);  mul_673 = None
    unsqueeze_498: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 2);  unsqueeze_497 = None
    unsqueeze_499: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 3);  unsqueeze_498 = None
    mul_674: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_496);  sub_188 = unsqueeze_496 = None
    sub_190: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(mul_667, mul_674);  mul_674 = None
    sub_191: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(sub_190, unsqueeze_493);  sub_190 = unsqueeze_493 = None
    mul_675: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_191, unsqueeze_499);  sub_191 = unsqueeze_499 = None
    mul_676: "f32[256]" = torch.ops.aten.mul.Tensor(sum_79, squeeze_28);  sum_79 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_675, mul_72, primals_102, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_675 = mul_72 = primals_102 = None
    getitem_196: "f32[8, 64, 64, 64]" = convolution_backward_36[0]
    getitem_197: "f32[256, 64, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_677: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_196, mul_71);  mul_71 = None
    mul_678: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_196, sigmoid_9);  getitem_196 = None
    sum_80: "f32[8, 64, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_677, [2, 3], True);  mul_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_192: "f32[8, 64, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_9)
    mul_679: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_9, sub_192);  sigmoid_9 = sub_192 = None
    mul_680: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(sum_80, mul_679);  sum_80 = mul_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_680, relu_1, primals_100, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_680 = primals_100 = None
    getitem_199: "f32[8, 8, 1, 1]" = convolution_backward_37[0]
    getitem_200: "f32[64, 8, 1, 1]" = convolution_backward_37[1]
    getitem_201: "f32[64]" = convolution_backward_37[2];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_4: "b8[8, 8, 1, 1]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
    where_4: "f32[8, 8, 1, 1]" = torch.ops.aten.where.self(le_4, full_default_29, getitem_199);  le_4 = getitem_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(where_4, mean_1, primals_98, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_4 = mean_1 = primals_98 = None
    getitem_202: "f32[8, 64, 1, 1]" = convolution_backward_38[0]
    getitem_203: "f32[8, 64, 1, 1]" = convolution_backward_38[1]
    getitem_204: "f32[8]" = convolution_backward_38[2];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_29: "f32[8, 64, 64, 64]" = torch.ops.aten.expand.default(getitem_202, [8, 64, 64, 64]);  getitem_202 = None
    div_9: "f32[8, 64, 64, 64]" = torch.ops.aten.div.Scalar(expand_29, 4096);  expand_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_254: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_678, div_9);  mul_678 = div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_66: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_45)
    full_default_55: "f32[8, 64, 64, 64]" = torch.ops.aten.full.default([8, 64, 64, 64], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_193: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(full_default_55, sigmoid_66)
    mul_681: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_45, sub_193);  add_45 = sub_193 = None
    add_255: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Scalar(mul_681, 1);  mul_681 = None
    mul_682: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_66, add_255);  sigmoid_66 = add_255 = None
    mul_683: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_254, mul_682);  add_254 = mul_682 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_81: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_683, [0, 2, 3])
    sub_194: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_502);  convolution_10 = unsqueeze_502 = None
    mul_684: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_683, sub_194)
    sum_82: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_684, [0, 2, 3]);  mul_684 = None
    mul_685: "f32[64]" = torch.ops.aten.mul.Tensor(sum_81, 3.0517578125e-05)
    unsqueeze_503: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_685, 0);  mul_685 = None
    unsqueeze_504: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 2);  unsqueeze_503 = None
    unsqueeze_505: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, 3);  unsqueeze_504 = None
    mul_686: "f32[64]" = torch.ops.aten.mul.Tensor(sum_82, 3.0517578125e-05)
    mul_687: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_688: "f32[64]" = torch.ops.aten.mul.Tensor(mul_686, mul_687);  mul_686 = mul_687 = None
    unsqueeze_506: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_688, 0);  mul_688 = None
    unsqueeze_507: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 2);  unsqueeze_506 = None
    unsqueeze_508: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 3);  unsqueeze_507 = None
    mul_689: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_17);  primals_17 = None
    unsqueeze_509: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_689, 0);  mul_689 = None
    unsqueeze_510: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 2);  unsqueeze_509 = None
    unsqueeze_511: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, 3);  unsqueeze_510 = None
    mul_690: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_194, unsqueeze_508);  sub_194 = unsqueeze_508 = None
    sub_196: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(mul_683, mul_690);  mul_683 = mul_690 = None
    sub_197: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_196, unsqueeze_505);  sub_196 = unsqueeze_505 = None
    mul_691: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_511);  sub_197 = unsqueeze_511 = None
    mul_692: "f32[64]" = torch.ops.aten.mul.Tensor(sum_82, squeeze_25);  sum_82 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_691, mul_63, primals_97, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_691 = mul_63 = primals_97 = None
    getitem_205: "f32[8, 64, 64, 64]" = convolution_backward_39[0]
    getitem_206: "f32[64, 64, 3, 3]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_695: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_205, mul_694);  getitem_205 = mul_694 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_83: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_695, [0, 2, 3])
    sub_199: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_514);  convolution_9 = unsqueeze_514 = None
    mul_696: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_695, sub_199)
    sum_84: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_696, [0, 2, 3]);  mul_696 = None
    mul_697: "f32[64]" = torch.ops.aten.mul.Tensor(sum_83, 3.0517578125e-05)
    unsqueeze_515: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_697, 0);  mul_697 = None
    unsqueeze_516: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 2);  unsqueeze_515 = None
    unsqueeze_517: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, 3);  unsqueeze_516 = None
    mul_698: "f32[64]" = torch.ops.aten.mul.Tensor(sum_84, 3.0517578125e-05)
    mul_699: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_700: "f32[64]" = torch.ops.aten.mul.Tensor(mul_698, mul_699);  mul_698 = mul_699 = None
    unsqueeze_518: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_700, 0);  mul_700 = None
    unsqueeze_519: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 2);  unsqueeze_518 = None
    unsqueeze_520: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_519, 3);  unsqueeze_519 = None
    mul_701: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_15);  primals_15 = None
    unsqueeze_521: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_701, 0);  mul_701 = None
    unsqueeze_522: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 2);  unsqueeze_521 = None
    unsqueeze_523: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, 3);  unsqueeze_522 = None
    mul_702: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_199, unsqueeze_520);  sub_199 = unsqueeze_520 = None
    sub_201: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(mul_695, mul_702);  mul_695 = mul_702 = None
    sub_202: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_201, unsqueeze_517);  sub_201 = unsqueeze_517 = None
    mul_703: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_202, unsqueeze_523);  sub_202 = unsqueeze_523 = None
    mul_704: "f32[64]" = torch.ops.aten.mul.Tensor(sum_84, squeeze_22);  sum_84 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_703, mul_55, primals_96, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_703 = mul_55 = primals_96 = None
    getitem_208: "f32[8, 256, 64, 64]" = convolution_backward_40[0]
    getitem_209: "f32[64, 256, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_257: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_667, getitem_208);  mul_667 = getitem_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    mul_707: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_257, mul_706);  add_257 = mul_706 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_85: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_707, [0, 2, 3])
    sub_204: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_526);  convolution_8 = unsqueeze_526 = None
    mul_708: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_707, sub_204)
    sum_86: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_708, [0, 2, 3]);  mul_708 = None
    mul_709: "f32[256]" = torch.ops.aten.mul.Tensor(sum_85, 3.0517578125e-05)
    unsqueeze_527: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_709, 0);  mul_709 = None
    unsqueeze_528: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 2);  unsqueeze_527 = None
    unsqueeze_529: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, 3);  unsqueeze_528 = None
    mul_710: "f32[256]" = torch.ops.aten.mul.Tensor(sum_86, 3.0517578125e-05)
    mul_711: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_712: "f32[256]" = torch.ops.aten.mul.Tensor(mul_710, mul_711);  mul_710 = mul_711 = None
    unsqueeze_530: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_712, 0);  mul_712 = None
    unsqueeze_531: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 2);  unsqueeze_530 = None
    unsqueeze_532: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_531, 3);  unsqueeze_531 = None
    mul_713: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_13);  primals_13 = None
    unsqueeze_533: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_713, 0);  mul_713 = None
    unsqueeze_534: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 2);  unsqueeze_533 = None
    unsqueeze_535: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, 3);  unsqueeze_534 = None
    mul_714: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_532);  sub_204 = unsqueeze_532 = None
    sub_206: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(mul_707, mul_714);  mul_714 = None
    sub_207: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(sub_206, unsqueeze_529);  sub_206 = None
    mul_715: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_207, unsqueeze_535);  sub_207 = unsqueeze_535 = None
    mul_716: "f32[256]" = torch.ops.aten.mul.Tensor(sum_86, squeeze_19);  sum_86 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_715, mul_23, primals_95, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_715 = primals_95 = None
    getitem_211: "f32[8, 64, 64, 64]" = convolution_backward_41[0]
    getitem_212: "f32[256, 64, 1, 1]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_208: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_538);  convolution_7 = unsqueeze_538 = None
    mul_717: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_707, sub_208)
    sum_88: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_717, [0, 2, 3]);  mul_717 = None
    mul_719: "f32[256]" = torch.ops.aten.mul.Tensor(sum_88, 3.0517578125e-05)
    mul_720: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_721: "f32[256]" = torch.ops.aten.mul.Tensor(mul_719, mul_720);  mul_719 = mul_720 = None
    unsqueeze_542: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_721, 0);  mul_721 = None
    unsqueeze_543: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 2);  unsqueeze_542 = None
    unsqueeze_544: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_543, 3);  unsqueeze_543 = None
    mul_722: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_11);  primals_11 = None
    unsqueeze_545: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_722, 0);  mul_722 = None
    unsqueeze_546: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 2);  unsqueeze_545 = None
    unsqueeze_547: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, 3);  unsqueeze_546 = None
    mul_723: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_544);  sub_208 = unsqueeze_544 = None
    sub_210: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(mul_707, mul_723);  mul_707 = mul_723 = None
    sub_211: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(sub_210, unsqueeze_529);  sub_210 = unsqueeze_529 = None
    mul_724: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_547);  sub_211 = unsqueeze_547 = None
    mul_725: "f32[256]" = torch.ops.aten.mul.Tensor(sum_88, squeeze_16);  sum_88 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_724, mul_40, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_724 = mul_40 = primals_94 = None
    getitem_214: "f32[8, 64, 64, 64]" = convolution_backward_42[0]
    getitem_215: "f32[256, 64, 1, 1]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_726: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_214, mul_39);  mul_39 = None
    mul_727: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_214, sigmoid_5);  getitem_214 = None
    sum_89: "f32[8, 64, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_726, [2, 3], True);  mul_726 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_212: "f32[8, 64, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_5)
    mul_728: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_5, sub_212);  sigmoid_5 = sub_212 = None
    mul_729: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(sum_89, mul_728);  sum_89 = mul_728 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_729, relu, primals_92, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_729 = primals_92 = None
    getitem_217: "f32[8, 8, 1, 1]" = convolution_backward_43[0]
    getitem_218: "f32[64, 8, 1, 1]" = convolution_backward_43[1]
    getitem_219: "f32[64]" = convolution_backward_43[2];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_5: "b8[8, 8, 1, 1]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
    where_5: "f32[8, 8, 1, 1]" = torch.ops.aten.where.self(le_5, full_default_29, getitem_217);  le_5 = full_default_29 = getitem_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(where_5, mean, primals_90, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_5 = mean = primals_90 = None
    getitem_220: "f32[8, 64, 1, 1]" = convolution_backward_44[0]
    getitem_221: "f32[8, 64, 1, 1]" = convolution_backward_44[1]
    getitem_222: "f32[8]" = convolution_backward_44[2];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_30: "f32[8, 64, 64, 64]" = torch.ops.aten.expand.default(getitem_220, [8, 64, 64, 64]);  getitem_220 = None
    div_10: "f32[8, 64, 64, 64]" = torch.ops.aten.div.Scalar(expand_30, 4096);  expand_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_259: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_727, div_10);  mul_727 = div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_69: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_24)
    sub_213: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(full_default_55, sigmoid_69);  full_default_55 = None
    mul_730: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_24, sub_213);  add_24 = sub_213 = None
    add_260: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Scalar(mul_730, 1);  mul_730 = None
    mul_731: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_69, add_260);  sigmoid_69 = add_260 = None
    mul_732: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_259, mul_731);  add_259 = mul_731 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_90: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_732, [0, 2, 3])
    sub_214: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_550);  convolution_4 = unsqueeze_550 = None
    mul_733: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_732, sub_214)
    sum_91: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_733, [0, 2, 3]);  mul_733 = None
    mul_734: "f32[64]" = torch.ops.aten.mul.Tensor(sum_90, 3.0517578125e-05)
    unsqueeze_551: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_734, 0);  mul_734 = None
    unsqueeze_552: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 2);  unsqueeze_551 = None
    unsqueeze_553: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, 3);  unsqueeze_552 = None
    mul_735: "f32[64]" = torch.ops.aten.mul.Tensor(sum_91, 3.0517578125e-05)
    mul_736: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_737: "f32[64]" = torch.ops.aten.mul.Tensor(mul_735, mul_736);  mul_735 = mul_736 = None
    unsqueeze_554: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_737, 0);  mul_737 = None
    unsqueeze_555: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 2);  unsqueeze_554 = None
    unsqueeze_556: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_555, 3);  unsqueeze_555 = None
    mul_738: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_9);  primals_9 = None
    unsqueeze_557: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_738, 0);  mul_738 = None
    unsqueeze_558: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 2);  unsqueeze_557 = None
    unsqueeze_559: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, 3);  unsqueeze_558 = None
    mul_739: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_214, unsqueeze_556);  sub_214 = unsqueeze_556 = None
    sub_216: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(mul_732, mul_739);  mul_732 = mul_739 = None
    sub_217: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_216, unsqueeze_553);  sub_216 = unsqueeze_553 = None
    mul_740: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_559);  sub_217 = unsqueeze_559 = None
    mul_741: "f32[64]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_13);  sum_91 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_740, mul_31, primals_89, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_740 = mul_31 = primals_89 = None
    getitem_223: "f32[8, 64, 64, 64]" = convolution_backward_45[0]
    getitem_224: "f32[64, 64, 3, 3]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_744: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_223, mul_743);  getitem_223 = mul_743 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_92: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_744, [0, 2, 3])
    sub_219: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_562);  convolution_3 = unsqueeze_562 = None
    mul_745: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_744, sub_219)
    sum_93: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_745, [0, 2, 3]);  mul_745 = None
    mul_746: "f32[64]" = torch.ops.aten.mul.Tensor(sum_92, 3.0517578125e-05)
    unsqueeze_563: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_746, 0);  mul_746 = None
    unsqueeze_564: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 2);  unsqueeze_563 = None
    unsqueeze_565: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, 3);  unsqueeze_564 = None
    mul_747: "f32[64]" = torch.ops.aten.mul.Tensor(sum_93, 3.0517578125e-05)
    mul_748: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_749: "f32[64]" = torch.ops.aten.mul.Tensor(mul_747, mul_748);  mul_747 = mul_748 = None
    unsqueeze_566: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_749, 0);  mul_749 = None
    unsqueeze_567: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 2);  unsqueeze_566 = None
    unsqueeze_568: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_567, 3);  unsqueeze_567 = None
    mul_750: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_7);  primals_7 = None
    unsqueeze_569: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_750, 0);  mul_750 = None
    unsqueeze_570: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 2);  unsqueeze_569 = None
    unsqueeze_571: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, 3);  unsqueeze_570 = None
    mul_751: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_568);  sub_219 = unsqueeze_568 = None
    sub_221: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(mul_744, mul_751);  mul_744 = mul_751 = None
    sub_222: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_221, unsqueeze_565);  sub_221 = unsqueeze_565 = None
    mul_752: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_222, unsqueeze_571);  sub_222 = unsqueeze_571 = None
    mul_753: "f32[64]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_10);  sum_93 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_752, mul_23, primals_88, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_752 = mul_23 = primals_88 = None
    getitem_226: "f32[8, 64, 64, 64]" = convolution_backward_46[0]
    getitem_227: "f32[64, 64, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_262: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(getitem_211, getitem_226);  getitem_211 = getitem_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_756: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_262, mul_755);  add_262 = mul_755 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_94: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_756, [0, 2, 3])
    sub_224: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_574);  convolution_2 = unsqueeze_574 = None
    mul_757: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_756, sub_224)
    sum_95: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_757, [0, 2, 3]);  mul_757 = None
    mul_758: "f32[64]" = torch.ops.aten.mul.Tensor(sum_94, 3.0517578125e-05)
    unsqueeze_575: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_758, 0);  mul_758 = None
    unsqueeze_576: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 2);  unsqueeze_575 = None
    unsqueeze_577: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, 3);  unsqueeze_576 = None
    mul_759: "f32[64]" = torch.ops.aten.mul.Tensor(sum_95, 3.0517578125e-05)
    mul_760: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_761: "f32[64]" = torch.ops.aten.mul.Tensor(mul_759, mul_760);  mul_759 = mul_760 = None
    unsqueeze_578: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_761, 0);  mul_761 = None
    unsqueeze_579: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 2);  unsqueeze_578 = None
    unsqueeze_580: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_579, 3);  unsqueeze_579 = None
    mul_762: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_5);  primals_5 = None
    unsqueeze_581: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_762, 0);  mul_762 = None
    unsqueeze_582: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 2);  unsqueeze_581 = None
    unsqueeze_583: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, 3);  unsqueeze_582 = None
    mul_763: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_580);  sub_224 = unsqueeze_580 = None
    sub_226: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(mul_756, mul_763);  mul_756 = mul_763 = None
    sub_227: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_226, unsqueeze_577);  sub_226 = unsqueeze_577 = None
    mul_764: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_583);  sub_227 = unsqueeze_583 = None
    mul_765: "f32[64]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_7);  sum_95 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_764, mul_15, primals_87, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_764 = mul_15 = primals_87 = None
    getitem_229: "f32[8, 32, 128, 128]" = convolution_backward_47[0]
    getitem_230: "f32[64, 32, 3, 3]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_768: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(getitem_229, mul_767);  getitem_229 = mul_767 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_96: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_768, [0, 2, 3])
    sub_229: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_586);  convolution_1 = unsqueeze_586 = None
    mul_769: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(mul_768, sub_229)
    sum_97: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_769, [0, 2, 3]);  mul_769 = None
    mul_770: "f32[32]" = torch.ops.aten.mul.Tensor(sum_96, 7.62939453125e-06)
    unsqueeze_587: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_770, 0);  mul_770 = None
    unsqueeze_588: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 2);  unsqueeze_587 = None
    unsqueeze_589: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, 3);  unsqueeze_588 = None
    mul_771: "f32[32]" = torch.ops.aten.mul.Tensor(sum_97, 7.62939453125e-06)
    mul_772: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_773: "f32[32]" = torch.ops.aten.mul.Tensor(mul_771, mul_772);  mul_771 = mul_772 = None
    unsqueeze_590: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_773, 0);  mul_773 = None
    unsqueeze_591: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 2);  unsqueeze_590 = None
    unsqueeze_592: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_591, 3);  unsqueeze_591 = None
    mul_774: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_3);  primals_3 = None
    unsqueeze_593: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_774, 0);  mul_774 = None
    unsqueeze_594: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 2);  unsqueeze_593 = None
    unsqueeze_595: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, 3);  unsqueeze_594 = None
    mul_775: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_592);  sub_229 = unsqueeze_592 = None
    sub_231: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(mul_768, mul_775);  mul_768 = mul_775 = None
    sub_232: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(sub_231, unsqueeze_589);  sub_231 = unsqueeze_589 = None
    mul_776: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_595);  sub_232 = unsqueeze_595 = None
    mul_777: "f32[32]" = torch.ops.aten.mul.Tensor(sum_97, squeeze_4);  sum_97 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_776, mul_7, primals_86, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_776 = mul_7 = primals_86 = None
    getitem_232: "f32[8, 24, 128, 128]" = convolution_backward_48[0]
    getitem_233: "f32[32, 24, 3, 3]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_780: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(getitem_232, mul_779);  getitem_232 = mul_779 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_98: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_780, [0, 2, 3])
    sub_234: "f32[8, 24, 128, 128]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_598);  convolution = unsqueeze_598 = None
    mul_781: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(mul_780, sub_234)
    sum_99: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_781, [0, 2, 3]);  mul_781 = None
    mul_782: "f32[24]" = torch.ops.aten.mul.Tensor(sum_98, 7.62939453125e-06)
    unsqueeze_599: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_782, 0);  mul_782 = None
    unsqueeze_600: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 2);  unsqueeze_599 = None
    unsqueeze_601: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, 3);  unsqueeze_600 = None
    mul_783: "f32[24]" = torch.ops.aten.mul.Tensor(sum_99, 7.62939453125e-06)
    mul_784: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_785: "f32[24]" = torch.ops.aten.mul.Tensor(mul_783, mul_784);  mul_783 = mul_784 = None
    unsqueeze_602: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_785, 0);  mul_785 = None
    unsqueeze_603: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 2);  unsqueeze_602 = None
    unsqueeze_604: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_603, 3);  unsqueeze_603 = None
    mul_786: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_1);  primals_1 = None
    unsqueeze_605: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_786, 0);  mul_786 = None
    unsqueeze_606: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 2);  unsqueeze_605 = None
    unsqueeze_607: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, 3);  unsqueeze_606 = None
    mul_787: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(sub_234, unsqueeze_604);  sub_234 = unsqueeze_604 = None
    sub_236: "f32[8, 24, 128, 128]" = torch.ops.aten.sub.Tensor(mul_780, mul_787);  mul_780 = mul_787 = None
    sub_237: "f32[8, 24, 128, 128]" = torch.ops.aten.sub.Tensor(sub_236, unsqueeze_601);  sub_236 = unsqueeze_601 = None
    mul_788: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(sub_237, unsqueeze_607);  sub_237 = unsqueeze_607 = None
    mul_789: "f32[24]" = torch.ops.aten.mul.Tensor(sum_99, squeeze_1);  sum_99 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_788, primals_263, primals_85, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_788 = primals_263 = primals_85 = None
    getitem_236: "f32[24, 3, 3, 3]" = convolution_backward_49[1];  convolution_backward_49 = None
    return [mul_789, sum_98, mul_777, sum_96, mul_765, sum_94, mul_753, sum_92, mul_741, sum_90, mul_725, sum_85, mul_716, sum_85, mul_704, sum_83, mul_692, sum_81, mul_676, sum_78, mul_664, sum_76, mul_652, sum_74, mul_636, sum_69, mul_627, sum_69, mul_615, sum_67, mul_603, sum_65, mul_587, sum_62, mul_575, sum_60, permute_122, permute_116, mul_560, sum_55, mul_548, sum_53, mul_536, sum_51, mul_524, sum_49, mul_508, sum_44, mul_499, sum_44, mul_487, sum_42, mul_475, sum_40, mul_459, sum_37, mul_447, sum_35, permute_95, permute_89, mul_432, sum_30, mul_420, sum_28, mul_408, sum_26, permute_74, permute_68, mul_393, sum_21, mul_381, sum_17, mul_372, sum_17, mul_360, sum_15, permute_53, permute_47, mul_345, sum_10, mul_333, sum_8, mul_321, sum_6, getitem_236, getitem_233, getitem_230, getitem_227, getitem_224, getitem_221, getitem_222, getitem_218, getitem_219, getitem_215, getitem_212, getitem_209, getitem_206, getitem_203, getitem_204, getitem_200, getitem_201, getitem_197, getitem_194, getitem_191, getitem_188, getitem_189, getitem_185, getitem_186, getitem_182, getitem_179, getitem_176, getitem_173, getitem_170, getitem_171, getitem_167, getitem_168, getitem_164, getitem_161, getitem_158, getitem_155, getitem_152, getitem_149, getitem_146, getitem_147, getitem_143, getitem_144, getitem_140, getitem_137, getitem_134, getitem_131, getitem_128, getitem_129, getitem_125, getitem_126, getitem_122, getitem_119, getitem_116, getitem_113, getitem_110, getitem_107, getitem_104, getitem_101, getitem_98, getitem_95, getitem_92, getitem_89, permute_36, view_97, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    