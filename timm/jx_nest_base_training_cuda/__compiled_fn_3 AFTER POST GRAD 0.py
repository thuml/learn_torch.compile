from __future__ import annotations



def forward(self, primals_1: "f32[1, 16, 196, 128]", primals_2: "f32[128]", primals_3: "f32[128]", primals_4: "f32[128]", primals_5: "f32[128]", primals_6: "f32[128]", primals_7: "f32[128]", primals_8: "f32[128]", primals_9: "f32[128]", primals_10: "f32[256]", primals_11: "f32[256]", primals_12: "f32[1, 4, 196, 256]", primals_13: "f32[256]", primals_14: "f32[256]", primals_15: "f32[256]", primals_16: "f32[256]", primals_17: "f32[256]", primals_18: "f32[256]", primals_19: "f32[256]", primals_20: "f32[256]", primals_21: "f32[512]", primals_22: "f32[512]", primals_23: "f32[1, 1, 196, 512]", primals_24: "f32[512]", primals_25: "f32[512]", primals_26: "f32[512]", primals_27: "f32[512]", primals_28: "f32[512]", primals_29: "f32[512]", primals_30: "f32[512]", primals_31: "f32[512]", primals_32: "f32[512]", primals_33: "f32[512]", primals_34: "f32[512]", primals_35: "f32[512]", primals_36: "f32[512]", primals_37: "f32[512]", primals_38: "f32[512]", primals_39: "f32[512]", primals_40: "f32[512]", primals_41: "f32[512]", primals_42: "f32[512]", primals_43: "f32[512]", primals_44: "f32[512]", primals_45: "f32[512]", primals_46: "f32[512]", primals_47: "f32[512]", primals_48: "f32[512]", primals_49: "f32[512]", primals_50: "f32[512]", primals_51: "f32[512]", primals_52: "f32[512]", primals_53: "f32[512]", primals_54: "f32[512]", primals_55: "f32[512]", primals_56: "f32[512]", primals_57: "f32[512]", primals_58: "f32[512]", primals_59: "f32[512]", primals_60: "f32[512]", primals_61: "f32[512]", primals_62: "f32[512]", primals_63: "f32[512]", primals_64: "f32[512]", primals_65: "f32[512]", primals_66: "f32[512]", primals_67: "f32[512]", primals_68: "f32[512]", primals_69: "f32[512]", primals_70: "f32[512]", primals_71: "f32[512]", primals_72: "f32[512]", primals_73: "f32[512]", primals_74: "f32[512]", primals_75: "f32[512]", primals_76: "f32[512]", primals_77: "f32[512]", primals_78: "f32[512]", primals_79: "f32[512]", primals_80: "f32[512]", primals_81: "f32[512]", primals_82: "f32[512]", primals_83: "f32[512]", primals_84: "f32[512]", primals_85: "f32[512]", primals_86: "f32[512]", primals_87: "f32[512]", primals_88: "f32[512]", primals_89: "f32[512]", primals_90: "f32[512]", primals_91: "f32[512]", primals_92: "f32[512]", primals_93: "f32[512]", primals_94: "f32[512]", primals_95: "f32[512]", primals_96: "f32[512]", primals_97: "f32[512]", primals_98: "f32[512]", primals_99: "f32[512]", primals_100: "f32[512]", primals_101: "f32[512]", primals_102: "f32[512]", primals_103: "f32[512]", primals_104: "f32[512]", primals_105: "f32[512]", primals_106: "f32[128, 3, 4, 4]", primals_107: "f32[128]", primals_108: "f32[384, 128]", primals_109: "f32[384]", primals_110: "f32[128, 128]", primals_111: "f32[128]", primals_112: "f32[512, 128]", primals_113: "f32[512]", primals_114: "f32[128, 512]", primals_115: "f32[128]", primals_116: "f32[384, 128]", primals_117: "f32[384]", primals_118: "f32[128, 128]", primals_119: "f32[128]", primals_120: "f32[512, 128]", primals_121: "f32[512]", primals_122: "f32[128, 512]", primals_123: "f32[128]", primals_124: "f32[256, 128, 3, 3]", primals_125: "f32[256]", primals_126: "f32[768, 256]", primals_127: "f32[768]", primals_128: "f32[256, 256]", primals_129: "f32[256]", primals_130: "f32[1024, 256]", primals_131: "f32[1024]", primals_132: "f32[256, 1024]", primals_133: "f32[256]", primals_134: "f32[768, 256]", primals_135: "f32[768]", primals_136: "f32[256, 256]", primals_137: "f32[256]", primals_138: "f32[1024, 256]", primals_139: "f32[1024]", primals_140: "f32[256, 1024]", primals_141: "f32[256]", primals_142: "f32[512, 256, 3, 3]", primals_143: "f32[512]", primals_144: "f32[1536, 512]", primals_145: "f32[1536]", primals_146: "f32[512, 512]", primals_147: "f32[512]", primals_148: "f32[2048, 512]", primals_149: "f32[2048]", primals_150: "f32[512, 2048]", primals_151: "f32[512]", primals_152: "f32[1536, 512]", primals_153: "f32[1536]", primals_154: "f32[512, 512]", primals_155: "f32[512]", primals_156: "f32[2048, 512]", primals_157: "f32[2048]", primals_158: "f32[512, 2048]", primals_159: "f32[512]", primals_160: "f32[1536, 512]", primals_161: "f32[1536]", primals_162: "f32[512, 512]", primals_163: "f32[512]", primals_164: "f32[2048, 512]", primals_165: "f32[2048]", primals_166: "f32[512, 2048]", primals_167: "f32[512]", primals_168: "f32[1536, 512]", primals_169: "f32[1536]", primals_170: "f32[512, 512]", primals_171: "f32[512]", primals_172: "f32[2048, 512]", primals_173: "f32[2048]", primals_174: "f32[512, 2048]", primals_175: "f32[512]", primals_176: "f32[1536, 512]", primals_177: "f32[1536]", primals_178: "f32[512, 512]", primals_179: "f32[512]", primals_180: "f32[2048, 512]", primals_181: "f32[2048]", primals_182: "f32[512, 2048]", primals_183: "f32[512]", primals_184: "f32[1536, 512]", primals_185: "f32[1536]", primals_186: "f32[512, 512]", primals_187: "f32[512]", primals_188: "f32[2048, 512]", primals_189: "f32[2048]", primals_190: "f32[512, 2048]", primals_191: "f32[512]", primals_192: "f32[1536, 512]", primals_193: "f32[1536]", primals_194: "f32[512, 512]", primals_195: "f32[512]", primals_196: "f32[2048, 512]", primals_197: "f32[2048]", primals_198: "f32[512, 2048]", primals_199: "f32[512]", primals_200: "f32[1536, 512]", primals_201: "f32[1536]", primals_202: "f32[512, 512]", primals_203: "f32[512]", primals_204: "f32[2048, 512]", primals_205: "f32[2048]", primals_206: "f32[512, 2048]", primals_207: "f32[512]", primals_208: "f32[1536, 512]", primals_209: "f32[1536]", primals_210: "f32[512, 512]", primals_211: "f32[512]", primals_212: "f32[2048, 512]", primals_213: "f32[2048]", primals_214: "f32[512, 2048]", primals_215: "f32[512]", primals_216: "f32[1536, 512]", primals_217: "f32[1536]", primals_218: "f32[512, 512]", primals_219: "f32[512]", primals_220: "f32[2048, 512]", primals_221: "f32[2048]", primals_222: "f32[512, 2048]", primals_223: "f32[512]", primals_224: "f32[1536, 512]", primals_225: "f32[1536]", primals_226: "f32[512, 512]", primals_227: "f32[512]", primals_228: "f32[2048, 512]", primals_229: "f32[2048]", primals_230: "f32[512, 2048]", primals_231: "f32[512]", primals_232: "f32[1536, 512]", primals_233: "f32[1536]", primals_234: "f32[512, 512]", primals_235: "f32[512]", primals_236: "f32[2048, 512]", primals_237: "f32[2048]", primals_238: "f32[512, 2048]", primals_239: "f32[512]", primals_240: "f32[1536, 512]", primals_241: "f32[1536]", primals_242: "f32[512, 512]", primals_243: "f32[512]", primals_244: "f32[2048, 512]", primals_245: "f32[2048]", primals_246: "f32[512, 2048]", primals_247: "f32[512]", primals_248: "f32[1536, 512]", primals_249: "f32[1536]", primals_250: "f32[512, 512]", primals_251: "f32[512]", primals_252: "f32[2048, 512]", primals_253: "f32[2048]", primals_254: "f32[512, 2048]", primals_255: "f32[512]", primals_256: "f32[1536, 512]", primals_257: "f32[1536]", primals_258: "f32[512, 512]", primals_259: "f32[512]", primals_260: "f32[2048, 512]", primals_261: "f32[2048]", primals_262: "f32[512, 2048]", primals_263: "f32[512]", primals_264: "f32[1536, 512]", primals_265: "f32[1536]", primals_266: "f32[512, 512]", primals_267: "f32[512]", primals_268: "f32[2048, 512]", primals_269: "f32[2048]", primals_270: "f32[512, 2048]", primals_271: "f32[512]", primals_272: "f32[1536, 512]", primals_273: "f32[1536]", primals_274: "f32[512, 512]", primals_275: "f32[512]", primals_276: "f32[2048, 512]", primals_277: "f32[2048]", primals_278: "f32[512, 2048]", primals_279: "f32[512]", primals_280: "f32[1536, 512]", primals_281: "f32[1536]", primals_282: "f32[512, 512]", primals_283: "f32[512]", primals_284: "f32[2048, 512]", primals_285: "f32[2048]", primals_286: "f32[512, 2048]", primals_287: "f32[512]", primals_288: "f32[1536, 512]", primals_289: "f32[1536]", primals_290: "f32[512, 512]", primals_291: "f32[512]", primals_292: "f32[2048, 512]", primals_293: "f32[2048]", primals_294: "f32[512, 2048]", primals_295: "f32[512]", primals_296: "f32[1536, 512]", primals_297: "f32[1536]", primals_298: "f32[512, 512]", primals_299: "f32[512]", primals_300: "f32[2048, 512]", primals_301: "f32[2048]", primals_302: "f32[512, 2048]", primals_303: "f32[512]", primals_304: "f32[1000, 512]", primals_305: "f32[1000]", primals_306: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution: "f32[8, 128, 56, 56]" = torch.ops.aten.convolution.default(primals_306, primals_106, primals_107, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  primals_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:233, code: x = x.permute(0, 2, 3, 1)  # (B, H', W', C), switch to channels last for transformer
    permute: "f32[8, 56, 56, 128]" = torch.ops.aten.permute.default(convolution, [0, 2, 3, 1]);  convolution = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:159, code: x = x.reshape(B, grid_height, block_size, grid_width, block_size, C)
    view: "f32[8, 4, 14, 4, 14, 128]" = torch.ops.aten.reshape.default(permute, [8, 4, 14, 4, 14, 128]);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:160, code: x = x.transpose(2, 3).reshape(B, grid_height * grid_width, -1, C)
    permute_1: "f32[8, 4, 4, 14, 14, 128]" = torch.ops.aten.permute.default(view, [0, 1, 3, 2, 4, 5]);  view = None
    clone: "f32[8, 4, 4, 14, 14, 128]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
    view_1: "f32[8, 16, 196, 128]" = torch.ops.aten.reshape.default(clone, [8, 16, 196, 128]);  clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:235, code: x = x + self.pos_embed
    add: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(view_1, primals_1);  view_1 = primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean = torch.ops.aten.var_mean.correction(add, [3], correction = 0, keepdim = True)
    getitem: "f32[8, 16, 196, 1]" = var_mean[0]
    getitem_1: "f32[8, 16, 196, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[8, 16, 196, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
    rsqrt: "f32[8, 16, 196, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 16, 196, 128]" = torch.ops.aten.sub.Tensor(add, getitem_1);  getitem_1 = None
    mul: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul, primals_2)
    add_2: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(mul_1, primals_3);  mul_1 = primals_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_2: "f32[25088, 128]" = torch.ops.aten.reshape.default(add_2, [25088, 128]);  add_2 = None
    permute_2: "f32[128, 384]" = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
    
    # No stacktrace found for following nodes
    mm_default_71: "f32[25088, 384]" = torch.ops.aten.mm.default(view_2, permute_2)
    add_tensor_71: "f32[25088, 384]" = torch.ops.aten.add.Tensor(mm_default_71, primals_109);  mm_default_71 = primals_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_3: "f32[8, 16, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_71, [8, 16, 196, 384]);  add_tensor_71 = None
    view_4: "f32[8, 16, 196, 3, 4, 32]" = torch.ops.aten.reshape.default(view_3, [8, 16, 196, 3, 4, 32]);  view_3 = None
    permute_3: "f32[3, 8, 4, 16, 196, 32]" = torch.ops.aten.permute.default(view_4, [3, 0, 4, 1, 2, 5]);  view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind = torch.ops.aten.unbind.int(permute_3);  permute_3 = None
    getitem_2: "f32[8, 4, 16, 196, 32]" = unbind[0]
    getitem_3: "f32[8, 4, 16, 196, 32]" = unbind[1]
    getitem_4: "f32[8, 4, 16, 196, 32]" = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_2: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_2, 0.42044820762685725);  getitem_2 = None
    permute_4: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.permute.default(getitem_3, [0, 1, 2, 4, 3]);  getitem_3 = None
    mul_3: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.mul.Scalar(permute_4, 0.42044820762685725);  permute_4 = None
    expand: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.expand.default(mul_2, [8, 4, 16, 196, 32]);  mul_2 = None
    clone_1: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
    view_5: "f32[512, 196, 32]" = torch.ops.aten.reshape.default(clone_1, [512, 196, 32]);  clone_1 = None
    expand_1: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.expand.default(mul_3, [8, 4, 16, 32, 196]);  mul_3 = None
    clone_2: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_6: "f32[512, 32, 196]" = torch.ops.aten.reshape.default(clone_2, [512, 32, 196]);  clone_2 = None
    bmm: "f32[512, 196, 196]" = torch.ops.aten.bmm.default(view_5, view_6)
    view_7: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.reshape.default(bmm, [8, 4, 16, 196, 196]);  bmm = None
    amax: "f32[8, 4, 16, 196, 1]" = torch.ops.aten.amax.default(view_7, [-1], True)
    sub_1: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.sub.Tensor(view_7, amax);  view_7 = amax = None
    exp: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_1: "f32[8, 4, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.alias.default(div)
    expand_2: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.expand.default(div, [8, 4, 16, 196, 196]);  div = None
    view_8: "f32[512, 196, 196]" = torch.ops.aten.reshape.default(expand_2, [512, 196, 196]);  expand_2 = None
    expand_3: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.expand.default(getitem_4, [8, 4, 16, 196, 32]);  getitem_4 = None
    clone_3: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
    view_9: "f32[512, 196, 32]" = torch.ops.aten.reshape.default(clone_3, [512, 196, 32]);  clone_3 = None
    bmm_1: "f32[512, 196, 32]" = torch.ops.aten.bmm.default(view_8, view_9)
    view_10: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.reshape.default(bmm_1, [8, 4, 16, 196, 32]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_5: "f32[8, 16, 196, 32, 4]" = torch.ops.aten.permute.default(view_10, [0, 2, 3, 4, 1]);  view_10 = None
    clone_4: "f32[8, 16, 196, 32, 4]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    view_11: "f32[8, 16, 196, 128]" = torch.ops.aten.reshape.default(clone_4, [8, 16, 196, 128]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_12: "f32[25088, 128]" = torch.ops.aten.reshape.default(view_11, [25088, 128]);  view_11 = None
    permute_6: "f32[128, 128]" = torch.ops.aten.permute.default(primals_110, [1, 0]);  primals_110 = None
    
    # No stacktrace found for following nodes
    mm_default_70: "f32[25088, 128]" = torch.ops.aten.mm.default(view_12, permute_6)
    add_tensor_70: "f32[25088, 128]" = torch.ops.aten.add.Tensor(mm_default_70, primals_111);  mm_default_70 = primals_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_13: "f32[8, 16, 196, 128]" = torch.ops.aten.reshape.default(add_tensor_70, [8, 16, 196, 128]);  add_tensor_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_3: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(add, view_13);  add = view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [3], correction = 0, keepdim = True)
    getitem_5: "f32[8, 16, 196, 1]" = var_mean_1[0]
    getitem_6: "f32[8, 16, 196, 1]" = var_mean_1[1];  var_mean_1 = None
    add_4: "f32[8, 16, 196, 1]" = torch.ops.aten.add.Tensor(getitem_5, 1e-06);  getitem_5 = None
    rsqrt_1: "f32[8, 16, 196, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_2: "f32[8, 16, 196, 128]" = torch.ops.aten.sub.Tensor(add_3, getitem_6);  getitem_6 = None
    mul_4: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
    mul_5: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul_4, primals_4)
    add_5: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(mul_5, primals_5);  mul_5 = primals_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_14: "f32[25088, 128]" = torch.ops.aten.reshape.default(add_5, [25088, 128]);  add_5 = None
    permute_7: "f32[128, 512]" = torch.ops.aten.permute.default(primals_112, [1, 0]);  primals_112 = None
    addmm_2: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_113, view_14, permute_7);  primals_113 = None
    view_15: "f32[8, 16, 196, 512]" = torch.ops.aten.reshape.default(addmm_2, [8, 16, 196, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_6: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(view_15, 0.5)
    mul_7: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(view_15, 0.7071067811865476);  view_15 = None
    erf: "f32[8, 16, 196, 512]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
    add_6: "f32[8, 16, 196, 512]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_8: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(mul_6, add_6);  mul_6 = add_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_16: "f32[25088, 512]" = torch.ops.aten.reshape.default(mul_8, [25088, 512]);  mul_8 = None
    permute_8: "f32[512, 128]" = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
    
    # No stacktrace found for following nodes
    mm_default_69: "f32[25088, 128]" = torch.ops.aten.mm.default(view_16, permute_8)
    add_tensor_69: "f32[25088, 128]" = torch.ops.aten.add.Tensor(mm_default_69, primals_115);  mm_default_69 = primals_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_17: "f32[8, 16, 196, 128]" = torch.ops.aten.reshape.default(add_tensor_69, [8, 16, 196, 128]);  add_tensor_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_7: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(add_3, view_17);  add_3 = view_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_7, [3], correction = 0, keepdim = True)
    getitem_7: "f32[8, 16, 196, 1]" = var_mean_2[0]
    getitem_8: "f32[8, 16, 196, 1]" = var_mean_2[1];  var_mean_2 = None
    add_8: "f32[8, 16, 196, 1]" = torch.ops.aten.add.Tensor(getitem_7, 1e-06);  getitem_7 = None
    rsqrt_2: "f32[8, 16, 196, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
    sub_3: "f32[8, 16, 196, 128]" = torch.ops.aten.sub.Tensor(add_7, getitem_8);  getitem_8 = None
    mul_9: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
    mul_10: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul_9, primals_6)
    add_9: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(mul_10, primals_7);  mul_10 = primals_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_18: "f32[25088, 128]" = torch.ops.aten.reshape.default(add_9, [25088, 128]);  add_9 = None
    permute_9: "f32[128, 384]" = torch.ops.aten.permute.default(primals_116, [1, 0]);  primals_116 = None
    
    # No stacktrace found for following nodes
    mm_default_68: "f32[25088, 384]" = torch.ops.aten.mm.default(view_18, permute_9)
    add_tensor_68: "f32[25088, 384]" = torch.ops.aten.add.Tensor(mm_default_68, primals_117);  mm_default_68 = primals_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_19: "f32[8, 16, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_68, [8, 16, 196, 384]);  add_tensor_68 = None
    view_20: "f32[8, 16, 196, 3, 4, 32]" = torch.ops.aten.reshape.default(view_19, [8, 16, 196, 3, 4, 32]);  view_19 = None
    permute_10: "f32[3, 8, 4, 16, 196, 32]" = torch.ops.aten.permute.default(view_20, [3, 0, 4, 1, 2, 5]);  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_1 = torch.ops.aten.unbind.int(permute_10);  permute_10 = None
    getitem_9: "f32[8, 4, 16, 196, 32]" = unbind_1[0]
    getitem_10: "f32[8, 4, 16, 196, 32]" = unbind_1[1]
    getitem_11: "f32[8, 4, 16, 196, 32]" = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_11: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_9, 0.42044820762685725);  getitem_9 = None
    permute_11: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.permute.default(getitem_10, [0, 1, 2, 4, 3]);  getitem_10 = None
    mul_12: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.mul.Scalar(permute_11, 0.42044820762685725);  permute_11 = None
    expand_4: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.expand.default(mul_11, [8, 4, 16, 196, 32]);  mul_11 = None
    clone_8: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
    view_21: "f32[512, 196, 32]" = torch.ops.aten.reshape.default(clone_8, [512, 196, 32]);  clone_8 = None
    expand_5: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.expand.default(mul_12, [8, 4, 16, 32, 196]);  mul_12 = None
    clone_9: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    view_22: "f32[512, 32, 196]" = torch.ops.aten.reshape.default(clone_9, [512, 32, 196]);  clone_9 = None
    bmm_2: "f32[512, 196, 196]" = torch.ops.aten.bmm.default(view_21, view_22)
    view_23: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.reshape.default(bmm_2, [8, 4, 16, 196, 196]);  bmm_2 = None
    amax_1: "f32[8, 4, 16, 196, 1]" = torch.ops.aten.amax.default(view_23, [-1], True)
    sub_4: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.sub.Tensor(view_23, amax_1);  view_23 = amax_1 = None
    exp_1: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_2: "f32[8, 4, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_1: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.alias.default(div_1)
    expand_6: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.expand.default(div_1, [8, 4, 16, 196, 196]);  div_1 = None
    view_24: "f32[512, 196, 196]" = torch.ops.aten.reshape.default(expand_6, [512, 196, 196]);  expand_6 = None
    expand_7: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.expand.default(getitem_11, [8, 4, 16, 196, 32]);  getitem_11 = None
    clone_10: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
    view_25: "f32[512, 196, 32]" = torch.ops.aten.reshape.default(clone_10, [512, 196, 32]);  clone_10 = None
    bmm_3: "f32[512, 196, 32]" = torch.ops.aten.bmm.default(view_24, view_25)
    view_26: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.reshape.default(bmm_3, [8, 4, 16, 196, 32]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_12: "f32[8, 16, 196, 32, 4]" = torch.ops.aten.permute.default(view_26, [0, 2, 3, 4, 1]);  view_26 = None
    clone_11: "f32[8, 16, 196, 32, 4]" = torch.ops.aten.clone.default(permute_12, memory_format = torch.contiguous_format);  permute_12 = None
    view_27: "f32[8, 16, 196, 128]" = torch.ops.aten.reshape.default(clone_11, [8, 16, 196, 128]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_28: "f32[25088, 128]" = torch.ops.aten.reshape.default(view_27, [25088, 128]);  view_27 = None
    permute_13: "f32[128, 128]" = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
    
    # No stacktrace found for following nodes
    mm_default_67: "f32[25088, 128]" = torch.ops.aten.mm.default(view_28, permute_13)
    add_tensor_67: "f32[25088, 128]" = torch.ops.aten.add.Tensor(mm_default_67, primals_119);  mm_default_67 = primals_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_29: "f32[8, 16, 196, 128]" = torch.ops.aten.reshape.default(add_tensor_67, [8, 16, 196, 128]);  add_tensor_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    bernoulli: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.9782608691602945)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_2: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli, 0.9782608691602945)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_13: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(view_29, div_2);  view_29 = div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_10: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(add_7, mul_13);  add_7 = mul_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_10, [3], correction = 0, keepdim = True)
    getitem_12: "f32[8, 16, 196, 1]" = var_mean_3[0]
    getitem_13: "f32[8, 16, 196, 1]" = var_mean_3[1];  var_mean_3 = None
    add_11: "f32[8, 16, 196, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-06);  getitem_12 = None
    rsqrt_3: "f32[8, 16, 196, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_5: "f32[8, 16, 196, 128]" = torch.ops.aten.sub.Tensor(add_10, getitem_13);  getitem_13 = None
    mul_14: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
    mul_15: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul_14, primals_8)
    add_12: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(mul_15, primals_9);  mul_15 = primals_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_30: "f32[25088, 128]" = torch.ops.aten.reshape.default(add_12, [25088, 128]);  add_12 = None
    permute_14: "f32[128, 512]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
    addmm_6: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_121, view_30, permute_14);  primals_121 = None
    view_31: "f32[8, 16, 196, 512]" = torch.ops.aten.reshape.default(addmm_6, [8, 16, 196, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_16: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(view_31, 0.5)
    mul_17: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(view_31, 0.7071067811865476);  view_31 = None
    erf_1: "f32[8, 16, 196, 512]" = torch.ops.aten.erf.default(mul_17);  mul_17 = None
    add_13: "f32[8, 16, 196, 512]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_18: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(mul_16, add_13);  mul_16 = add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_32: "f32[25088, 512]" = torch.ops.aten.reshape.default(mul_18, [25088, 512]);  mul_18 = None
    permute_15: "f32[512, 128]" = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
    
    # No stacktrace found for following nodes
    mm_default_66: "f32[25088, 128]" = torch.ops.aten.mm.default(view_32, permute_15)
    add_tensor_66: "f32[25088, 128]" = torch.ops.aten.add.Tensor(mm_default_66, primals_123);  mm_default_66 = primals_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_33: "f32[8, 16, 196, 128]" = torch.ops.aten.reshape.default(add_tensor_66, [8, 16, 196, 128]);  add_tensor_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_1: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.9782608691602945)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_3: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_1, 0.9782608691602945)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_19: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(view_33, div_3);  view_33 = div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_14: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(add_10, mul_19);  add_10 = mul_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:174, code: x = x.reshape(B, grid_size, grid_size, block_size, block_size, C)
    view_34: "f32[8, 4, 4, 14, 14, 128]" = torch.ops.aten.reshape.default(add_14, [8, 4, 4, 14, 14, 128]);  add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:175, code: x = x.transpose(2, 3).reshape(B, height, width, C)
    permute_16: "f32[8, 4, 14, 4, 14, 128]" = torch.ops.aten.permute.default(view_34, [0, 1, 3, 2, 4, 5]);  view_34 = None
    clone_15: "f32[8, 4, 14, 4, 14, 128]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
    view_35: "f32[8, 56, 56, 128]" = torch.ops.aten.reshape.default(clone_15, [8, 56, 56, 128]);  clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:242, code: return x.permute(0, 3, 1, 2)  # (B, C, H', W')
    permute_17: "f32[8, 128, 56, 56]" = torch.ops.aten.permute.default(view_35, [0, 3, 1, 2]);  view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:141, code: x = self.conv(x)
    convolution_1: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(permute_17, primals_124, primals_125, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:143, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_18: "f32[8, 56, 56, 256]" = torch.ops.aten.permute.default(convolution_1, [0, 2, 3, 1]);  convolution_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_16: "f32[8, 56, 56, 256]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    var_mean_4 = torch.ops.aten.var_mean.correction(clone_16, [3], correction = 0, keepdim = True)
    getitem_14: "f32[8, 56, 56, 1]" = var_mean_4[0]
    getitem_15: "f32[8, 56, 56, 1]" = var_mean_4[1];  var_mean_4 = None
    add_15: "f32[8, 56, 56, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
    rsqrt_4: "f32[8, 56, 56, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
    sub_6: "f32[8, 56, 56, 256]" = torch.ops.aten.sub.Tensor(clone_16, getitem_15);  clone_16 = getitem_15 = None
    mul_20: "f32[8, 56, 56, 256]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
    mul_21: "f32[8, 56, 56, 256]" = torch.ops.aten.mul.Tensor(mul_20, primals_10)
    add_16: "f32[8, 56, 56, 256]" = torch.ops.aten.add.Tensor(mul_21, primals_11);  mul_21 = primals_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:143, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_19: "f32[8, 256, 56, 56]" = torch.ops.aten.permute.default(add_16, [0, 3, 1, 2]);  add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd: "f32[8, 256, 57, 57]" = torch.ops.aten.constant_pad_nd.default(permute_19, [0, 1, 0, 1], -inf);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(constant_pad_nd, [3, 3], [2, 2])
    getitem_16: "f32[8, 256, 28, 28]" = max_pool2d_with_indices[0]
    getitem_17: "i64[8, 256, 28, 28]" = max_pool2d_with_indices[1];  max_pool2d_with_indices = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:233, code: x = x.permute(0, 2, 3, 1)  # (B, H', W', C), switch to channels last for transformer
    permute_20: "f32[8, 28, 28, 256]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 3, 1]);  getitem_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:159, code: x = x.reshape(B, grid_height, block_size, grid_width, block_size, C)
    view_36: "f32[8, 2, 14, 2, 14, 256]" = torch.ops.aten.reshape.default(permute_20, [8, 2, 14, 2, 14, 256]);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:160, code: x = x.transpose(2, 3).reshape(B, grid_height * grid_width, -1, C)
    permute_21: "f32[8, 2, 2, 14, 14, 256]" = torch.ops.aten.permute.default(view_36, [0, 1, 3, 2, 4, 5]);  view_36 = None
    clone_17: "f32[8, 2, 2, 14, 14, 256]" = torch.ops.aten.clone.default(permute_21, memory_format = torch.contiguous_format);  permute_21 = None
    view_37: "f32[8, 4, 196, 256]" = torch.ops.aten.reshape.default(clone_17, [8, 4, 196, 256]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:235, code: x = x + self.pos_embed
    add_17: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(view_37, primals_12);  view_37 = primals_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_17, [3], correction = 0, keepdim = True)
    getitem_18: "f32[8, 4, 196, 1]" = var_mean_5[0]
    getitem_19: "f32[8, 4, 196, 1]" = var_mean_5[1];  var_mean_5 = None
    add_18: "f32[8, 4, 196, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-06);  getitem_18 = None
    rsqrt_5: "f32[8, 4, 196, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_7: "f32[8, 4, 196, 256]" = torch.ops.aten.sub.Tensor(add_17, getitem_19);  getitem_19 = None
    mul_22: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_5);  sub_7 = None
    mul_23: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_22, primals_13)
    add_19: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(mul_23, primals_14);  mul_23 = primals_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_38: "f32[6272, 256]" = torch.ops.aten.reshape.default(add_19, [6272, 256]);  add_19 = None
    permute_22: "f32[256, 768]" = torch.ops.aten.permute.default(primals_126, [1, 0]);  primals_126 = None
    
    # No stacktrace found for following nodes
    mm_default_65: "f32[6272, 768]" = torch.ops.aten.mm.default(view_38, permute_22)
    add_tensor_65: "f32[6272, 768]" = torch.ops.aten.add.Tensor(mm_default_65, primals_127);  mm_default_65 = primals_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_39: "f32[8, 4, 196, 768]" = torch.ops.aten.reshape.default(add_tensor_65, [8, 4, 196, 768]);  add_tensor_65 = None
    view_40: "f32[8, 4, 196, 3, 8, 32]" = torch.ops.aten.reshape.default(view_39, [8, 4, 196, 3, 8, 32]);  view_39 = None
    permute_23: "f32[3, 8, 8, 4, 196, 32]" = torch.ops.aten.permute.default(view_40, [3, 0, 4, 1, 2, 5]);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_2 = torch.ops.aten.unbind.int(permute_23);  permute_23 = None
    getitem_20: "f32[8, 8, 4, 196, 32]" = unbind_2[0]
    getitem_21: "f32[8, 8, 4, 196, 32]" = unbind_2[1]
    getitem_22: "f32[8, 8, 4, 196, 32]" = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_24: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_20, 0.42044820762685725);  getitem_20 = None
    permute_24: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.permute.default(getitem_21, [0, 1, 2, 4, 3]);  getitem_21 = None
    mul_25: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.mul.Scalar(permute_24, 0.42044820762685725);  permute_24 = None
    expand_8: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.expand.default(mul_24, [8, 8, 4, 196, 32]);  mul_24 = None
    clone_18: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
    view_41: "f32[256, 196, 32]" = torch.ops.aten.reshape.default(clone_18, [256, 196, 32]);  clone_18 = None
    expand_9: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.expand.default(mul_25, [8, 8, 4, 32, 196]);  mul_25 = None
    clone_19: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    view_42: "f32[256, 32, 196]" = torch.ops.aten.reshape.default(clone_19, [256, 32, 196]);  clone_19 = None
    bmm_4: "f32[256, 196, 196]" = torch.ops.aten.bmm.default(view_41, view_42)
    view_43: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.reshape.default(bmm_4, [8, 8, 4, 196, 196]);  bmm_4 = None
    amax_2: "f32[8, 8, 4, 196, 1]" = torch.ops.aten.amax.default(view_43, [-1], True)
    sub_8: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.sub.Tensor(view_43, amax_2);  view_43 = amax_2 = None
    exp_2: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
    sum_3: "f32[8, 8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_4: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_2: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.alias.default(div_4)
    expand_10: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.expand.default(div_4, [8, 8, 4, 196, 196]);  div_4 = None
    view_44: "f32[256, 196, 196]" = torch.ops.aten.reshape.default(expand_10, [256, 196, 196]);  expand_10 = None
    expand_11: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.expand.default(getitem_22, [8, 8, 4, 196, 32]);  getitem_22 = None
    clone_20: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
    view_45: "f32[256, 196, 32]" = torch.ops.aten.reshape.default(clone_20, [256, 196, 32]);  clone_20 = None
    bmm_5: "f32[256, 196, 32]" = torch.ops.aten.bmm.default(view_44, view_45)
    view_46: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.reshape.default(bmm_5, [8, 8, 4, 196, 32]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_25: "f32[8, 4, 196, 32, 8]" = torch.ops.aten.permute.default(view_46, [0, 2, 3, 4, 1]);  view_46 = None
    clone_21: "f32[8, 4, 196, 32, 8]" = torch.ops.aten.clone.default(permute_25, memory_format = torch.contiguous_format);  permute_25 = None
    view_47: "f32[8, 4, 196, 256]" = torch.ops.aten.reshape.default(clone_21, [8, 4, 196, 256]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_48: "f32[6272, 256]" = torch.ops.aten.reshape.default(view_47, [6272, 256]);  view_47 = None
    permute_26: "f32[256, 256]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
    
    # No stacktrace found for following nodes
    mm_default_64: "f32[6272, 256]" = torch.ops.aten.mm.default(view_48, permute_26)
    add_tensor_64: "f32[6272, 256]" = torch.ops.aten.add.Tensor(mm_default_64, primals_129);  mm_default_64 = primals_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_49: "f32[8, 4, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_64, [8, 4, 196, 256]);  add_tensor_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_2: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.9565217383205891)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_5: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_2, 0.9565217383205891)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_26: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(view_49, div_5);  view_49 = div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_20: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(add_17, mul_26);  add_17 = mul_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_20, [3], correction = 0, keepdim = True)
    getitem_23: "f32[8, 4, 196, 1]" = var_mean_6[0]
    getitem_24: "f32[8, 4, 196, 1]" = var_mean_6[1];  var_mean_6 = None
    add_21: "f32[8, 4, 196, 1]" = torch.ops.aten.add.Tensor(getitem_23, 1e-06);  getitem_23 = None
    rsqrt_6: "f32[8, 4, 196, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_9: "f32[8, 4, 196, 256]" = torch.ops.aten.sub.Tensor(add_20, getitem_24);  getitem_24 = None
    mul_27: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = None
    mul_28: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_27, primals_15)
    add_22: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(mul_28, primals_16);  mul_28 = primals_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_50: "f32[6272, 256]" = torch.ops.aten.reshape.default(add_22, [6272, 256]);  add_22 = None
    permute_27: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_130, [1, 0]);  primals_130 = None
    addmm_10: "f32[6272, 1024]" = torch.ops.aten.addmm.default(primals_131, view_50, permute_27);  primals_131 = None
    view_51: "f32[8, 4, 196, 1024]" = torch.ops.aten.reshape.default(addmm_10, [8, 4, 196, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_29: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(view_51, 0.5)
    mul_30: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(view_51, 0.7071067811865476);  view_51 = None
    erf_2: "f32[8, 4, 196, 1024]" = torch.ops.aten.erf.default(mul_30);  mul_30 = None
    add_23: "f32[8, 4, 196, 1024]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_31: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(mul_29, add_23);  mul_29 = add_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_52: "f32[6272, 1024]" = torch.ops.aten.reshape.default(mul_31, [6272, 1024]);  mul_31 = None
    permute_28: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_132, [1, 0]);  primals_132 = None
    
    # No stacktrace found for following nodes
    mm_default_63: "f32[6272, 256]" = torch.ops.aten.mm.default(view_52, permute_28)
    add_tensor_63: "f32[6272, 256]" = torch.ops.aten.add.Tensor(mm_default_63, primals_133);  mm_default_63 = primals_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_53: "f32[8, 4, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_63, [8, 4, 196, 256]);  add_tensor_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_3: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.9565217383205891)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_6: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_3, 0.9565217383205891)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_32: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(view_53, div_6);  view_53 = div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_24: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(add_20, mul_32);  add_20 = mul_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_24, [3], correction = 0, keepdim = True)
    getitem_25: "f32[8, 4, 196, 1]" = var_mean_7[0]
    getitem_26: "f32[8, 4, 196, 1]" = var_mean_7[1];  var_mean_7 = None
    add_25: "f32[8, 4, 196, 1]" = torch.ops.aten.add.Tensor(getitem_25, 1e-06);  getitem_25 = None
    rsqrt_7: "f32[8, 4, 196, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    sub_10: "f32[8, 4, 196, 256]" = torch.ops.aten.sub.Tensor(add_24, getitem_26);  getitem_26 = None
    mul_33: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_7);  sub_10 = None
    mul_34: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_33, primals_17)
    add_26: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(mul_34, primals_18);  mul_34 = primals_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_54: "f32[6272, 256]" = torch.ops.aten.reshape.default(add_26, [6272, 256]);  add_26 = None
    permute_29: "f32[256, 768]" = torch.ops.aten.permute.default(primals_134, [1, 0]);  primals_134 = None
    
    # No stacktrace found for following nodes
    mm_default_62: "f32[6272, 768]" = torch.ops.aten.mm.default(view_54, permute_29)
    add_tensor_62: "f32[6272, 768]" = torch.ops.aten.add.Tensor(mm_default_62, primals_135);  mm_default_62 = primals_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_55: "f32[8, 4, 196, 768]" = torch.ops.aten.reshape.default(add_tensor_62, [8, 4, 196, 768]);  add_tensor_62 = None
    view_56: "f32[8, 4, 196, 3, 8, 32]" = torch.ops.aten.reshape.default(view_55, [8, 4, 196, 3, 8, 32]);  view_55 = None
    permute_30: "f32[3, 8, 8, 4, 196, 32]" = torch.ops.aten.permute.default(view_56, [3, 0, 4, 1, 2, 5]);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_3 = torch.ops.aten.unbind.int(permute_30);  permute_30 = None
    getitem_27: "f32[8, 8, 4, 196, 32]" = unbind_3[0]
    getitem_28: "f32[8, 8, 4, 196, 32]" = unbind_3[1]
    getitem_29: "f32[8, 8, 4, 196, 32]" = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_35: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_27, 0.42044820762685725);  getitem_27 = None
    permute_31: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.permute.default(getitem_28, [0, 1, 2, 4, 3]);  getitem_28 = None
    mul_36: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.mul.Scalar(permute_31, 0.42044820762685725);  permute_31 = None
    expand_12: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.expand.default(mul_35, [8, 8, 4, 196, 32]);  mul_35 = None
    clone_25: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
    view_57: "f32[256, 196, 32]" = torch.ops.aten.reshape.default(clone_25, [256, 196, 32]);  clone_25 = None
    expand_13: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.expand.default(mul_36, [8, 8, 4, 32, 196]);  mul_36 = None
    clone_26: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
    view_58: "f32[256, 32, 196]" = torch.ops.aten.reshape.default(clone_26, [256, 32, 196]);  clone_26 = None
    bmm_6: "f32[256, 196, 196]" = torch.ops.aten.bmm.default(view_57, view_58)
    view_59: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.reshape.default(bmm_6, [8, 8, 4, 196, 196]);  bmm_6 = None
    amax_3: "f32[8, 8, 4, 196, 1]" = torch.ops.aten.amax.default(view_59, [-1], True)
    sub_11: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.sub.Tensor(view_59, amax_3);  view_59 = amax_3 = None
    exp_3: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
    sum_4: "f32[8, 8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_7: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_3: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.alias.default(div_7)
    expand_14: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.expand.default(div_7, [8, 8, 4, 196, 196]);  div_7 = None
    view_60: "f32[256, 196, 196]" = torch.ops.aten.reshape.default(expand_14, [256, 196, 196]);  expand_14 = None
    expand_15: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.expand.default(getitem_29, [8, 8, 4, 196, 32]);  getitem_29 = None
    clone_27: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
    view_61: "f32[256, 196, 32]" = torch.ops.aten.reshape.default(clone_27, [256, 196, 32]);  clone_27 = None
    bmm_7: "f32[256, 196, 32]" = torch.ops.aten.bmm.default(view_60, view_61)
    view_62: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.reshape.default(bmm_7, [8, 8, 4, 196, 32]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_32: "f32[8, 4, 196, 32, 8]" = torch.ops.aten.permute.default(view_62, [0, 2, 3, 4, 1]);  view_62 = None
    clone_28: "f32[8, 4, 196, 32, 8]" = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
    view_63: "f32[8, 4, 196, 256]" = torch.ops.aten.reshape.default(clone_28, [8, 4, 196, 256]);  clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_64: "f32[6272, 256]" = torch.ops.aten.reshape.default(view_63, [6272, 256]);  view_63 = None
    permute_33: "f32[256, 256]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
    
    # No stacktrace found for following nodes
    mm_default_61: "f32[6272, 256]" = torch.ops.aten.mm.default(view_64, permute_33)
    add_tensor_61: "f32[6272, 256]" = torch.ops.aten.add.Tensor(mm_default_61, primals_137);  mm_default_61 = primals_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_65: "f32[8, 4, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_61, [8, 4, 196, 256]);  add_tensor_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_4: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.9347826093435287)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_8: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_4, 0.9347826093435287)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_37: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(view_65, div_8);  view_65 = div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_27: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(add_24, mul_37);  add_24 = mul_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_27, [3], correction = 0, keepdim = True)
    getitem_30: "f32[8, 4, 196, 1]" = var_mean_8[0]
    getitem_31: "f32[8, 4, 196, 1]" = var_mean_8[1];  var_mean_8 = None
    add_28: "f32[8, 4, 196, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-06);  getitem_30 = None
    rsqrt_8: "f32[8, 4, 196, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_12: "f32[8, 4, 196, 256]" = torch.ops.aten.sub.Tensor(add_27, getitem_31);  getitem_31 = None
    mul_38: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = None
    mul_39: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_38, primals_19)
    add_29: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(mul_39, primals_20);  mul_39 = primals_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_66: "f32[6272, 256]" = torch.ops.aten.reshape.default(add_29, [6272, 256]);  add_29 = None
    permute_34: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
    addmm_14: "f32[6272, 1024]" = torch.ops.aten.addmm.default(primals_139, view_66, permute_34);  primals_139 = None
    view_67: "f32[8, 4, 196, 1024]" = torch.ops.aten.reshape.default(addmm_14, [8, 4, 196, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_40: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(view_67, 0.5)
    mul_41: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(view_67, 0.7071067811865476);  view_67 = None
    erf_3: "f32[8, 4, 196, 1024]" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
    add_30: "f32[8, 4, 196, 1024]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_42: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(mul_40, add_30);  mul_40 = add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_68: "f32[6272, 1024]" = torch.ops.aten.reshape.default(mul_42, [6272, 1024]);  mul_42 = None
    permute_35: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_140, [1, 0]);  primals_140 = None
    
    # No stacktrace found for following nodes
    mm_default_60: "f32[6272, 256]" = torch.ops.aten.mm.default(view_68, permute_35)
    add_tensor_60: "f32[6272, 256]" = torch.ops.aten.add.Tensor(mm_default_60, primals_141);  mm_default_60 = primals_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_69: "f32[8, 4, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_60, [8, 4, 196, 256]);  add_tensor_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_5: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.9347826093435287)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_9: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_5, 0.9347826093435287)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_43: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(view_69, div_9);  view_69 = div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_31: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(add_27, mul_43);  add_27 = mul_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:174, code: x = x.reshape(B, grid_size, grid_size, block_size, block_size, C)
    view_70: "f32[8, 2, 2, 14, 14, 256]" = torch.ops.aten.reshape.default(add_31, [8, 2, 2, 14, 14, 256]);  add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:175, code: x = x.transpose(2, 3).reshape(B, height, width, C)
    permute_36: "f32[8, 2, 14, 2, 14, 256]" = torch.ops.aten.permute.default(view_70, [0, 1, 3, 2, 4, 5]);  view_70 = None
    clone_32: "f32[8, 2, 14, 2, 14, 256]" = torch.ops.aten.clone.default(permute_36, memory_format = torch.contiguous_format);  permute_36 = None
    view_71: "f32[8, 28, 28, 256]" = torch.ops.aten.reshape.default(clone_32, [8, 28, 28, 256]);  clone_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:242, code: return x.permute(0, 3, 1, 2)  # (B, C, H', W')
    permute_37: "f32[8, 256, 28, 28]" = torch.ops.aten.permute.default(view_71, [0, 3, 1, 2]);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:141, code: x = self.conv(x)
    convolution_2: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(permute_37, primals_142, primals_143, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:143, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_38: "f32[8, 28, 28, 512]" = torch.ops.aten.permute.default(convolution_2, [0, 2, 3, 1]);  convolution_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_33: "f32[8, 28, 28, 512]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    var_mean_9 = torch.ops.aten.var_mean.correction(clone_33, [3], correction = 0, keepdim = True)
    getitem_32: "f32[8, 28, 28, 1]" = var_mean_9[0]
    getitem_33: "f32[8, 28, 28, 1]" = var_mean_9[1];  var_mean_9 = None
    add_32: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
    rsqrt_9: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_13: "f32[8, 28, 28, 512]" = torch.ops.aten.sub.Tensor(clone_33, getitem_33);  clone_33 = getitem_33 = None
    mul_44: "f32[8, 28, 28, 512]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_9);  sub_13 = None
    mul_45: "f32[8, 28, 28, 512]" = torch.ops.aten.mul.Tensor(mul_44, primals_21)
    add_33: "f32[8, 28, 28, 512]" = torch.ops.aten.add.Tensor(mul_45, primals_22);  mul_45 = primals_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:143, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_39: "f32[8, 512, 28, 28]" = torch.ops.aten.permute.default(add_33, [0, 3, 1, 2]);  add_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_1: "f32[8, 512, 29, 29]" = torch.ops.aten.constant_pad_nd.default(permute_39, [0, 1, 0, 1], -inf);  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    max_pool2d_with_indices_1 = torch.ops.aten.max_pool2d_with_indices.default(constant_pad_nd_1, [3, 3], [2, 2])
    getitem_34: "f32[8, 512, 14, 14]" = max_pool2d_with_indices_1[0]
    getitem_35: "i64[8, 512, 14, 14]" = max_pool2d_with_indices_1[1];  max_pool2d_with_indices_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:233, code: x = x.permute(0, 2, 3, 1)  # (B, H', W', C), switch to channels last for transformer
    permute_40: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(getitem_34, [0, 2, 3, 1]);  getitem_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:159, code: x = x.reshape(B, grid_height, block_size, grid_width, block_size, C)
    view_72: "f32[8, 1, 14, 1, 14, 512]" = torch.ops.aten.reshape.default(permute_40, [8, 1, 14, 1, 14, 512]);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:160, code: x = x.transpose(2, 3).reshape(B, grid_height * grid_width, -1, C)
    permute_41: "f32[8, 1, 1, 14, 14, 512]" = torch.ops.aten.permute.default(view_72, [0, 1, 3, 2, 4, 5]);  view_72 = None
    view_73: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(permute_41, [8, 1, -1, 512]);  permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:235, code: x = x + self.pos_embed
    add_34: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(view_73, primals_23);  view_73 = primals_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_34, [3], correction = 0, keepdim = True)
    getitem_36: "f32[8, 1, 196, 1]" = var_mean_10[0]
    getitem_37: "f32[8, 1, 196, 1]" = var_mean_10[1];  var_mean_10 = None
    add_35: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-06);  getitem_36 = None
    rsqrt_10: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
    sub_14: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_34, getitem_37);  getitem_37 = None
    mul_46: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_10);  sub_14 = None
    mul_47: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_46, primals_24)
    add_36: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_47, primals_25);  mul_47 = primals_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_74: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_36, [1568, 512]);  add_36 = None
    permute_42: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
    
    # No stacktrace found for following nodes
    mm_default_59: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_74, permute_42)
    add_tensor_59: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_59, primals_145);  mm_default_59 = primals_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_75: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_59, [8, 1, 196, 1536]);  add_tensor_59 = None
    view_76: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_75, [8, 1, 196, 3, 16, 32]);  view_75 = None
    permute_43: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_76, [3, 0, 4, 1, 2, 5]);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_4 = torch.ops.aten.unbind.int(permute_43);  permute_43 = None
    getitem_38: "f32[8, 16, 1, 196, 32]" = unbind_4[0]
    getitem_39: "f32[8, 16, 1, 196, 32]" = unbind_4[1]
    getitem_40: "f32[8, 16, 1, 196, 32]" = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_48: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_38, 0.42044820762685725);  getitem_38 = None
    permute_44: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_39, [0, 1, 2, 4, 3]);  getitem_39 = None
    mul_49: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_44, 0.42044820762685725);  permute_44 = None
    expand_16: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_48, [8, 16, 1, 196, 32]);  mul_48 = None
    clone_34: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
    view_77: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_34, [128, 196, 32]);  clone_34 = None
    expand_17: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_49, [8, 16, 1, 32, 196]);  mul_49 = None
    clone_35: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    view_78: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_35, [128, 32, 196]);  clone_35 = None
    bmm_8: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_77, view_78)
    view_79: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_8, [8, 16, 1, 196, 196]);  bmm_8 = None
    amax_4: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_79, [-1], True)
    sub_15: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_79, amax_4);  view_79 = amax_4 = None
    exp_4: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
    sum_5: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_10: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    alias_4: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_10)
    expand_18: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_10, [8, 16, 1, 196, 196]);  div_10 = None
    view_80: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_18, [128, 196, 196]);  expand_18 = None
    expand_19: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_40, [8, 16, 1, 196, 32]);  getitem_40 = None
    clone_36: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
    view_81: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_36, [128, 196, 32]);  clone_36 = None
    bmm_9: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_80, view_81)
    view_82: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_9, [8, 16, 1, 196, 32]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_45: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_82, [0, 2, 3, 4, 1]);  view_82 = None
    clone_37: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
    view_83: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_37, [8, 1, 196, 512]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_84: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_83, [1568, 512]);  view_83 = None
    permute_46: "f32[512, 512]" = torch.ops.aten.permute.default(primals_146, [1, 0]);  primals_146 = None
    
    # No stacktrace found for following nodes
    mm_default_58: "f32[1568, 512]" = torch.ops.aten.mm.default(view_84, permute_46)
    add_tensor_58: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_58, primals_147);  mm_default_58 = primals_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_85: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_58, [8, 1, 196, 512]);  add_tensor_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_6: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.9130434766411781)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_11: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_6, 0.9130434766411781)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_50: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_85, div_11);  view_85 = div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_37: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_34, mul_50);  add_34 = mul_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_37, [3], correction = 0, keepdim = True)
    getitem_41: "f32[8, 1, 196, 1]" = var_mean_11[0]
    getitem_42: "f32[8, 1, 196, 1]" = var_mean_11[1];  var_mean_11 = None
    add_38: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_41, 1e-06);  getitem_41 = None
    rsqrt_11: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_16: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_37, getitem_42);  getitem_42 = None
    mul_51: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_11);  sub_16 = None
    mul_52: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_51, primals_26)
    add_39: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_52, primals_27);  mul_52 = primals_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_86: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_39, [1568, 512]);  add_39 = None
    permute_47: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_148, [1, 0]);  primals_148 = None
    addmm_18: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_149, view_86, permute_47);  primals_149 = None
    view_87: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(addmm_18, [8, 1, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_53: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_87, 0.5)
    mul_54: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_87, 0.7071067811865476);  view_87 = None
    erf_4: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_54);  mul_54 = None
    add_40: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_55: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_53, add_40);  mul_53 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_88: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_55, [1568, 2048]);  mul_55 = None
    permute_48: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
    
    # No stacktrace found for following nodes
    mm_default_57: "f32[1568, 512]" = torch.ops.aten.mm.default(view_88, permute_48)
    add_tensor_57: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_57, primals_151);  mm_default_57 = primals_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_89: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_57, [8, 1, 196, 512]);  add_tensor_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_7: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.9130434766411781)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_12: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_7, 0.9130434766411781)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_56: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_89, div_12);  view_89 = div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_41: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_37, mul_56);  add_37 = mul_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_41, [3], correction = 0, keepdim = True)
    getitem_43: "f32[8, 1, 196, 1]" = var_mean_12[0]
    getitem_44: "f32[8, 1, 196, 1]" = var_mean_12[1];  var_mean_12 = None
    add_42: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_43, 1e-06);  getitem_43 = None
    rsqrt_12: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_17: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_41, getitem_44);  getitem_44 = None
    mul_57: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_12);  sub_17 = None
    mul_58: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_57, primals_28)
    add_43: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_58, primals_29);  mul_58 = primals_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_90: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_43, [1568, 512]);  add_43 = None
    permute_49: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
    
    # No stacktrace found for following nodes
    mm_default_56: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_90, permute_49)
    add_tensor_56: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_56, primals_153);  mm_default_56 = primals_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_91: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_56, [8, 1, 196, 1536]);  add_tensor_56 = None
    view_92: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_91, [8, 1, 196, 3, 16, 32]);  view_91 = None
    permute_50: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_92, [3, 0, 4, 1, 2, 5]);  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_5 = torch.ops.aten.unbind.int(permute_50);  permute_50 = None
    getitem_45: "f32[8, 16, 1, 196, 32]" = unbind_5[0]
    getitem_46: "f32[8, 16, 1, 196, 32]" = unbind_5[1]
    getitem_47: "f32[8, 16, 1, 196, 32]" = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_59: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_45, 0.42044820762685725);  getitem_45 = None
    permute_51: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_46, [0, 1, 2, 4, 3]);  getitem_46 = None
    mul_60: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_51, 0.42044820762685725);  permute_51 = None
    expand_20: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_59, [8, 16, 1, 196, 32]);  mul_59 = None
    clone_41: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
    view_93: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_41, [128, 196, 32]);  clone_41 = None
    expand_21: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_60, [8, 16, 1, 32, 196]);  mul_60 = None
    clone_42: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    view_94: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_42, [128, 32, 196]);  clone_42 = None
    bmm_10: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_93, view_94)
    view_95: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_10, [8, 16, 1, 196, 196]);  bmm_10 = None
    amax_5: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_95, [-1], True)
    sub_18: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_95, amax_5);  view_95 = amax_5 = None
    exp_5: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
    sum_6: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_13: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    alias_5: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_13)
    expand_22: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_13, [8, 16, 1, 196, 196]);  div_13 = None
    view_96: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_22, [128, 196, 196]);  expand_22 = None
    expand_23: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_47, [8, 16, 1, 196, 32]);  getitem_47 = None
    clone_43: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
    view_97: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_43, [128, 196, 32]);  clone_43 = None
    bmm_11: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_96, view_97)
    view_98: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_11, [8, 16, 1, 196, 32]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_52: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_98, [0, 2, 3, 4, 1]);  view_98 = None
    clone_44: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None
    view_99: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_44, [8, 1, 196, 512]);  clone_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_100: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_99, [1568, 512]);  view_99 = None
    permute_53: "f32[512, 512]" = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
    
    # No stacktrace found for following nodes
    mm_default_55: "f32[1568, 512]" = torch.ops.aten.mm.default(view_100, permute_53)
    add_tensor_55: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_55, primals_155);  mm_default_55 = primals_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_101: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_55, [8, 1, 196, 512]);  add_tensor_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_8: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.8913043439388275)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_14: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_8, 0.8913043439388275)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_61: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_101, div_14);  view_101 = div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_44: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_41, mul_61);  add_41 = mul_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_13 = torch.ops.aten.var_mean.correction(add_44, [3], correction = 0, keepdim = True)
    getitem_48: "f32[8, 1, 196, 1]" = var_mean_13[0]
    getitem_49: "f32[8, 1, 196, 1]" = var_mean_13[1];  var_mean_13 = None
    add_45: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-06);  getitem_48 = None
    rsqrt_13: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
    sub_19: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_44, getitem_49);  getitem_49 = None
    mul_62: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_13);  sub_19 = None
    mul_63: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_62, primals_30)
    add_46: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_63, primals_31);  mul_63 = primals_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_102: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_46, [1568, 512]);  add_46 = None
    permute_54: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
    addmm_22: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_157, view_102, permute_54);  primals_157 = None
    view_103: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(addmm_22, [8, 1, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_64: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_103, 0.5)
    mul_65: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_103, 0.7071067811865476);  view_103 = None
    erf_5: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_65);  mul_65 = None
    add_47: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_66: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_64, add_47);  mul_64 = add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_104: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_66, [1568, 2048]);  mul_66 = None
    permute_55: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_158, [1, 0]);  primals_158 = None
    
    # No stacktrace found for following nodes
    mm_default_54: "f32[1568, 512]" = torch.ops.aten.mm.default(view_104, permute_55)
    add_tensor_54: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_54, primals_159);  mm_default_54 = primals_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_105: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_54, [8, 1, 196, 512]);  add_tensor_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_9: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.8913043439388275)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_15: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_9, 0.8913043439388275)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_67: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_105, div_15);  view_105 = div_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_48: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_44, mul_67);  add_44 = mul_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_48, [3], correction = 0, keepdim = True)
    getitem_50: "f32[8, 1, 196, 1]" = var_mean_14[0]
    getitem_51: "f32[8, 1, 196, 1]" = var_mean_14[1];  var_mean_14 = None
    add_49: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-06);  getitem_50 = None
    rsqrt_14: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    sub_20: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_48, getitem_51);  getitem_51 = None
    mul_68: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_14);  sub_20 = None
    mul_69: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_68, primals_32)
    add_50: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_69, primals_33);  mul_69 = primals_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_106: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_50, [1568, 512]);  add_50 = None
    permute_56: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None
    
    # No stacktrace found for following nodes
    mm_default_53: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_106, permute_56)
    add_tensor_53: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_53, primals_161);  mm_default_53 = primals_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_107: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_53, [8, 1, 196, 1536]);  add_tensor_53 = None
    view_108: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_107, [8, 1, 196, 3, 16, 32]);  view_107 = None
    permute_57: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_108, [3, 0, 4, 1, 2, 5]);  view_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_6 = torch.ops.aten.unbind.int(permute_57);  permute_57 = None
    getitem_52: "f32[8, 16, 1, 196, 32]" = unbind_6[0]
    getitem_53: "f32[8, 16, 1, 196, 32]" = unbind_6[1]
    getitem_54: "f32[8, 16, 1, 196, 32]" = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_70: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_52, 0.42044820762685725);  getitem_52 = None
    permute_58: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_53, [0, 1, 2, 4, 3]);  getitem_53 = None
    mul_71: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_58, 0.42044820762685725);  permute_58 = None
    expand_24: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_70, [8, 16, 1, 196, 32]);  mul_70 = None
    clone_48: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
    view_109: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_48, [128, 196, 32]);  clone_48 = None
    expand_25: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_71, [8, 16, 1, 32, 196]);  mul_71 = None
    clone_49: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    view_110: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_49, [128, 32, 196]);  clone_49 = None
    bmm_12: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_109, view_110)
    view_111: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_12, [8, 16, 1, 196, 196]);  bmm_12 = None
    amax_6: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_111, [-1], True)
    sub_21: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_111, amax_6);  view_111 = amax_6 = None
    exp_6: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
    sum_7: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_16: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    alias_6: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_16)
    expand_26: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_16, [8, 16, 1, 196, 196]);  div_16 = None
    view_112: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_26, [128, 196, 196]);  expand_26 = None
    expand_27: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_54, [8, 16, 1, 196, 32]);  getitem_54 = None
    clone_50: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
    view_113: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_50, [128, 196, 32]);  clone_50 = None
    bmm_13: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_112, view_113)
    view_114: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_13, [8, 16, 1, 196, 32]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_59: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_114, [0, 2, 3, 4, 1]);  view_114 = None
    clone_51: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    view_115: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_51, [8, 1, 196, 512]);  clone_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_116: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_115, [1568, 512]);  view_115 = None
    permute_60: "f32[512, 512]" = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
    
    # No stacktrace found for following nodes
    mm_default_52: "f32[1568, 512]" = torch.ops.aten.mm.default(view_116, permute_60)
    add_tensor_52: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_52, primals_163);  mm_default_52 = primals_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_117: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_52, [8, 1, 196, 512]);  add_tensor_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_10: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.8695652186870575)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_17: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_10, 0.8695652186870575)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_72: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_117, div_17);  view_117 = div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_51: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_48, mul_72);  add_48 = mul_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_15 = torch.ops.aten.var_mean.correction(add_51, [3], correction = 0, keepdim = True)
    getitem_55: "f32[8, 1, 196, 1]" = var_mean_15[0]
    getitem_56: "f32[8, 1, 196, 1]" = var_mean_15[1];  var_mean_15 = None
    add_52: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_55, 1e-06);  getitem_55 = None
    rsqrt_15: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_22: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_51, getitem_56);  getitem_56 = None
    mul_73: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_15);  sub_22 = None
    mul_74: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_73, primals_34)
    add_53: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_74, primals_35);  mul_74 = primals_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_118: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_53, [1568, 512]);  add_53 = None
    permute_61: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_164, [1, 0]);  primals_164 = None
    addmm_26: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_165, view_118, permute_61);  primals_165 = None
    view_119: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(addmm_26, [8, 1, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_75: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_119, 0.5)
    mul_76: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_119, 0.7071067811865476);  view_119 = None
    erf_6: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
    add_54: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_77: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_75, add_54);  mul_75 = add_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_120: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_77, [1568, 2048]);  mul_77 = None
    permute_62: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_166, [1, 0]);  primals_166 = None
    
    # No stacktrace found for following nodes
    mm_default_51: "f32[1568, 512]" = torch.ops.aten.mm.default(view_120, permute_62)
    add_tensor_51: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_51, primals_167);  mm_default_51 = primals_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_121: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_51, [8, 1, 196, 512]);  add_tensor_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_11: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.8695652186870575)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_18: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_11, 0.8695652186870575)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_78: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_121, div_18);  view_121 = div_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_55: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_51, mul_78);  add_51 = mul_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_16 = torch.ops.aten.var_mean.correction(add_55, [3], correction = 0, keepdim = True)
    getitem_57: "f32[8, 1, 196, 1]" = var_mean_16[0]
    getitem_58: "f32[8, 1, 196, 1]" = var_mean_16[1];  var_mean_16 = None
    add_56: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_57, 1e-06);  getitem_57 = None
    rsqrt_16: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_23: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_55, getitem_58);  getitem_58 = None
    mul_79: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_16);  sub_23 = None
    mul_80: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_79, primals_36)
    add_57: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_80, primals_37);  mul_80 = primals_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_122: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_57, [1568, 512]);  add_57 = None
    permute_63: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_168, [1, 0]);  primals_168 = None
    
    # No stacktrace found for following nodes
    mm_default_50: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_122, permute_63)
    add_tensor_50: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_50, primals_169);  mm_default_50 = primals_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_123: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_50, [8, 1, 196, 1536]);  add_tensor_50 = None
    view_124: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_123, [8, 1, 196, 3, 16, 32]);  view_123 = None
    permute_64: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_124, [3, 0, 4, 1, 2, 5]);  view_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_7 = torch.ops.aten.unbind.int(permute_64);  permute_64 = None
    getitem_59: "f32[8, 16, 1, 196, 32]" = unbind_7[0]
    getitem_60: "f32[8, 16, 1, 196, 32]" = unbind_7[1]
    getitem_61: "f32[8, 16, 1, 196, 32]" = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_81: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_59, 0.42044820762685725);  getitem_59 = None
    permute_65: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_60, [0, 1, 2, 4, 3]);  getitem_60 = None
    mul_82: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_65, 0.42044820762685725);  permute_65 = None
    expand_28: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_81, [8, 16, 1, 196, 32]);  mul_81 = None
    clone_55: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
    view_125: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_55, [128, 196, 32]);  clone_55 = None
    expand_29: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_82, [8, 16, 1, 32, 196]);  mul_82 = None
    clone_56: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
    view_126: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_56, [128, 32, 196]);  clone_56 = None
    bmm_14: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_125, view_126)
    view_127: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_14, [8, 16, 1, 196, 196]);  bmm_14 = None
    amax_7: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_127, [-1], True)
    sub_24: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_127, amax_7);  view_127 = amax_7 = None
    exp_7: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_24);  sub_24 = None
    sum_8: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_19: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    alias_7: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_19)
    expand_30: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_19, [8, 16, 1, 196, 196]);  div_19 = None
    view_128: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_30, [128, 196, 196]);  expand_30 = None
    expand_31: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_61, [8, 16, 1, 196, 32]);  getitem_61 = None
    clone_57: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
    view_129: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_57, [128, 196, 32]);  clone_57 = None
    bmm_15: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_128, view_129)
    view_130: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_15, [8, 16, 1, 196, 32]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_66: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_130, [0, 2, 3, 4, 1]);  view_130 = None
    clone_58: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_66, memory_format = torch.contiguous_format);  permute_66 = None
    view_131: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_58, [8, 1, 196, 512]);  clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_132: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_131, [1568, 512]);  view_131 = None
    permute_67: "f32[512, 512]" = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
    
    # No stacktrace found for following nodes
    mm_default_49: "f32[1568, 512]" = torch.ops.aten.mm.default(view_132, permute_67)
    add_tensor_49: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_49, primals_171);  mm_default_49 = primals_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_133: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_49, [8, 1, 196, 512]);  add_tensor_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_12: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.8478260785341263)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_20: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_12, 0.8478260785341263)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_83: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_133, div_20);  view_133 = div_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_58: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_55, mul_83);  add_55 = mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_17 = torch.ops.aten.var_mean.correction(add_58, [3], correction = 0, keepdim = True)
    getitem_62: "f32[8, 1, 196, 1]" = var_mean_17[0]
    getitem_63: "f32[8, 1, 196, 1]" = var_mean_17[1];  var_mean_17 = None
    add_59: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-06);  getitem_62 = None
    rsqrt_17: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
    sub_25: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_58, getitem_63);  getitem_63 = None
    mul_84: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_17);  sub_25 = None
    mul_85: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_84, primals_38)
    add_60: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_85, primals_39);  mul_85 = primals_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_134: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_60, [1568, 512]);  add_60 = None
    permute_68: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_172, [1, 0]);  primals_172 = None
    addmm_30: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_173, view_134, permute_68);  primals_173 = None
    view_135: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(addmm_30, [8, 1, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_86: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_135, 0.5)
    mul_87: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_135, 0.7071067811865476);  view_135 = None
    erf_7: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
    add_61: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_88: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_86, add_61);  mul_86 = add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_136: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_88, [1568, 2048]);  mul_88 = None
    permute_69: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_174, [1, 0]);  primals_174 = None
    
    # No stacktrace found for following nodes
    mm_default_48: "f32[1568, 512]" = torch.ops.aten.mm.default(view_136, permute_69)
    add_tensor_48: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_48, primals_175);  mm_default_48 = primals_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_137: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_48, [8, 1, 196, 512]);  add_tensor_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_13: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.8478260785341263)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_21: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_13, 0.8478260785341263)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_89: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_137, div_21);  view_137 = div_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_62: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_58, mul_89);  add_58 = mul_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_18 = torch.ops.aten.var_mean.correction(add_62, [3], correction = 0, keepdim = True)
    getitem_64: "f32[8, 1, 196, 1]" = var_mean_18[0]
    getitem_65: "f32[8, 1, 196, 1]" = var_mean_18[1];  var_mean_18 = None
    add_63: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-06);  getitem_64 = None
    rsqrt_18: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    sub_26: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_62, getitem_65);  getitem_65 = None
    mul_90: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_18);  sub_26 = None
    mul_91: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_90, primals_40)
    add_64: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_91, primals_41);  mul_91 = primals_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_138: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_64, [1568, 512]);  add_64 = None
    permute_70: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_176, [1, 0]);  primals_176 = None
    
    # No stacktrace found for following nodes
    mm_default_47: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_138, permute_70)
    add_tensor_47: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_47, primals_177);  mm_default_47 = primals_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_139: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_47, [8, 1, 196, 1536]);  add_tensor_47 = None
    view_140: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_139, [8, 1, 196, 3, 16, 32]);  view_139 = None
    permute_71: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_140, [3, 0, 4, 1, 2, 5]);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_8 = torch.ops.aten.unbind.int(permute_71);  permute_71 = None
    getitem_66: "f32[8, 16, 1, 196, 32]" = unbind_8[0]
    getitem_67: "f32[8, 16, 1, 196, 32]" = unbind_8[1]
    getitem_68: "f32[8, 16, 1, 196, 32]" = unbind_8[2];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_92: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_66, 0.42044820762685725);  getitem_66 = None
    permute_72: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_67, [0, 1, 2, 4, 3]);  getitem_67 = None
    mul_93: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_72, 0.42044820762685725);  permute_72 = None
    expand_32: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_92, [8, 16, 1, 196, 32]);  mul_92 = None
    clone_62: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
    view_141: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_62, [128, 196, 32]);  clone_62 = None
    expand_33: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_93, [8, 16, 1, 32, 196]);  mul_93 = None
    clone_63: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
    view_142: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_63, [128, 32, 196]);  clone_63 = None
    bmm_16: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_141, view_142)
    view_143: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_16, [8, 16, 1, 196, 196]);  bmm_16 = None
    amax_8: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_143, [-1], True)
    sub_27: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_143, amax_8);  view_143 = amax_8 = None
    exp_8: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_27);  sub_27 = None
    sum_9: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_22: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    alias_8: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_22)
    expand_34: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_22, [8, 16, 1, 196, 196]);  div_22 = None
    view_144: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_34, [128, 196, 196]);  expand_34 = None
    expand_35: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_68, [8, 16, 1, 196, 32]);  getitem_68 = None
    clone_64: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
    view_145: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_64, [128, 196, 32]);  clone_64 = None
    bmm_17: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_144, view_145)
    view_146: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_17, [8, 16, 1, 196, 32]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_73: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_146, [0, 2, 3, 4, 1]);  view_146 = None
    clone_65: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    view_147: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_65, [8, 1, 196, 512]);  clone_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_148: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_147, [1568, 512]);  view_147 = None
    permute_74: "f32[512, 512]" = torch.ops.aten.permute.default(primals_178, [1, 0]);  primals_178 = None
    
    # No stacktrace found for following nodes
    mm_default_46: "f32[1568, 512]" = torch.ops.aten.mm.default(view_148, permute_74)
    add_tensor_46: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_46, primals_179);  mm_default_46 = primals_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_149: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_46, [8, 1, 196, 512]);  add_tensor_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_14: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.8260869532823563)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_23: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_14, 0.8260869532823563)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_94: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_149, div_23);  view_149 = div_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_65: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_62, mul_94);  add_62 = mul_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_19 = torch.ops.aten.var_mean.correction(add_65, [3], correction = 0, keepdim = True)
    getitem_69: "f32[8, 1, 196, 1]" = var_mean_19[0]
    getitem_70: "f32[8, 1, 196, 1]" = var_mean_19[1];  var_mean_19 = None
    add_66: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_69, 1e-06);  getitem_69 = None
    rsqrt_19: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    sub_28: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_65, getitem_70);  getitem_70 = None
    mul_95: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_19);  sub_28 = None
    mul_96: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_95, primals_42)
    add_67: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_96, primals_43);  mul_96 = primals_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_150: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_67, [1568, 512]);  add_67 = None
    permute_75: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_180, [1, 0]);  primals_180 = None
    addmm_34: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_181, view_150, permute_75);  primals_181 = None
    view_151: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(addmm_34, [8, 1, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_97: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_151, 0.5)
    mul_98: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476);  view_151 = None
    erf_8: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_98);  mul_98 = None
    add_68: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_99: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_97, add_68);  mul_97 = add_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_152: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_99, [1568, 2048]);  mul_99 = None
    permute_76: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_182, [1, 0]);  primals_182 = None
    
    # No stacktrace found for following nodes
    mm_default_45: "f32[1568, 512]" = torch.ops.aten.mm.default(view_152, permute_76)
    add_tensor_45: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_45, primals_183);  mm_default_45 = primals_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_153: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_45, [8, 1, 196, 512]);  add_tensor_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_15: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.8260869532823563)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_24: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_15, 0.8260869532823563)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_100: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_153, div_24);  view_153 = div_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_69: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_65, mul_100);  add_65 = mul_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_69, [3], correction = 0, keepdim = True)
    getitem_71: "f32[8, 1, 196, 1]" = var_mean_20[0]
    getitem_72: "f32[8, 1, 196, 1]" = var_mean_20[1];  var_mean_20 = None
    add_70: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_71, 1e-06);  getitem_71 = None
    rsqrt_20: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    sub_29: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_69, getitem_72);  getitem_72 = None
    mul_101: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_20);  sub_29 = None
    mul_102: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_101, primals_44)
    add_71: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_102, primals_45);  mul_102 = primals_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_154: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_71, [1568, 512]);  add_71 = None
    permute_77: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None
    
    # No stacktrace found for following nodes
    mm_default_44: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_154, permute_77)
    add_tensor_44: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_44, primals_185);  mm_default_44 = primals_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_155: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_44, [8, 1, 196, 1536]);  add_tensor_44 = None
    view_156: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_155, [8, 1, 196, 3, 16, 32]);  view_155 = None
    permute_78: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_156, [3, 0, 4, 1, 2, 5]);  view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_9 = torch.ops.aten.unbind.int(permute_78);  permute_78 = None
    getitem_73: "f32[8, 16, 1, 196, 32]" = unbind_9[0]
    getitem_74: "f32[8, 16, 1, 196, 32]" = unbind_9[1]
    getitem_75: "f32[8, 16, 1, 196, 32]" = unbind_9[2];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_103: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_73, 0.42044820762685725);  getitem_73 = None
    permute_79: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_74, [0, 1, 2, 4, 3]);  getitem_74 = None
    mul_104: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_79, 0.42044820762685725);  permute_79 = None
    expand_36: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_103, [8, 16, 1, 196, 32]);  mul_103 = None
    clone_69: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
    view_157: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_69, [128, 196, 32]);  clone_69 = None
    expand_37: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_104, [8, 16, 1, 32, 196]);  mul_104 = None
    clone_70: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
    view_158: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_70, [128, 32, 196]);  clone_70 = None
    bmm_18: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_157, view_158)
    view_159: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_18, [8, 16, 1, 196, 196]);  bmm_18 = None
    amax_9: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_159, [-1], True)
    sub_30: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_159, amax_9);  view_159 = amax_9 = None
    exp_9: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_30);  sub_30 = None
    sum_10: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_25: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    alias_9: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_25)
    expand_38: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_25, [8, 16, 1, 196, 196]);  div_25 = None
    view_160: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_38, [128, 196, 196]);  expand_38 = None
    expand_39: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_75, [8, 16, 1, 196, 32]);  getitem_75 = None
    clone_71: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
    view_161: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_71, [128, 196, 32]);  clone_71 = None
    bmm_19: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_160, view_161)
    view_162: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_19, [8, 16, 1, 196, 32]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_80: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_162, [0, 2, 3, 4, 1]);  view_162 = None
    clone_72: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
    view_163: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_72, [8, 1, 196, 512]);  clone_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_164: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_163, [1568, 512]);  view_163 = None
    permute_81: "f32[512, 512]" = torch.ops.aten.permute.default(primals_186, [1, 0]);  primals_186 = None
    
    # No stacktrace found for following nodes
    mm_default_43: "f32[1568, 512]" = torch.ops.aten.mm.default(view_164, permute_81)
    add_tensor_43: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_43, primals_187);  mm_default_43 = primals_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_165: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_43, [8, 1, 196, 512]);  add_tensor_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_16: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.8043478280305862)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_26: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_16, 0.8043478280305862)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_105: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_165, div_26);  view_165 = div_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_72: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_69, mul_105);  add_69 = mul_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_72, [3], correction = 0, keepdim = True)
    getitem_76: "f32[8, 1, 196, 1]" = var_mean_21[0]
    getitem_77: "f32[8, 1, 196, 1]" = var_mean_21[1];  var_mean_21 = None
    add_73: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-06);  getitem_76 = None
    rsqrt_21: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    sub_31: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_72, getitem_77);  getitem_77 = None
    mul_106: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_21);  sub_31 = None
    mul_107: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_106, primals_46)
    add_74: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_107, primals_47);  mul_107 = primals_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_166: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_74, [1568, 512]);  add_74 = None
    permute_82: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
    addmm_38: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_189, view_166, permute_82);  primals_189 = None
    view_167: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(addmm_38, [8, 1, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_108: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_167, 0.5)
    mul_109: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_167, 0.7071067811865476);  view_167 = None
    erf_9: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_109);  mul_109 = None
    add_75: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_110: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_108, add_75);  mul_108 = add_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_168: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_110, [1568, 2048]);  mul_110 = None
    permute_83: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_190, [1, 0]);  primals_190 = None
    
    # No stacktrace found for following nodes
    mm_default_42: "f32[1568, 512]" = torch.ops.aten.mm.default(view_168, permute_83)
    add_tensor_42: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_42, primals_191);  mm_default_42 = primals_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_169: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_42, [8, 1, 196, 512]);  add_tensor_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_17: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.8043478280305862)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_27: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_17, 0.8043478280305862)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_111: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_169, div_27);  view_169 = div_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_76: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_72, mul_111);  add_72 = mul_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_76, [3], correction = 0, keepdim = True)
    getitem_78: "f32[8, 1, 196, 1]" = var_mean_22[0]
    getitem_79: "f32[8, 1, 196, 1]" = var_mean_22[1];  var_mean_22 = None
    add_77: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-06);  getitem_78 = None
    rsqrt_22: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
    sub_32: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_76, getitem_79);  getitem_79 = None
    mul_112: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_22);  sub_32 = None
    mul_113: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_112, primals_48)
    add_78: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_113, primals_49);  mul_113 = primals_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_170: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_78, [1568, 512]);  add_78 = None
    permute_84: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_192, [1, 0]);  primals_192 = None
    
    # No stacktrace found for following nodes
    mm_default_41: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_170, permute_84)
    add_tensor_41: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_41, primals_193);  mm_default_41 = primals_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_171: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_41, [8, 1, 196, 1536]);  add_tensor_41 = None
    view_172: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_171, [8, 1, 196, 3, 16, 32]);  view_171 = None
    permute_85: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_172, [3, 0, 4, 1, 2, 5]);  view_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_10 = torch.ops.aten.unbind.int(permute_85);  permute_85 = None
    getitem_80: "f32[8, 16, 1, 196, 32]" = unbind_10[0]
    getitem_81: "f32[8, 16, 1, 196, 32]" = unbind_10[1]
    getitem_82: "f32[8, 16, 1, 196, 32]" = unbind_10[2];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_114: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_80, 0.42044820762685725);  getitem_80 = None
    permute_86: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_81, [0, 1, 2, 4, 3]);  getitem_81 = None
    mul_115: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_86, 0.42044820762685725);  permute_86 = None
    expand_40: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_114, [8, 16, 1, 196, 32]);  mul_114 = None
    clone_76: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
    view_173: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_76, [128, 196, 32]);  clone_76 = None
    expand_41: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_115, [8, 16, 1, 32, 196]);  mul_115 = None
    clone_77: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
    view_174: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_77, [128, 32, 196]);  clone_77 = None
    bmm_20: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_173, view_174)
    view_175: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_20, [8, 16, 1, 196, 196]);  bmm_20 = None
    amax_10: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_175, [-1], True)
    sub_33: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_175, amax_10);  view_175 = amax_10 = None
    exp_10: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_33);  sub_33 = None
    sum_11: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_28: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    alias_10: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_28)
    expand_42: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_28, [8, 16, 1, 196, 196]);  div_28 = None
    view_176: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_42, [128, 196, 196]);  expand_42 = None
    expand_43: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_82, [8, 16, 1, 196, 32]);  getitem_82 = None
    clone_78: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
    view_177: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_78, [128, 196, 32]);  clone_78 = None
    bmm_21: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_176, view_177)
    view_178: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_21, [8, 16, 1, 196, 32]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_87: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_178, [0, 2, 3, 4, 1]);  view_178 = None
    clone_79: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_87, memory_format = torch.contiguous_format);  permute_87 = None
    view_179: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_79, [8, 1, 196, 512]);  clone_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_180: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_179, [1568, 512]);  view_179 = None
    permute_88: "f32[512, 512]" = torch.ops.aten.permute.default(primals_194, [1, 0]);  primals_194 = None
    
    # No stacktrace found for following nodes
    mm_default_40: "f32[1568, 512]" = torch.ops.aten.mm.default(view_180, permute_88)
    add_tensor_40: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_40, primals_195);  mm_default_40 = primals_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_181: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_40, [8, 1, 196, 512]);  add_tensor_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_18: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.782608687877655)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_29: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_18, 0.782608687877655)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_116: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_181, div_29);  view_181 = div_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_79: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_76, mul_116);  add_76 = mul_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_79, [3], correction = 0, keepdim = True)
    getitem_83: "f32[8, 1, 196, 1]" = var_mean_23[0]
    getitem_84: "f32[8, 1, 196, 1]" = var_mean_23[1];  var_mean_23 = None
    add_80: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_83, 1e-06);  getitem_83 = None
    rsqrt_23: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
    sub_34: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_79, getitem_84);  getitem_84 = None
    mul_117: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_23);  sub_34 = None
    mul_118: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_117, primals_50)
    add_81: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_118, primals_51);  mul_118 = primals_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_182: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_81, [1568, 512]);  add_81 = None
    permute_89: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_196, [1, 0]);  primals_196 = None
    addmm_42: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_197, view_182, permute_89);  primals_197 = None
    view_183: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(addmm_42, [8, 1, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_119: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_183, 0.5)
    mul_120: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_183, 0.7071067811865476);  view_183 = None
    erf_10: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_120);  mul_120 = None
    add_82: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_121: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_119, add_82);  mul_119 = add_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_184: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_121, [1568, 2048]);  mul_121 = None
    permute_90: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_198, [1, 0]);  primals_198 = None
    
    # No stacktrace found for following nodes
    mm_default_39: "f32[1568, 512]" = torch.ops.aten.mm.default(view_184, permute_90)
    add_tensor_39: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_39, primals_199);  mm_default_39 = primals_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_185: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_39, [8, 1, 196, 512]);  add_tensor_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_19: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.782608687877655)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_30: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_19, 0.782608687877655)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_122: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_185, div_30);  view_185 = div_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_83: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_79, mul_122);  add_79 = mul_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_83, [3], correction = 0, keepdim = True)
    getitem_85: "f32[8, 1, 196, 1]" = var_mean_24[0]
    getitem_86: "f32[8, 1, 196, 1]" = var_mean_24[1];  var_mean_24 = None
    add_84: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_85, 1e-06);  getitem_85 = None
    rsqrt_24: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_35: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_83, getitem_86);  getitem_86 = None
    mul_123: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_24);  sub_35 = None
    mul_124: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_123, primals_52)
    add_85: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_124, primals_53);  mul_124 = primals_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_186: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_85, [1568, 512]);  add_85 = None
    permute_91: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_200, [1, 0]);  primals_200 = None
    
    # No stacktrace found for following nodes
    mm_default_38: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_186, permute_91)
    add_tensor_38: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_38, primals_201);  mm_default_38 = primals_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_187: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_38, [8, 1, 196, 1536]);  add_tensor_38 = None
    view_188: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_187, [8, 1, 196, 3, 16, 32]);  view_187 = None
    permute_92: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_188, [3, 0, 4, 1, 2, 5]);  view_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_11 = torch.ops.aten.unbind.int(permute_92);  permute_92 = None
    getitem_87: "f32[8, 16, 1, 196, 32]" = unbind_11[0]
    getitem_88: "f32[8, 16, 1, 196, 32]" = unbind_11[1]
    getitem_89: "f32[8, 16, 1, 196, 32]" = unbind_11[2];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_125: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_87, 0.42044820762685725);  getitem_87 = None
    permute_93: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_88, [0, 1, 2, 4, 3]);  getitem_88 = None
    mul_126: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_93, 0.42044820762685725);  permute_93 = None
    expand_44: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_125, [8, 16, 1, 196, 32]);  mul_125 = None
    clone_83: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
    view_189: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_83, [128, 196, 32]);  clone_83 = None
    expand_45: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_126, [8, 16, 1, 32, 196]);  mul_126 = None
    clone_84: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
    view_190: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_84, [128, 32, 196]);  clone_84 = None
    bmm_22: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_189, view_190)
    view_191: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_22, [8, 16, 1, 196, 196]);  bmm_22 = None
    amax_11: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_191, [-1], True)
    sub_36: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_191, amax_11);  view_191 = amax_11 = None
    exp_11: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_36);  sub_36 = None
    sum_12: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_31: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    alias_11: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_31)
    expand_46: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_31, [8, 16, 1, 196, 196]);  div_31 = None
    view_192: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_46, [128, 196, 196]);  expand_46 = None
    expand_47: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_89, [8, 16, 1, 196, 32]);  getitem_89 = None
    clone_85: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
    view_193: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_85, [128, 196, 32]);  clone_85 = None
    bmm_23: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_192, view_193)
    view_194: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_23, [8, 16, 1, 196, 32]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_94: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_194, [0, 2, 3, 4, 1]);  view_194 = None
    clone_86: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_94, memory_format = torch.contiguous_format);  permute_94 = None
    view_195: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_86, [8, 1, 196, 512]);  clone_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_196: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_195, [1568, 512]);  view_195 = None
    permute_95: "f32[512, 512]" = torch.ops.aten.permute.default(primals_202, [1, 0]);  primals_202 = None
    
    # No stacktrace found for following nodes
    mm_default_37: "f32[1568, 512]" = torch.ops.aten.mm.default(view_196, permute_95)
    add_tensor_37: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_37, primals_203);  mm_default_37 = primals_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_197: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_37, [8, 1, 196, 512]);  add_tensor_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_20: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.760869562625885)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_32: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_20, 0.760869562625885)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_127: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_197, div_32);  view_197 = div_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_86: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_83, mul_127);  add_83 = mul_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_25 = torch.ops.aten.var_mean.correction(add_86, [3], correction = 0, keepdim = True)
    getitem_90: "f32[8, 1, 196, 1]" = var_mean_25[0]
    getitem_91: "f32[8, 1, 196, 1]" = var_mean_25[1];  var_mean_25 = None
    add_87: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-06);  getitem_90 = None
    rsqrt_25: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
    sub_37: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_86, getitem_91);  getitem_91 = None
    mul_128: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_25);  sub_37 = None
    mul_129: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_128, primals_54)
    add_88: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_129, primals_55);  mul_129 = primals_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_198: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_88, [1568, 512]);  add_88 = None
    permute_96: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_204, [1, 0]);  primals_204 = None
    addmm_46: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_205, view_198, permute_96);  primals_205 = None
    view_199: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(addmm_46, [8, 1, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_130: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_199, 0.5)
    mul_131: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_199, 0.7071067811865476);  view_199 = None
    erf_11: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_131);  mul_131 = None
    add_89: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_132: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_130, add_89);  mul_130 = add_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_200: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_132, [1568, 2048]);  mul_132 = None
    permute_97: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_206, [1, 0]);  primals_206 = None
    
    # No stacktrace found for following nodes
    mm_default_36: "f32[1568, 512]" = torch.ops.aten.mm.default(view_200, permute_97)
    add_tensor_36: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_36, primals_207);  mm_default_36 = primals_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_201: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_36, [8, 1, 196, 512]);  add_tensor_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_21: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.760869562625885)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_33: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_21, 0.760869562625885)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_133: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_201, div_33);  view_201 = div_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_90: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_86, mul_133);  add_86 = mul_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_26 = torch.ops.aten.var_mean.correction(add_90, [3], correction = 0, keepdim = True)
    getitem_92: "f32[8, 1, 196, 1]" = var_mean_26[0]
    getitem_93: "f32[8, 1, 196, 1]" = var_mean_26[1];  var_mean_26 = None
    add_91: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-06);  getitem_92 = None
    rsqrt_26: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
    sub_38: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_90, getitem_93);  getitem_93 = None
    mul_134: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_26);  sub_38 = None
    mul_135: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_134, primals_56)
    add_92: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_135, primals_57);  mul_135 = primals_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_202: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_92, [1568, 512]);  add_92 = None
    permute_98: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_208, [1, 0]);  primals_208 = None
    
    # No stacktrace found for following nodes
    mm_default_35: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_202, permute_98)
    add_tensor_35: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_35, primals_209);  mm_default_35 = primals_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_203: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_35, [8, 1, 196, 1536]);  add_tensor_35 = None
    view_204: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_203, [8, 1, 196, 3, 16, 32]);  view_203 = None
    permute_99: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_204, [3, 0, 4, 1, 2, 5]);  view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_12 = torch.ops.aten.unbind.int(permute_99);  permute_99 = None
    getitem_94: "f32[8, 16, 1, 196, 32]" = unbind_12[0]
    getitem_95: "f32[8, 16, 1, 196, 32]" = unbind_12[1]
    getitem_96: "f32[8, 16, 1, 196, 32]" = unbind_12[2];  unbind_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_136: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_94, 0.42044820762685725);  getitem_94 = None
    permute_100: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_95, [0, 1, 2, 4, 3]);  getitem_95 = None
    mul_137: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_100, 0.42044820762685725);  permute_100 = None
    expand_48: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_136, [8, 16, 1, 196, 32]);  mul_136 = None
    clone_90: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
    view_205: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_90, [128, 196, 32]);  clone_90 = None
    expand_49: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_137, [8, 16, 1, 32, 196]);  mul_137 = None
    clone_91: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
    view_206: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_91, [128, 32, 196]);  clone_91 = None
    bmm_24: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_205, view_206)
    view_207: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_24, [8, 16, 1, 196, 196]);  bmm_24 = None
    amax_12: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_207, [-1], True)
    sub_39: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_207, amax_12);  view_207 = amax_12 = None
    exp_12: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_39);  sub_39 = None
    sum_13: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_34: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    alias_12: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_34)
    expand_50: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_34, [8, 16, 1, 196, 196]);  div_34 = None
    view_208: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_50, [128, 196, 196]);  expand_50 = None
    expand_51: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_96, [8, 16, 1, 196, 32]);  getitem_96 = None
    clone_92: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
    view_209: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_92, [128, 196, 32]);  clone_92 = None
    bmm_25: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_208, view_209)
    view_210: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_25, [8, 16, 1, 196, 32]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_101: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_210, [0, 2, 3, 4, 1]);  view_210 = None
    clone_93: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_101, memory_format = torch.contiguous_format);  permute_101 = None
    view_211: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_93, [8, 1, 196, 512]);  clone_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_212: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_211, [1568, 512]);  view_211 = None
    permute_102: "f32[512, 512]" = torch.ops.aten.permute.default(primals_210, [1, 0]);  primals_210 = None
    
    # No stacktrace found for following nodes
    mm_default_34: "f32[1568, 512]" = torch.ops.aten.mm.default(view_212, permute_102)
    add_tensor_34: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_34, primals_211);  mm_default_34 = primals_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_213: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_34, [8, 1, 196, 512]);  add_tensor_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_22: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.739130437374115)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_35: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_22, 0.739130437374115)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_138: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_213, div_35);  view_213 = div_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_93: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_90, mul_138);  add_90 = mul_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_27 = torch.ops.aten.var_mean.correction(add_93, [3], correction = 0, keepdim = True)
    getitem_97: "f32[8, 1, 196, 1]" = var_mean_27[0]
    getitem_98: "f32[8, 1, 196, 1]" = var_mean_27[1];  var_mean_27 = None
    add_94: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_97, 1e-06);  getitem_97 = None
    rsqrt_27: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
    sub_40: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_93, getitem_98);  getitem_98 = None
    mul_139: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_27);  sub_40 = None
    mul_140: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_139, primals_58)
    add_95: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_140, primals_59);  mul_140 = primals_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_214: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_95, [1568, 512]);  add_95 = None
    permute_103: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_212, [1, 0]);  primals_212 = None
    addmm_50: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_213, view_214, permute_103);  primals_213 = None
    view_215: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(addmm_50, [8, 1, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_141: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_215, 0.5)
    mul_142: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_215, 0.7071067811865476);  view_215 = None
    erf_12: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_142);  mul_142 = None
    add_96: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_143: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_141, add_96);  mul_141 = add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_216: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_143, [1568, 2048]);  mul_143 = None
    permute_104: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_214, [1, 0]);  primals_214 = None
    
    # No stacktrace found for following nodes
    mm_default_33: "f32[1568, 512]" = torch.ops.aten.mm.default(view_216, permute_104)
    add_tensor_33: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_33, primals_215);  mm_default_33 = primals_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_217: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_33, [8, 1, 196, 512]);  add_tensor_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_23: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.739130437374115)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_36: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_23, 0.739130437374115)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_144: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_217, div_36);  view_217 = div_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_97: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_93, mul_144);  add_93 = mul_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_28 = torch.ops.aten.var_mean.correction(add_97, [3], correction = 0, keepdim = True)
    getitem_99: "f32[8, 1, 196, 1]" = var_mean_28[0]
    getitem_100: "f32[8, 1, 196, 1]" = var_mean_28[1];  var_mean_28 = None
    add_98: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_99, 1e-06);  getitem_99 = None
    rsqrt_28: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    sub_41: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_97, getitem_100);  getitem_100 = None
    mul_145: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_28);  sub_41 = None
    mul_146: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_145, primals_60)
    add_99: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_146, primals_61);  mul_146 = primals_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_218: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_99, [1568, 512]);  add_99 = None
    permute_105: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_216, [1, 0]);  primals_216 = None
    
    # No stacktrace found for following nodes
    mm_default_32: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_218, permute_105)
    add_tensor_32: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_32, primals_217);  mm_default_32 = primals_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_219: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_32, [8, 1, 196, 1536]);  add_tensor_32 = None
    view_220: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_219, [8, 1, 196, 3, 16, 32]);  view_219 = None
    permute_106: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_220, [3, 0, 4, 1, 2, 5]);  view_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_13 = torch.ops.aten.unbind.int(permute_106);  permute_106 = None
    getitem_101: "f32[8, 16, 1, 196, 32]" = unbind_13[0]
    getitem_102: "f32[8, 16, 1, 196, 32]" = unbind_13[1]
    getitem_103: "f32[8, 16, 1, 196, 32]" = unbind_13[2];  unbind_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_147: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_101, 0.42044820762685725);  getitem_101 = None
    permute_107: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_102, [0, 1, 2, 4, 3]);  getitem_102 = None
    mul_148: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_107, 0.42044820762685725);  permute_107 = None
    expand_52: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_147, [8, 16, 1, 196, 32]);  mul_147 = None
    clone_97: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
    view_221: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_97, [128, 196, 32]);  clone_97 = None
    expand_53: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_148, [8, 16, 1, 32, 196]);  mul_148 = None
    clone_98: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
    view_222: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_98, [128, 32, 196]);  clone_98 = None
    bmm_26: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_221, view_222)
    view_223: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_26, [8, 16, 1, 196, 196]);  bmm_26 = None
    amax_13: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_223, [-1], True)
    sub_42: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_223, amax_13);  view_223 = amax_13 = None
    exp_13: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_42);  sub_42 = None
    sum_14: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_37: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
    alias_13: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_37)
    expand_54: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_37, [8, 16, 1, 196, 196]);  div_37 = None
    view_224: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_54, [128, 196, 196]);  expand_54 = None
    expand_55: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_103, [8, 16, 1, 196, 32]);  getitem_103 = None
    clone_99: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
    view_225: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_99, [128, 196, 32]);  clone_99 = None
    bmm_27: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_224, view_225)
    view_226: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_27, [8, 16, 1, 196, 32]);  bmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_108: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_226, [0, 2, 3, 4, 1]);  view_226 = None
    clone_100: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_108, memory_format = torch.contiguous_format);  permute_108 = None
    view_227: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_100, [8, 1, 196, 512]);  clone_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_228: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_227, [1568, 512]);  view_227 = None
    permute_109: "f32[512, 512]" = torch.ops.aten.permute.default(primals_218, [1, 0]);  primals_218 = None
    
    # No stacktrace found for following nodes
    mm_default_31: "f32[1568, 512]" = torch.ops.aten.mm.default(view_228, permute_109)
    add_tensor_31: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_31, primals_219);  mm_default_31 = primals_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_229: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_31, [8, 1, 196, 512]);  add_tensor_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_24: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.717391312122345)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_38: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_24, 0.717391312122345)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_149: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_229, div_38);  view_229 = div_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_100: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_97, mul_149);  add_97 = mul_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_29 = torch.ops.aten.var_mean.correction(add_100, [3], correction = 0, keepdim = True)
    getitem_104: "f32[8, 1, 196, 1]" = var_mean_29[0]
    getitem_105: "f32[8, 1, 196, 1]" = var_mean_29[1];  var_mean_29 = None
    add_101: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-06);  getitem_104 = None
    rsqrt_29: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
    sub_43: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_100, getitem_105);  getitem_105 = None
    mul_150: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_29);  sub_43 = None
    mul_151: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_150, primals_62)
    add_102: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_151, primals_63);  mul_151 = primals_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_230: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_102, [1568, 512]);  add_102 = None
    permute_110: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_220, [1, 0]);  primals_220 = None
    addmm_54: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_221, view_230, permute_110);  primals_221 = None
    view_231: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(addmm_54, [8, 1, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_152: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_231, 0.5)
    mul_153: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_231, 0.7071067811865476);  view_231 = None
    erf_13: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_153);  mul_153 = None
    add_103: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_154: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_152, add_103);  mul_152 = add_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_232: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_154, [1568, 2048]);  mul_154 = None
    permute_111: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_222, [1, 0]);  primals_222 = None
    
    # No stacktrace found for following nodes
    mm_default_30: "f32[1568, 512]" = torch.ops.aten.mm.default(view_232, permute_111)
    add_tensor_30: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_30, primals_223);  mm_default_30 = primals_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_233: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_30, [8, 1, 196, 512]);  add_tensor_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_25: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.717391312122345)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_39: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_25, 0.717391312122345)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_155: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_233, div_39);  view_233 = div_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_104: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_100, mul_155);  add_100 = mul_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_30 = torch.ops.aten.var_mean.correction(add_104, [3], correction = 0, keepdim = True)
    getitem_106: "f32[8, 1, 196, 1]" = var_mean_30[0]
    getitem_107: "f32[8, 1, 196, 1]" = var_mean_30[1];  var_mean_30 = None
    add_105: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-06);  getitem_106 = None
    rsqrt_30: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
    sub_44: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_104, getitem_107);  getitem_107 = None
    mul_156: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_30);  sub_44 = None
    mul_157: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_156, primals_64)
    add_106: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_157, primals_65);  mul_157 = primals_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_234: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_106, [1568, 512]);  add_106 = None
    permute_112: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_224, [1, 0]);  primals_224 = None
    
    # No stacktrace found for following nodes
    mm_default_29: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_234, permute_112)
    add_tensor_29: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_29, primals_225);  mm_default_29 = primals_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_235: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_29, [8, 1, 196, 1536]);  add_tensor_29 = None
    view_236: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_235, [8, 1, 196, 3, 16, 32]);  view_235 = None
    permute_113: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_236, [3, 0, 4, 1, 2, 5]);  view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_14 = torch.ops.aten.unbind.int(permute_113);  permute_113 = None
    getitem_108: "f32[8, 16, 1, 196, 32]" = unbind_14[0]
    getitem_109: "f32[8, 16, 1, 196, 32]" = unbind_14[1]
    getitem_110: "f32[8, 16, 1, 196, 32]" = unbind_14[2];  unbind_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_158: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_108, 0.42044820762685725);  getitem_108 = None
    permute_114: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_109, [0, 1, 2, 4, 3]);  getitem_109 = None
    mul_159: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_114, 0.42044820762685725);  permute_114 = None
    expand_56: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_158, [8, 16, 1, 196, 32]);  mul_158 = None
    clone_104: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
    view_237: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_104, [128, 196, 32]);  clone_104 = None
    expand_57: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_159, [8, 16, 1, 32, 196]);  mul_159 = None
    clone_105: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
    view_238: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_105, [128, 32, 196]);  clone_105 = None
    bmm_28: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_237, view_238)
    view_239: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_28, [8, 16, 1, 196, 196]);  bmm_28 = None
    amax_14: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_239, [-1], True)
    sub_45: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_239, amax_14);  view_239 = amax_14 = None
    exp_14: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_45);  sub_45 = None
    sum_15: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_40: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
    alias_14: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_40)
    expand_58: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_40, [8, 16, 1, 196, 196]);  div_40 = None
    view_240: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_58, [128, 196, 196]);  expand_58 = None
    expand_59: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_110, [8, 16, 1, 196, 32]);  getitem_110 = None
    clone_106: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
    view_241: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_106, [128, 196, 32]);  clone_106 = None
    bmm_29: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_240, view_241)
    view_242: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_29, [8, 16, 1, 196, 32]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_115: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_242, [0, 2, 3, 4, 1]);  view_242 = None
    clone_107: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
    view_243: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_107, [8, 1, 196, 512]);  clone_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_244: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_243, [1568, 512]);  view_243 = None
    permute_116: "f32[512, 512]" = torch.ops.aten.permute.default(primals_226, [1, 0]);  primals_226 = None
    
    # No stacktrace found for following nodes
    mm_default_28: "f32[1568, 512]" = torch.ops.aten.mm.default(view_244, permute_116)
    add_tensor_28: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_28, primals_227);  mm_default_28 = primals_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_245: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_28, [8, 1, 196, 512]);  add_tensor_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_26: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.695652186870575)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_41: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_26, 0.695652186870575)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_160: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_245, div_41);  view_245 = div_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_107: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_104, mul_160);  add_104 = mul_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_31 = torch.ops.aten.var_mean.correction(add_107, [3], correction = 0, keepdim = True)
    getitem_111: "f32[8, 1, 196, 1]" = var_mean_31[0]
    getitem_112: "f32[8, 1, 196, 1]" = var_mean_31[1];  var_mean_31 = None
    add_108: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_111, 1e-06);  getitem_111 = None
    rsqrt_31: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    sub_46: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_107, getitem_112);  getitem_112 = None
    mul_161: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_31);  sub_46 = None
    mul_162: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_161, primals_66)
    add_109: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_162, primals_67);  mul_162 = primals_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_246: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_109, [1568, 512]);  add_109 = None
    permute_117: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_228, [1, 0]);  primals_228 = None
    addmm_58: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_229, view_246, permute_117);  primals_229 = None
    view_247: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(addmm_58, [8, 1, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_163: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_247, 0.5)
    mul_164: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_247, 0.7071067811865476);  view_247 = None
    erf_14: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_164);  mul_164 = None
    add_110: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_165: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_163, add_110);  mul_163 = add_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_248: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_165, [1568, 2048]);  mul_165 = None
    permute_118: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_230, [1, 0]);  primals_230 = None
    
    # No stacktrace found for following nodes
    mm_default_27: "f32[1568, 512]" = torch.ops.aten.mm.default(view_248, permute_118)
    add_tensor_27: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_27, primals_231);  mm_default_27 = primals_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_249: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_27, [8, 1, 196, 512]);  add_tensor_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_27: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.695652186870575)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_42: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_27, 0.695652186870575)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_166: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_249, div_42);  view_249 = div_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_111: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_107, mul_166);  add_107 = mul_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_32 = torch.ops.aten.var_mean.correction(add_111, [3], correction = 0, keepdim = True)
    getitem_113: "f32[8, 1, 196, 1]" = var_mean_32[0]
    getitem_114: "f32[8, 1, 196, 1]" = var_mean_32[1];  var_mean_32 = None
    add_112: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_113, 1e-06);  getitem_113 = None
    rsqrt_32: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
    sub_47: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_111, getitem_114);  getitem_114 = None
    mul_167: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_32);  sub_47 = None
    mul_168: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_167, primals_68)
    add_113: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_168, primals_69);  mul_168 = primals_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_250: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_113, [1568, 512]);  add_113 = None
    permute_119: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_232, [1, 0]);  primals_232 = None
    
    # No stacktrace found for following nodes
    mm_default_26: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_250, permute_119)
    add_tensor_26: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_26, primals_233);  mm_default_26 = primals_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_251: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_26, [8, 1, 196, 1536]);  add_tensor_26 = None
    view_252: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_251, [8, 1, 196, 3, 16, 32]);  view_251 = None
    permute_120: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_252, [3, 0, 4, 1, 2, 5]);  view_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_15 = torch.ops.aten.unbind.int(permute_120);  permute_120 = None
    getitem_115: "f32[8, 16, 1, 196, 32]" = unbind_15[0]
    getitem_116: "f32[8, 16, 1, 196, 32]" = unbind_15[1]
    getitem_117: "f32[8, 16, 1, 196, 32]" = unbind_15[2];  unbind_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_169: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_115, 0.42044820762685725);  getitem_115 = None
    permute_121: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_116, [0, 1, 2, 4, 3]);  getitem_116 = None
    mul_170: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_121, 0.42044820762685725);  permute_121 = None
    expand_60: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_169, [8, 16, 1, 196, 32]);  mul_169 = None
    clone_111: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
    view_253: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_111, [128, 196, 32]);  clone_111 = None
    expand_61: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_170, [8, 16, 1, 32, 196]);  mul_170 = None
    clone_112: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
    view_254: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_112, [128, 32, 196]);  clone_112 = None
    bmm_30: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_253, view_254)
    view_255: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_30, [8, 16, 1, 196, 196]);  bmm_30 = None
    amax_15: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_255, [-1], True)
    sub_48: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_255, amax_15);  view_255 = amax_15 = None
    exp_15: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_48);  sub_48 = None
    sum_16: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
    div_43: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
    alias_15: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_43)
    expand_62: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_43, [8, 16, 1, 196, 196]);  div_43 = None
    view_256: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_62, [128, 196, 196]);  expand_62 = None
    expand_63: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_117, [8, 16, 1, 196, 32]);  getitem_117 = None
    clone_113: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
    view_257: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_113, [128, 196, 32]);  clone_113 = None
    bmm_31: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_256, view_257)
    view_258: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_31, [8, 16, 1, 196, 32]);  bmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_122: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_258, [0, 2, 3, 4, 1]);  view_258 = None
    clone_114: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
    view_259: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_114, [8, 1, 196, 512]);  clone_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_260: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_259, [1568, 512]);  view_259 = None
    permute_123: "f32[512, 512]" = torch.ops.aten.permute.default(primals_234, [1, 0]);  primals_234 = None
    
    # No stacktrace found for following nodes
    mm_default_25: "f32[1568, 512]" = torch.ops.aten.mm.default(view_260, permute_123)
    add_tensor_25: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_25, primals_235);  mm_default_25 = primals_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_261: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_25, [8, 1, 196, 512]);  add_tensor_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_28: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.6739130616188049)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_44: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_28, 0.6739130616188049)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_171: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_261, div_44);  view_261 = div_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_114: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_111, mul_171);  add_111 = mul_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_33 = torch.ops.aten.var_mean.correction(add_114, [3], correction = 0, keepdim = True)
    getitem_118: "f32[8, 1, 196, 1]" = var_mean_33[0]
    getitem_119: "f32[8, 1, 196, 1]" = var_mean_33[1];  var_mean_33 = None
    add_115: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-06);  getitem_118 = None
    rsqrt_33: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
    sub_49: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_114, getitem_119);  getitem_119 = None
    mul_172: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_33);  sub_49 = None
    mul_173: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_172, primals_70)
    add_116: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_173, primals_71);  mul_173 = primals_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_262: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_116, [1568, 512]);  add_116 = None
    permute_124: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_236, [1, 0]);  primals_236 = None
    addmm_62: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_237, view_262, permute_124);  primals_237 = None
    view_263: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(addmm_62, [8, 1, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_174: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_263, 0.5)
    mul_175: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_263, 0.7071067811865476);  view_263 = None
    erf_15: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_175);  mul_175 = None
    add_117: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_176: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_174, add_117);  mul_174 = add_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_264: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_176, [1568, 2048]);  mul_176 = None
    permute_125: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_238, [1, 0]);  primals_238 = None
    
    # No stacktrace found for following nodes
    mm_default_24: "f32[1568, 512]" = torch.ops.aten.mm.default(view_264, permute_125)
    add_tensor_24: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_24, primals_239);  mm_default_24 = primals_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_265: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_24, [8, 1, 196, 512]);  add_tensor_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_29: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.6739130616188049)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_45: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_29, 0.6739130616188049)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_177: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_265, div_45);  view_265 = div_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_118: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_114, mul_177);  add_114 = mul_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_34 = torch.ops.aten.var_mean.correction(add_118, [3], correction = 0, keepdim = True)
    getitem_120: "f32[8, 1, 196, 1]" = var_mean_34[0]
    getitem_121: "f32[8, 1, 196, 1]" = var_mean_34[1];  var_mean_34 = None
    add_119: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-06);  getitem_120 = None
    rsqrt_34: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
    sub_50: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_118, getitem_121);  getitem_121 = None
    mul_178: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_34);  sub_50 = None
    mul_179: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_178, primals_72)
    add_120: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_179, primals_73);  mul_179 = primals_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_266: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_120, [1568, 512]);  add_120 = None
    permute_126: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_240, [1, 0]);  primals_240 = None
    
    # No stacktrace found for following nodes
    mm_default_23: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_266, permute_126)
    add_tensor_23: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_23, primals_241);  mm_default_23 = primals_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_267: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_23, [8, 1, 196, 1536]);  add_tensor_23 = None
    view_268: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_267, [8, 1, 196, 3, 16, 32]);  view_267 = None
    permute_127: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_268, [3, 0, 4, 1, 2, 5]);  view_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_16 = torch.ops.aten.unbind.int(permute_127);  permute_127 = None
    getitem_122: "f32[8, 16, 1, 196, 32]" = unbind_16[0]
    getitem_123: "f32[8, 16, 1, 196, 32]" = unbind_16[1]
    getitem_124: "f32[8, 16, 1, 196, 32]" = unbind_16[2];  unbind_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_180: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_122, 0.42044820762685725);  getitem_122 = None
    permute_128: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_123, [0, 1, 2, 4, 3]);  getitem_123 = None
    mul_181: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_128, 0.42044820762685725);  permute_128 = None
    expand_64: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_180, [8, 16, 1, 196, 32]);  mul_180 = None
    clone_118: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
    view_269: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_118, [128, 196, 32]);  clone_118 = None
    expand_65: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_181, [8, 16, 1, 32, 196]);  mul_181 = None
    clone_119: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
    view_270: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_119, [128, 32, 196]);  clone_119 = None
    bmm_32: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_269, view_270)
    view_271: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_32, [8, 16, 1, 196, 196]);  bmm_32 = None
    amax_16: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_271, [-1], True)
    sub_51: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_271, amax_16);  view_271 = amax_16 = None
    exp_16: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_51);  sub_51 = None
    sum_17: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_46: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
    alias_16: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_46)
    expand_66: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_46, [8, 16, 1, 196, 196]);  div_46 = None
    view_272: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_66, [128, 196, 196]);  expand_66 = None
    expand_67: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_124, [8, 16, 1, 196, 32]);  getitem_124 = None
    clone_120: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
    view_273: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_120, [128, 196, 32]);  clone_120 = None
    bmm_33: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_272, view_273)
    view_274: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_33, [8, 16, 1, 196, 32]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_129: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_274, [0, 2, 3, 4, 1]);  view_274 = None
    clone_121: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None
    view_275: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_121, [8, 1, 196, 512]);  clone_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_276: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_275, [1568, 512]);  view_275 = None
    permute_130: "f32[512, 512]" = torch.ops.aten.permute.default(primals_242, [1, 0]);  primals_242 = None
    
    # No stacktrace found for following nodes
    mm_default_22: "f32[1568, 512]" = torch.ops.aten.mm.default(view_276, permute_130)
    add_tensor_22: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_22, primals_243);  mm_default_22 = primals_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_277: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_22, [8, 1, 196, 512]);  add_tensor_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_30: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.6521739065647125)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_47: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_30, 0.6521739065647125)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_182: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_277, div_47);  view_277 = div_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_121: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_118, mul_182);  add_118 = mul_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_35 = torch.ops.aten.var_mean.correction(add_121, [3], correction = 0, keepdim = True)
    getitem_125: "f32[8, 1, 196, 1]" = var_mean_35[0]
    getitem_126: "f32[8, 1, 196, 1]" = var_mean_35[1];  var_mean_35 = None
    add_122: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_125, 1e-06);  getitem_125 = None
    rsqrt_35: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    sub_52: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_121, getitem_126);  getitem_126 = None
    mul_183: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_35);  sub_52 = None
    mul_184: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_183, primals_74)
    add_123: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_184, primals_75);  mul_184 = primals_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_278: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_123, [1568, 512]);  add_123 = None
    permute_131: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_244, [1, 0]);  primals_244 = None
    addmm_66: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_245, view_278, permute_131);  primals_245 = None
    view_279: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(addmm_66, [8, 1, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_185: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_279, 0.5)
    mul_186: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_279, 0.7071067811865476);  view_279 = None
    erf_16: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_186);  mul_186 = None
    add_124: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_187: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_185, add_124);  mul_185 = add_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_280: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_187, [1568, 2048]);  mul_187 = None
    permute_132: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_246, [1, 0]);  primals_246 = None
    
    # No stacktrace found for following nodes
    mm_default_21: "f32[1568, 512]" = torch.ops.aten.mm.default(view_280, permute_132)
    add_tensor_21: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_21, primals_247);  mm_default_21 = primals_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_281: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_21, [8, 1, 196, 512]);  add_tensor_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_31: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.6521739065647125)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_48: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_31, 0.6521739065647125)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_188: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_281, div_48);  view_281 = div_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_125: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_121, mul_188);  add_121 = mul_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_36 = torch.ops.aten.var_mean.correction(add_125, [3], correction = 0, keepdim = True)
    getitem_127: "f32[8, 1, 196, 1]" = var_mean_36[0]
    getitem_128: "f32[8, 1, 196, 1]" = var_mean_36[1];  var_mean_36 = None
    add_126: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_127, 1e-06);  getitem_127 = None
    rsqrt_36: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
    sub_53: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_125, getitem_128);  getitem_128 = None
    mul_189: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_36);  sub_53 = None
    mul_190: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_189, primals_76)
    add_127: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_190, primals_77);  mul_190 = primals_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_282: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_127, [1568, 512]);  add_127 = None
    permute_133: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_248, [1, 0]);  primals_248 = None
    
    # No stacktrace found for following nodes
    mm_default_20: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_282, permute_133)
    add_tensor_20: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_20, primals_249);  mm_default_20 = primals_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_283: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_20, [8, 1, 196, 1536]);  add_tensor_20 = None
    view_284: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_283, [8, 1, 196, 3, 16, 32]);  view_283 = None
    permute_134: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_284, [3, 0, 4, 1, 2, 5]);  view_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_17 = torch.ops.aten.unbind.int(permute_134);  permute_134 = None
    getitem_129: "f32[8, 16, 1, 196, 32]" = unbind_17[0]
    getitem_130: "f32[8, 16, 1, 196, 32]" = unbind_17[1]
    getitem_131: "f32[8, 16, 1, 196, 32]" = unbind_17[2];  unbind_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_191: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_129, 0.42044820762685725);  getitem_129 = None
    permute_135: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_130, [0, 1, 2, 4, 3]);  getitem_130 = None
    mul_192: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_135, 0.42044820762685725);  permute_135 = None
    expand_68: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_191, [8, 16, 1, 196, 32]);  mul_191 = None
    clone_125: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
    view_285: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_125, [128, 196, 32]);  clone_125 = None
    expand_69: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_192, [8, 16, 1, 32, 196]);  mul_192 = None
    clone_126: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
    view_286: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_126, [128, 32, 196]);  clone_126 = None
    bmm_34: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_285, view_286)
    view_287: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_34, [8, 16, 1, 196, 196]);  bmm_34 = None
    amax_17: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_287, [-1], True)
    sub_54: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_287, amax_17);  view_287 = amax_17 = None
    exp_17: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_54);  sub_54 = None
    sum_18: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
    div_49: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
    alias_17: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_49)
    expand_70: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_49, [8, 16, 1, 196, 196]);  div_49 = None
    view_288: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_70, [128, 196, 196]);  expand_70 = None
    expand_71: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_131, [8, 16, 1, 196, 32]);  getitem_131 = None
    clone_127: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
    view_289: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_127, [128, 196, 32]);  clone_127 = None
    bmm_35: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_288, view_289)
    view_290: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_35, [8, 16, 1, 196, 32]);  bmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_136: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_290, [0, 2, 3, 4, 1]);  view_290 = None
    clone_128: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_136, memory_format = torch.contiguous_format);  permute_136 = None
    view_291: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_128, [8, 1, 196, 512]);  clone_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_292: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_291, [1568, 512]);  view_291 = None
    permute_137: "f32[512, 512]" = torch.ops.aten.permute.default(primals_250, [1, 0]);  primals_250 = None
    
    # No stacktrace found for following nodes
    mm_default_19: "f32[1568, 512]" = torch.ops.aten.mm.default(view_292, permute_137)
    add_tensor_19: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_19, primals_251);  mm_default_19 = primals_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_293: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_19, [8, 1, 196, 512]);  add_tensor_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_32: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.6304347813129425)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_50: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_32, 0.6304347813129425)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_193: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_293, div_50);  view_293 = div_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_128: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_125, mul_193);  add_125 = mul_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_37 = torch.ops.aten.var_mean.correction(add_128, [3], correction = 0, keepdim = True)
    getitem_132: "f32[8, 1, 196, 1]" = var_mean_37[0]
    getitem_133: "f32[8, 1, 196, 1]" = var_mean_37[1];  var_mean_37 = None
    add_129: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-06);  getitem_132 = None
    rsqrt_37: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
    sub_55: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_128, getitem_133);  getitem_133 = None
    mul_194: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_37);  sub_55 = None
    mul_195: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_194, primals_78)
    add_130: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_195, primals_79);  mul_195 = primals_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_294: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_130, [1568, 512]);  add_130 = None
    permute_138: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_252, [1, 0]);  primals_252 = None
    addmm_70: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_253, view_294, permute_138);  primals_253 = None
    view_295: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(addmm_70, [8, 1, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_196: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_295, 0.5)
    mul_197: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_295, 0.7071067811865476);  view_295 = None
    erf_17: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_197);  mul_197 = None
    add_131: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_198: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_196, add_131);  mul_196 = add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_296: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_198, [1568, 2048]);  mul_198 = None
    permute_139: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_254, [1, 0]);  primals_254 = None
    
    # No stacktrace found for following nodes
    mm_default_18: "f32[1568, 512]" = torch.ops.aten.mm.default(view_296, permute_139)
    add_tensor_18: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_18, primals_255);  mm_default_18 = primals_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_297: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_18, [8, 1, 196, 512]);  add_tensor_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_33: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.6304347813129425)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_51: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_33, 0.6304347813129425)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_199: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_297, div_51);  view_297 = div_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_132: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_128, mul_199);  add_128 = mul_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_38 = torch.ops.aten.var_mean.correction(add_132, [3], correction = 0, keepdim = True)
    getitem_134: "f32[8, 1, 196, 1]" = var_mean_38[0]
    getitem_135: "f32[8, 1, 196, 1]" = var_mean_38[1];  var_mean_38 = None
    add_133: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_134, 1e-06);  getitem_134 = None
    rsqrt_38: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
    sub_56: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_132, getitem_135);  getitem_135 = None
    mul_200: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_38);  sub_56 = None
    mul_201: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_200, primals_80)
    add_134: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_201, primals_81);  mul_201 = primals_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_298: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_134, [1568, 512]);  add_134 = None
    permute_140: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_256, [1, 0]);  primals_256 = None
    
    # No stacktrace found for following nodes
    mm_default_17: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_298, permute_140)
    add_tensor_17: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_17, primals_257);  mm_default_17 = primals_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_299: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_17, [8, 1, 196, 1536]);  add_tensor_17 = None
    view_300: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_299, [8, 1, 196, 3, 16, 32]);  view_299 = None
    permute_141: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_300, [3, 0, 4, 1, 2, 5]);  view_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_18 = torch.ops.aten.unbind.int(permute_141);  permute_141 = None
    getitem_136: "f32[8, 16, 1, 196, 32]" = unbind_18[0]
    getitem_137: "f32[8, 16, 1, 196, 32]" = unbind_18[1]
    getitem_138: "f32[8, 16, 1, 196, 32]" = unbind_18[2];  unbind_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_202: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_136, 0.42044820762685725);  getitem_136 = None
    permute_142: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_137, [0, 1, 2, 4, 3]);  getitem_137 = None
    mul_203: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_142, 0.42044820762685725);  permute_142 = None
    expand_72: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_202, [8, 16, 1, 196, 32]);  mul_202 = None
    clone_132: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
    view_301: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_132, [128, 196, 32]);  clone_132 = None
    expand_73: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_203, [8, 16, 1, 32, 196]);  mul_203 = None
    clone_133: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
    view_302: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_133, [128, 32, 196]);  clone_133 = None
    bmm_36: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_301, view_302)
    view_303: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_36, [8, 16, 1, 196, 196]);  bmm_36 = None
    amax_18: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_303, [-1], True)
    sub_57: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_303, amax_18);  view_303 = amax_18 = None
    exp_18: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_57);  sub_57 = None
    sum_19: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
    div_52: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
    alias_18: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_52)
    expand_74: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_52, [8, 16, 1, 196, 196]);  div_52 = None
    view_304: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_74, [128, 196, 196]);  expand_74 = None
    expand_75: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_138, [8, 16, 1, 196, 32]);  getitem_138 = None
    clone_134: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
    view_305: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_134, [128, 196, 32]);  clone_134 = None
    bmm_37: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_304, view_305)
    view_306: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_37, [8, 16, 1, 196, 32]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_143: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_306, [0, 2, 3, 4, 1]);  view_306 = None
    clone_135: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_143, memory_format = torch.contiguous_format);  permute_143 = None
    view_307: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_135, [8, 1, 196, 512]);  clone_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_308: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_307, [1568, 512]);  view_307 = None
    permute_144: "f32[512, 512]" = torch.ops.aten.permute.default(primals_258, [1, 0]);  primals_258 = None
    
    # No stacktrace found for following nodes
    mm_default_16: "f32[1568, 512]" = torch.ops.aten.mm.default(view_308, permute_144)
    add_tensor_16: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_16, primals_259);  mm_default_16 = primals_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_309: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_16, [8, 1, 196, 512]);  add_tensor_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_34: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.6086956560611725)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_53: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_34, 0.6086956560611725)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_204: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_309, div_53);  view_309 = div_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_135: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_132, mul_204);  add_132 = mul_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_39 = torch.ops.aten.var_mean.correction(add_135, [3], correction = 0, keepdim = True)
    getitem_139: "f32[8, 1, 196, 1]" = var_mean_39[0]
    getitem_140: "f32[8, 1, 196, 1]" = var_mean_39[1];  var_mean_39 = None
    add_136: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_139, 1e-06);  getitem_139 = None
    rsqrt_39: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
    sub_58: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_135, getitem_140);  getitem_140 = None
    mul_205: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_39);  sub_58 = None
    mul_206: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_205, primals_82)
    add_137: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_206, primals_83);  mul_206 = primals_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_310: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_137, [1568, 512]);  add_137 = None
    permute_145: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_260, [1, 0]);  primals_260 = None
    addmm_74: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_261, view_310, permute_145);  primals_261 = None
    view_311: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(addmm_74, [8, 1, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_207: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_311, 0.5)
    mul_208: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_311, 0.7071067811865476);  view_311 = None
    erf_18: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_208);  mul_208 = None
    add_138: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_209: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_207, add_138);  mul_207 = add_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_312: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_209, [1568, 2048]);  mul_209 = None
    permute_146: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_262, [1, 0]);  primals_262 = None
    
    # No stacktrace found for following nodes
    mm_default_15: "f32[1568, 512]" = torch.ops.aten.mm.default(view_312, permute_146)
    add_tensor_15: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_15, primals_263);  mm_default_15 = primals_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_313: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_15, [8, 1, 196, 512]);  add_tensor_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_35: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.6086956560611725)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_54: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_35, 0.6086956560611725)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_210: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_313, div_54);  view_313 = div_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_139: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_135, mul_210);  add_135 = mul_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_40 = torch.ops.aten.var_mean.correction(add_139, [3], correction = 0, keepdim = True)
    getitem_141: "f32[8, 1, 196, 1]" = var_mean_40[0]
    getitem_142: "f32[8, 1, 196, 1]" = var_mean_40[1];  var_mean_40 = None
    add_140: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_141, 1e-06);  getitem_141 = None
    rsqrt_40: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    sub_59: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_139, getitem_142);  getitem_142 = None
    mul_211: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_40);  sub_59 = None
    mul_212: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_211, primals_84)
    add_141: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_212, primals_85);  mul_212 = primals_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_314: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_141, [1568, 512]);  add_141 = None
    permute_147: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_264, [1, 0]);  primals_264 = None
    
    # No stacktrace found for following nodes
    mm_default_14: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_314, permute_147)
    add_tensor_14: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_14, primals_265);  mm_default_14 = primals_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_315: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_14, [8, 1, 196, 1536]);  add_tensor_14 = None
    view_316: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_315, [8, 1, 196, 3, 16, 32]);  view_315 = None
    permute_148: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_316, [3, 0, 4, 1, 2, 5]);  view_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_19 = torch.ops.aten.unbind.int(permute_148);  permute_148 = None
    getitem_143: "f32[8, 16, 1, 196, 32]" = unbind_19[0]
    getitem_144: "f32[8, 16, 1, 196, 32]" = unbind_19[1]
    getitem_145: "f32[8, 16, 1, 196, 32]" = unbind_19[2];  unbind_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_213: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_143, 0.42044820762685725);  getitem_143 = None
    permute_149: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_144, [0, 1, 2, 4, 3]);  getitem_144 = None
    mul_214: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_149, 0.42044820762685725);  permute_149 = None
    expand_76: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_213, [8, 16, 1, 196, 32]);  mul_213 = None
    clone_139: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
    view_317: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_139, [128, 196, 32]);  clone_139 = None
    expand_77: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_214, [8, 16, 1, 32, 196]);  mul_214 = None
    clone_140: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_77, memory_format = torch.contiguous_format);  expand_77 = None
    view_318: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_140, [128, 32, 196]);  clone_140 = None
    bmm_38: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_317, view_318)
    view_319: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_38, [8, 16, 1, 196, 196]);  bmm_38 = None
    amax_19: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_319, [-1], True)
    sub_60: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_319, amax_19);  view_319 = amax_19 = None
    exp_19: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_60);  sub_60 = None
    sum_20: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
    div_55: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
    alias_19: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_55)
    expand_78: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_55, [8, 16, 1, 196, 196]);  div_55 = None
    view_320: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_78, [128, 196, 196]);  expand_78 = None
    expand_79: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_145, [8, 16, 1, 196, 32]);  getitem_145 = None
    clone_141: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_79, memory_format = torch.contiguous_format);  expand_79 = None
    view_321: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_141, [128, 196, 32]);  clone_141 = None
    bmm_39: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_320, view_321)
    view_322: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_39, [8, 16, 1, 196, 32]);  bmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_150: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_322, [0, 2, 3, 4, 1]);  view_322 = None
    clone_142: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
    view_323: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_142, [8, 1, 196, 512]);  clone_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_324: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_323, [1568, 512]);  view_323 = None
    permute_151: "f32[512, 512]" = torch.ops.aten.permute.default(primals_266, [1, 0]);  primals_266 = None
    
    # No stacktrace found for following nodes
    mm_default_13: "f32[1568, 512]" = torch.ops.aten.mm.default(view_324, permute_151)
    add_tensor_13: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_13, primals_267);  mm_default_13 = primals_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_325: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_13, [8, 1, 196, 512]);  add_tensor_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_36: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.5869565308094025)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_56: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_36, 0.5869565308094025)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_215: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_325, div_56);  view_325 = div_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_142: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_139, mul_215);  add_139 = mul_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_41 = torch.ops.aten.var_mean.correction(add_142, [3], correction = 0, keepdim = True)
    getitem_146: "f32[8, 1, 196, 1]" = var_mean_41[0]
    getitem_147: "f32[8, 1, 196, 1]" = var_mean_41[1];  var_mean_41 = None
    add_143: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_146, 1e-06);  getitem_146 = None
    rsqrt_41: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
    sub_61: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_142, getitem_147);  getitem_147 = None
    mul_216: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_41);  sub_61 = None
    mul_217: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_216, primals_86)
    add_144: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_217, primals_87);  mul_217 = primals_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_326: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_144, [1568, 512]);  add_144 = None
    permute_152: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_268, [1, 0]);  primals_268 = None
    addmm_78: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_269, view_326, permute_152);  primals_269 = None
    view_327: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(addmm_78, [8, 1, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_218: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_327, 0.5)
    mul_219: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_327, 0.7071067811865476);  view_327 = None
    erf_19: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_219);  mul_219 = None
    add_145: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_220: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_218, add_145);  mul_218 = add_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_328: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_220, [1568, 2048]);  mul_220 = None
    permute_153: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_270, [1, 0]);  primals_270 = None
    
    # No stacktrace found for following nodes
    mm_default_12: "f32[1568, 512]" = torch.ops.aten.mm.default(view_328, permute_153)
    add_tensor_12: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_12, primals_271);  mm_default_12 = primals_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_329: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_12, [8, 1, 196, 512]);  add_tensor_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_37: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.5869565308094025)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_57: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_37, 0.5869565308094025)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_221: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_329, div_57);  view_329 = div_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_146: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_142, mul_221);  add_142 = mul_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_42 = torch.ops.aten.var_mean.correction(add_146, [3], correction = 0, keepdim = True)
    getitem_148: "f32[8, 1, 196, 1]" = var_mean_42[0]
    getitem_149: "f32[8, 1, 196, 1]" = var_mean_42[1];  var_mean_42 = None
    add_147: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_148, 1e-06);  getitem_148 = None
    rsqrt_42: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
    sub_62: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_146, getitem_149);  getitem_149 = None
    mul_222: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_42);  sub_62 = None
    mul_223: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_222, primals_88)
    add_148: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_223, primals_89);  mul_223 = primals_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_330: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_148, [1568, 512]);  add_148 = None
    permute_154: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_272, [1, 0]);  primals_272 = None
    
    # No stacktrace found for following nodes
    mm_default_11: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_330, permute_154)
    add_tensor_11: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_11, primals_273);  mm_default_11 = primals_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_331: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_11, [8, 1, 196, 1536]);  add_tensor_11 = None
    view_332: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_331, [8, 1, 196, 3, 16, 32]);  view_331 = None
    permute_155: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_332, [3, 0, 4, 1, 2, 5]);  view_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_20 = torch.ops.aten.unbind.int(permute_155);  permute_155 = None
    getitem_150: "f32[8, 16, 1, 196, 32]" = unbind_20[0]
    getitem_151: "f32[8, 16, 1, 196, 32]" = unbind_20[1]
    getitem_152: "f32[8, 16, 1, 196, 32]" = unbind_20[2];  unbind_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_224: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_150, 0.42044820762685725);  getitem_150 = None
    permute_156: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_151, [0, 1, 2, 4, 3]);  getitem_151 = None
    mul_225: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_156, 0.42044820762685725);  permute_156 = None
    expand_80: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_224, [8, 16, 1, 196, 32]);  mul_224 = None
    clone_146: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
    view_333: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_146, [128, 196, 32]);  clone_146 = None
    expand_81: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_225, [8, 16, 1, 32, 196]);  mul_225 = None
    clone_147: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
    view_334: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_147, [128, 32, 196]);  clone_147 = None
    bmm_40: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_333, view_334)
    view_335: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_40, [8, 16, 1, 196, 196]);  bmm_40 = None
    amax_20: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_335, [-1], True)
    sub_63: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_335, amax_20);  view_335 = amax_20 = None
    exp_20: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_63);  sub_63 = None
    sum_21: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
    div_58: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
    alias_20: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_58)
    expand_82: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_58, [8, 16, 1, 196, 196]);  div_58 = None
    view_336: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_82, [128, 196, 196]);  expand_82 = None
    expand_83: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_152, [8, 16, 1, 196, 32]);  getitem_152 = None
    clone_148: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_83, memory_format = torch.contiguous_format);  expand_83 = None
    view_337: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_148, [128, 196, 32]);  clone_148 = None
    bmm_41: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_336, view_337)
    view_338: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_41, [8, 16, 1, 196, 32]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_157: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_338, [0, 2, 3, 4, 1]);  view_338 = None
    clone_149: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_157, memory_format = torch.contiguous_format);  permute_157 = None
    view_339: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_149, [8, 1, 196, 512]);  clone_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_340: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_339, [1568, 512]);  view_339 = None
    permute_158: "f32[512, 512]" = torch.ops.aten.permute.default(primals_274, [1, 0]);  primals_274 = None
    
    # No stacktrace found for following nodes
    mm_default_10: "f32[1568, 512]" = torch.ops.aten.mm.default(view_340, permute_158)
    add_tensor_10: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_10, primals_275);  mm_default_10 = primals_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_341: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_10, [8, 1, 196, 512]);  add_tensor_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_38: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.5652174055576324)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_59: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_38, 0.5652174055576324)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_226: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_341, div_59);  view_341 = div_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_149: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_146, mul_226);  add_146 = mul_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_43 = torch.ops.aten.var_mean.correction(add_149, [3], correction = 0, keepdim = True)
    getitem_153: "f32[8, 1, 196, 1]" = var_mean_43[0]
    getitem_154: "f32[8, 1, 196, 1]" = var_mean_43[1];  var_mean_43 = None
    add_150: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_153, 1e-06);  getitem_153 = None
    rsqrt_43: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
    sub_64: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_149, getitem_154);  getitem_154 = None
    mul_227: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_43);  sub_64 = None
    mul_228: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_227, primals_90)
    add_151: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_228, primals_91);  mul_228 = primals_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_342: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_151, [1568, 512]);  add_151 = None
    permute_159: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_276, [1, 0]);  primals_276 = None
    addmm_82: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_277, view_342, permute_159);  primals_277 = None
    view_343: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(addmm_82, [8, 1, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_229: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_343, 0.5)
    mul_230: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_343, 0.7071067811865476);  view_343 = None
    erf_20: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_230);  mul_230 = None
    add_152: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_231: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_229, add_152);  mul_229 = add_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_344: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_231, [1568, 2048]);  mul_231 = None
    permute_160: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_278, [1, 0]);  primals_278 = None
    
    # No stacktrace found for following nodes
    mm_default_9: "f32[1568, 512]" = torch.ops.aten.mm.default(view_344, permute_160)
    add_tensor_9: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_9, primals_279);  mm_default_9 = primals_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_345: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_9, [8, 1, 196, 512]);  add_tensor_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_39: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.5652174055576324)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_60: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_39, 0.5652174055576324)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_232: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_345, div_60);  view_345 = div_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_153: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_149, mul_232);  add_149 = mul_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_44 = torch.ops.aten.var_mean.correction(add_153, [3], correction = 0, keepdim = True)
    getitem_155: "f32[8, 1, 196, 1]" = var_mean_44[0]
    getitem_156: "f32[8, 1, 196, 1]" = var_mean_44[1];  var_mean_44 = None
    add_154: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_155, 1e-06);  getitem_155 = None
    rsqrt_44: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
    sub_65: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_153, getitem_156);  getitem_156 = None
    mul_233: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_44);  sub_65 = None
    mul_234: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_233, primals_92)
    add_155: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_234, primals_93);  mul_234 = primals_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_346: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_155, [1568, 512]);  add_155 = None
    permute_161: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_280, [1, 0]);  primals_280 = None
    
    # No stacktrace found for following nodes
    mm_default_8: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_346, permute_161)
    add_tensor_8: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_8, primals_281);  mm_default_8 = primals_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_347: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_8, [8, 1, 196, 1536]);  add_tensor_8 = None
    view_348: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_347, [8, 1, 196, 3, 16, 32]);  view_347 = None
    permute_162: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_348, [3, 0, 4, 1, 2, 5]);  view_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_21 = torch.ops.aten.unbind.int(permute_162);  permute_162 = None
    getitem_157: "f32[8, 16, 1, 196, 32]" = unbind_21[0]
    getitem_158: "f32[8, 16, 1, 196, 32]" = unbind_21[1]
    getitem_159: "f32[8, 16, 1, 196, 32]" = unbind_21[2];  unbind_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_235: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_157, 0.42044820762685725);  getitem_157 = None
    permute_163: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_158, [0, 1, 2, 4, 3]);  getitem_158 = None
    mul_236: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_163, 0.42044820762685725);  permute_163 = None
    expand_84: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_235, [8, 16, 1, 196, 32]);  mul_235 = None
    clone_153: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_84, memory_format = torch.contiguous_format);  expand_84 = None
    view_349: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_153, [128, 196, 32]);  clone_153 = None
    expand_85: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_236, [8, 16, 1, 32, 196]);  mul_236 = None
    clone_154: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_85, memory_format = torch.contiguous_format);  expand_85 = None
    view_350: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_154, [128, 32, 196]);  clone_154 = None
    bmm_42: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_349, view_350)
    view_351: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_42, [8, 16, 1, 196, 196]);  bmm_42 = None
    amax_21: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_351, [-1], True)
    sub_66: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_351, amax_21);  view_351 = amax_21 = None
    exp_21: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_66);  sub_66 = None
    sum_22: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
    div_61: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
    alias_21: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_61)
    expand_86: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_61, [8, 16, 1, 196, 196]);  div_61 = None
    view_352: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_86, [128, 196, 196]);  expand_86 = None
    expand_87: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_159, [8, 16, 1, 196, 32]);  getitem_159 = None
    clone_155: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_87, memory_format = torch.contiguous_format);  expand_87 = None
    view_353: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_155, [128, 196, 32]);  clone_155 = None
    bmm_43: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_352, view_353)
    view_354: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_43, [8, 16, 1, 196, 32]);  bmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_164: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_354, [0, 2, 3, 4, 1]);  view_354 = None
    clone_156: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_164, memory_format = torch.contiguous_format);  permute_164 = None
    view_355: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_156, [8, 1, 196, 512]);  clone_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_356: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_355, [1568, 512]);  view_355 = None
    permute_165: "f32[512, 512]" = torch.ops.aten.permute.default(primals_282, [1, 0]);  primals_282 = None
    
    # No stacktrace found for following nodes
    mm_default_7: "f32[1568, 512]" = torch.ops.aten.mm.default(view_356, permute_165)
    add_tensor_7: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_7, primals_283);  mm_default_7 = primals_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_357: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_7, [8, 1, 196, 512]);  add_tensor_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_40: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.54347825050354)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_62: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_40, 0.54347825050354)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_237: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_357, div_62);  view_357 = div_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_156: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_153, mul_237);  add_153 = mul_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_45 = torch.ops.aten.var_mean.correction(add_156, [3], correction = 0, keepdim = True)
    getitem_160: "f32[8, 1, 196, 1]" = var_mean_45[0]
    getitem_161: "f32[8, 1, 196, 1]" = var_mean_45[1];  var_mean_45 = None
    add_157: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_160, 1e-06);  getitem_160 = None
    rsqrt_45: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
    sub_67: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_156, getitem_161);  getitem_161 = None
    mul_238: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_45);  sub_67 = None
    mul_239: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_238, primals_94)
    add_158: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_239, primals_95);  mul_239 = primals_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_358: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_158, [1568, 512]);  add_158 = None
    permute_166: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_284, [1, 0]);  primals_284 = None
    addmm_86: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_285, view_358, permute_166);  primals_285 = None
    view_359: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(addmm_86, [8, 1, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_240: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_359, 0.5)
    mul_241: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_359, 0.7071067811865476);  view_359 = None
    erf_21: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_241);  mul_241 = None
    add_159: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_242: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_240, add_159);  mul_240 = add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_360: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_242, [1568, 2048]);  mul_242 = None
    permute_167: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_286, [1, 0]);  primals_286 = None
    
    # No stacktrace found for following nodes
    mm_default_6: "f32[1568, 512]" = torch.ops.aten.mm.default(view_360, permute_167)
    add_tensor_6: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_6, primals_287);  mm_default_6 = primals_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_361: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_6, [8, 1, 196, 512]);  add_tensor_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_41: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.54347825050354)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_63: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_41, 0.54347825050354)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_243: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_361, div_63);  view_361 = div_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_160: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_156, mul_243);  add_156 = mul_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_46 = torch.ops.aten.var_mean.correction(add_160, [3], correction = 0, keepdim = True)
    getitem_162: "f32[8, 1, 196, 1]" = var_mean_46[0]
    getitem_163: "f32[8, 1, 196, 1]" = var_mean_46[1];  var_mean_46 = None
    add_161: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_162, 1e-06);  getitem_162 = None
    rsqrt_46: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    sub_68: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_160, getitem_163);  getitem_163 = None
    mul_244: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_46);  sub_68 = None
    mul_245: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_244, primals_96)
    add_162: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_245, primals_97);  mul_245 = primals_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_362: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_162, [1568, 512]);  add_162 = None
    permute_168: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_288, [1, 0]);  primals_288 = None
    
    # No stacktrace found for following nodes
    mm_default_5: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_362, permute_168)
    add_tensor_5: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_5, primals_289);  mm_default_5 = primals_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_363: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_5, [8, 1, 196, 1536]);  add_tensor_5 = None
    view_364: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_363, [8, 1, 196, 3, 16, 32]);  view_363 = None
    permute_169: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_364, [3, 0, 4, 1, 2, 5]);  view_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_22 = torch.ops.aten.unbind.int(permute_169);  permute_169 = None
    getitem_164: "f32[8, 16, 1, 196, 32]" = unbind_22[0]
    getitem_165: "f32[8, 16, 1, 196, 32]" = unbind_22[1]
    getitem_166: "f32[8, 16, 1, 196, 32]" = unbind_22[2];  unbind_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_246: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_164, 0.42044820762685725);  getitem_164 = None
    permute_170: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_165, [0, 1, 2, 4, 3]);  getitem_165 = None
    mul_247: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_170, 0.42044820762685725);  permute_170 = None
    expand_88: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_246, [8, 16, 1, 196, 32]);  mul_246 = None
    clone_160: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
    view_365: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_160, [128, 196, 32]);  clone_160 = None
    expand_89: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_247, [8, 16, 1, 32, 196]);  mul_247 = None
    clone_161: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
    view_366: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_161, [128, 32, 196]);  clone_161 = None
    bmm_44: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_365, view_366)
    view_367: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_44, [8, 16, 1, 196, 196]);  bmm_44 = None
    amax_22: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_367, [-1], True)
    sub_69: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_367, amax_22);  view_367 = amax_22 = None
    exp_22: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_69);  sub_69 = None
    sum_23: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
    div_64: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
    alias_22: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_64)
    expand_90: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_64, [8, 16, 1, 196, 196]);  div_64 = None
    view_368: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_90, [128, 196, 196]);  expand_90 = None
    expand_91: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_166, [8, 16, 1, 196, 32]);  getitem_166 = None
    clone_162: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_91, memory_format = torch.contiguous_format);  expand_91 = None
    view_369: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_162, [128, 196, 32]);  clone_162 = None
    bmm_45: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_368, view_369)
    view_370: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_45, [8, 16, 1, 196, 32]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_171: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_370, [0, 2, 3, 4, 1]);  view_370 = None
    clone_163: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_171, memory_format = torch.contiguous_format);  permute_171 = None
    view_371: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_163, [8, 1, 196, 512]);  clone_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_372: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_371, [1568, 512]);  view_371 = None
    permute_172: "f32[512, 512]" = torch.ops.aten.permute.default(primals_290, [1, 0]);  primals_290 = None
    
    # No stacktrace found for following nodes
    mm_default_4: "f32[1568, 512]" = torch.ops.aten.mm.default(view_372, permute_172)
    add_tensor_4: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_4, primals_291);  mm_default_4 = primals_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_373: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_4, [8, 1, 196, 512]);  add_tensor_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_42: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.52173912525177)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_65: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_42, 0.52173912525177)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_248: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_373, div_65);  view_373 = div_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_163: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_160, mul_248);  add_160 = mul_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_47 = torch.ops.aten.var_mean.correction(add_163, [3], correction = 0, keepdim = True)
    getitem_167: "f32[8, 1, 196, 1]" = var_mean_47[0]
    getitem_168: "f32[8, 1, 196, 1]" = var_mean_47[1];  var_mean_47 = None
    add_164: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_167, 1e-06);  getitem_167 = None
    rsqrt_47: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
    sub_70: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_163, getitem_168);  getitem_168 = None
    mul_249: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_47);  sub_70 = None
    mul_250: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_249, primals_98)
    add_165: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_250, primals_99);  mul_250 = primals_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_374: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_165, [1568, 512]);  add_165 = None
    permute_173: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_292, [1, 0]);  primals_292 = None
    addmm_90: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_293, view_374, permute_173);  primals_293 = None
    view_375: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(addmm_90, [8, 1, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_251: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_375, 0.5)
    mul_252: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_375, 0.7071067811865476);  view_375 = None
    erf_22: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_252);  mul_252 = None
    add_166: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_253: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_251, add_166);  mul_251 = add_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_376: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_253, [1568, 2048]);  mul_253 = None
    permute_174: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_294, [1, 0]);  primals_294 = None
    
    # No stacktrace found for following nodes
    mm_default_3: "f32[1568, 512]" = torch.ops.aten.mm.default(view_376, permute_174)
    add_tensor_3: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_3, primals_295);  mm_default_3 = primals_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_377: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_3, [8, 1, 196, 512]);  add_tensor_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_43: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.52173912525177)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_66: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_43, 0.52173912525177)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_254: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_377, div_66);  view_377 = div_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_167: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_163, mul_254);  add_163 = mul_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_48 = torch.ops.aten.var_mean.correction(add_167, [3], correction = 0, keepdim = True)
    getitem_169: "f32[8, 1, 196, 1]" = var_mean_48[0]
    getitem_170: "f32[8, 1, 196, 1]" = var_mean_48[1];  var_mean_48 = None
    add_168: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_169, 1e-06);  getitem_169 = None
    rsqrt_48: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
    sub_71: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_167, getitem_170);  getitem_170 = None
    mul_255: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_48);  sub_71 = None
    mul_256: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_255, primals_100)
    add_169: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_256, primals_101);  mul_256 = primals_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_378: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_169, [1568, 512]);  add_169 = None
    permute_175: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_296, [1, 0]);  primals_296 = None
    
    # No stacktrace found for following nodes
    mm_default_2: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_378, permute_175)
    add_tensor_2: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_2, primals_297);  mm_default_2 = primals_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_379: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_2, [8, 1, 196, 1536]);  add_tensor_2 = None
    view_380: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_379, [8, 1, 196, 3, 16, 32]);  view_379 = None
    permute_176: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_380, [3, 0, 4, 1, 2, 5]);  view_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_23 = torch.ops.aten.unbind.int(permute_176);  permute_176 = None
    getitem_171: "f32[8, 16, 1, 196, 32]" = unbind_23[0]
    getitem_172: "f32[8, 16, 1, 196, 32]" = unbind_23[1]
    getitem_173: "f32[8, 16, 1, 196, 32]" = unbind_23[2];  unbind_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_257: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_171, 0.42044820762685725);  getitem_171 = None
    permute_177: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_172, [0, 1, 2, 4, 3]);  getitem_172 = None
    mul_258: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_177, 0.42044820762685725);  permute_177 = None
    expand_92: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_257, [8, 16, 1, 196, 32]);  mul_257 = None
    clone_167: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
    view_381: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_167, [128, 196, 32]);  clone_167 = None
    expand_93: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_258, [8, 16, 1, 32, 196]);  mul_258 = None
    clone_168: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_93, memory_format = torch.contiguous_format);  expand_93 = None
    view_382: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_168, [128, 32, 196]);  clone_168 = None
    bmm_46: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_381, view_382)
    view_383: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_46, [8, 16, 1, 196, 196]);  bmm_46 = None
    amax_23: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_383, [-1], True)
    sub_72: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_383, amax_23);  view_383 = amax_23 = None
    exp_23: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_72);  sub_72 = None
    sum_24: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
    div_67: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
    alias_23: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_67)
    expand_94: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_67, [8, 16, 1, 196, 196]);  div_67 = None
    view_384: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_94, [128, 196, 196]);  expand_94 = None
    expand_95: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_173, [8, 16, 1, 196, 32]);  getitem_173 = None
    clone_169: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_95, memory_format = torch.contiguous_format);  expand_95 = None
    view_385: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_169, [128, 196, 32]);  clone_169 = None
    bmm_47: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_384, view_385)
    view_386: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_47, [8, 16, 1, 196, 32]);  bmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_178: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_386, [0, 2, 3, 4, 1]);  view_386 = None
    clone_170: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_178, memory_format = torch.contiguous_format);  permute_178 = None
    view_387: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_170, [8, 1, 196, 512]);  clone_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_388: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_387, [1568, 512]);  view_387 = None
    permute_179: "f32[512, 512]" = torch.ops.aten.permute.default(primals_298, [1, 0]);  primals_298 = None
    
    # No stacktrace found for following nodes
    mm_default_1: "f32[1568, 512]" = torch.ops.aten.mm.default(view_388, permute_179)
    add_tensor_1: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_1, primals_299);  mm_default_1 = primals_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_389: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_1, [8, 1, 196, 512]);  add_tensor_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_44: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_68: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_44, 0.5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_259: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_389, div_68);  view_389 = div_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_170: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_167, mul_259);  add_167 = mul_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_49 = torch.ops.aten.var_mean.correction(add_170, [3], correction = 0, keepdim = True)
    getitem_174: "f32[8, 1, 196, 1]" = var_mean_49[0]
    getitem_175: "f32[8, 1, 196, 1]" = var_mean_49[1];  var_mean_49 = None
    add_171: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_174, 1e-06);  getitem_174 = None
    rsqrt_49: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
    sub_73: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_170, getitem_175);  getitem_175 = None
    mul_260: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_49);  sub_73 = None
    mul_261: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_260, primals_102)
    add_172: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_261, primals_103);  mul_261 = primals_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_390: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_172, [1568, 512]);  add_172 = None
    permute_180: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_300, [1, 0]);  primals_300 = None
    addmm_94: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_301, view_390, permute_180);  primals_301 = None
    view_391: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(addmm_94, [8, 1, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_262: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_391, 0.5)
    mul_263: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_391, 0.7071067811865476);  view_391 = None
    erf_23: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_263);  mul_263 = None
    add_173: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_264: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_262, add_173);  mul_262 = add_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_392: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_264, [1568, 2048]);  mul_264 = None
    permute_181: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_302, [1, 0]);  primals_302 = None
    
    # No stacktrace found for following nodes
    mm_default: "f32[1568, 512]" = torch.ops.aten.mm.default(view_392, permute_181)
    add_tensor: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default, primals_303);  mm_default = primals_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_393: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor, [8, 1, 196, 512]);  add_tensor = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    bernoulli_45: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.5);  empty = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_69: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_45, 0.5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_265: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_393, div_69);  view_393 = div_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_174: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_170, mul_265);  add_170 = mul_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:174, code: x = x.reshape(B, grid_size, grid_size, block_size, block_size, C)
    view_394: "f32[8, 1, 1, 14, 14, 512]" = torch.ops.aten.reshape.default(add_174, [8, 1, 1, 14, 14, 512]);  add_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:175, code: x = x.transpose(2, 3).reshape(B, height, width, C)
    permute_182: "f32[8, 1, 14, 1, 14, 512]" = torch.ops.aten.permute.default(view_394, [0, 1, 3, 2, 4, 5]);  view_394 = None
    view_395: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(permute_182, [8, 14, 14, 512]);  permute_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:242, code: return x.permute(0, 3, 1, 2)  # (B, C, H', W')
    permute_183: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(view_395, [0, 3, 1, 2]);  view_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:427, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_184: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(permute_183, [0, 2, 3, 1]);  permute_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_50 = torch.ops.aten.var_mean.correction(permute_184, [3], correction = 0, keepdim = True)
    getitem_176: "f32[8, 14, 14, 1]" = var_mean_50[0]
    getitem_177: "f32[8, 14, 14, 1]" = var_mean_50[1];  var_mean_50 = None
    add_175: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-06);  getitem_176 = None
    rsqrt_50: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
    sub_74: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(permute_184, getitem_177);  permute_184 = getitem_177 = None
    mul_266: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_50);  sub_74 = None
    mul_267: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_266, primals_104)
    add_176: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_267, primals_105);  mul_267 = primals_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:427, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_185: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(add_176, [0, 3, 1, 2]);  add_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean: "f32[8, 512, 1, 1]" = torch.ops.aten.mean.dim(permute_185, [-1, -2], True);  permute_185 = None
    as_strided: "f32[8, 512, 1, 1]" = torch.ops.aten.as_strided.default(mean, [8, 512, 1, 1], [512, 1, 512, 512]);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_396: "f32[8, 512]" = torch.ops.aten.reshape.default(as_strided, [8, 512]);  as_strided = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:433, code: return x if pre_logits else self.head(x)
    permute_186: "f32[512, 1000]" = torch.ops.aten.permute.default(primals_304, [1, 0]);  primals_304 = None
    addmm_96: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_305, view_396, permute_186);  primals_305 = None
    permute_187: "f32[1000, 512]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_71: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_50, 512);  rsqrt_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_195: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_199: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_180, [1, 0]);  permute_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_72: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_49, 512);  rsqrt_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    permute_203: "f32[512, 512]" = torch.ops.aten.permute.default(permute_179, [1, 0]);  permute_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    permute_208: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_384, [0, 2, 1]);  view_384 = None
    permute_209: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_385, [0, 2, 1]);  view_385 = None
    alias_24: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_23);  alias_23 = None
    permute_210: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_381, [0, 2, 1]);  view_381 = None
    permute_211: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_382, [0, 2, 1]);  view_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_214: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_175, [1, 0]);  permute_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_73: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_48, 512);  rsqrt_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_218: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_222: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_74: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_47, 512);  rsqrt_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    permute_226: "f32[512, 512]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    permute_231: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_368, [0, 2, 1]);  view_368 = None
    permute_232: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_369, [0, 2, 1]);  view_369 = None
    alias_25: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_22);  alias_22 = None
    permute_233: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_365, [0, 2, 1]);  view_365 = None
    permute_234: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_366, [0, 2, 1]);  view_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_237: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_168, [1, 0]);  permute_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_75: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_46, 512);  rsqrt_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_241: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_245: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_166, [1, 0]);  permute_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_76: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_45, 512);  rsqrt_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    permute_249: "f32[512, 512]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    permute_254: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_352, [0, 2, 1]);  view_352 = None
    permute_255: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_353, [0, 2, 1]);  view_353 = None
    alias_26: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_21);  alias_21 = None
    permute_256: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_349, [0, 2, 1]);  view_349 = None
    permute_257: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_350, [0, 2, 1]);  view_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_260: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_77: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 512);  rsqrt_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_264: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_160, [1, 0]);  permute_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_268: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_159, [1, 0]);  permute_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_78: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 512);  rsqrt_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    permute_272: "f32[512, 512]" = torch.ops.aten.permute.default(permute_158, [1, 0]);  permute_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    permute_277: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_336, [0, 2, 1]);  view_336 = None
    permute_278: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_337, [0, 2, 1]);  view_337 = None
    alias_27: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_20);  alias_20 = None
    permute_279: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_333, [0, 2, 1]);  view_333 = None
    permute_280: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_334, [0, 2, 1]);  view_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_283: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_79: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_42, 512);  rsqrt_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_287: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_291: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_80: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_41, 512);  rsqrt_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    permute_295: "f32[512, 512]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    permute_300: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_320, [0, 2, 1]);  view_320 = None
    permute_301: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_321, [0, 2, 1]);  view_321 = None
    alias_28: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_19);  alias_19 = None
    permute_302: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_317, [0, 2, 1]);  view_317 = None
    permute_303: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_318, [0, 2, 1]);  view_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_306: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_147, [1, 0]);  permute_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_81: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 512);  rsqrt_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_310: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_146, [1, 0]);  permute_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_314: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_82: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 512);  rsqrt_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    permute_318: "f32[512, 512]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    permute_323: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_304, [0, 2, 1]);  view_304 = None
    permute_324: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_305, [0, 2, 1]);  view_305 = None
    alias_29: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_18);  alias_18 = None
    permute_325: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_301, [0, 2, 1]);  view_301 = None
    permute_326: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_302, [0, 2, 1]);  view_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_329: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_83: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 512);  rsqrt_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_333: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_139, [1, 0]);  permute_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_337: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_138, [1, 0]);  permute_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_84: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 512);  rsqrt_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    permute_341: "f32[512, 512]" = torch.ops.aten.permute.default(permute_137, [1, 0]);  permute_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    permute_346: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_288, [0, 2, 1]);  view_288 = None
    permute_347: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_289, [0, 2, 1]);  view_289 = None
    alias_30: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    permute_348: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_285, [0, 2, 1]);  view_285 = None
    permute_349: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_286, [0, 2, 1]);  view_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_352: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_85: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 512);  rsqrt_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_356: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_360: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_86: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 512);  rsqrt_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    permute_364: "f32[512, 512]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    permute_369: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_272, [0, 2, 1]);  view_272 = None
    permute_370: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_273, [0, 2, 1]);  view_273 = None
    alias_31: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_16);  alias_16 = None
    permute_371: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_269, [0, 2, 1]);  view_269 = None
    permute_372: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_270, [0, 2, 1]);  view_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_375: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_126, [1, 0]);  permute_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_87: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 512);  rsqrt_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_379: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_383: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_88: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 512);  rsqrt_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    permute_387: "f32[512, 512]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    permute_392: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_256, [0, 2, 1]);  view_256 = None
    permute_393: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_257, [0, 2, 1]);  view_257 = None
    alias_32: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    permute_394: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_253, [0, 2, 1]);  view_253 = None
    permute_395: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_398: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_89: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 512);  rsqrt_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_402: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_406: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_117, [1, 0]);  permute_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_90: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 512);  rsqrt_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    permute_410: "f32[512, 512]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    permute_415: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_240, [0, 2, 1]);  view_240 = None
    permute_416: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_241, [0, 2, 1]);  view_241 = None
    alias_33: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    permute_417: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_237, [0, 2, 1]);  view_237 = None
    permute_418: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_238, [0, 2, 1]);  view_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_421: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_91: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 512);  rsqrt_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_425: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_429: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_92: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 512);  rsqrt_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    permute_433: "f32[512, 512]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    permute_438: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_224, [0, 2, 1]);  view_224 = None
    permute_439: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_225, [0, 2, 1]);  view_225 = None
    alias_34: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    permute_440: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_221, [0, 2, 1]);  view_221 = None
    permute_441: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_222, [0, 2, 1]);  view_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_444: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_105, [1, 0]);  permute_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_93: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 512);  rsqrt_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_448: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_104, [1, 0]);  permute_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_452: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_94: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 512);  rsqrt_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    permute_456: "f32[512, 512]" = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    permute_461: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_208, [0, 2, 1]);  view_208 = None
    permute_462: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_209, [0, 2, 1]);  view_209 = None
    alias_35: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    permute_463: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_205, [0, 2, 1]);  view_205 = None
    permute_464: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_206, [0, 2, 1]);  view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_467: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_95: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 512);  rsqrt_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_471: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_475: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_96: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 512);  rsqrt_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    permute_479: "f32[512, 512]" = torch.ops.aten.permute.default(permute_95, [1, 0]);  permute_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    permute_484: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_192, [0, 2, 1]);  view_192 = None
    permute_485: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_193, [0, 2, 1]);  view_193 = None
    alias_36: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    permute_486: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_189, [0, 2, 1]);  view_189 = None
    permute_487: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_190, [0, 2, 1]);  view_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_490: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_91, [1, 0]);  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_97: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 512);  rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_494: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_498: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_98: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 512);  rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    permute_502: "f32[512, 512]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    permute_507: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_176, [0, 2, 1]);  view_176 = None
    permute_508: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_177, [0, 2, 1]);  view_177 = None
    alias_37: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    permute_509: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_173, [0, 2, 1]);  view_173 = None
    permute_510: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_174, [0, 2, 1]);  view_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_513: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_84, [1, 0]);  permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_99: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 512);  rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_517: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_521: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_82, [1, 0]);  permute_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_100: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 512);  rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    permute_525: "f32[512, 512]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    permute_530: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_160, [0, 2, 1]);  view_160 = None
    permute_531: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_161, [0, 2, 1]);  view_161 = None
    alias_38: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    permute_532: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_157, [0, 2, 1]);  view_157 = None
    permute_533: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_158, [0, 2, 1]);  view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_536: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_101: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 512);  rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_540: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_544: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_102: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 512);  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    permute_548: "f32[512, 512]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    permute_553: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
    permute_554: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None
    alias_39: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    permute_555: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_141, [0, 2, 1]);  view_141 = None
    permute_556: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_142, [0, 2, 1]);  view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_559: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_103: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 512);  rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_563: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_567: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_104: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 512);  rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    permute_571: "f32[512, 512]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    permute_576: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_128, [0, 2, 1]);  view_128 = None
    permute_577: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_129, [0, 2, 1]);  view_129 = None
    alias_40: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    permute_578: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_125, [0, 2, 1]);  view_125 = None
    permute_579: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_126, [0, 2, 1]);  view_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_582: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_105: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 512);  rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_586: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_62, [1, 0]);  permute_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_590: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_106: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 512);  rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    permute_594: "f32[512, 512]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    permute_599: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_112, [0, 2, 1]);  view_112 = None
    permute_600: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_113, [0, 2, 1]);  view_113 = None
    alias_41: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    permute_601: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_109, [0, 2, 1]);  view_109 = None
    permute_602: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_110, [0, 2, 1]);  view_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_605: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_107: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 512);  rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_609: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_613: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_108: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 512);  rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    permute_617: "f32[512, 512]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    permute_622: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_96, [0, 2, 1]);  view_96 = None
    permute_623: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
    alias_42: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    permute_624: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_93, [0, 2, 1]);  view_93 = None
    permute_625: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_94, [0, 2, 1]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_628: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_109: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 512);  rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_632: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_636: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_110: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 512);  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    permute_640: "f32[512, 512]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    permute_645: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_80, [0, 2, 1]);  view_80 = None
    permute_646: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_81, [0, 2, 1]);  view_81 = None
    alias_43: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    permute_647: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_77, [0, 2, 1]);  view_77 = None
    permute_648: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_651: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_111: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 512);  rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_112: "f32[8, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 512);  rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_661: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_665: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_113: "f32[8, 4, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 256);  rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    permute_669: "f32[256, 256]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    permute_674: "f32[256, 196, 196]" = torch.ops.aten.permute.default(view_60, [0, 2, 1]);  view_60 = None
    permute_675: "f32[256, 32, 196]" = torch.ops.aten.permute.default(view_61, [0, 2, 1]);  view_61 = None
    alias_44: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    permute_676: "f32[256, 32, 196]" = torch.ops.aten.permute.default(view_57, [0, 2, 1]);  view_57 = None
    permute_677: "f32[256, 196, 32]" = torch.ops.aten.permute.default(view_58, [0, 2, 1]);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_680: "f32[768, 256]" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_114: "f32[8, 4, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 256);  rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_684: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_28, [1, 0]);  permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_688: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_115: "f32[8, 4, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 256);  rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    permute_692: "f32[256, 256]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    permute_697: "f32[256, 196, 196]" = torch.ops.aten.permute.default(view_44, [0, 2, 1]);  view_44 = None
    permute_698: "f32[256, 32, 196]" = torch.ops.aten.permute.default(view_45, [0, 2, 1]);  view_45 = None
    alias_45: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    permute_699: "f32[256, 32, 196]" = torch.ops.aten.permute.default(view_41, [0, 2, 1]);  view_41 = None
    permute_700: "f32[256, 196, 32]" = torch.ops.aten.permute.default(view_42, [0, 2, 1]);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_703: "f32[768, 256]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_116: "f32[8, 4, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 256);  rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_117: "f32[8, 56, 56, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 256);  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_713: "f32[128, 512]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_717: "f32[512, 128]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_118: "f32[8, 16, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 128);  rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    permute_721: "f32[128, 128]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    permute_726: "f32[512, 196, 196]" = torch.ops.aten.permute.default(view_24, [0, 2, 1]);  view_24 = None
    permute_727: "f32[512, 32, 196]" = torch.ops.aten.permute.default(view_25, [0, 2, 1]);  view_25 = None
    alias_46: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    permute_728: "f32[512, 32, 196]" = torch.ops.aten.permute.default(view_21, [0, 2, 1]);  view_21 = None
    permute_729: "f32[512, 196, 32]" = torch.ops.aten.permute.default(view_22, [0, 2, 1]);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_732: "f32[384, 128]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_119: "f32[8, 16, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 128);  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_736: "f32[128, 512]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_740: "f32[512, 128]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_120: "f32[8, 16, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 128);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    permute_744: "f32[128, 128]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    permute_749: "f32[512, 196, 196]" = torch.ops.aten.permute.default(view_8, [0, 2, 1]);  view_8 = None
    permute_750: "f32[512, 32, 196]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
    alias_47: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.alias.default(alias);  alias = None
    permute_751: "f32[512, 32, 196]" = torch.ops.aten.permute.default(view_5, [0, 2, 1]);  view_5 = None
    permute_752: "f32[512, 196, 32]" = torch.ops.aten.permute.default(view_6, [0, 2, 1]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_755: "f32[384, 128]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_121: "f32[8, 16, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt, 128);  rsqrt = None
    return [addmm_96, primals_2, primals_4, primals_6, primals_8, primals_10, primals_13, primals_15, primals_17, primals_19, primals_21, primals_24, primals_26, primals_28, primals_30, primals_32, primals_34, primals_36, primals_38, primals_40, primals_42, primals_44, primals_46, primals_48, primals_50, primals_52, primals_54, primals_56, primals_58, primals_60, primals_62, primals_64, primals_66, primals_68, primals_70, primals_72, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, primals_106, primals_124, primals_142, primals_306, mul, view_2, view_12, mul_4, view_14, addmm_2, view_16, mul_9, view_18, view_28, bernoulli, mul_14, view_30, addmm_6, view_32, bernoulli_1, permute_17, mul_20, constant_pad_nd, getitem_17, mul_22, view_38, view_48, bernoulli_2, mul_27, view_50, addmm_10, view_52, bernoulli_3, mul_33, view_54, view_64, bernoulli_4, mul_38, view_66, addmm_14, view_68, bernoulli_5, permute_37, mul_44, constant_pad_nd_1, getitem_35, mul_46, view_74, view_84, bernoulli_6, mul_51, view_86, addmm_18, view_88, bernoulli_7, mul_57, view_90, view_100, bernoulli_8, mul_62, view_102, addmm_22, view_104, bernoulli_9, mul_68, view_106, view_116, bernoulli_10, mul_73, view_118, addmm_26, view_120, bernoulli_11, mul_79, view_122, view_132, bernoulli_12, mul_84, view_134, addmm_30, view_136, bernoulli_13, mul_90, view_138, view_148, bernoulli_14, mul_95, view_150, addmm_34, view_152, bernoulli_15, mul_101, view_154, view_164, bernoulli_16, mul_106, view_166, addmm_38, view_168, bernoulli_17, mul_112, view_170, view_180, bernoulli_18, mul_117, view_182, addmm_42, view_184, bernoulli_19, mul_123, view_186, view_196, bernoulli_20, mul_128, view_198, addmm_46, view_200, bernoulli_21, mul_134, view_202, view_212, bernoulli_22, mul_139, view_214, addmm_50, view_216, bernoulli_23, mul_145, view_218, view_228, bernoulli_24, mul_150, view_230, addmm_54, view_232, bernoulli_25, mul_156, view_234, view_244, bernoulli_26, mul_161, view_246, addmm_58, view_248, bernoulli_27, mul_167, view_250, view_260, bernoulli_28, mul_172, view_262, addmm_62, view_264, bernoulli_29, mul_178, view_266, view_276, bernoulli_30, mul_183, view_278, addmm_66, view_280, bernoulli_31, mul_189, view_282, view_292, bernoulli_32, mul_194, view_294, addmm_70, view_296, bernoulli_33, mul_200, view_298, view_308, bernoulli_34, mul_205, view_310, addmm_74, view_312, bernoulli_35, mul_211, view_314, view_324, bernoulli_36, mul_216, view_326, addmm_78, view_328, bernoulli_37, mul_222, view_330, view_340, bernoulli_38, mul_227, view_342, addmm_82, view_344, bernoulli_39, mul_233, view_346, view_356, bernoulli_40, mul_238, view_358, addmm_86, view_360, bernoulli_41, mul_244, view_362, view_372, bernoulli_42, mul_249, view_374, addmm_90, view_376, bernoulli_43, mul_255, view_378, view_388, bernoulli_44, mul_260, view_390, addmm_94, view_392, bernoulli_45, mul_266, view_396, permute_187, div_71, permute_195, permute_199, div_72, permute_203, permute_208, permute_209, alias_24, permute_210, permute_211, permute_214, div_73, permute_218, permute_222, div_74, permute_226, permute_231, permute_232, alias_25, permute_233, permute_234, permute_237, div_75, permute_241, permute_245, div_76, permute_249, permute_254, permute_255, alias_26, permute_256, permute_257, permute_260, div_77, permute_264, permute_268, div_78, permute_272, permute_277, permute_278, alias_27, permute_279, permute_280, permute_283, div_79, permute_287, permute_291, div_80, permute_295, permute_300, permute_301, alias_28, permute_302, permute_303, permute_306, div_81, permute_310, permute_314, div_82, permute_318, permute_323, permute_324, alias_29, permute_325, permute_326, permute_329, div_83, permute_333, permute_337, div_84, permute_341, permute_346, permute_347, alias_30, permute_348, permute_349, permute_352, div_85, permute_356, permute_360, div_86, permute_364, permute_369, permute_370, alias_31, permute_371, permute_372, permute_375, div_87, permute_379, permute_383, div_88, permute_387, permute_392, permute_393, alias_32, permute_394, permute_395, permute_398, div_89, permute_402, permute_406, div_90, permute_410, permute_415, permute_416, alias_33, permute_417, permute_418, permute_421, div_91, permute_425, permute_429, div_92, permute_433, permute_438, permute_439, alias_34, permute_440, permute_441, permute_444, div_93, permute_448, permute_452, div_94, permute_456, permute_461, permute_462, alias_35, permute_463, permute_464, permute_467, div_95, permute_471, permute_475, div_96, permute_479, permute_484, permute_485, alias_36, permute_486, permute_487, permute_490, div_97, permute_494, permute_498, div_98, permute_502, permute_507, permute_508, alias_37, permute_509, permute_510, permute_513, div_99, permute_517, permute_521, div_100, permute_525, permute_530, permute_531, alias_38, permute_532, permute_533, permute_536, div_101, permute_540, permute_544, div_102, permute_548, permute_553, permute_554, alias_39, permute_555, permute_556, permute_559, div_103, permute_563, permute_567, div_104, permute_571, permute_576, permute_577, alias_40, permute_578, permute_579, permute_582, div_105, permute_586, permute_590, div_106, permute_594, permute_599, permute_600, alias_41, permute_601, permute_602, permute_605, div_107, permute_609, permute_613, div_108, permute_617, permute_622, permute_623, alias_42, permute_624, permute_625, permute_628, div_109, permute_632, permute_636, div_110, permute_640, permute_645, permute_646, alias_43, permute_647, permute_648, permute_651, div_111, div_112, permute_661, permute_665, div_113, permute_669, permute_674, permute_675, alias_44, permute_676, permute_677, permute_680, div_114, permute_684, permute_688, div_115, permute_692, permute_697, permute_698, alias_45, permute_699, permute_700, permute_703, div_116, div_117, permute_713, permute_717, div_118, permute_721, permute_726, permute_727, alias_46, permute_728, permute_729, permute_732, div_119, permute_736, permute_740, div_120, permute_744, permute_749, permute_750, alias_47, permute_751, permute_752, permute_755, div_121]
    