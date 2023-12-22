from __future__ import annotations



def forward(self, primals_1: "f32[1, 24, 4, 4]", primals_2: "f32[1, 1, 384]", primals_3: "f32[1, 197, 384]", primals_4: "f32[24, 3, 7, 7]", primals_5: "f32[24]", primals_6: "f32[384]", primals_7: "f32[384]", primals_8: "f32[384, 384]", primals_9: "f32[384]", primals_10: "f32[384]", primals_11: "f32[384]", primals_12: "f32[24]", primals_13: "f32[24]", primals_14: "f32[48, 24]", primals_15: "f32[24, 24]", primals_16: "f32[24, 24]", primals_17: "f32[24]", primals_18: "f32[24]", primals_19: "f32[24]", primals_20: "f32[96, 24]", primals_21: "f32[96]", primals_22: "f32[24, 96]", primals_23: "f32[24]", primals_24: "f32[24]", primals_25: "f32[24]", primals_26: "f32[384, 384]", primals_27: "f32[384]", primals_28: "f32[384]", primals_29: "f32[384]", primals_30: "f32[768, 384]", primals_31: "f32[384, 384]", primals_32: "f32[384, 384]", primals_33: "f32[384]", primals_34: "f32[384]", primals_35: "f32[384]", primals_36: "f32[1536, 384]", primals_37: "f32[1536]", primals_38: "f32[384, 1536]", primals_39: "f32[384]", primals_40: "f32[24]", primals_41: "f32[24]", primals_42: "f32[48, 24]", primals_43: "f32[24, 24]", primals_44: "f32[24, 24]", primals_45: "f32[24]", primals_46: "f32[24]", primals_47: "f32[24]", primals_48: "f32[96, 24]", primals_49: "f32[96]", primals_50: "f32[24, 96]", primals_51: "f32[24]", primals_52: "f32[24]", primals_53: "f32[24]", primals_54: "f32[384, 384]", primals_55: "f32[384]", primals_56: "f32[384]", primals_57: "f32[384]", primals_58: "f32[768, 384]", primals_59: "f32[384, 384]", primals_60: "f32[384, 384]", primals_61: "f32[384]", primals_62: "f32[384]", primals_63: "f32[384]", primals_64: "f32[1536, 384]", primals_65: "f32[1536]", primals_66: "f32[384, 1536]", primals_67: "f32[384]", primals_68: "f32[24]", primals_69: "f32[24]", primals_70: "f32[48, 24]", primals_71: "f32[24, 24]", primals_72: "f32[24, 24]", primals_73: "f32[24]", primals_74: "f32[24]", primals_75: "f32[24]", primals_76: "f32[96, 24]", primals_77: "f32[96]", primals_78: "f32[24, 96]", primals_79: "f32[24]", primals_80: "f32[24]", primals_81: "f32[24]", primals_82: "f32[384, 384]", primals_83: "f32[384]", primals_84: "f32[384]", primals_85: "f32[384]", primals_86: "f32[768, 384]", primals_87: "f32[384, 384]", primals_88: "f32[384, 384]", primals_89: "f32[384]", primals_90: "f32[384]", primals_91: "f32[384]", primals_92: "f32[1536, 384]", primals_93: "f32[1536]", primals_94: "f32[384, 1536]", primals_95: "f32[384]", primals_96: "f32[24]", primals_97: "f32[24]", primals_98: "f32[48, 24]", primals_99: "f32[24, 24]", primals_100: "f32[24, 24]", primals_101: "f32[24]", primals_102: "f32[24]", primals_103: "f32[24]", primals_104: "f32[96, 24]", primals_105: "f32[96]", primals_106: "f32[24, 96]", primals_107: "f32[24]", primals_108: "f32[24]", primals_109: "f32[24]", primals_110: "f32[384, 384]", primals_111: "f32[384]", primals_112: "f32[384]", primals_113: "f32[384]", primals_114: "f32[768, 384]", primals_115: "f32[384, 384]", primals_116: "f32[384, 384]", primals_117: "f32[384]", primals_118: "f32[384]", primals_119: "f32[384]", primals_120: "f32[1536, 384]", primals_121: "f32[1536]", primals_122: "f32[384, 1536]", primals_123: "f32[384]", primals_124: "f32[24]", primals_125: "f32[24]", primals_126: "f32[48, 24]", primals_127: "f32[24, 24]", primals_128: "f32[24, 24]", primals_129: "f32[24]", primals_130: "f32[24]", primals_131: "f32[24]", primals_132: "f32[96, 24]", primals_133: "f32[96]", primals_134: "f32[24, 96]", primals_135: "f32[24]", primals_136: "f32[24]", primals_137: "f32[24]", primals_138: "f32[384, 384]", primals_139: "f32[384]", primals_140: "f32[384]", primals_141: "f32[384]", primals_142: "f32[768, 384]", primals_143: "f32[384, 384]", primals_144: "f32[384, 384]", primals_145: "f32[384]", primals_146: "f32[384]", primals_147: "f32[384]", primals_148: "f32[1536, 384]", primals_149: "f32[1536]", primals_150: "f32[384, 1536]", primals_151: "f32[384]", primals_152: "f32[24]", primals_153: "f32[24]", primals_154: "f32[48, 24]", primals_155: "f32[24, 24]", primals_156: "f32[24, 24]", primals_157: "f32[24]", primals_158: "f32[24]", primals_159: "f32[24]", primals_160: "f32[96, 24]", primals_161: "f32[96]", primals_162: "f32[24, 96]", primals_163: "f32[24]", primals_164: "f32[24]", primals_165: "f32[24]", primals_166: "f32[384, 384]", primals_167: "f32[384]", primals_168: "f32[384]", primals_169: "f32[384]", primals_170: "f32[768, 384]", primals_171: "f32[384, 384]", primals_172: "f32[384, 384]", primals_173: "f32[384]", primals_174: "f32[384]", primals_175: "f32[384]", primals_176: "f32[1536, 384]", primals_177: "f32[1536]", primals_178: "f32[384, 1536]", primals_179: "f32[384]", primals_180: "f32[24]", primals_181: "f32[24]", primals_182: "f32[48, 24]", primals_183: "f32[24, 24]", primals_184: "f32[24, 24]", primals_185: "f32[24]", primals_186: "f32[24]", primals_187: "f32[24]", primals_188: "f32[96, 24]", primals_189: "f32[96]", primals_190: "f32[24, 96]", primals_191: "f32[24]", primals_192: "f32[24]", primals_193: "f32[24]", primals_194: "f32[384, 384]", primals_195: "f32[384]", primals_196: "f32[384]", primals_197: "f32[384]", primals_198: "f32[768, 384]", primals_199: "f32[384, 384]", primals_200: "f32[384, 384]", primals_201: "f32[384]", primals_202: "f32[384]", primals_203: "f32[384]", primals_204: "f32[1536, 384]", primals_205: "f32[1536]", primals_206: "f32[384, 1536]", primals_207: "f32[384]", primals_208: "f32[24]", primals_209: "f32[24]", primals_210: "f32[48, 24]", primals_211: "f32[24, 24]", primals_212: "f32[24, 24]", primals_213: "f32[24]", primals_214: "f32[24]", primals_215: "f32[24]", primals_216: "f32[96, 24]", primals_217: "f32[96]", primals_218: "f32[24, 96]", primals_219: "f32[24]", primals_220: "f32[24]", primals_221: "f32[24]", primals_222: "f32[384, 384]", primals_223: "f32[384]", primals_224: "f32[384]", primals_225: "f32[384]", primals_226: "f32[768, 384]", primals_227: "f32[384, 384]", primals_228: "f32[384, 384]", primals_229: "f32[384]", primals_230: "f32[384]", primals_231: "f32[384]", primals_232: "f32[1536, 384]", primals_233: "f32[1536]", primals_234: "f32[384, 1536]", primals_235: "f32[384]", primals_236: "f32[24]", primals_237: "f32[24]", primals_238: "f32[48, 24]", primals_239: "f32[24, 24]", primals_240: "f32[24, 24]", primals_241: "f32[24]", primals_242: "f32[24]", primals_243: "f32[24]", primals_244: "f32[96, 24]", primals_245: "f32[96]", primals_246: "f32[24, 96]", primals_247: "f32[24]", primals_248: "f32[24]", primals_249: "f32[24]", primals_250: "f32[384, 384]", primals_251: "f32[384]", primals_252: "f32[384]", primals_253: "f32[384]", primals_254: "f32[768, 384]", primals_255: "f32[384, 384]", primals_256: "f32[384, 384]", primals_257: "f32[384]", primals_258: "f32[384]", primals_259: "f32[384]", primals_260: "f32[1536, 384]", primals_261: "f32[1536]", primals_262: "f32[384, 1536]", primals_263: "f32[384]", primals_264: "f32[24]", primals_265: "f32[24]", primals_266: "f32[48, 24]", primals_267: "f32[24, 24]", primals_268: "f32[24, 24]", primals_269: "f32[24]", primals_270: "f32[24]", primals_271: "f32[24]", primals_272: "f32[96, 24]", primals_273: "f32[96]", primals_274: "f32[24, 96]", primals_275: "f32[24]", primals_276: "f32[24]", primals_277: "f32[24]", primals_278: "f32[384, 384]", primals_279: "f32[384]", primals_280: "f32[384]", primals_281: "f32[384]", primals_282: "f32[768, 384]", primals_283: "f32[384, 384]", primals_284: "f32[384, 384]", primals_285: "f32[384]", primals_286: "f32[384]", primals_287: "f32[384]", primals_288: "f32[1536, 384]", primals_289: "f32[1536]", primals_290: "f32[384, 1536]", primals_291: "f32[384]", primals_292: "f32[24]", primals_293: "f32[24]", primals_294: "f32[48, 24]", primals_295: "f32[24, 24]", primals_296: "f32[24, 24]", primals_297: "f32[24]", primals_298: "f32[24]", primals_299: "f32[24]", primals_300: "f32[96, 24]", primals_301: "f32[96]", primals_302: "f32[24, 96]", primals_303: "f32[24]", primals_304: "f32[24]", primals_305: "f32[24]", primals_306: "f32[384, 384]", primals_307: "f32[384]", primals_308: "f32[384]", primals_309: "f32[384]", primals_310: "f32[768, 384]", primals_311: "f32[384, 384]", primals_312: "f32[384, 384]", primals_313: "f32[384]", primals_314: "f32[384]", primals_315: "f32[384]", primals_316: "f32[1536, 384]", primals_317: "f32[1536]", primals_318: "f32[384, 1536]", primals_319: "f32[384]", primals_320: "f32[24]", primals_321: "f32[24]", primals_322: "f32[48, 24]", primals_323: "f32[24, 24]", primals_324: "f32[24, 24]", primals_325: "f32[24]", primals_326: "f32[24]", primals_327: "f32[24]", primals_328: "f32[96, 24]", primals_329: "f32[96]", primals_330: "f32[24, 96]", primals_331: "f32[24]", primals_332: "f32[24]", primals_333: "f32[24]", primals_334: "f32[384, 384]", primals_335: "f32[384]", primals_336: "f32[384]", primals_337: "f32[384]", primals_338: "f32[768, 384]", primals_339: "f32[384, 384]", primals_340: "f32[384, 384]", primals_341: "f32[384]", primals_342: "f32[384]", primals_343: "f32[384]", primals_344: "f32[1536, 384]", primals_345: "f32[1536]", primals_346: "f32[384, 1536]", primals_347: "f32[384]", primals_348: "f32[384]", primals_349: "f32[384]", primals_350: "f32[1000, 384]", primals_351: "f32[1000]", primals_352: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:181, code: x = self.proj(x)
    convolution: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(primals_352, primals_4, primals_5, [4, 4], [3, 3], [1, 1], False, [0, 0], 1);  primals_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:182, code: x = self.unfold(x)
    iota: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 4, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
    iota_1: "i64[4]" = torch.ops.prims.iota.default(4, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_1: "i64[4, 1]" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
    add: "i64[4, 14]" = torch.ops.aten.add.Tensor(unsqueeze, unsqueeze_1);  unsqueeze = unsqueeze_1 = None
    unsqueeze_4: "i64[4, 14, 1]" = torch.ops.aten.unsqueeze.default(add, -1)
    unsqueeze_5: "i64[4, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    index: "f32[8, 24, 4, 14, 4, 14]" = torch.ops.aten.index.Tensor(convolution, [None, None, unsqueeze_5, add]);  convolution = None
    permute: "f32[8, 24, 4, 4, 14, 14]" = torch.ops.aten.permute.default(index, [0, 1, 2, 4, 3, 5]);  index = None
    clone: "f32[8, 24, 4, 4, 14, 14]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
    view: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(clone, [8, 384, 196]);  clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:183, code: x = x.transpose(1, 2).reshape(B * self.num_patches, self.in_dim, self.new_patch_size[0], self.new_patch_size[1])
    permute_1: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    clone_1: "f32[8, 196, 384]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
    view_1: "f32[1568, 24, 4, 4]" = torch.ops.aten.reshape.default(clone_1, [1568, 24, 4, 4]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:184, code: x = x + pixel_pos
    add_2: "f32[1568, 24, 4, 4]" = torch.ops.aten.add.Tensor(view_1, primals_1);  view_1 = primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:185, code: x = x.reshape(B * self.num_patches, self.in_dim, -1).transpose(1, 2)
    view_2: "f32[1568, 24, 16]" = torch.ops.aten.reshape.default(add_2, [1568, 24, -1]);  add_2 = None
    permute_2: "f32[1568, 16, 24]" = torch.ops.aten.permute.default(view_2, [0, 2, 1]);  view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:311, code: patch_embed = self.norm2_proj(self.proj(self.norm1_proj(pixel_embed.reshape(B, self.num_patches, -1))))
    clone_2: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format)
    view_3: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(clone_2, [8, 196, 384])
    var_mean = torch.ops.aten.var_mean.correction(view_3, [2], correction = 0, keepdim = True)
    getitem: "f32[8, 196, 1]" = var_mean[0]
    getitem_1: "f32[8, 196, 1]" = var_mean[1];  var_mean = None
    add_3: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
    sub: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(view_3, getitem_1);  view_3 = None
    mul: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul, primals_6);  mul = None
    add_4: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_1, primals_7);  mul_1 = primals_7 = None
    view_4: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_4, [1568, 384]);  add_4 = None
    permute_3: "f32[384, 384]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
    addmm: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_9, view_4, permute_3);  primals_9 = None
    view_5: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(addmm, [8, 196, 384])
    var_mean_1 = torch.ops.aten.var_mean.correction(view_5, [2], correction = 0, keepdim = True)
    getitem_2: "f32[8, 196, 1]" = var_mean_1[0]
    getitem_3: "f32[8, 196, 1]" = var_mean_1[1];  var_mean_1 = None
    add_5: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
    rsqrt_1: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
    sub_1: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(view_5, getitem_3);  view_5 = None
    mul_2: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    mul_3: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_2, primals_10);  mul_2 = None
    add_6: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_3, primals_11);  mul_3 = primals_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:312, code: patch_embed = torch.cat((self.cls_token.expand(B, -1, -1), patch_embed), dim=1)
    expand: "f32[8, 1, 384]" = torch.ops.aten.expand.default(primals_2, [8, -1, -1]);  primals_2 = None
    cat: "f32[8, 197, 384]" = torch.ops.aten.cat.default([expand, add_6], 1);  expand = add_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:313, code: patch_embed = patch_embed + self.patch_pos
    add_7: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat, primals_3);  cat = primals_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    var_mean_2 = torch.ops.aten.var_mean.correction(clone_2, [2], correction = 0, keepdim = True)
    getitem_4: "f32[1568, 16, 1]" = var_mean_2[0]
    getitem_5: "f32[1568, 16, 1]" = var_mean_2[1];  var_mean_2 = None
    add_8: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_2: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
    sub_2: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_2, getitem_5)
    mul_4: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    mul_5: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_4, primals_12);  mul_4 = None
    add_9: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_5, primals_13);  mul_5 = primals_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_4: "f32[24, 48]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
    view_6: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_9, [25088, 24]);  add_9 = None
    mm: "f32[25088, 48]" = torch.ops.aten.mm.default(view_6, permute_4)
    view_7: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(mm, [1568, 16, 48]);  mm = None
    view_8: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.reshape.default(view_7, [1568, 16, 2, 4, 6]);  view_7 = None
    permute_5: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_8, [2, 0, 3, 1, 4]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind = torch.ops.aten.unbind.int(permute_5);  permute_5 = None
    getitem_6: "f32[1568, 4, 16, 6]" = unbind[0]
    getitem_7: "f32[1568, 4, 16, 6]" = unbind[1];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_6: "f32[24, 24]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
    mm_1: "f32[25088, 24]" = torch.ops.aten.mm.default(view_6, permute_6)
    view_10: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_1, [1568, 16, 24]);  mm_1 = None
    view_11: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_10, [1568, 16, 4, -1]);  view_10 = None
    permute_7: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_11, [0, 2, 1, 3]);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_8: "f32[1568, 4, 6, 16]" = torch.ops.aten.permute.default(getitem_7, [0, 1, 3, 2]);  getitem_7 = None
    expand_1: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(getitem_6, [1568, 4, 16, 6]);  getitem_6 = None
    clone_5: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_12: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_5, [6272, 16, 6]);  clone_5 = None
    expand_2: "f32[1568, 4, 6, 16]" = torch.ops.aten.expand.default(permute_8, [1568, 4, 6, 16]);  permute_8 = None
    clone_6: "f32[1568, 4, 6, 16]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
    view_13: "f32[6272, 6, 16]" = torch.ops.aten.reshape.default(clone_6, [6272, 6, 16]);  clone_6 = None
    bmm: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_12, view_13)
    view_14: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm, [1568, 4, 16, 16]);  bmm = None
    mul_6: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_14, 0.408248290463863);  view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax: "f32[1568, 4, 16, 1]" = torch.ops.aten.amax.default(mul_6, [-1], True)
    sub_3: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_6, amax);  mul_6 = amax = None
    exp: "f32[1568, 4, 16, 16]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
    sum_1: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[1568, 4, 16, 16]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias: "f32[1568, 4, 16, 16]" = torch.ops.aten.alias.default(div)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_3: "f32[1568, 4, 16, 16]" = torch.ops.aten.expand.default(div, [1568, 4, 16, 16]);  div = None
    view_15: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(expand_3, [6272, 16, 16]);  expand_3 = None
    expand_4: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(permute_7, [1568, 4, 16, 6]);  permute_7 = None
    clone_7: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
    view_16: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_7, [6272, 16, 6]);  clone_7 = None
    bmm_1: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_15, view_16)
    view_17: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_1, [1568, 4, 16, 6]);  bmm_1 = None
    permute_9: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None
    clone_8: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_9, memory_format = torch.contiguous_format);  permute_9 = None
    view_18: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_8, [1568, 16, 24]);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_19: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_18, [25088, 24]);  view_18 = None
    permute_10: "f32[24, 24]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
    addmm_1: "f32[25088, 24]" = torch.ops.aten.addmm.default(primals_17, view_19, permute_10);  primals_17 = None
    view_20: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(addmm_1, [1568, 16, 24]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    add_10: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(permute_2, view_20);  permute_2 = view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    clone_9: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_10, memory_format = torch.contiguous_format)
    var_mean_3 = torch.ops.aten.var_mean.correction(clone_9, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1568, 16, 1]" = var_mean_3[0]
    getitem_9: "f32[1568, 16, 1]" = var_mean_3[1];  var_mean_3 = None
    add_11: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_3: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_4: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_9, getitem_9);  clone_9 = getitem_9 = None
    mul_7: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = None
    mul_8: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_7, primals_18)
    add_12: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_8, primals_19);  mul_8 = primals_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_21: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_12, [25088, 24]);  add_12 = None
    permute_11: "f32[24, 96]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
    addmm_2: "f32[25088, 96]" = torch.ops.aten.addmm.default(primals_21, view_21, permute_11);  primals_21 = None
    view_22: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(addmm_2, [1568, 16, 96])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_9: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_22, 0.5)
    mul_10: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_22, 0.7071067811865476);  view_22 = None
    erf: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_10);  mul_10 = None
    add_13: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_11: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_9, add_13);  mul_9 = add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_23: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_11, [25088, 96]);  mul_11 = None
    permute_12: "f32[96, 24]" = torch.ops.aten.permute.default(primals_22, [1, 0]);  primals_22 = None
    addmm_3: "f32[25088, 24]" = torch.ops.aten.addmm.default(primals_23, view_23, permute_12);  primals_23 = None
    view_24: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(addmm_3, [1568, 16, 24]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    add_14: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_10, view_24);  add_10 = view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    slice_4: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_7, 1, 0, 1)
    slice_6: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_7, 1, 1, 9223372036854775807);  add_7 = None
    clone_12: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_14, memory_format = torch.contiguous_format)
    var_mean_4 = torch.ops.aten.var_mean.correction(clone_12, [2], correction = 0, keepdim = True)
    getitem_10: "f32[1568, 16, 1]" = var_mean_4[0]
    getitem_11: "f32[1568, 16, 1]" = var_mean_4[1];  var_mean_4 = None
    add_15: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
    rsqrt_4: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
    sub_5: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_12, getitem_11);  clone_12 = getitem_11 = None
    mul_12: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_4);  sub_5 = None
    mul_13: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_12, primals_24)
    add_16: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_13, primals_25);  mul_13 = primals_25 = None
    view_25: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_16, [8, 196, -1]);  add_16 = None
    view_26: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_25, [1568, 384]);  view_25 = None
    permute_13: "f32[384, 384]" = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
    addmm_4: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_27, view_26, permute_13);  primals_27 = None
    view_27: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(addmm_4, [8, 196, 384]);  addmm_4 = None
    add_17: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(slice_6, view_27);  slice_6 = view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    cat_1: "f32[8, 197, 384]" = torch.ops.aten.cat.default([slice_4, add_17], 1);  slice_4 = add_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    var_mean_5 = torch.ops.aten.var_mean.correction(cat_1, [2], correction = 0, keepdim = True)
    getitem_12: "f32[8, 197, 1]" = var_mean_5[0]
    getitem_13: "f32[8, 197, 1]" = var_mean_5[1];  var_mean_5 = None
    add_18: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_5: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_6: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_1, getitem_13)
    mul_14: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_5);  sub_6 = None
    mul_15: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_14, primals_28);  mul_14 = None
    add_19: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_15, primals_29);  mul_15 = primals_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_14: "f32[384, 768]" = torch.ops.aten.permute.default(primals_30, [1, 0]);  primals_30 = None
    view_28: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_19, [1576, 384]);  add_19 = None
    mm_2: "f32[1576, 768]" = torch.ops.aten.mm.default(view_28, permute_14)
    view_29: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_2, [8, 197, 768]);  mm_2 = None
    view_30: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.reshape.default(view_29, [8, 197, 2, 6, 64]);  view_29 = None
    permute_15: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.permute.default(view_30, [2, 0, 3, 1, 4]);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_1 = torch.ops.aten.unbind.int(permute_15);  permute_15 = None
    getitem_14: "f32[8, 6, 197, 64]" = unbind_1[0]
    getitem_15: "f32[8, 6, 197, 64]" = unbind_1[1];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_16: "f32[384, 384]" = torch.ops.aten.permute.default(primals_31, [1, 0]);  primals_31 = None
    mm_3: "f32[1576, 384]" = torch.ops.aten.mm.default(view_28, permute_16)
    view_32: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_3, [8, 197, 384]);  mm_3 = None
    view_33: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_32, [8, 197, 6, -1]);  view_32 = None
    permute_17: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_18: "f32[8, 6, 64, 197]" = torch.ops.aten.permute.default(getitem_15, [0, 1, 3, 2]);  getitem_15 = None
    expand_5: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(getitem_14, [8, 6, 197, 64]);  getitem_14 = None
    clone_13: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    view_34: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_13, [48, 197, 64]);  clone_13 = None
    expand_6: "f32[8, 6, 64, 197]" = torch.ops.aten.expand.default(permute_18, [8, 6, 64, 197]);  permute_18 = None
    clone_14: "f32[8, 6, 64, 197]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
    view_35: "f32[48, 64, 197]" = torch.ops.aten.reshape.default(clone_14, [48, 64, 197]);  clone_14 = None
    bmm_2: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_34, view_35)
    view_36: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_2, [8, 6, 197, 197]);  bmm_2 = None
    mul_16: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_36, 0.125);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_1: "f32[8, 6, 197, 1]" = torch.ops.aten.amax.default(mul_16, [-1], True)
    sub_7: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_16, amax_1);  mul_16 = amax_1 = None
    exp_1: "f32[8, 6, 197, 197]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_2: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[8, 6, 197, 197]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_1: "f32[8, 6, 197, 197]" = torch.ops.aten.alias.default(div_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_7: "f32[8, 6, 197, 197]" = torch.ops.aten.expand.default(div_1, [8, 6, 197, 197]);  div_1 = None
    view_37: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(expand_7, [48, 197, 197]);  expand_7 = None
    expand_8: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(permute_17, [8, 6, 197, 64]);  permute_17 = None
    clone_15: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
    view_38: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_15, [48, 197, 64]);  clone_15 = None
    bmm_3: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_37, view_38)
    view_39: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_3, [8, 6, 197, 64]);  bmm_3 = None
    permute_19: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_39, [0, 2, 1, 3]);  view_39 = None
    clone_16: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
    view_40: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_16, [8, 197, 384]);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_41: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_40, [1576, 384]);  view_40 = None
    permute_20: "f32[384, 384]" = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
    addmm_5: "f32[1576, 384]" = torch.ops.aten.addmm.default(primals_33, view_41, permute_20);  primals_33 = None
    view_42: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(addmm_5, [8, 197, 384]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    add_20: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_1, view_42);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    var_mean_6 = torch.ops.aten.var_mean.correction(add_20, [2], correction = 0, keepdim = True)
    getitem_16: "f32[8, 197, 1]" = var_mean_6[0]
    getitem_17: "f32[8, 197, 1]" = var_mean_6[1];  var_mean_6 = None
    add_21: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_6: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_8: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_20, getitem_17);  getitem_17 = None
    mul_17: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_6);  sub_8 = None
    mul_18: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_17, primals_34)
    add_22: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_18, primals_35);  mul_18 = primals_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_43: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_22, [1576, 384]);  add_22 = None
    permute_21: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_36, [1, 0]);  primals_36 = None
    addmm_6: "f32[1576, 1536]" = torch.ops.aten.addmm.default(primals_37, view_43, permute_21);  primals_37 = None
    view_44: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(addmm_6, [8, 197, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_19: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_44, 0.5)
    mul_20: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_44, 0.7071067811865476);  view_44 = None
    erf_1: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
    add_23: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_21: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_19, add_23);  mul_19 = add_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_45: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_21, [1576, 1536]);  mul_21 = None
    permute_22: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
    addmm_7: "f32[1576, 384]" = torch.ops.aten.addmm.default(primals_39, view_45, permute_22);  primals_39 = None
    view_46: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(addmm_7, [8, 197, 384]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    add_24: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_20, view_46);  add_20 = view_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    mul_23: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_12, primals_40)
    add_26: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_23, primals_41);  mul_23 = primals_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_23: "f32[24, 48]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
    view_47: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_26, [25088, 24]);  add_26 = None
    mm_4: "f32[25088, 48]" = torch.ops.aten.mm.default(view_47, permute_23)
    view_48: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(mm_4, [1568, 16, 48]);  mm_4 = None
    view_49: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.reshape.default(view_48, [1568, 16, 2, 4, 6]);  view_48 = None
    permute_24: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_49, [2, 0, 3, 1, 4]);  view_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_2 = torch.ops.aten.unbind.int(permute_24);  permute_24 = None
    getitem_20: "f32[1568, 4, 16, 6]" = unbind_2[0]
    getitem_21: "f32[1568, 4, 16, 6]" = unbind_2[1];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_25: "f32[24, 24]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
    mm_5: "f32[25088, 24]" = torch.ops.aten.mm.default(view_47, permute_25)
    view_51: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_5, [1568, 16, 24]);  mm_5 = None
    view_52: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_51, [1568, 16, 4, -1]);  view_51 = None
    permute_26: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_27: "f32[1568, 4, 6, 16]" = torch.ops.aten.permute.default(getitem_21, [0, 1, 3, 2]);  getitem_21 = None
    expand_9: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(getitem_20, [1568, 4, 16, 6]);  getitem_20 = None
    clone_20: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    view_53: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_20, [6272, 16, 6]);  clone_20 = None
    expand_10: "f32[1568, 4, 6, 16]" = torch.ops.aten.expand.default(permute_27, [1568, 4, 6, 16]);  permute_27 = None
    clone_21: "f32[1568, 4, 6, 16]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
    view_54: "f32[6272, 6, 16]" = torch.ops.aten.reshape.default(clone_21, [6272, 6, 16]);  clone_21 = None
    bmm_4: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_53, view_54)
    view_55: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_4, [1568, 4, 16, 16]);  bmm_4 = None
    mul_24: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_55, 0.408248290463863);  view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_2: "f32[1568, 4, 16, 1]" = torch.ops.aten.amax.default(mul_24, [-1], True)
    sub_10: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_24, amax_2);  mul_24 = amax_2 = None
    exp_2: "f32[1568, 4, 16, 16]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_3: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_2: "f32[1568, 4, 16, 16]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_2: "f32[1568, 4, 16, 16]" = torch.ops.aten.alias.default(div_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_11: "f32[1568, 4, 16, 16]" = torch.ops.aten.expand.default(div_2, [1568, 4, 16, 16]);  div_2 = None
    view_56: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(expand_11, [6272, 16, 16]);  expand_11 = None
    expand_12: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(permute_26, [1568, 4, 16, 6]);  permute_26 = None
    clone_22: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
    view_57: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_22, [6272, 16, 6]);  clone_22 = None
    bmm_5: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_56, view_57)
    view_58: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_5, [1568, 4, 16, 6]);  bmm_5 = None
    permute_28: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
    clone_23: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_28, memory_format = torch.contiguous_format);  permute_28 = None
    view_59: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_23, [1568, 16, 24]);  clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_60: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_59, [25088, 24]);  view_59 = None
    permute_29: "f32[24, 24]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
    addmm_8: "f32[25088, 24]" = torch.ops.aten.addmm.default(primals_45, view_60, permute_29);  primals_45 = None
    view_61: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(addmm_8, [1568, 16, 24]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    add_27: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_14, view_61);  add_14 = view_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    clone_24: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_27, memory_format = torch.contiguous_format)
    var_mean_8 = torch.ops.aten.var_mean.correction(clone_24, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1568, 16, 1]" = var_mean_8[0]
    getitem_23: "f32[1568, 16, 1]" = var_mean_8[1];  var_mean_8 = None
    add_28: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_8: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_11: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_24, getitem_23);  clone_24 = getitem_23 = None
    mul_25: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_8);  sub_11 = None
    mul_26: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_25, primals_46)
    add_29: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_26, primals_47);  mul_26 = primals_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_62: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_29, [25088, 24]);  add_29 = None
    permute_30: "f32[24, 96]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
    addmm_9: "f32[25088, 96]" = torch.ops.aten.addmm.default(primals_49, view_62, permute_30);  primals_49 = None
    view_63: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(addmm_9, [1568, 16, 96])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_27: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_63, 0.5)
    mul_28: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476);  view_63 = None
    erf_2: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
    add_30: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_29: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_27, add_30);  mul_27 = add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_64: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_29, [25088, 96]);  mul_29 = None
    permute_31: "f32[96, 24]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
    addmm_10: "f32[25088, 24]" = torch.ops.aten.addmm.default(primals_51, view_64, permute_31);  primals_51 = None
    view_65: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(addmm_10, [1568, 16, 24]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    add_31: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_27, view_65);  add_27 = view_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    slice_8: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_24, 1, 0, 1)
    slice_10: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_24, 1, 1, 9223372036854775807);  add_24 = None
    clone_27: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_31, memory_format = torch.contiguous_format)
    var_mean_9 = torch.ops.aten.var_mean.correction(clone_27, [2], correction = 0, keepdim = True)
    getitem_24: "f32[1568, 16, 1]" = var_mean_9[0]
    getitem_25: "f32[1568, 16, 1]" = var_mean_9[1];  var_mean_9 = None
    add_32: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
    rsqrt_9: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_12: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_27, getitem_25);  clone_27 = getitem_25 = None
    mul_30: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_9);  sub_12 = None
    mul_31: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_30, primals_52)
    add_33: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_31, primals_53);  mul_31 = primals_53 = None
    view_66: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_33, [8, 196, -1]);  add_33 = None
    view_67: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_66, [1568, 384]);  view_66 = None
    permute_32: "f32[384, 384]" = torch.ops.aten.permute.default(primals_54, [1, 0]);  primals_54 = None
    addmm_11: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_55, view_67, permute_32);  primals_55 = None
    view_68: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(addmm_11, [8, 196, 384]);  addmm_11 = None
    add_34: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(slice_10, view_68);  slice_10 = view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    cat_2: "f32[8, 197, 384]" = torch.ops.aten.cat.default([slice_8, add_34], 1);  slice_8 = add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    var_mean_10 = torch.ops.aten.var_mean.correction(cat_2, [2], correction = 0, keepdim = True)
    getitem_26: "f32[8, 197, 1]" = var_mean_10[0]
    getitem_27: "f32[8, 197, 1]" = var_mean_10[1];  var_mean_10 = None
    add_35: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
    rsqrt_10: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
    sub_13: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_2, getitem_27)
    mul_32: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_10);  sub_13 = None
    mul_33: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_32, primals_56);  mul_32 = None
    add_36: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_33, primals_57);  mul_33 = primals_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_33: "f32[384, 768]" = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
    view_69: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_36, [1576, 384]);  add_36 = None
    mm_6: "f32[1576, 768]" = torch.ops.aten.mm.default(view_69, permute_33)
    view_70: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_6, [8, 197, 768]);  mm_6 = None
    view_71: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.reshape.default(view_70, [8, 197, 2, 6, 64]);  view_70 = None
    permute_34: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.permute.default(view_71, [2, 0, 3, 1, 4]);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_3 = torch.ops.aten.unbind.int(permute_34);  permute_34 = None
    getitem_28: "f32[8, 6, 197, 64]" = unbind_3[0]
    getitem_29: "f32[8, 6, 197, 64]" = unbind_3[1];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_35: "f32[384, 384]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
    mm_7: "f32[1576, 384]" = torch.ops.aten.mm.default(view_69, permute_35)
    view_73: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_7, [8, 197, 384]);  mm_7 = None
    view_74: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_73, [8, 197, 6, -1]);  view_73 = None
    permute_36: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_37: "f32[8, 6, 64, 197]" = torch.ops.aten.permute.default(getitem_29, [0, 1, 3, 2]);  getitem_29 = None
    expand_13: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(getitem_28, [8, 6, 197, 64]);  getitem_28 = None
    clone_28: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
    view_75: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_28, [48, 197, 64]);  clone_28 = None
    expand_14: "f32[8, 6, 64, 197]" = torch.ops.aten.expand.default(permute_37, [8, 6, 64, 197]);  permute_37 = None
    clone_29: "f32[8, 6, 64, 197]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
    view_76: "f32[48, 64, 197]" = torch.ops.aten.reshape.default(clone_29, [48, 64, 197]);  clone_29 = None
    bmm_6: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_75, view_76)
    view_77: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_6, [8, 6, 197, 197]);  bmm_6 = None
    mul_34: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_77, 0.125);  view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_3: "f32[8, 6, 197, 1]" = torch.ops.aten.amax.default(mul_34, [-1], True)
    sub_14: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_34, amax_3);  mul_34 = amax_3 = None
    exp_3: "f32[8, 6, 197, 197]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
    sum_4: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_3: "f32[8, 6, 197, 197]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_3: "f32[8, 6, 197, 197]" = torch.ops.aten.alias.default(div_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_15: "f32[8, 6, 197, 197]" = torch.ops.aten.expand.default(div_3, [8, 6, 197, 197]);  div_3 = None
    view_78: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(expand_15, [48, 197, 197]);  expand_15 = None
    expand_16: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(permute_36, [8, 6, 197, 64]);  permute_36 = None
    clone_30: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
    view_79: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_30, [48, 197, 64]);  clone_30 = None
    bmm_7: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_78, view_79)
    view_80: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_7, [8, 6, 197, 64]);  bmm_7 = None
    permute_38: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
    clone_31: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    view_81: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_31, [8, 197, 384]);  clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_82: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_81, [1576, 384]);  view_81 = None
    permute_39: "f32[384, 384]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
    addmm_12: "f32[1576, 384]" = torch.ops.aten.addmm.default(primals_61, view_82, permute_39);  primals_61 = None
    view_83: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(addmm_12, [8, 197, 384]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    add_37: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_2, view_83);  view_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    var_mean_11 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
    getitem_30: "f32[8, 197, 1]" = var_mean_11[0]
    getitem_31: "f32[8, 197, 1]" = var_mean_11[1];  var_mean_11 = None
    add_38: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
    rsqrt_11: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_15: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_37, getitem_31);  getitem_31 = None
    mul_35: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_11);  sub_15 = None
    mul_36: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_35, primals_62)
    add_39: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_36, primals_63);  mul_36 = primals_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_84: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_39, [1576, 384]);  add_39 = None
    permute_40: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
    addmm_13: "f32[1576, 1536]" = torch.ops.aten.addmm.default(primals_65, view_84, permute_40);  primals_65 = None
    view_85: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(addmm_13, [8, 197, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_37: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_85, 0.5)
    mul_38: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476);  view_85 = None
    erf_3: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_38);  mul_38 = None
    add_40: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_39: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_37, add_40);  mul_37 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_86: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_39, [1576, 1536]);  mul_39 = None
    permute_41: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
    addmm_14: "f32[1576, 384]" = torch.ops.aten.addmm.default(primals_67, view_86, permute_41);  primals_67 = None
    view_87: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(addmm_14, [8, 197, 384]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    add_41: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_37, view_87);  add_37 = view_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    mul_41: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_30, primals_68)
    add_43: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_41, primals_69);  mul_41 = primals_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_42: "f32[24, 48]" = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
    view_88: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_43, [25088, 24]);  add_43 = None
    mm_8: "f32[25088, 48]" = torch.ops.aten.mm.default(view_88, permute_42)
    view_89: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(mm_8, [1568, 16, 48]);  mm_8 = None
    view_90: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.reshape.default(view_89, [1568, 16, 2, 4, 6]);  view_89 = None
    permute_43: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_90, [2, 0, 3, 1, 4]);  view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_4 = torch.ops.aten.unbind.int(permute_43);  permute_43 = None
    getitem_34: "f32[1568, 4, 16, 6]" = unbind_4[0]
    getitem_35: "f32[1568, 4, 16, 6]" = unbind_4[1];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_44: "f32[24, 24]" = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
    mm_9: "f32[25088, 24]" = torch.ops.aten.mm.default(view_88, permute_44)
    view_92: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_9, [1568, 16, 24]);  mm_9 = None
    view_93: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_92, [1568, 16, 4, -1]);  view_92 = None
    permute_45: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_46: "f32[1568, 4, 6, 16]" = torch.ops.aten.permute.default(getitem_35, [0, 1, 3, 2]);  getitem_35 = None
    expand_17: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(getitem_34, [1568, 4, 16, 6]);  getitem_34 = None
    clone_35: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    view_94: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_35, [6272, 16, 6]);  clone_35 = None
    expand_18: "f32[1568, 4, 6, 16]" = torch.ops.aten.expand.default(permute_46, [1568, 4, 6, 16]);  permute_46 = None
    clone_36: "f32[1568, 4, 6, 16]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
    view_95: "f32[6272, 6, 16]" = torch.ops.aten.reshape.default(clone_36, [6272, 6, 16]);  clone_36 = None
    bmm_8: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_94, view_95)
    view_96: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_8, [1568, 4, 16, 16]);  bmm_8 = None
    mul_42: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_96, 0.408248290463863);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_4: "f32[1568, 4, 16, 1]" = torch.ops.aten.amax.default(mul_42, [-1], True)
    sub_17: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_42, amax_4);  mul_42 = amax_4 = None
    exp_4: "f32[1568, 4, 16, 16]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
    sum_5: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_4: "f32[1568, 4, 16, 16]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    alias_4: "f32[1568, 4, 16, 16]" = torch.ops.aten.alias.default(div_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_19: "f32[1568, 4, 16, 16]" = torch.ops.aten.expand.default(div_4, [1568, 4, 16, 16]);  div_4 = None
    view_97: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(expand_19, [6272, 16, 16]);  expand_19 = None
    expand_20: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(permute_45, [1568, 4, 16, 6]);  permute_45 = None
    clone_37: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
    view_98: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_37, [6272, 16, 6]);  clone_37 = None
    bmm_9: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_97, view_98)
    view_99: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_9, [1568, 4, 16, 6]);  bmm_9 = None
    permute_47: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None
    clone_38: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
    view_100: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_38, [1568, 16, 24]);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_101: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_100, [25088, 24]);  view_100 = None
    permute_48: "f32[24, 24]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
    addmm_15: "f32[25088, 24]" = torch.ops.aten.addmm.default(primals_73, view_101, permute_48);  primals_73 = None
    view_102: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(addmm_15, [1568, 16, 24]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    add_44: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_31, view_102);  add_31 = view_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    clone_39: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_44, memory_format = torch.contiguous_format)
    var_mean_13 = torch.ops.aten.var_mean.correction(clone_39, [2], correction = 0, keepdim = True)
    getitem_36: "f32[1568, 16, 1]" = var_mean_13[0]
    getitem_37: "f32[1568, 16, 1]" = var_mean_13[1];  var_mean_13 = None
    add_45: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
    rsqrt_13: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
    sub_18: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_39, getitem_37);  clone_39 = getitem_37 = None
    mul_43: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_13);  sub_18 = None
    mul_44: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_43, primals_74)
    add_46: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_44, primals_75);  mul_44 = primals_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_103: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_46, [25088, 24]);  add_46 = None
    permute_49: "f32[24, 96]" = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
    addmm_16: "f32[25088, 96]" = torch.ops.aten.addmm.default(primals_77, view_103, permute_49);  primals_77 = None
    view_104: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(addmm_16, [1568, 16, 96])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_45: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_104, 0.5)
    mul_46: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_104, 0.7071067811865476);  view_104 = None
    erf_4: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_46);  mul_46 = None
    add_47: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_47: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_45, add_47);  mul_45 = add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_105: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_47, [25088, 96]);  mul_47 = None
    permute_50: "f32[96, 24]" = torch.ops.aten.permute.default(primals_78, [1, 0]);  primals_78 = None
    addmm_17: "f32[25088, 24]" = torch.ops.aten.addmm.default(primals_79, view_105, permute_50);  primals_79 = None
    view_106: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(addmm_17, [1568, 16, 24]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    add_48: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_44, view_106);  add_44 = view_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    slice_12: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_41, 1, 0, 1)
    slice_14: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_41, 1, 1, 9223372036854775807);  add_41 = None
    clone_42: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_48, memory_format = torch.contiguous_format)
    var_mean_14 = torch.ops.aten.var_mean.correction(clone_42, [2], correction = 0, keepdim = True)
    getitem_38: "f32[1568, 16, 1]" = var_mean_14[0]
    getitem_39: "f32[1568, 16, 1]" = var_mean_14[1];  var_mean_14 = None
    add_49: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
    rsqrt_14: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    sub_19: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_42, getitem_39);  clone_42 = getitem_39 = None
    mul_48: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_14);  sub_19 = None
    mul_49: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_48, primals_80)
    add_50: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_49, primals_81);  mul_49 = primals_81 = None
    view_107: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_50, [8, 196, -1]);  add_50 = None
    view_108: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_107, [1568, 384]);  view_107 = None
    permute_51: "f32[384, 384]" = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
    addmm_18: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_83, view_108, permute_51);  primals_83 = None
    view_109: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(addmm_18, [8, 196, 384]);  addmm_18 = None
    add_51: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(slice_14, view_109);  slice_14 = view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    cat_3: "f32[8, 197, 384]" = torch.ops.aten.cat.default([slice_12, add_51], 1);  slice_12 = add_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    var_mean_15 = torch.ops.aten.var_mean.correction(cat_3, [2], correction = 0, keepdim = True)
    getitem_40: "f32[8, 197, 1]" = var_mean_15[0]
    getitem_41: "f32[8, 197, 1]" = var_mean_15[1];  var_mean_15 = None
    add_52: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
    rsqrt_15: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_20: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_3, getitem_41)
    mul_50: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_15);  sub_20 = None
    mul_51: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_50, primals_84);  mul_50 = None
    add_53: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_51, primals_85);  mul_51 = primals_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_52: "f32[384, 768]" = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
    view_110: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_53, [1576, 384]);  add_53 = None
    mm_10: "f32[1576, 768]" = torch.ops.aten.mm.default(view_110, permute_52)
    view_111: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_10, [8, 197, 768]);  mm_10 = None
    view_112: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.reshape.default(view_111, [8, 197, 2, 6, 64]);  view_111 = None
    permute_53: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.permute.default(view_112, [2, 0, 3, 1, 4]);  view_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_5 = torch.ops.aten.unbind.int(permute_53);  permute_53 = None
    getitem_42: "f32[8, 6, 197, 64]" = unbind_5[0]
    getitem_43: "f32[8, 6, 197, 64]" = unbind_5[1];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_54: "f32[384, 384]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
    mm_11: "f32[1576, 384]" = torch.ops.aten.mm.default(view_110, permute_54)
    view_114: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_11, [8, 197, 384]);  mm_11 = None
    view_115: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_114, [8, 197, 6, -1]);  view_114 = None
    permute_55: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_115, [0, 2, 1, 3]);  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_56: "f32[8, 6, 64, 197]" = torch.ops.aten.permute.default(getitem_43, [0, 1, 3, 2]);  getitem_43 = None
    expand_21: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(getitem_42, [8, 6, 197, 64]);  getitem_42 = None
    clone_43: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    view_116: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_43, [48, 197, 64]);  clone_43 = None
    expand_22: "f32[8, 6, 64, 197]" = torch.ops.aten.expand.default(permute_56, [8, 6, 64, 197]);  permute_56 = None
    clone_44: "f32[8, 6, 64, 197]" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
    view_117: "f32[48, 64, 197]" = torch.ops.aten.reshape.default(clone_44, [48, 64, 197]);  clone_44 = None
    bmm_10: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_116, view_117)
    view_118: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_10, [8, 6, 197, 197]);  bmm_10 = None
    mul_52: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_118, 0.125);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_5: "f32[8, 6, 197, 1]" = torch.ops.aten.amax.default(mul_52, [-1], True)
    sub_21: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_52, amax_5);  mul_52 = amax_5 = None
    exp_5: "f32[8, 6, 197, 197]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
    sum_6: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_5: "f32[8, 6, 197, 197]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    alias_5: "f32[8, 6, 197, 197]" = torch.ops.aten.alias.default(div_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_23: "f32[8, 6, 197, 197]" = torch.ops.aten.expand.default(div_5, [8, 6, 197, 197]);  div_5 = None
    view_119: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(expand_23, [48, 197, 197]);  expand_23 = None
    expand_24: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(permute_55, [8, 6, 197, 64]);  permute_55 = None
    clone_45: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
    view_120: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_45, [48, 197, 64]);  clone_45 = None
    bmm_11: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_119, view_120)
    view_121: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_11, [8, 6, 197, 64]);  bmm_11 = None
    permute_57: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_121, [0, 2, 1, 3]);  view_121 = None
    clone_46: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
    view_122: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_46, [8, 197, 384]);  clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_123: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_122, [1576, 384]);  view_122 = None
    permute_58: "f32[384, 384]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
    addmm_19: "f32[1576, 384]" = torch.ops.aten.addmm.default(primals_89, view_123, permute_58);  primals_89 = None
    view_124: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(addmm_19, [8, 197, 384]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    add_54: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_3, view_124);  view_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    var_mean_16 = torch.ops.aten.var_mean.correction(add_54, [2], correction = 0, keepdim = True)
    getitem_44: "f32[8, 197, 1]" = var_mean_16[0]
    getitem_45: "f32[8, 197, 1]" = var_mean_16[1];  var_mean_16 = None
    add_55: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
    rsqrt_16: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
    sub_22: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_54, getitem_45);  getitem_45 = None
    mul_53: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_16);  sub_22 = None
    mul_54: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_53, primals_90)
    add_56: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_54, primals_91);  mul_54 = primals_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_125: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_56, [1576, 384]);  add_56 = None
    permute_59: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
    addmm_20: "f32[1576, 1536]" = torch.ops.aten.addmm.default(primals_93, view_125, permute_59);  primals_93 = None
    view_126: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(addmm_20, [8, 197, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_55: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_126, 0.5)
    mul_56: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_126, 0.7071067811865476);  view_126 = None
    erf_5: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_56);  mul_56 = None
    add_57: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_57: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_55, add_57);  mul_55 = add_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_127: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_57, [1576, 1536]);  mul_57 = None
    permute_60: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_94, [1, 0]);  primals_94 = None
    addmm_21: "f32[1576, 384]" = torch.ops.aten.addmm.default(primals_95, view_127, permute_60);  primals_95 = None
    view_128: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(addmm_21, [8, 197, 384]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    add_58: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_54, view_128);  add_54 = view_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    mul_59: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_48, primals_96)
    add_60: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_59, primals_97);  mul_59 = primals_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_61: "f32[24, 48]" = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
    view_129: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_60, [25088, 24]);  add_60 = None
    mm_12: "f32[25088, 48]" = torch.ops.aten.mm.default(view_129, permute_61)
    view_130: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(mm_12, [1568, 16, 48]);  mm_12 = None
    view_131: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.reshape.default(view_130, [1568, 16, 2, 4, 6]);  view_130 = None
    permute_62: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_131, [2, 0, 3, 1, 4]);  view_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_6 = torch.ops.aten.unbind.int(permute_62);  permute_62 = None
    getitem_48: "f32[1568, 4, 16, 6]" = unbind_6[0]
    getitem_49: "f32[1568, 4, 16, 6]" = unbind_6[1];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_63: "f32[24, 24]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
    mm_13: "f32[25088, 24]" = torch.ops.aten.mm.default(view_129, permute_63)
    view_133: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_13, [1568, 16, 24]);  mm_13 = None
    view_134: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_133, [1568, 16, 4, -1]);  view_133 = None
    permute_64: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_65: "f32[1568, 4, 6, 16]" = torch.ops.aten.permute.default(getitem_49, [0, 1, 3, 2]);  getitem_49 = None
    expand_25: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(getitem_48, [1568, 4, 16, 6]);  getitem_48 = None
    clone_50: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    view_135: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_50, [6272, 16, 6]);  clone_50 = None
    expand_26: "f32[1568, 4, 6, 16]" = torch.ops.aten.expand.default(permute_65, [1568, 4, 6, 16]);  permute_65 = None
    clone_51: "f32[1568, 4, 6, 16]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
    view_136: "f32[6272, 6, 16]" = torch.ops.aten.reshape.default(clone_51, [6272, 6, 16]);  clone_51 = None
    bmm_12: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_135, view_136)
    view_137: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_12, [1568, 4, 16, 16]);  bmm_12 = None
    mul_60: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_137, 0.408248290463863);  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_6: "f32[1568, 4, 16, 1]" = torch.ops.aten.amax.default(mul_60, [-1], True)
    sub_24: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_60, amax_6);  mul_60 = amax_6 = None
    exp_6: "f32[1568, 4, 16, 16]" = torch.ops.aten.exp.default(sub_24);  sub_24 = None
    sum_7: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_6: "f32[1568, 4, 16, 16]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    alias_6: "f32[1568, 4, 16, 16]" = torch.ops.aten.alias.default(div_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_27: "f32[1568, 4, 16, 16]" = torch.ops.aten.expand.default(div_6, [1568, 4, 16, 16]);  div_6 = None
    view_138: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(expand_27, [6272, 16, 16]);  expand_27 = None
    expand_28: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(permute_64, [1568, 4, 16, 6]);  permute_64 = None
    clone_52: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
    view_139: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_52, [6272, 16, 6]);  clone_52 = None
    bmm_13: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_138, view_139)
    view_140: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_13, [1568, 4, 16, 6]);  bmm_13 = None
    permute_66: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
    clone_53: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_66, memory_format = torch.contiguous_format);  permute_66 = None
    view_141: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_53, [1568, 16, 24]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_142: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_141, [25088, 24]);  view_141 = None
    permute_67: "f32[24, 24]" = torch.ops.aten.permute.default(primals_100, [1, 0]);  primals_100 = None
    addmm_22: "f32[25088, 24]" = torch.ops.aten.addmm.default(primals_101, view_142, permute_67);  primals_101 = None
    view_143: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(addmm_22, [1568, 16, 24]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    add_61: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_48, view_143);  add_48 = view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    clone_54: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_61, memory_format = torch.contiguous_format)
    var_mean_18 = torch.ops.aten.var_mean.correction(clone_54, [2], correction = 0, keepdim = True)
    getitem_50: "f32[1568, 16, 1]" = var_mean_18[0]
    getitem_51: "f32[1568, 16, 1]" = var_mean_18[1];  var_mean_18 = None
    add_62: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
    rsqrt_18: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    sub_25: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_54, getitem_51);  clone_54 = getitem_51 = None
    mul_61: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_18);  sub_25 = None
    mul_62: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_61, primals_102)
    add_63: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_62, primals_103);  mul_62 = primals_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_144: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_63, [25088, 24]);  add_63 = None
    permute_68: "f32[24, 96]" = torch.ops.aten.permute.default(primals_104, [1, 0]);  primals_104 = None
    addmm_23: "f32[25088, 96]" = torch.ops.aten.addmm.default(primals_105, view_144, permute_68);  primals_105 = None
    view_145: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(addmm_23, [1568, 16, 96])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_63: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_145, 0.5)
    mul_64: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_145, 0.7071067811865476);  view_145 = None
    erf_6: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_64);  mul_64 = None
    add_64: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_65: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_63, add_64);  mul_63 = add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_146: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_65, [25088, 96]);  mul_65 = None
    permute_69: "f32[96, 24]" = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
    addmm_24: "f32[25088, 24]" = torch.ops.aten.addmm.default(primals_107, view_146, permute_69);  primals_107 = None
    view_147: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(addmm_24, [1568, 16, 24]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    add_65: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_61, view_147);  add_61 = view_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    slice_16: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_58, 1, 0, 1)
    slice_18: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_58, 1, 1, 9223372036854775807);  add_58 = None
    clone_57: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_65, memory_format = torch.contiguous_format)
    var_mean_19 = torch.ops.aten.var_mean.correction(clone_57, [2], correction = 0, keepdim = True)
    getitem_52: "f32[1568, 16, 1]" = var_mean_19[0]
    getitem_53: "f32[1568, 16, 1]" = var_mean_19[1];  var_mean_19 = None
    add_66: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
    rsqrt_19: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    sub_26: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_57, getitem_53);  clone_57 = getitem_53 = None
    mul_66: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_19);  sub_26 = None
    mul_67: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_66, primals_108)
    add_67: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_67, primals_109);  mul_67 = primals_109 = None
    view_148: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_67, [8, 196, -1]);  add_67 = None
    view_149: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_148, [1568, 384]);  view_148 = None
    permute_70: "f32[384, 384]" = torch.ops.aten.permute.default(primals_110, [1, 0]);  primals_110 = None
    addmm_25: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_111, view_149, permute_70);  primals_111 = None
    view_150: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(addmm_25, [8, 196, 384]);  addmm_25 = None
    add_68: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(slice_18, view_150);  slice_18 = view_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    cat_4: "f32[8, 197, 384]" = torch.ops.aten.cat.default([slice_16, add_68], 1);  slice_16 = add_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    var_mean_20 = torch.ops.aten.var_mean.correction(cat_4, [2], correction = 0, keepdim = True)
    getitem_54: "f32[8, 197, 1]" = var_mean_20[0]
    getitem_55: "f32[8, 197, 1]" = var_mean_20[1];  var_mean_20 = None
    add_69: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
    rsqrt_20: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
    sub_27: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_4, getitem_55)
    mul_68: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_20);  sub_27 = None
    mul_69: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_68, primals_112);  mul_68 = None
    add_70: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_69, primals_113);  mul_69 = primals_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_71: "f32[384, 768]" = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
    view_151: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_70, [1576, 384]);  add_70 = None
    mm_14: "f32[1576, 768]" = torch.ops.aten.mm.default(view_151, permute_71)
    view_152: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_14, [8, 197, 768]);  mm_14 = None
    view_153: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.reshape.default(view_152, [8, 197, 2, 6, 64]);  view_152 = None
    permute_72: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.permute.default(view_153, [2, 0, 3, 1, 4]);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_7 = torch.ops.aten.unbind.int(permute_72);  permute_72 = None
    getitem_56: "f32[8, 6, 197, 64]" = unbind_7[0]
    getitem_57: "f32[8, 6, 197, 64]" = unbind_7[1];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_73: "f32[384, 384]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
    mm_15: "f32[1576, 384]" = torch.ops.aten.mm.default(view_151, permute_73)
    view_155: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_15, [8, 197, 384]);  mm_15 = None
    view_156: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_155, [8, 197, 6, -1]);  view_155 = None
    permute_74: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_75: "f32[8, 6, 64, 197]" = torch.ops.aten.permute.default(getitem_57, [0, 1, 3, 2]);  getitem_57 = None
    expand_29: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(getitem_56, [8, 6, 197, 64]);  getitem_56 = None
    clone_58: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
    view_157: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_58, [48, 197, 64]);  clone_58 = None
    expand_30: "f32[8, 6, 64, 197]" = torch.ops.aten.expand.default(permute_75, [8, 6, 64, 197]);  permute_75 = None
    clone_59: "f32[8, 6, 64, 197]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
    view_158: "f32[48, 64, 197]" = torch.ops.aten.reshape.default(clone_59, [48, 64, 197]);  clone_59 = None
    bmm_14: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_157, view_158)
    view_159: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_14, [8, 6, 197, 197]);  bmm_14 = None
    mul_70: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_159, 0.125);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_7: "f32[8, 6, 197, 1]" = torch.ops.aten.amax.default(mul_70, [-1], True)
    sub_28: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_70, amax_7);  mul_70 = amax_7 = None
    exp_7: "f32[8, 6, 197, 197]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
    sum_8: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_7: "f32[8, 6, 197, 197]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    alias_7: "f32[8, 6, 197, 197]" = torch.ops.aten.alias.default(div_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_31: "f32[8, 6, 197, 197]" = torch.ops.aten.expand.default(div_7, [8, 6, 197, 197]);  div_7 = None
    view_160: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(expand_31, [48, 197, 197]);  expand_31 = None
    expand_32: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(permute_74, [8, 6, 197, 64]);  permute_74 = None
    clone_60: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
    view_161: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_60, [48, 197, 64]);  clone_60 = None
    bmm_15: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_160, view_161)
    view_162: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_15, [8, 6, 197, 64]);  bmm_15 = None
    permute_76: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
    clone_61: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_76, memory_format = torch.contiguous_format);  permute_76 = None
    view_163: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_61, [8, 197, 384]);  clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_164: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_163, [1576, 384]);  view_163 = None
    permute_77: "f32[384, 384]" = torch.ops.aten.permute.default(primals_116, [1, 0]);  primals_116 = None
    addmm_26: "f32[1576, 384]" = torch.ops.aten.addmm.default(primals_117, view_164, permute_77);  primals_117 = None
    view_165: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(addmm_26, [8, 197, 384]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    add_71: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_4, view_165);  view_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    var_mean_21 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
    getitem_58: "f32[8, 197, 1]" = var_mean_21[0]
    getitem_59: "f32[8, 197, 1]" = var_mean_21[1];  var_mean_21 = None
    add_72: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
    rsqrt_21: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
    sub_29: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_71, getitem_59);  getitem_59 = None
    mul_71: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_21);  sub_29 = None
    mul_72: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_71, primals_118)
    add_73: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_72, primals_119);  mul_72 = primals_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_166: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_73, [1576, 384]);  add_73 = None
    permute_78: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
    addmm_27: "f32[1576, 1536]" = torch.ops.aten.addmm.default(primals_121, view_166, permute_78);  primals_121 = None
    view_167: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(addmm_27, [8, 197, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_73: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_167, 0.5)
    mul_74: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_167, 0.7071067811865476);  view_167 = None
    erf_7: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_74);  mul_74 = None
    add_74: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_75: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_73, add_74);  mul_73 = add_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_168: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_75, [1576, 1536]);  mul_75 = None
    permute_79: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
    addmm_28: "f32[1576, 384]" = torch.ops.aten.addmm.default(primals_123, view_168, permute_79);  primals_123 = None
    view_169: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(addmm_28, [8, 197, 384]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    add_75: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_71, view_169);  add_71 = view_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    mul_77: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_66, primals_124)
    add_77: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_77, primals_125);  mul_77 = primals_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_80: "f32[24, 48]" = torch.ops.aten.permute.default(primals_126, [1, 0]);  primals_126 = None
    view_170: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_77, [25088, 24]);  add_77 = None
    mm_16: "f32[25088, 48]" = torch.ops.aten.mm.default(view_170, permute_80)
    view_171: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(mm_16, [1568, 16, 48]);  mm_16 = None
    view_172: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.reshape.default(view_171, [1568, 16, 2, 4, 6]);  view_171 = None
    permute_81: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_172, [2, 0, 3, 1, 4]);  view_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_8 = torch.ops.aten.unbind.int(permute_81);  permute_81 = None
    getitem_62: "f32[1568, 4, 16, 6]" = unbind_8[0]
    getitem_63: "f32[1568, 4, 16, 6]" = unbind_8[1];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_82: "f32[24, 24]" = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
    mm_17: "f32[25088, 24]" = torch.ops.aten.mm.default(view_170, permute_82)
    view_174: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_17, [1568, 16, 24]);  mm_17 = None
    view_175: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_174, [1568, 16, 4, -1]);  view_174 = None
    permute_83: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_175, [0, 2, 1, 3]);  view_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_84: "f32[1568, 4, 6, 16]" = torch.ops.aten.permute.default(getitem_63, [0, 1, 3, 2]);  getitem_63 = None
    expand_33: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(getitem_62, [1568, 4, 16, 6]);  getitem_62 = None
    clone_65: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
    view_176: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_65, [6272, 16, 6]);  clone_65 = None
    expand_34: "f32[1568, 4, 6, 16]" = torch.ops.aten.expand.default(permute_84, [1568, 4, 6, 16]);  permute_84 = None
    clone_66: "f32[1568, 4, 6, 16]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
    view_177: "f32[6272, 6, 16]" = torch.ops.aten.reshape.default(clone_66, [6272, 6, 16]);  clone_66 = None
    bmm_16: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_176, view_177)
    view_178: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_16, [1568, 4, 16, 16]);  bmm_16 = None
    mul_78: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_178, 0.408248290463863);  view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_8: "f32[1568, 4, 16, 1]" = torch.ops.aten.amax.default(mul_78, [-1], True)
    sub_31: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_78, amax_8);  mul_78 = amax_8 = None
    exp_8: "f32[1568, 4, 16, 16]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_9: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_8: "f32[1568, 4, 16, 16]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    alias_8: "f32[1568, 4, 16, 16]" = torch.ops.aten.alias.default(div_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_35: "f32[1568, 4, 16, 16]" = torch.ops.aten.expand.default(div_8, [1568, 4, 16, 16]);  div_8 = None
    view_179: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(expand_35, [6272, 16, 16]);  expand_35 = None
    expand_36: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(permute_83, [1568, 4, 16, 6]);  permute_83 = None
    clone_67: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
    view_180: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_67, [6272, 16, 6]);  clone_67 = None
    bmm_17: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_179, view_180)
    view_181: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_17, [1568, 4, 16, 6]);  bmm_17 = None
    permute_85: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None
    clone_68: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None
    view_182: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_68, [1568, 16, 24]);  clone_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_183: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_182, [25088, 24]);  view_182 = None
    permute_86: "f32[24, 24]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
    addmm_29: "f32[25088, 24]" = torch.ops.aten.addmm.default(primals_129, view_183, permute_86);  primals_129 = None
    view_184: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(addmm_29, [1568, 16, 24]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    add_78: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_65, view_184);  add_65 = view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    clone_69: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_78, memory_format = torch.contiguous_format)
    var_mean_23 = torch.ops.aten.var_mean.correction(clone_69, [2], correction = 0, keepdim = True)
    getitem_64: "f32[1568, 16, 1]" = var_mean_23[0]
    getitem_65: "f32[1568, 16, 1]" = var_mean_23[1];  var_mean_23 = None
    add_79: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
    rsqrt_23: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
    sub_32: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_69, getitem_65);  clone_69 = getitem_65 = None
    mul_79: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_23);  sub_32 = None
    mul_80: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_79, primals_130)
    add_80: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_80, primals_131);  mul_80 = primals_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_185: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_80, [25088, 24]);  add_80 = None
    permute_87: "f32[24, 96]" = torch.ops.aten.permute.default(primals_132, [1, 0]);  primals_132 = None
    addmm_30: "f32[25088, 96]" = torch.ops.aten.addmm.default(primals_133, view_185, permute_87);  primals_133 = None
    view_186: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(addmm_30, [1568, 16, 96])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_81: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_186, 0.5)
    mul_82: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_186, 0.7071067811865476);  view_186 = None
    erf_8: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_82);  mul_82 = None
    add_81: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_83: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_81, add_81);  mul_81 = add_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_187: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_83, [25088, 96]);  mul_83 = None
    permute_88: "f32[96, 24]" = torch.ops.aten.permute.default(primals_134, [1, 0]);  primals_134 = None
    addmm_31: "f32[25088, 24]" = torch.ops.aten.addmm.default(primals_135, view_187, permute_88);  primals_135 = None
    view_188: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(addmm_31, [1568, 16, 24]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    add_82: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_78, view_188);  add_78 = view_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    slice_20: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_75, 1, 0, 1)
    slice_22: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_75, 1, 1, 9223372036854775807);  add_75 = None
    clone_72: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_82, memory_format = torch.contiguous_format)
    var_mean_24 = torch.ops.aten.var_mean.correction(clone_72, [2], correction = 0, keepdim = True)
    getitem_66: "f32[1568, 16, 1]" = var_mean_24[0]
    getitem_67: "f32[1568, 16, 1]" = var_mean_24[1];  var_mean_24 = None
    add_83: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
    rsqrt_24: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
    sub_33: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_72, getitem_67);  clone_72 = getitem_67 = None
    mul_84: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_24);  sub_33 = None
    mul_85: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_84, primals_136)
    add_84: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_85, primals_137);  mul_85 = primals_137 = None
    view_189: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_84, [8, 196, -1]);  add_84 = None
    view_190: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_189, [1568, 384]);  view_189 = None
    permute_89: "f32[384, 384]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
    addmm_32: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_139, view_190, permute_89);  primals_139 = None
    view_191: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(addmm_32, [8, 196, 384]);  addmm_32 = None
    add_85: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(slice_22, view_191);  slice_22 = view_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    cat_5: "f32[8, 197, 384]" = torch.ops.aten.cat.default([slice_20, add_85], 1);  slice_20 = add_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    var_mean_25 = torch.ops.aten.var_mean.correction(cat_5, [2], correction = 0, keepdim = True)
    getitem_68: "f32[8, 197, 1]" = var_mean_25[0]
    getitem_69: "f32[8, 197, 1]" = var_mean_25[1];  var_mean_25 = None
    add_86: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
    rsqrt_25: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
    sub_34: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_5, getitem_69)
    mul_86: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_25);  sub_34 = None
    mul_87: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_86, primals_140);  mul_86 = None
    add_87: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_87, primals_141);  mul_87 = primals_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_90: "f32[384, 768]" = torch.ops.aten.permute.default(primals_142, [1, 0]);  primals_142 = None
    view_192: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_87, [1576, 384]);  add_87 = None
    mm_18: "f32[1576, 768]" = torch.ops.aten.mm.default(view_192, permute_90)
    view_193: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_18, [8, 197, 768]);  mm_18 = None
    view_194: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.reshape.default(view_193, [8, 197, 2, 6, 64]);  view_193 = None
    permute_91: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.permute.default(view_194, [2, 0, 3, 1, 4]);  view_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_9 = torch.ops.aten.unbind.int(permute_91);  permute_91 = None
    getitem_70: "f32[8, 6, 197, 64]" = unbind_9[0]
    getitem_71: "f32[8, 6, 197, 64]" = unbind_9[1];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_92: "f32[384, 384]" = torch.ops.aten.permute.default(primals_143, [1, 0]);  primals_143 = None
    mm_19: "f32[1576, 384]" = torch.ops.aten.mm.default(view_192, permute_92)
    view_196: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_19, [8, 197, 384]);  mm_19 = None
    view_197: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_196, [8, 197, 6, -1]);  view_196 = None
    permute_93: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_197, [0, 2, 1, 3]);  view_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_94: "f32[8, 6, 64, 197]" = torch.ops.aten.permute.default(getitem_71, [0, 1, 3, 2]);  getitem_71 = None
    expand_37: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(getitem_70, [8, 6, 197, 64]);  getitem_70 = None
    clone_73: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
    view_198: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_73, [48, 197, 64]);  clone_73 = None
    expand_38: "f32[8, 6, 64, 197]" = torch.ops.aten.expand.default(permute_94, [8, 6, 64, 197]);  permute_94 = None
    clone_74: "f32[8, 6, 64, 197]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
    view_199: "f32[48, 64, 197]" = torch.ops.aten.reshape.default(clone_74, [48, 64, 197]);  clone_74 = None
    bmm_18: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_198, view_199)
    view_200: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_18, [8, 6, 197, 197]);  bmm_18 = None
    mul_88: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_200, 0.125);  view_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_9: "f32[8, 6, 197, 1]" = torch.ops.aten.amax.default(mul_88, [-1], True)
    sub_35: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_88, amax_9);  mul_88 = amax_9 = None
    exp_9: "f32[8, 6, 197, 197]" = torch.ops.aten.exp.default(sub_35);  sub_35 = None
    sum_10: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_9: "f32[8, 6, 197, 197]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    alias_9: "f32[8, 6, 197, 197]" = torch.ops.aten.alias.default(div_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_39: "f32[8, 6, 197, 197]" = torch.ops.aten.expand.default(div_9, [8, 6, 197, 197]);  div_9 = None
    view_201: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(expand_39, [48, 197, 197]);  expand_39 = None
    expand_40: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(permute_93, [8, 6, 197, 64]);  permute_93 = None
    clone_75: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
    view_202: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_75, [48, 197, 64]);  clone_75 = None
    bmm_19: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_201, view_202)
    view_203: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_19, [8, 6, 197, 64]);  bmm_19 = None
    permute_95: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_203, [0, 2, 1, 3]);  view_203 = None
    clone_76: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    view_204: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_76, [8, 197, 384]);  clone_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_205: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_204, [1576, 384]);  view_204 = None
    permute_96: "f32[384, 384]" = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
    addmm_33: "f32[1576, 384]" = torch.ops.aten.addmm.default(primals_145, view_205, permute_96);  primals_145 = None
    view_206: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(addmm_33, [8, 197, 384]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    add_88: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_5, view_206);  view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    var_mean_26 = torch.ops.aten.var_mean.correction(add_88, [2], correction = 0, keepdim = True)
    getitem_72: "f32[8, 197, 1]" = var_mean_26[0]
    getitem_73: "f32[8, 197, 1]" = var_mean_26[1];  var_mean_26 = None
    add_89: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
    rsqrt_26: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
    sub_36: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_88, getitem_73);  getitem_73 = None
    mul_89: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_26);  sub_36 = None
    mul_90: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_89, primals_146)
    add_90: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_90, primals_147);  mul_90 = primals_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_207: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_90, [1576, 384]);  add_90 = None
    permute_97: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_148, [1, 0]);  primals_148 = None
    addmm_34: "f32[1576, 1536]" = torch.ops.aten.addmm.default(primals_149, view_207, permute_97);  primals_149 = None
    view_208: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(addmm_34, [8, 197, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_91: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_208, 0.5)
    mul_92: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_208, 0.7071067811865476);  view_208 = None
    erf_9: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_92);  mul_92 = None
    add_91: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_93: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_91, add_91);  mul_91 = add_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_209: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_93, [1576, 1536]);  mul_93 = None
    permute_98: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
    addmm_35: "f32[1576, 384]" = torch.ops.aten.addmm.default(primals_151, view_209, permute_98);  primals_151 = None
    view_210: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(addmm_35, [8, 197, 384]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    add_92: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_88, view_210);  add_88 = view_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    mul_95: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_84, primals_152)
    add_94: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_95, primals_153);  mul_95 = primals_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_99: "f32[24, 48]" = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
    view_211: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_94, [25088, 24]);  add_94 = None
    mm_20: "f32[25088, 48]" = torch.ops.aten.mm.default(view_211, permute_99)
    view_212: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(mm_20, [1568, 16, 48]);  mm_20 = None
    view_213: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.reshape.default(view_212, [1568, 16, 2, 4, 6]);  view_212 = None
    permute_100: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_213, [2, 0, 3, 1, 4]);  view_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_10 = torch.ops.aten.unbind.int(permute_100);  permute_100 = None
    getitem_76: "f32[1568, 4, 16, 6]" = unbind_10[0]
    getitem_77: "f32[1568, 4, 16, 6]" = unbind_10[1];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_101: "f32[24, 24]" = torch.ops.aten.permute.default(primals_155, [1, 0]);  primals_155 = None
    mm_21: "f32[25088, 24]" = torch.ops.aten.mm.default(view_211, permute_101)
    view_215: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_21, [1568, 16, 24]);  mm_21 = None
    view_216: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_215, [1568, 16, 4, -1]);  view_215 = None
    permute_102: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_216, [0, 2, 1, 3]);  view_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_103: "f32[1568, 4, 6, 16]" = torch.ops.aten.permute.default(getitem_77, [0, 1, 3, 2]);  getitem_77 = None
    expand_41: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(getitem_76, [1568, 4, 16, 6]);  getitem_76 = None
    clone_80: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
    view_217: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_80, [6272, 16, 6]);  clone_80 = None
    expand_42: "f32[1568, 4, 6, 16]" = torch.ops.aten.expand.default(permute_103, [1568, 4, 6, 16]);  permute_103 = None
    clone_81: "f32[1568, 4, 6, 16]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
    view_218: "f32[6272, 6, 16]" = torch.ops.aten.reshape.default(clone_81, [6272, 6, 16]);  clone_81 = None
    bmm_20: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_217, view_218)
    view_219: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_20, [1568, 4, 16, 16]);  bmm_20 = None
    mul_96: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_219, 0.408248290463863);  view_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_10: "f32[1568, 4, 16, 1]" = torch.ops.aten.amax.default(mul_96, [-1], True)
    sub_38: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_96, amax_10);  mul_96 = amax_10 = None
    exp_10: "f32[1568, 4, 16, 16]" = torch.ops.aten.exp.default(sub_38);  sub_38 = None
    sum_11: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_10: "f32[1568, 4, 16, 16]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    alias_10: "f32[1568, 4, 16, 16]" = torch.ops.aten.alias.default(div_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_43: "f32[1568, 4, 16, 16]" = torch.ops.aten.expand.default(div_10, [1568, 4, 16, 16]);  div_10 = None
    view_220: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(expand_43, [6272, 16, 16]);  expand_43 = None
    expand_44: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(permute_102, [1568, 4, 16, 6]);  permute_102 = None
    clone_82: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
    view_221: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_82, [6272, 16, 6]);  clone_82 = None
    bmm_21: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_220, view_221)
    view_222: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_21, [1568, 4, 16, 6]);  bmm_21 = None
    permute_104: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None
    clone_83: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
    view_223: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_83, [1568, 16, 24]);  clone_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_224: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_223, [25088, 24]);  view_223 = None
    permute_105: "f32[24, 24]" = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
    addmm_36: "f32[25088, 24]" = torch.ops.aten.addmm.default(primals_157, view_224, permute_105);  primals_157 = None
    view_225: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(addmm_36, [1568, 16, 24]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    add_95: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_82, view_225);  add_82 = view_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    clone_84: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_95, memory_format = torch.contiguous_format)
    var_mean_28 = torch.ops.aten.var_mean.correction(clone_84, [2], correction = 0, keepdim = True)
    getitem_78: "f32[1568, 16, 1]" = var_mean_28[0]
    getitem_79: "f32[1568, 16, 1]" = var_mean_28[1];  var_mean_28 = None
    add_96: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
    rsqrt_28: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
    sub_39: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_84, getitem_79);  clone_84 = getitem_79 = None
    mul_97: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_28);  sub_39 = None
    mul_98: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_97, primals_158)
    add_97: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_98, primals_159);  mul_98 = primals_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_226: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_97, [25088, 24]);  add_97 = None
    permute_106: "f32[24, 96]" = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None
    addmm_37: "f32[25088, 96]" = torch.ops.aten.addmm.default(primals_161, view_226, permute_106);  primals_161 = None
    view_227: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(addmm_37, [1568, 16, 96])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_99: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_227, 0.5)
    mul_100: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_227, 0.7071067811865476);  view_227 = None
    erf_10: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_100);  mul_100 = None
    add_98: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_101: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_99, add_98);  mul_99 = add_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_228: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_101, [25088, 96]);  mul_101 = None
    permute_107: "f32[96, 24]" = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
    addmm_38: "f32[25088, 24]" = torch.ops.aten.addmm.default(primals_163, view_228, permute_107);  primals_163 = None
    view_229: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(addmm_38, [1568, 16, 24]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    add_99: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_95, view_229);  add_95 = view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    slice_24: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_92, 1, 0, 1)
    slice_26: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_92, 1, 1, 9223372036854775807);  add_92 = None
    clone_87: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_99, memory_format = torch.contiguous_format)
    var_mean_29 = torch.ops.aten.var_mean.correction(clone_87, [2], correction = 0, keepdim = True)
    getitem_80: "f32[1568, 16, 1]" = var_mean_29[0]
    getitem_81: "f32[1568, 16, 1]" = var_mean_29[1];  var_mean_29 = None
    add_100: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
    rsqrt_29: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
    sub_40: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_87, getitem_81);  clone_87 = getitem_81 = None
    mul_102: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_29);  sub_40 = None
    mul_103: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_102, primals_164)
    add_101: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_103, primals_165);  mul_103 = primals_165 = None
    view_230: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_101, [8, 196, -1]);  add_101 = None
    view_231: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_230, [1568, 384]);  view_230 = None
    permute_108: "f32[384, 384]" = torch.ops.aten.permute.default(primals_166, [1, 0]);  primals_166 = None
    addmm_39: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_167, view_231, permute_108);  primals_167 = None
    view_232: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(addmm_39, [8, 196, 384]);  addmm_39 = None
    add_102: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(slice_26, view_232);  slice_26 = view_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    cat_6: "f32[8, 197, 384]" = torch.ops.aten.cat.default([slice_24, add_102], 1);  slice_24 = add_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    var_mean_30 = torch.ops.aten.var_mean.correction(cat_6, [2], correction = 0, keepdim = True)
    getitem_82: "f32[8, 197, 1]" = var_mean_30[0]
    getitem_83: "f32[8, 197, 1]" = var_mean_30[1];  var_mean_30 = None
    add_103: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
    rsqrt_30: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
    sub_41: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_6, getitem_83)
    mul_104: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_30);  sub_41 = None
    mul_105: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_104, primals_168);  mul_104 = None
    add_104: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_105, primals_169);  mul_105 = primals_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_109: "f32[384, 768]" = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
    view_233: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_104, [1576, 384]);  add_104 = None
    mm_22: "f32[1576, 768]" = torch.ops.aten.mm.default(view_233, permute_109)
    view_234: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_22, [8, 197, 768]);  mm_22 = None
    view_235: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.reshape.default(view_234, [8, 197, 2, 6, 64]);  view_234 = None
    permute_110: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.permute.default(view_235, [2, 0, 3, 1, 4]);  view_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_11 = torch.ops.aten.unbind.int(permute_110);  permute_110 = None
    getitem_84: "f32[8, 6, 197, 64]" = unbind_11[0]
    getitem_85: "f32[8, 6, 197, 64]" = unbind_11[1];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_111: "f32[384, 384]" = torch.ops.aten.permute.default(primals_171, [1, 0]);  primals_171 = None
    mm_23: "f32[1576, 384]" = torch.ops.aten.mm.default(view_233, permute_111)
    view_237: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_23, [8, 197, 384]);  mm_23 = None
    view_238: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_237, [8, 197, 6, -1]);  view_237 = None
    permute_112: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_238, [0, 2, 1, 3]);  view_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_113: "f32[8, 6, 64, 197]" = torch.ops.aten.permute.default(getitem_85, [0, 1, 3, 2]);  getitem_85 = None
    expand_45: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(getitem_84, [8, 6, 197, 64]);  getitem_84 = None
    clone_88: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
    view_239: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_88, [48, 197, 64]);  clone_88 = None
    expand_46: "f32[8, 6, 64, 197]" = torch.ops.aten.expand.default(permute_113, [8, 6, 64, 197]);  permute_113 = None
    clone_89: "f32[8, 6, 64, 197]" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
    view_240: "f32[48, 64, 197]" = torch.ops.aten.reshape.default(clone_89, [48, 64, 197]);  clone_89 = None
    bmm_22: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_239, view_240)
    view_241: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_22, [8, 6, 197, 197]);  bmm_22 = None
    mul_106: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_241, 0.125);  view_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_11: "f32[8, 6, 197, 1]" = torch.ops.aten.amax.default(mul_106, [-1], True)
    sub_42: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_106, amax_11);  mul_106 = amax_11 = None
    exp_11: "f32[8, 6, 197, 197]" = torch.ops.aten.exp.default(sub_42);  sub_42 = None
    sum_12: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_11: "f32[8, 6, 197, 197]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    alias_11: "f32[8, 6, 197, 197]" = torch.ops.aten.alias.default(div_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_47: "f32[8, 6, 197, 197]" = torch.ops.aten.expand.default(div_11, [8, 6, 197, 197]);  div_11 = None
    view_242: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(expand_47, [48, 197, 197]);  expand_47 = None
    expand_48: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(permute_112, [8, 6, 197, 64]);  permute_112 = None
    clone_90: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
    view_243: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_90, [48, 197, 64]);  clone_90 = None
    bmm_23: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_242, view_243)
    view_244: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_23, [8, 6, 197, 64]);  bmm_23 = None
    permute_114: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None
    clone_91: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
    view_245: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_91, [8, 197, 384]);  clone_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_246: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_245, [1576, 384]);  view_245 = None
    permute_115: "f32[384, 384]" = torch.ops.aten.permute.default(primals_172, [1, 0]);  primals_172 = None
    addmm_40: "f32[1576, 384]" = torch.ops.aten.addmm.default(primals_173, view_246, permute_115);  primals_173 = None
    view_247: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(addmm_40, [8, 197, 384]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    add_105: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_6, view_247);  view_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    var_mean_31 = torch.ops.aten.var_mean.correction(add_105, [2], correction = 0, keepdim = True)
    getitem_86: "f32[8, 197, 1]" = var_mean_31[0]
    getitem_87: "f32[8, 197, 1]" = var_mean_31[1];  var_mean_31 = None
    add_106: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
    rsqrt_31: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
    sub_43: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_105, getitem_87);  getitem_87 = None
    mul_107: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_31);  sub_43 = None
    mul_108: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_107, primals_174)
    add_107: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_108, primals_175);  mul_108 = primals_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_248: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_107, [1576, 384]);  add_107 = None
    permute_116: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_176, [1, 0]);  primals_176 = None
    addmm_41: "f32[1576, 1536]" = torch.ops.aten.addmm.default(primals_177, view_248, permute_116);  primals_177 = None
    view_249: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(addmm_41, [8, 197, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_109: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_249, 0.5)
    mul_110: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_249, 0.7071067811865476);  view_249 = None
    erf_11: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_110);  mul_110 = None
    add_108: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_111: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_109, add_108);  mul_109 = add_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_250: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_111, [1576, 1536]);  mul_111 = None
    permute_117: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_178, [1, 0]);  primals_178 = None
    addmm_42: "f32[1576, 384]" = torch.ops.aten.addmm.default(primals_179, view_250, permute_117);  primals_179 = None
    view_251: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(addmm_42, [8, 197, 384]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    add_109: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_105, view_251);  add_105 = view_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    mul_113: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_102, primals_180)
    add_111: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_113, primals_181);  mul_113 = primals_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_118: "f32[24, 48]" = torch.ops.aten.permute.default(primals_182, [1, 0]);  primals_182 = None
    view_252: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_111, [25088, 24]);  add_111 = None
    mm_24: "f32[25088, 48]" = torch.ops.aten.mm.default(view_252, permute_118)
    view_253: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(mm_24, [1568, 16, 48]);  mm_24 = None
    view_254: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.reshape.default(view_253, [1568, 16, 2, 4, 6]);  view_253 = None
    permute_119: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_254, [2, 0, 3, 1, 4]);  view_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_12 = torch.ops.aten.unbind.int(permute_119);  permute_119 = None
    getitem_90: "f32[1568, 4, 16, 6]" = unbind_12[0]
    getitem_91: "f32[1568, 4, 16, 6]" = unbind_12[1];  unbind_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_120: "f32[24, 24]" = torch.ops.aten.permute.default(primals_183, [1, 0]);  primals_183 = None
    mm_25: "f32[25088, 24]" = torch.ops.aten.mm.default(view_252, permute_120)
    view_256: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_25, [1568, 16, 24]);  mm_25 = None
    view_257: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_256, [1568, 16, 4, -1]);  view_256 = None
    permute_121: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_257, [0, 2, 1, 3]);  view_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_122: "f32[1568, 4, 6, 16]" = torch.ops.aten.permute.default(getitem_91, [0, 1, 3, 2]);  getitem_91 = None
    expand_49: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(getitem_90, [1568, 4, 16, 6]);  getitem_90 = None
    clone_95: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
    view_258: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_95, [6272, 16, 6]);  clone_95 = None
    expand_50: "f32[1568, 4, 6, 16]" = torch.ops.aten.expand.default(permute_122, [1568, 4, 6, 16]);  permute_122 = None
    clone_96: "f32[1568, 4, 6, 16]" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
    view_259: "f32[6272, 6, 16]" = torch.ops.aten.reshape.default(clone_96, [6272, 6, 16]);  clone_96 = None
    bmm_24: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_258, view_259)
    view_260: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_24, [1568, 4, 16, 16]);  bmm_24 = None
    mul_114: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_260, 0.408248290463863);  view_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_12: "f32[1568, 4, 16, 1]" = torch.ops.aten.amax.default(mul_114, [-1], True)
    sub_45: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_114, amax_12);  mul_114 = amax_12 = None
    exp_12: "f32[1568, 4, 16, 16]" = torch.ops.aten.exp.default(sub_45);  sub_45 = None
    sum_13: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_12: "f32[1568, 4, 16, 16]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    alias_12: "f32[1568, 4, 16, 16]" = torch.ops.aten.alias.default(div_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_51: "f32[1568, 4, 16, 16]" = torch.ops.aten.expand.default(div_12, [1568, 4, 16, 16]);  div_12 = None
    view_261: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(expand_51, [6272, 16, 16]);  expand_51 = None
    expand_52: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(permute_121, [1568, 4, 16, 6]);  permute_121 = None
    clone_97: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
    view_262: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_97, [6272, 16, 6]);  clone_97 = None
    bmm_25: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_261, view_262)
    view_263: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_25, [1568, 4, 16, 6]);  bmm_25 = None
    permute_123: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_263, [0, 2, 1, 3]);  view_263 = None
    clone_98: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_123, memory_format = torch.contiguous_format);  permute_123 = None
    view_264: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_98, [1568, 16, 24]);  clone_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_265: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_264, [25088, 24]);  view_264 = None
    permute_124: "f32[24, 24]" = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None
    addmm_43: "f32[25088, 24]" = torch.ops.aten.addmm.default(primals_185, view_265, permute_124);  primals_185 = None
    view_266: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(addmm_43, [1568, 16, 24]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    add_112: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_99, view_266);  add_99 = view_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    clone_99: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_112, memory_format = torch.contiguous_format)
    var_mean_33 = torch.ops.aten.var_mean.correction(clone_99, [2], correction = 0, keepdim = True)
    getitem_92: "f32[1568, 16, 1]" = var_mean_33[0]
    getitem_93: "f32[1568, 16, 1]" = var_mean_33[1];  var_mean_33 = None
    add_113: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
    rsqrt_33: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
    sub_46: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_99, getitem_93);  clone_99 = getitem_93 = None
    mul_115: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_33);  sub_46 = None
    mul_116: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_115, primals_186)
    add_114: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_116, primals_187);  mul_116 = primals_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_267: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_114, [25088, 24]);  add_114 = None
    permute_125: "f32[24, 96]" = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
    addmm_44: "f32[25088, 96]" = torch.ops.aten.addmm.default(primals_189, view_267, permute_125);  primals_189 = None
    view_268: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(addmm_44, [1568, 16, 96])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_117: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_268, 0.5)
    mul_118: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_268, 0.7071067811865476);  view_268 = None
    erf_12: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_118);  mul_118 = None
    add_115: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_119: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_117, add_115);  mul_117 = add_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_269: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_119, [25088, 96]);  mul_119 = None
    permute_126: "f32[96, 24]" = torch.ops.aten.permute.default(primals_190, [1, 0]);  primals_190 = None
    addmm_45: "f32[25088, 24]" = torch.ops.aten.addmm.default(primals_191, view_269, permute_126);  primals_191 = None
    view_270: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(addmm_45, [1568, 16, 24]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    add_116: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_112, view_270);  add_112 = view_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    slice_28: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_109, 1, 0, 1)
    slice_30: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_109, 1, 1, 9223372036854775807);  add_109 = None
    clone_102: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_116, memory_format = torch.contiguous_format)
    var_mean_34 = torch.ops.aten.var_mean.correction(clone_102, [2], correction = 0, keepdim = True)
    getitem_94: "f32[1568, 16, 1]" = var_mean_34[0]
    getitem_95: "f32[1568, 16, 1]" = var_mean_34[1];  var_mean_34 = None
    add_117: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
    rsqrt_34: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
    sub_47: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_102, getitem_95);  clone_102 = getitem_95 = None
    mul_120: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_34);  sub_47 = None
    mul_121: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_120, primals_192)
    add_118: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_121, primals_193);  mul_121 = primals_193 = None
    view_271: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_118, [8, 196, -1]);  add_118 = None
    view_272: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_271, [1568, 384]);  view_271 = None
    permute_127: "f32[384, 384]" = torch.ops.aten.permute.default(primals_194, [1, 0]);  primals_194 = None
    addmm_46: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_195, view_272, permute_127);  primals_195 = None
    view_273: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(addmm_46, [8, 196, 384]);  addmm_46 = None
    add_119: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(slice_30, view_273);  slice_30 = view_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    cat_7: "f32[8, 197, 384]" = torch.ops.aten.cat.default([slice_28, add_119], 1);  slice_28 = add_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    var_mean_35 = torch.ops.aten.var_mean.correction(cat_7, [2], correction = 0, keepdim = True)
    getitem_96: "f32[8, 197, 1]" = var_mean_35[0]
    getitem_97: "f32[8, 197, 1]" = var_mean_35[1];  var_mean_35 = None
    add_120: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
    rsqrt_35: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
    sub_48: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_7, getitem_97)
    mul_122: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_35);  sub_48 = None
    mul_123: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_122, primals_196);  mul_122 = None
    add_121: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_123, primals_197);  mul_123 = primals_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_128: "f32[384, 768]" = torch.ops.aten.permute.default(primals_198, [1, 0]);  primals_198 = None
    view_274: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_121, [1576, 384]);  add_121 = None
    mm_26: "f32[1576, 768]" = torch.ops.aten.mm.default(view_274, permute_128)
    view_275: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_26, [8, 197, 768]);  mm_26 = None
    view_276: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.reshape.default(view_275, [8, 197, 2, 6, 64]);  view_275 = None
    permute_129: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.permute.default(view_276, [2, 0, 3, 1, 4]);  view_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_13 = torch.ops.aten.unbind.int(permute_129);  permute_129 = None
    getitem_98: "f32[8, 6, 197, 64]" = unbind_13[0]
    getitem_99: "f32[8, 6, 197, 64]" = unbind_13[1];  unbind_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_130: "f32[384, 384]" = torch.ops.aten.permute.default(primals_199, [1, 0]);  primals_199 = None
    mm_27: "f32[1576, 384]" = torch.ops.aten.mm.default(view_274, permute_130)
    view_278: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_27, [8, 197, 384]);  mm_27 = None
    view_279: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_278, [8, 197, 6, -1]);  view_278 = None
    permute_131: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_279, [0, 2, 1, 3]);  view_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_132: "f32[8, 6, 64, 197]" = torch.ops.aten.permute.default(getitem_99, [0, 1, 3, 2]);  getitem_99 = None
    expand_53: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(getitem_98, [8, 6, 197, 64]);  getitem_98 = None
    clone_103: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
    view_280: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_103, [48, 197, 64]);  clone_103 = None
    expand_54: "f32[8, 6, 64, 197]" = torch.ops.aten.expand.default(permute_132, [8, 6, 64, 197]);  permute_132 = None
    clone_104: "f32[8, 6, 64, 197]" = torch.ops.aten.clone.default(expand_54, memory_format = torch.contiguous_format);  expand_54 = None
    view_281: "f32[48, 64, 197]" = torch.ops.aten.reshape.default(clone_104, [48, 64, 197]);  clone_104 = None
    bmm_26: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_280, view_281)
    view_282: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_26, [8, 6, 197, 197]);  bmm_26 = None
    mul_124: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_282, 0.125);  view_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_13: "f32[8, 6, 197, 1]" = torch.ops.aten.amax.default(mul_124, [-1], True)
    sub_49: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_124, amax_13);  mul_124 = amax_13 = None
    exp_13: "f32[8, 6, 197, 197]" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
    sum_14: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_13: "f32[8, 6, 197, 197]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
    alias_13: "f32[8, 6, 197, 197]" = torch.ops.aten.alias.default(div_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_55: "f32[8, 6, 197, 197]" = torch.ops.aten.expand.default(div_13, [8, 6, 197, 197]);  div_13 = None
    view_283: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(expand_55, [48, 197, 197]);  expand_55 = None
    expand_56: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(permute_131, [8, 6, 197, 64]);  permute_131 = None
    clone_105: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
    view_284: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_105, [48, 197, 64]);  clone_105 = None
    bmm_27: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_283, view_284)
    view_285: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_27, [8, 6, 197, 64]);  bmm_27 = None
    permute_133: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_285, [0, 2, 1, 3]);  view_285 = None
    clone_106: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_133, memory_format = torch.contiguous_format);  permute_133 = None
    view_286: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_106, [8, 197, 384]);  clone_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_287: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_286, [1576, 384]);  view_286 = None
    permute_134: "f32[384, 384]" = torch.ops.aten.permute.default(primals_200, [1, 0]);  primals_200 = None
    addmm_47: "f32[1576, 384]" = torch.ops.aten.addmm.default(primals_201, view_287, permute_134);  primals_201 = None
    view_288: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(addmm_47, [8, 197, 384]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    add_122: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_7, view_288);  view_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    var_mean_36 = torch.ops.aten.var_mean.correction(add_122, [2], correction = 0, keepdim = True)
    getitem_100: "f32[8, 197, 1]" = var_mean_36[0]
    getitem_101: "f32[8, 197, 1]" = var_mean_36[1];  var_mean_36 = None
    add_123: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05);  getitem_100 = None
    rsqrt_36: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
    sub_50: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_122, getitem_101);  getitem_101 = None
    mul_125: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_36);  sub_50 = None
    mul_126: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_125, primals_202)
    add_124: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_126, primals_203);  mul_126 = primals_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_289: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_124, [1576, 384]);  add_124 = None
    permute_135: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_204, [1, 0]);  primals_204 = None
    addmm_48: "f32[1576, 1536]" = torch.ops.aten.addmm.default(primals_205, view_289, permute_135);  primals_205 = None
    view_290: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(addmm_48, [8, 197, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_127: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_290, 0.5)
    mul_128: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_290, 0.7071067811865476);  view_290 = None
    erf_13: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_128);  mul_128 = None
    add_125: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_129: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_127, add_125);  mul_127 = add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_291: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_129, [1576, 1536]);  mul_129 = None
    permute_136: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_206, [1, 0]);  primals_206 = None
    addmm_49: "f32[1576, 384]" = torch.ops.aten.addmm.default(primals_207, view_291, permute_136);  primals_207 = None
    view_292: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(addmm_49, [8, 197, 384]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    add_126: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_122, view_292);  add_122 = view_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    mul_131: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_120, primals_208)
    add_128: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_131, primals_209);  mul_131 = primals_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_137: "f32[24, 48]" = torch.ops.aten.permute.default(primals_210, [1, 0]);  primals_210 = None
    view_293: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_128, [25088, 24]);  add_128 = None
    mm_28: "f32[25088, 48]" = torch.ops.aten.mm.default(view_293, permute_137)
    view_294: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(mm_28, [1568, 16, 48]);  mm_28 = None
    view_295: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.reshape.default(view_294, [1568, 16, 2, 4, 6]);  view_294 = None
    permute_138: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_295, [2, 0, 3, 1, 4]);  view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_14 = torch.ops.aten.unbind.int(permute_138);  permute_138 = None
    getitem_104: "f32[1568, 4, 16, 6]" = unbind_14[0]
    getitem_105: "f32[1568, 4, 16, 6]" = unbind_14[1];  unbind_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_139: "f32[24, 24]" = torch.ops.aten.permute.default(primals_211, [1, 0]);  primals_211 = None
    mm_29: "f32[25088, 24]" = torch.ops.aten.mm.default(view_293, permute_139)
    view_297: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_29, [1568, 16, 24]);  mm_29 = None
    view_298: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_297, [1568, 16, 4, -1]);  view_297 = None
    permute_140: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_298, [0, 2, 1, 3]);  view_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_141: "f32[1568, 4, 6, 16]" = torch.ops.aten.permute.default(getitem_105, [0, 1, 3, 2]);  getitem_105 = None
    expand_57: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(getitem_104, [1568, 4, 16, 6]);  getitem_104 = None
    clone_110: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
    view_299: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_110, [6272, 16, 6]);  clone_110 = None
    expand_58: "f32[1568, 4, 6, 16]" = torch.ops.aten.expand.default(permute_141, [1568, 4, 6, 16]);  permute_141 = None
    clone_111: "f32[1568, 4, 6, 16]" = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
    view_300: "f32[6272, 6, 16]" = torch.ops.aten.reshape.default(clone_111, [6272, 6, 16]);  clone_111 = None
    bmm_28: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_299, view_300)
    view_301: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_28, [1568, 4, 16, 16]);  bmm_28 = None
    mul_132: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_301, 0.408248290463863);  view_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_14: "f32[1568, 4, 16, 1]" = torch.ops.aten.amax.default(mul_132, [-1], True)
    sub_52: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_132, amax_14);  mul_132 = amax_14 = None
    exp_14: "f32[1568, 4, 16, 16]" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
    sum_15: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_14: "f32[1568, 4, 16, 16]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
    alias_14: "f32[1568, 4, 16, 16]" = torch.ops.aten.alias.default(div_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_59: "f32[1568, 4, 16, 16]" = torch.ops.aten.expand.default(div_14, [1568, 4, 16, 16]);  div_14 = None
    view_302: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(expand_59, [6272, 16, 16]);  expand_59 = None
    expand_60: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(permute_140, [1568, 4, 16, 6]);  permute_140 = None
    clone_112: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
    view_303: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_112, [6272, 16, 6]);  clone_112 = None
    bmm_29: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_302, view_303)
    view_304: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_29, [1568, 4, 16, 6]);  bmm_29 = None
    permute_142: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_304, [0, 2, 1, 3]);  view_304 = None
    clone_113: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_142, memory_format = torch.contiguous_format);  permute_142 = None
    view_305: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_113, [1568, 16, 24]);  clone_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_306: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_305, [25088, 24]);  view_305 = None
    permute_143: "f32[24, 24]" = torch.ops.aten.permute.default(primals_212, [1, 0]);  primals_212 = None
    addmm_50: "f32[25088, 24]" = torch.ops.aten.addmm.default(primals_213, view_306, permute_143);  primals_213 = None
    view_307: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(addmm_50, [1568, 16, 24]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    add_129: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_116, view_307);  add_116 = view_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    clone_114: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_129, memory_format = torch.contiguous_format)
    var_mean_38 = torch.ops.aten.var_mean.correction(clone_114, [2], correction = 0, keepdim = True)
    getitem_106: "f32[1568, 16, 1]" = var_mean_38[0]
    getitem_107: "f32[1568, 16, 1]" = var_mean_38[1];  var_mean_38 = None
    add_130: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05);  getitem_106 = None
    rsqrt_38: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
    sub_53: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_114, getitem_107);  clone_114 = getitem_107 = None
    mul_133: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_38);  sub_53 = None
    mul_134: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_133, primals_214)
    add_131: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_134, primals_215);  mul_134 = primals_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_308: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_131, [25088, 24]);  add_131 = None
    permute_144: "f32[24, 96]" = torch.ops.aten.permute.default(primals_216, [1, 0]);  primals_216 = None
    addmm_51: "f32[25088, 96]" = torch.ops.aten.addmm.default(primals_217, view_308, permute_144);  primals_217 = None
    view_309: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(addmm_51, [1568, 16, 96])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_135: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_309, 0.5)
    mul_136: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_309, 0.7071067811865476);  view_309 = None
    erf_14: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_136);  mul_136 = None
    add_132: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_137: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_135, add_132);  mul_135 = add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_310: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_137, [25088, 96]);  mul_137 = None
    permute_145: "f32[96, 24]" = torch.ops.aten.permute.default(primals_218, [1, 0]);  primals_218 = None
    addmm_52: "f32[25088, 24]" = torch.ops.aten.addmm.default(primals_219, view_310, permute_145);  primals_219 = None
    view_311: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(addmm_52, [1568, 16, 24]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    add_133: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_129, view_311);  add_129 = view_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    slice_32: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_126, 1, 0, 1)
    slice_34: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_126, 1, 1, 9223372036854775807);  add_126 = None
    clone_117: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_133, memory_format = torch.contiguous_format)
    var_mean_39 = torch.ops.aten.var_mean.correction(clone_117, [2], correction = 0, keepdim = True)
    getitem_108: "f32[1568, 16, 1]" = var_mean_39[0]
    getitem_109: "f32[1568, 16, 1]" = var_mean_39[1];  var_mean_39 = None
    add_134: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
    rsqrt_39: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
    sub_54: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_117, getitem_109);  clone_117 = getitem_109 = None
    mul_138: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_39);  sub_54 = None
    mul_139: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_138, primals_220)
    add_135: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_139, primals_221);  mul_139 = primals_221 = None
    view_312: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_135, [8, 196, -1]);  add_135 = None
    view_313: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_312, [1568, 384]);  view_312 = None
    permute_146: "f32[384, 384]" = torch.ops.aten.permute.default(primals_222, [1, 0]);  primals_222 = None
    addmm_53: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_223, view_313, permute_146);  primals_223 = None
    view_314: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(addmm_53, [8, 196, 384]);  addmm_53 = None
    add_136: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(slice_34, view_314);  slice_34 = view_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    cat_8: "f32[8, 197, 384]" = torch.ops.aten.cat.default([slice_32, add_136], 1);  slice_32 = add_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    var_mean_40 = torch.ops.aten.var_mean.correction(cat_8, [2], correction = 0, keepdim = True)
    getitem_110: "f32[8, 197, 1]" = var_mean_40[0]
    getitem_111: "f32[8, 197, 1]" = var_mean_40[1];  var_mean_40 = None
    add_137: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
    rsqrt_40: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
    sub_55: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_8, getitem_111)
    mul_140: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_40);  sub_55 = None
    mul_141: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_140, primals_224);  mul_140 = None
    add_138: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_141, primals_225);  mul_141 = primals_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_147: "f32[384, 768]" = torch.ops.aten.permute.default(primals_226, [1, 0]);  primals_226 = None
    view_315: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_138, [1576, 384]);  add_138 = None
    mm_30: "f32[1576, 768]" = torch.ops.aten.mm.default(view_315, permute_147)
    view_316: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_30, [8, 197, 768]);  mm_30 = None
    view_317: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.reshape.default(view_316, [8, 197, 2, 6, 64]);  view_316 = None
    permute_148: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.permute.default(view_317, [2, 0, 3, 1, 4]);  view_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_15 = torch.ops.aten.unbind.int(permute_148);  permute_148 = None
    getitem_112: "f32[8, 6, 197, 64]" = unbind_15[0]
    getitem_113: "f32[8, 6, 197, 64]" = unbind_15[1];  unbind_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_149: "f32[384, 384]" = torch.ops.aten.permute.default(primals_227, [1, 0]);  primals_227 = None
    mm_31: "f32[1576, 384]" = torch.ops.aten.mm.default(view_315, permute_149)
    view_319: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_31, [8, 197, 384]);  mm_31 = None
    view_320: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_319, [8, 197, 6, -1]);  view_319 = None
    permute_150: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_320, [0, 2, 1, 3]);  view_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_151: "f32[8, 6, 64, 197]" = torch.ops.aten.permute.default(getitem_113, [0, 1, 3, 2]);  getitem_113 = None
    expand_61: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(getitem_112, [8, 6, 197, 64]);  getitem_112 = None
    clone_118: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
    view_321: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_118, [48, 197, 64]);  clone_118 = None
    expand_62: "f32[8, 6, 64, 197]" = torch.ops.aten.expand.default(permute_151, [8, 6, 64, 197]);  permute_151 = None
    clone_119: "f32[8, 6, 64, 197]" = torch.ops.aten.clone.default(expand_62, memory_format = torch.contiguous_format);  expand_62 = None
    view_322: "f32[48, 64, 197]" = torch.ops.aten.reshape.default(clone_119, [48, 64, 197]);  clone_119 = None
    bmm_30: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_321, view_322)
    view_323: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_30, [8, 6, 197, 197]);  bmm_30 = None
    mul_142: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_323, 0.125);  view_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_15: "f32[8, 6, 197, 1]" = torch.ops.aten.amax.default(mul_142, [-1], True)
    sub_56: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_142, amax_15);  mul_142 = amax_15 = None
    exp_15: "f32[8, 6, 197, 197]" = torch.ops.aten.exp.default(sub_56);  sub_56 = None
    sum_16: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
    div_15: "f32[8, 6, 197, 197]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
    alias_15: "f32[8, 6, 197, 197]" = torch.ops.aten.alias.default(div_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_63: "f32[8, 6, 197, 197]" = torch.ops.aten.expand.default(div_15, [8, 6, 197, 197]);  div_15 = None
    view_324: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(expand_63, [48, 197, 197]);  expand_63 = None
    expand_64: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(permute_150, [8, 6, 197, 64]);  permute_150 = None
    clone_120: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
    view_325: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_120, [48, 197, 64]);  clone_120 = None
    bmm_31: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_324, view_325)
    view_326: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_31, [8, 6, 197, 64]);  bmm_31 = None
    permute_152: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_326, [0, 2, 1, 3]);  view_326 = None
    clone_121: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_152, memory_format = torch.contiguous_format);  permute_152 = None
    view_327: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_121, [8, 197, 384]);  clone_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_328: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_327, [1576, 384]);  view_327 = None
    permute_153: "f32[384, 384]" = torch.ops.aten.permute.default(primals_228, [1, 0]);  primals_228 = None
    addmm_54: "f32[1576, 384]" = torch.ops.aten.addmm.default(primals_229, view_328, permute_153);  primals_229 = None
    view_329: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(addmm_54, [8, 197, 384]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    add_139: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_8, view_329);  view_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    var_mean_41 = torch.ops.aten.var_mean.correction(add_139, [2], correction = 0, keepdim = True)
    getitem_114: "f32[8, 197, 1]" = var_mean_41[0]
    getitem_115: "f32[8, 197, 1]" = var_mean_41[1];  var_mean_41 = None
    add_140: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-05);  getitem_114 = None
    rsqrt_41: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    sub_57: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_139, getitem_115);  getitem_115 = None
    mul_143: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_41);  sub_57 = None
    mul_144: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_143, primals_230)
    add_141: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_144, primals_231);  mul_144 = primals_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_330: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_141, [1576, 384]);  add_141 = None
    permute_154: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_232, [1, 0]);  primals_232 = None
    addmm_55: "f32[1576, 1536]" = torch.ops.aten.addmm.default(primals_233, view_330, permute_154);  primals_233 = None
    view_331: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(addmm_55, [8, 197, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_145: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_331, 0.5)
    mul_146: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_331, 0.7071067811865476);  view_331 = None
    erf_15: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_146);  mul_146 = None
    add_142: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_147: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_145, add_142);  mul_145 = add_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_332: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_147, [1576, 1536]);  mul_147 = None
    permute_155: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_234, [1, 0]);  primals_234 = None
    addmm_56: "f32[1576, 384]" = torch.ops.aten.addmm.default(primals_235, view_332, permute_155);  primals_235 = None
    view_333: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(addmm_56, [8, 197, 384]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    add_143: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_139, view_333);  add_139 = view_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    mul_149: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_138, primals_236)
    add_145: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_149, primals_237);  mul_149 = primals_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_156: "f32[24, 48]" = torch.ops.aten.permute.default(primals_238, [1, 0]);  primals_238 = None
    view_334: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_145, [25088, 24]);  add_145 = None
    mm_32: "f32[25088, 48]" = torch.ops.aten.mm.default(view_334, permute_156)
    view_335: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(mm_32, [1568, 16, 48]);  mm_32 = None
    view_336: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.reshape.default(view_335, [1568, 16, 2, 4, 6]);  view_335 = None
    permute_157: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_336, [2, 0, 3, 1, 4]);  view_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_16 = torch.ops.aten.unbind.int(permute_157);  permute_157 = None
    getitem_118: "f32[1568, 4, 16, 6]" = unbind_16[0]
    getitem_119: "f32[1568, 4, 16, 6]" = unbind_16[1];  unbind_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_158: "f32[24, 24]" = torch.ops.aten.permute.default(primals_239, [1, 0]);  primals_239 = None
    mm_33: "f32[25088, 24]" = torch.ops.aten.mm.default(view_334, permute_158)
    view_338: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_33, [1568, 16, 24]);  mm_33 = None
    view_339: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_338, [1568, 16, 4, -1]);  view_338 = None
    permute_159: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_339, [0, 2, 1, 3]);  view_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_160: "f32[1568, 4, 6, 16]" = torch.ops.aten.permute.default(getitem_119, [0, 1, 3, 2]);  getitem_119 = None
    expand_65: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(getitem_118, [1568, 4, 16, 6]);  getitem_118 = None
    clone_125: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
    view_340: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_125, [6272, 16, 6]);  clone_125 = None
    expand_66: "f32[1568, 4, 6, 16]" = torch.ops.aten.expand.default(permute_160, [1568, 4, 6, 16]);  permute_160 = None
    clone_126: "f32[1568, 4, 6, 16]" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
    view_341: "f32[6272, 6, 16]" = torch.ops.aten.reshape.default(clone_126, [6272, 6, 16]);  clone_126 = None
    bmm_32: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_340, view_341)
    view_342: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_32, [1568, 4, 16, 16]);  bmm_32 = None
    mul_150: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_342, 0.408248290463863);  view_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_16: "f32[1568, 4, 16, 1]" = torch.ops.aten.amax.default(mul_150, [-1], True)
    sub_59: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_150, amax_16);  mul_150 = amax_16 = None
    exp_16: "f32[1568, 4, 16, 16]" = torch.ops.aten.exp.default(sub_59);  sub_59 = None
    sum_17: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_16: "f32[1568, 4, 16, 16]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
    alias_16: "f32[1568, 4, 16, 16]" = torch.ops.aten.alias.default(div_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_67: "f32[1568, 4, 16, 16]" = torch.ops.aten.expand.default(div_16, [1568, 4, 16, 16]);  div_16 = None
    view_343: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(expand_67, [6272, 16, 16]);  expand_67 = None
    expand_68: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(permute_159, [1568, 4, 16, 6]);  permute_159 = None
    clone_127: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
    view_344: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_127, [6272, 16, 6]);  clone_127 = None
    bmm_33: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_343, view_344)
    view_345: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_33, [1568, 4, 16, 6]);  bmm_33 = None
    permute_161: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_345, [0, 2, 1, 3]);  view_345 = None
    clone_128: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
    view_346: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_128, [1568, 16, 24]);  clone_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_347: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_346, [25088, 24]);  view_346 = None
    permute_162: "f32[24, 24]" = torch.ops.aten.permute.default(primals_240, [1, 0]);  primals_240 = None
    addmm_57: "f32[25088, 24]" = torch.ops.aten.addmm.default(primals_241, view_347, permute_162);  primals_241 = None
    view_348: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(addmm_57, [1568, 16, 24]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    add_146: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_133, view_348);  add_133 = view_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    clone_129: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_146, memory_format = torch.contiguous_format)
    var_mean_43 = torch.ops.aten.var_mean.correction(clone_129, [2], correction = 0, keepdim = True)
    getitem_120: "f32[1568, 16, 1]" = var_mean_43[0]
    getitem_121: "f32[1568, 16, 1]" = var_mean_43[1];  var_mean_43 = None
    add_147: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-05);  getitem_120 = None
    rsqrt_43: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
    sub_60: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_129, getitem_121);  clone_129 = getitem_121 = None
    mul_151: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_43);  sub_60 = None
    mul_152: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_151, primals_242)
    add_148: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_152, primals_243);  mul_152 = primals_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_349: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_148, [25088, 24]);  add_148 = None
    permute_163: "f32[24, 96]" = torch.ops.aten.permute.default(primals_244, [1, 0]);  primals_244 = None
    addmm_58: "f32[25088, 96]" = torch.ops.aten.addmm.default(primals_245, view_349, permute_163);  primals_245 = None
    view_350: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(addmm_58, [1568, 16, 96])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_153: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_350, 0.5)
    mul_154: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_350, 0.7071067811865476);  view_350 = None
    erf_16: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_154);  mul_154 = None
    add_149: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_155: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_153, add_149);  mul_153 = add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_351: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_155, [25088, 96]);  mul_155 = None
    permute_164: "f32[96, 24]" = torch.ops.aten.permute.default(primals_246, [1, 0]);  primals_246 = None
    addmm_59: "f32[25088, 24]" = torch.ops.aten.addmm.default(primals_247, view_351, permute_164);  primals_247 = None
    view_352: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(addmm_59, [1568, 16, 24]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    add_150: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_146, view_352);  add_146 = view_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    slice_36: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_143, 1, 0, 1)
    slice_38: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_143, 1, 1, 9223372036854775807);  add_143 = None
    clone_132: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_150, memory_format = torch.contiguous_format)
    var_mean_44 = torch.ops.aten.var_mean.correction(clone_132, [2], correction = 0, keepdim = True)
    getitem_122: "f32[1568, 16, 1]" = var_mean_44[0]
    getitem_123: "f32[1568, 16, 1]" = var_mean_44[1];  var_mean_44 = None
    add_151: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-05);  getitem_122 = None
    rsqrt_44: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
    sub_61: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_132, getitem_123);  clone_132 = getitem_123 = None
    mul_156: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_44);  sub_61 = None
    mul_157: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_156, primals_248)
    add_152: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_157, primals_249);  mul_157 = primals_249 = None
    view_353: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_152, [8, 196, -1]);  add_152 = None
    view_354: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_353, [1568, 384]);  view_353 = None
    permute_165: "f32[384, 384]" = torch.ops.aten.permute.default(primals_250, [1, 0]);  primals_250 = None
    addmm_60: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_251, view_354, permute_165);  primals_251 = None
    view_355: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(addmm_60, [8, 196, 384]);  addmm_60 = None
    add_153: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(slice_38, view_355);  slice_38 = view_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    cat_9: "f32[8, 197, 384]" = torch.ops.aten.cat.default([slice_36, add_153], 1);  slice_36 = add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    var_mean_45 = torch.ops.aten.var_mean.correction(cat_9, [2], correction = 0, keepdim = True)
    getitem_124: "f32[8, 197, 1]" = var_mean_45[0]
    getitem_125: "f32[8, 197, 1]" = var_mean_45[1];  var_mean_45 = None
    add_154: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-05);  getitem_124 = None
    rsqrt_45: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
    sub_62: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_9, getitem_125)
    mul_158: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_45);  sub_62 = None
    mul_159: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_158, primals_252);  mul_158 = None
    add_155: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_159, primals_253);  mul_159 = primals_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_166: "f32[384, 768]" = torch.ops.aten.permute.default(primals_254, [1, 0]);  primals_254 = None
    view_356: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_155, [1576, 384]);  add_155 = None
    mm_34: "f32[1576, 768]" = torch.ops.aten.mm.default(view_356, permute_166)
    view_357: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_34, [8, 197, 768]);  mm_34 = None
    view_358: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.reshape.default(view_357, [8, 197, 2, 6, 64]);  view_357 = None
    permute_167: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.permute.default(view_358, [2, 0, 3, 1, 4]);  view_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_17 = torch.ops.aten.unbind.int(permute_167);  permute_167 = None
    getitem_126: "f32[8, 6, 197, 64]" = unbind_17[0]
    getitem_127: "f32[8, 6, 197, 64]" = unbind_17[1];  unbind_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_168: "f32[384, 384]" = torch.ops.aten.permute.default(primals_255, [1, 0]);  primals_255 = None
    mm_35: "f32[1576, 384]" = torch.ops.aten.mm.default(view_356, permute_168)
    view_360: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_35, [8, 197, 384]);  mm_35 = None
    view_361: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_360, [8, 197, 6, -1]);  view_360 = None
    permute_169: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_361, [0, 2, 1, 3]);  view_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_170: "f32[8, 6, 64, 197]" = torch.ops.aten.permute.default(getitem_127, [0, 1, 3, 2]);  getitem_127 = None
    expand_69: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(getitem_126, [8, 6, 197, 64]);  getitem_126 = None
    clone_133: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
    view_362: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_133, [48, 197, 64]);  clone_133 = None
    expand_70: "f32[8, 6, 64, 197]" = torch.ops.aten.expand.default(permute_170, [8, 6, 64, 197]);  permute_170 = None
    clone_134: "f32[8, 6, 64, 197]" = torch.ops.aten.clone.default(expand_70, memory_format = torch.contiguous_format);  expand_70 = None
    view_363: "f32[48, 64, 197]" = torch.ops.aten.reshape.default(clone_134, [48, 64, 197]);  clone_134 = None
    bmm_34: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_362, view_363)
    view_364: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_34, [8, 6, 197, 197]);  bmm_34 = None
    mul_160: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_364, 0.125);  view_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_17: "f32[8, 6, 197, 1]" = torch.ops.aten.amax.default(mul_160, [-1], True)
    sub_63: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_160, amax_17);  mul_160 = amax_17 = None
    exp_17: "f32[8, 6, 197, 197]" = torch.ops.aten.exp.default(sub_63);  sub_63 = None
    sum_18: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
    div_17: "f32[8, 6, 197, 197]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
    alias_17: "f32[8, 6, 197, 197]" = torch.ops.aten.alias.default(div_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_71: "f32[8, 6, 197, 197]" = torch.ops.aten.expand.default(div_17, [8, 6, 197, 197]);  div_17 = None
    view_365: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(expand_71, [48, 197, 197]);  expand_71 = None
    expand_72: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(permute_169, [8, 6, 197, 64]);  permute_169 = None
    clone_135: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
    view_366: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_135, [48, 197, 64]);  clone_135 = None
    bmm_35: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_365, view_366)
    view_367: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_35, [8, 6, 197, 64]);  bmm_35 = None
    permute_171: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_367, [0, 2, 1, 3]);  view_367 = None
    clone_136: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_171, memory_format = torch.contiguous_format);  permute_171 = None
    view_368: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_136, [8, 197, 384]);  clone_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_369: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_368, [1576, 384]);  view_368 = None
    permute_172: "f32[384, 384]" = torch.ops.aten.permute.default(primals_256, [1, 0]);  primals_256 = None
    addmm_61: "f32[1576, 384]" = torch.ops.aten.addmm.default(primals_257, view_369, permute_172);  primals_257 = None
    view_370: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(addmm_61, [8, 197, 384]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    add_156: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_9, view_370);  view_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    var_mean_46 = torch.ops.aten.var_mean.correction(add_156, [2], correction = 0, keepdim = True)
    getitem_128: "f32[8, 197, 1]" = var_mean_46[0]
    getitem_129: "f32[8, 197, 1]" = var_mean_46[1];  var_mean_46 = None
    add_157: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-05);  getitem_128 = None
    rsqrt_46: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
    sub_64: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_156, getitem_129);  getitem_129 = None
    mul_161: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_46);  sub_64 = None
    mul_162: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_161, primals_258)
    add_158: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_162, primals_259);  mul_162 = primals_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_371: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_158, [1576, 384]);  add_158 = None
    permute_173: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_260, [1, 0]);  primals_260 = None
    addmm_62: "f32[1576, 1536]" = torch.ops.aten.addmm.default(primals_261, view_371, permute_173);  primals_261 = None
    view_372: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(addmm_62, [8, 197, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_163: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_372, 0.5)
    mul_164: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_372, 0.7071067811865476);  view_372 = None
    erf_17: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_164);  mul_164 = None
    add_159: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_165: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_163, add_159);  mul_163 = add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_373: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_165, [1576, 1536]);  mul_165 = None
    permute_174: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_262, [1, 0]);  primals_262 = None
    addmm_63: "f32[1576, 384]" = torch.ops.aten.addmm.default(primals_263, view_373, permute_174);  primals_263 = None
    view_374: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(addmm_63, [8, 197, 384]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    add_160: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_156, view_374);  add_156 = view_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    mul_167: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_156, primals_264)
    add_162: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_167, primals_265);  mul_167 = primals_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_175: "f32[24, 48]" = torch.ops.aten.permute.default(primals_266, [1, 0]);  primals_266 = None
    view_375: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_162, [25088, 24]);  add_162 = None
    mm_36: "f32[25088, 48]" = torch.ops.aten.mm.default(view_375, permute_175)
    view_376: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(mm_36, [1568, 16, 48]);  mm_36 = None
    view_377: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.reshape.default(view_376, [1568, 16, 2, 4, 6]);  view_376 = None
    permute_176: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_377, [2, 0, 3, 1, 4]);  view_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_18 = torch.ops.aten.unbind.int(permute_176);  permute_176 = None
    getitem_132: "f32[1568, 4, 16, 6]" = unbind_18[0]
    getitem_133: "f32[1568, 4, 16, 6]" = unbind_18[1];  unbind_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_177: "f32[24, 24]" = torch.ops.aten.permute.default(primals_267, [1, 0]);  primals_267 = None
    mm_37: "f32[25088, 24]" = torch.ops.aten.mm.default(view_375, permute_177)
    view_379: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_37, [1568, 16, 24]);  mm_37 = None
    view_380: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_379, [1568, 16, 4, -1]);  view_379 = None
    permute_178: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_380, [0, 2, 1, 3]);  view_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_179: "f32[1568, 4, 6, 16]" = torch.ops.aten.permute.default(getitem_133, [0, 1, 3, 2]);  getitem_133 = None
    expand_73: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(getitem_132, [1568, 4, 16, 6]);  getitem_132 = None
    clone_140: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
    view_381: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_140, [6272, 16, 6]);  clone_140 = None
    expand_74: "f32[1568, 4, 6, 16]" = torch.ops.aten.expand.default(permute_179, [1568, 4, 6, 16]);  permute_179 = None
    clone_141: "f32[1568, 4, 6, 16]" = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
    view_382: "f32[6272, 6, 16]" = torch.ops.aten.reshape.default(clone_141, [6272, 6, 16]);  clone_141 = None
    bmm_36: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_381, view_382)
    view_383: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_36, [1568, 4, 16, 16]);  bmm_36 = None
    mul_168: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_383, 0.408248290463863);  view_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_18: "f32[1568, 4, 16, 1]" = torch.ops.aten.amax.default(mul_168, [-1], True)
    sub_66: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_168, amax_18);  mul_168 = amax_18 = None
    exp_18: "f32[1568, 4, 16, 16]" = torch.ops.aten.exp.default(sub_66);  sub_66 = None
    sum_19: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
    div_18: "f32[1568, 4, 16, 16]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
    alias_18: "f32[1568, 4, 16, 16]" = torch.ops.aten.alias.default(div_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_75: "f32[1568, 4, 16, 16]" = torch.ops.aten.expand.default(div_18, [1568, 4, 16, 16]);  div_18 = None
    view_384: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(expand_75, [6272, 16, 16]);  expand_75 = None
    expand_76: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(permute_178, [1568, 4, 16, 6]);  permute_178 = None
    clone_142: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
    view_385: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_142, [6272, 16, 6]);  clone_142 = None
    bmm_37: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_384, view_385)
    view_386: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_37, [1568, 4, 16, 6]);  bmm_37 = None
    permute_180: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_386, [0, 2, 1, 3]);  view_386 = None
    clone_143: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_180, memory_format = torch.contiguous_format);  permute_180 = None
    view_387: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_143, [1568, 16, 24]);  clone_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_388: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_387, [25088, 24]);  view_387 = None
    permute_181: "f32[24, 24]" = torch.ops.aten.permute.default(primals_268, [1, 0]);  primals_268 = None
    addmm_64: "f32[25088, 24]" = torch.ops.aten.addmm.default(primals_269, view_388, permute_181);  primals_269 = None
    view_389: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(addmm_64, [1568, 16, 24]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    add_163: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_150, view_389);  add_150 = view_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    clone_144: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_163, memory_format = torch.contiguous_format)
    var_mean_48 = torch.ops.aten.var_mean.correction(clone_144, [2], correction = 0, keepdim = True)
    getitem_134: "f32[1568, 16, 1]" = var_mean_48[0]
    getitem_135: "f32[1568, 16, 1]" = var_mean_48[1];  var_mean_48 = None
    add_164: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_134, 1e-05);  getitem_134 = None
    rsqrt_48: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
    sub_67: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_144, getitem_135);  clone_144 = getitem_135 = None
    mul_169: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_48);  sub_67 = None
    mul_170: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_169, primals_270)
    add_165: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_170, primals_271);  mul_170 = primals_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_390: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_165, [25088, 24]);  add_165 = None
    permute_182: "f32[24, 96]" = torch.ops.aten.permute.default(primals_272, [1, 0]);  primals_272 = None
    addmm_65: "f32[25088, 96]" = torch.ops.aten.addmm.default(primals_273, view_390, permute_182);  primals_273 = None
    view_391: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(addmm_65, [1568, 16, 96])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_171: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_391, 0.5)
    mul_172: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_391, 0.7071067811865476);  view_391 = None
    erf_18: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_172);  mul_172 = None
    add_166: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_173: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_171, add_166);  mul_171 = add_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_392: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_173, [25088, 96]);  mul_173 = None
    permute_183: "f32[96, 24]" = torch.ops.aten.permute.default(primals_274, [1, 0]);  primals_274 = None
    addmm_66: "f32[25088, 24]" = torch.ops.aten.addmm.default(primals_275, view_392, permute_183);  primals_275 = None
    view_393: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(addmm_66, [1568, 16, 24]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    add_167: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_163, view_393);  add_163 = view_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    slice_40: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_160, 1, 0, 1)
    slice_42: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_160, 1, 1, 9223372036854775807);  add_160 = None
    clone_147: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_167, memory_format = torch.contiguous_format)
    var_mean_49 = torch.ops.aten.var_mean.correction(clone_147, [2], correction = 0, keepdim = True)
    getitem_136: "f32[1568, 16, 1]" = var_mean_49[0]
    getitem_137: "f32[1568, 16, 1]" = var_mean_49[1];  var_mean_49 = None
    add_168: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_136, 1e-05);  getitem_136 = None
    rsqrt_49: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
    sub_68: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_147, getitem_137);  clone_147 = getitem_137 = None
    mul_174: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_49);  sub_68 = None
    mul_175: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_174, primals_276)
    add_169: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_175, primals_277);  mul_175 = primals_277 = None
    view_394: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_169, [8, 196, -1]);  add_169 = None
    view_395: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_394, [1568, 384]);  view_394 = None
    permute_184: "f32[384, 384]" = torch.ops.aten.permute.default(primals_278, [1, 0]);  primals_278 = None
    addmm_67: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_279, view_395, permute_184);  primals_279 = None
    view_396: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(addmm_67, [8, 196, 384]);  addmm_67 = None
    add_170: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(slice_42, view_396);  slice_42 = view_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    cat_10: "f32[8, 197, 384]" = torch.ops.aten.cat.default([slice_40, add_170], 1);  slice_40 = add_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    var_mean_50 = torch.ops.aten.var_mean.correction(cat_10, [2], correction = 0, keepdim = True)
    getitem_138: "f32[8, 197, 1]" = var_mean_50[0]
    getitem_139: "f32[8, 197, 1]" = var_mean_50[1];  var_mean_50 = None
    add_171: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_138, 1e-05);  getitem_138 = None
    rsqrt_50: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
    sub_69: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_10, getitem_139)
    mul_176: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_50);  sub_69 = None
    mul_177: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_176, primals_280);  mul_176 = None
    add_172: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_177, primals_281);  mul_177 = primals_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_185: "f32[384, 768]" = torch.ops.aten.permute.default(primals_282, [1, 0]);  primals_282 = None
    view_397: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_172, [1576, 384]);  add_172 = None
    mm_38: "f32[1576, 768]" = torch.ops.aten.mm.default(view_397, permute_185)
    view_398: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_38, [8, 197, 768]);  mm_38 = None
    view_399: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.reshape.default(view_398, [8, 197, 2, 6, 64]);  view_398 = None
    permute_186: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.permute.default(view_399, [2, 0, 3, 1, 4]);  view_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_19 = torch.ops.aten.unbind.int(permute_186);  permute_186 = None
    getitem_140: "f32[8, 6, 197, 64]" = unbind_19[0]
    getitem_141: "f32[8, 6, 197, 64]" = unbind_19[1];  unbind_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_187: "f32[384, 384]" = torch.ops.aten.permute.default(primals_283, [1, 0]);  primals_283 = None
    mm_39: "f32[1576, 384]" = torch.ops.aten.mm.default(view_397, permute_187)
    view_401: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_39, [8, 197, 384]);  mm_39 = None
    view_402: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_401, [8, 197, 6, -1]);  view_401 = None
    permute_188: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_189: "f32[8, 6, 64, 197]" = torch.ops.aten.permute.default(getitem_141, [0, 1, 3, 2]);  getitem_141 = None
    expand_77: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(getitem_140, [8, 6, 197, 64]);  getitem_140 = None
    clone_148: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_77, memory_format = torch.contiguous_format);  expand_77 = None
    view_403: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_148, [48, 197, 64]);  clone_148 = None
    expand_78: "f32[8, 6, 64, 197]" = torch.ops.aten.expand.default(permute_189, [8, 6, 64, 197]);  permute_189 = None
    clone_149: "f32[8, 6, 64, 197]" = torch.ops.aten.clone.default(expand_78, memory_format = torch.contiguous_format);  expand_78 = None
    view_404: "f32[48, 64, 197]" = torch.ops.aten.reshape.default(clone_149, [48, 64, 197]);  clone_149 = None
    bmm_38: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_403, view_404)
    view_405: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_38, [8, 6, 197, 197]);  bmm_38 = None
    mul_178: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_405, 0.125);  view_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_19: "f32[8, 6, 197, 1]" = torch.ops.aten.amax.default(mul_178, [-1], True)
    sub_70: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_178, amax_19);  mul_178 = amax_19 = None
    exp_19: "f32[8, 6, 197, 197]" = torch.ops.aten.exp.default(sub_70);  sub_70 = None
    sum_20: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
    div_19: "f32[8, 6, 197, 197]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
    alias_19: "f32[8, 6, 197, 197]" = torch.ops.aten.alias.default(div_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_79: "f32[8, 6, 197, 197]" = torch.ops.aten.expand.default(div_19, [8, 6, 197, 197]);  div_19 = None
    view_406: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(expand_79, [48, 197, 197]);  expand_79 = None
    expand_80: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(permute_188, [8, 6, 197, 64]);  permute_188 = None
    clone_150: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
    view_407: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_150, [48, 197, 64]);  clone_150 = None
    bmm_39: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_406, view_407)
    view_408: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_39, [8, 6, 197, 64]);  bmm_39 = None
    permute_190: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_408, [0, 2, 1, 3]);  view_408 = None
    clone_151: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_190, memory_format = torch.contiguous_format);  permute_190 = None
    view_409: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_151, [8, 197, 384]);  clone_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_410: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_409, [1576, 384]);  view_409 = None
    permute_191: "f32[384, 384]" = torch.ops.aten.permute.default(primals_284, [1, 0]);  primals_284 = None
    addmm_68: "f32[1576, 384]" = torch.ops.aten.addmm.default(primals_285, view_410, permute_191);  primals_285 = None
    view_411: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(addmm_68, [8, 197, 384]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    add_173: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_10, view_411);  view_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    var_mean_51 = torch.ops.aten.var_mean.correction(add_173, [2], correction = 0, keepdim = True)
    getitem_142: "f32[8, 197, 1]" = var_mean_51[0]
    getitem_143: "f32[8, 197, 1]" = var_mean_51[1];  var_mean_51 = None
    add_174: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_142, 1e-05);  getitem_142 = None
    rsqrt_51: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
    sub_71: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_173, getitem_143);  getitem_143 = None
    mul_179: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_51);  sub_71 = None
    mul_180: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_179, primals_286)
    add_175: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_180, primals_287);  mul_180 = primals_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_412: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_175, [1576, 384]);  add_175 = None
    permute_192: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_288, [1, 0]);  primals_288 = None
    addmm_69: "f32[1576, 1536]" = torch.ops.aten.addmm.default(primals_289, view_412, permute_192);  primals_289 = None
    view_413: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(addmm_69, [8, 197, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_181: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_413, 0.5)
    mul_182: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_413, 0.7071067811865476);  view_413 = None
    erf_19: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_182);  mul_182 = None
    add_176: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_183: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_181, add_176);  mul_181 = add_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_414: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_183, [1576, 1536]);  mul_183 = None
    permute_193: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_290, [1, 0]);  primals_290 = None
    addmm_70: "f32[1576, 384]" = torch.ops.aten.addmm.default(primals_291, view_414, permute_193);  primals_291 = None
    view_415: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(addmm_70, [8, 197, 384]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    add_177: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_173, view_415);  add_173 = view_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    mul_185: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_174, primals_292)
    add_179: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_185, primals_293);  mul_185 = primals_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_194: "f32[24, 48]" = torch.ops.aten.permute.default(primals_294, [1, 0]);  primals_294 = None
    view_416: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_179, [25088, 24]);  add_179 = None
    mm_40: "f32[25088, 48]" = torch.ops.aten.mm.default(view_416, permute_194)
    view_417: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(mm_40, [1568, 16, 48]);  mm_40 = None
    view_418: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.reshape.default(view_417, [1568, 16, 2, 4, 6]);  view_417 = None
    permute_195: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_418, [2, 0, 3, 1, 4]);  view_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_20 = torch.ops.aten.unbind.int(permute_195);  permute_195 = None
    getitem_146: "f32[1568, 4, 16, 6]" = unbind_20[0]
    getitem_147: "f32[1568, 4, 16, 6]" = unbind_20[1];  unbind_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_196: "f32[24, 24]" = torch.ops.aten.permute.default(primals_295, [1, 0]);  primals_295 = None
    mm_41: "f32[25088, 24]" = torch.ops.aten.mm.default(view_416, permute_196)
    view_420: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_41, [1568, 16, 24]);  mm_41 = None
    view_421: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_420, [1568, 16, 4, -1]);  view_420 = None
    permute_197: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_421, [0, 2, 1, 3]);  view_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_198: "f32[1568, 4, 6, 16]" = torch.ops.aten.permute.default(getitem_147, [0, 1, 3, 2]);  getitem_147 = None
    expand_81: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(getitem_146, [1568, 4, 16, 6]);  getitem_146 = None
    clone_155: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
    view_422: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_155, [6272, 16, 6]);  clone_155 = None
    expand_82: "f32[1568, 4, 6, 16]" = torch.ops.aten.expand.default(permute_198, [1568, 4, 6, 16]);  permute_198 = None
    clone_156: "f32[1568, 4, 6, 16]" = torch.ops.aten.clone.default(expand_82, memory_format = torch.contiguous_format);  expand_82 = None
    view_423: "f32[6272, 6, 16]" = torch.ops.aten.reshape.default(clone_156, [6272, 6, 16]);  clone_156 = None
    bmm_40: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_422, view_423)
    view_424: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_40, [1568, 4, 16, 16]);  bmm_40 = None
    mul_186: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_424, 0.408248290463863);  view_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_20: "f32[1568, 4, 16, 1]" = torch.ops.aten.amax.default(mul_186, [-1], True)
    sub_73: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_186, amax_20);  mul_186 = amax_20 = None
    exp_20: "f32[1568, 4, 16, 16]" = torch.ops.aten.exp.default(sub_73);  sub_73 = None
    sum_21: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
    div_20: "f32[1568, 4, 16, 16]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
    alias_20: "f32[1568, 4, 16, 16]" = torch.ops.aten.alias.default(div_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_83: "f32[1568, 4, 16, 16]" = torch.ops.aten.expand.default(div_20, [1568, 4, 16, 16]);  div_20 = None
    view_425: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(expand_83, [6272, 16, 16]);  expand_83 = None
    expand_84: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(permute_197, [1568, 4, 16, 6]);  permute_197 = None
    clone_157: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_84, memory_format = torch.contiguous_format);  expand_84 = None
    view_426: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_157, [6272, 16, 6]);  clone_157 = None
    bmm_41: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_425, view_426)
    view_427: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_41, [1568, 4, 16, 6]);  bmm_41 = None
    permute_199: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_427, [0, 2, 1, 3]);  view_427 = None
    clone_158: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None
    view_428: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_158, [1568, 16, 24]);  clone_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_429: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_428, [25088, 24]);  view_428 = None
    permute_200: "f32[24, 24]" = torch.ops.aten.permute.default(primals_296, [1, 0]);  primals_296 = None
    addmm_71: "f32[25088, 24]" = torch.ops.aten.addmm.default(primals_297, view_429, permute_200);  primals_297 = None
    view_430: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(addmm_71, [1568, 16, 24]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    add_180: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_167, view_430);  add_167 = view_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    clone_159: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_180, memory_format = torch.contiguous_format)
    var_mean_53 = torch.ops.aten.var_mean.correction(clone_159, [2], correction = 0, keepdim = True)
    getitem_148: "f32[1568, 16, 1]" = var_mean_53[0]
    getitem_149: "f32[1568, 16, 1]" = var_mean_53[1];  var_mean_53 = None
    add_181: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_148, 1e-05);  getitem_148 = None
    rsqrt_53: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
    sub_74: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_159, getitem_149);  clone_159 = getitem_149 = None
    mul_187: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_53);  sub_74 = None
    mul_188: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_187, primals_298)
    add_182: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_188, primals_299);  mul_188 = primals_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_431: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_182, [25088, 24]);  add_182 = None
    permute_201: "f32[24, 96]" = torch.ops.aten.permute.default(primals_300, [1, 0]);  primals_300 = None
    addmm_72: "f32[25088, 96]" = torch.ops.aten.addmm.default(primals_301, view_431, permute_201);  primals_301 = None
    view_432: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(addmm_72, [1568, 16, 96])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_189: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_432, 0.5)
    mul_190: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_432, 0.7071067811865476);  view_432 = None
    erf_20: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_190);  mul_190 = None
    add_183: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_191: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_189, add_183);  mul_189 = add_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_433: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_191, [25088, 96]);  mul_191 = None
    permute_202: "f32[96, 24]" = torch.ops.aten.permute.default(primals_302, [1, 0]);  primals_302 = None
    addmm_73: "f32[25088, 24]" = torch.ops.aten.addmm.default(primals_303, view_433, permute_202);  primals_303 = None
    view_434: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(addmm_73, [1568, 16, 24]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    add_184: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_180, view_434);  add_180 = view_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    slice_44: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_177, 1, 0, 1)
    slice_46: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_177, 1, 1, 9223372036854775807);  add_177 = None
    clone_162: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_184, memory_format = torch.contiguous_format)
    var_mean_54 = torch.ops.aten.var_mean.correction(clone_162, [2], correction = 0, keepdim = True)
    getitem_150: "f32[1568, 16, 1]" = var_mean_54[0]
    getitem_151: "f32[1568, 16, 1]" = var_mean_54[1];  var_mean_54 = None
    add_185: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_150, 1e-05);  getitem_150 = None
    rsqrt_54: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
    sub_75: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_162, getitem_151);  clone_162 = getitem_151 = None
    mul_192: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_54);  sub_75 = None
    mul_193: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_192, primals_304)
    add_186: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_193, primals_305);  mul_193 = primals_305 = None
    view_435: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_186, [8, 196, -1]);  add_186 = None
    view_436: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_435, [1568, 384]);  view_435 = None
    permute_203: "f32[384, 384]" = torch.ops.aten.permute.default(primals_306, [1, 0]);  primals_306 = None
    addmm_74: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_307, view_436, permute_203);  primals_307 = None
    view_437: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(addmm_74, [8, 196, 384]);  addmm_74 = None
    add_187: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(slice_46, view_437);  slice_46 = view_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    cat_11: "f32[8, 197, 384]" = torch.ops.aten.cat.default([slice_44, add_187], 1);  slice_44 = add_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    var_mean_55 = torch.ops.aten.var_mean.correction(cat_11, [2], correction = 0, keepdim = True)
    getitem_152: "f32[8, 197, 1]" = var_mean_55[0]
    getitem_153: "f32[8, 197, 1]" = var_mean_55[1];  var_mean_55 = None
    add_188: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_152, 1e-05);  getitem_152 = None
    rsqrt_55: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
    sub_76: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_11, getitem_153)
    mul_194: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_55);  sub_76 = None
    mul_195: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_194, primals_308);  mul_194 = None
    add_189: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_195, primals_309);  mul_195 = primals_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_204: "f32[384, 768]" = torch.ops.aten.permute.default(primals_310, [1, 0]);  primals_310 = None
    view_438: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_189, [1576, 384]);  add_189 = None
    mm_42: "f32[1576, 768]" = torch.ops.aten.mm.default(view_438, permute_204)
    view_439: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_42, [8, 197, 768]);  mm_42 = None
    view_440: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.reshape.default(view_439, [8, 197, 2, 6, 64]);  view_439 = None
    permute_205: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.permute.default(view_440, [2, 0, 3, 1, 4]);  view_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_21 = torch.ops.aten.unbind.int(permute_205);  permute_205 = None
    getitem_154: "f32[8, 6, 197, 64]" = unbind_21[0]
    getitem_155: "f32[8, 6, 197, 64]" = unbind_21[1];  unbind_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_206: "f32[384, 384]" = torch.ops.aten.permute.default(primals_311, [1, 0]);  primals_311 = None
    mm_43: "f32[1576, 384]" = torch.ops.aten.mm.default(view_438, permute_206)
    view_442: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_43, [8, 197, 384]);  mm_43 = None
    view_443: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_442, [8, 197, 6, -1]);  view_442 = None
    permute_207: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_443, [0, 2, 1, 3]);  view_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_208: "f32[8, 6, 64, 197]" = torch.ops.aten.permute.default(getitem_155, [0, 1, 3, 2]);  getitem_155 = None
    expand_85: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(getitem_154, [8, 6, 197, 64]);  getitem_154 = None
    clone_163: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_85, memory_format = torch.contiguous_format);  expand_85 = None
    view_444: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_163, [48, 197, 64]);  clone_163 = None
    expand_86: "f32[8, 6, 64, 197]" = torch.ops.aten.expand.default(permute_208, [8, 6, 64, 197]);  permute_208 = None
    clone_164: "f32[8, 6, 64, 197]" = torch.ops.aten.clone.default(expand_86, memory_format = torch.contiguous_format);  expand_86 = None
    view_445: "f32[48, 64, 197]" = torch.ops.aten.reshape.default(clone_164, [48, 64, 197]);  clone_164 = None
    bmm_42: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_444, view_445)
    view_446: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_42, [8, 6, 197, 197]);  bmm_42 = None
    mul_196: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_446, 0.125);  view_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_21: "f32[8, 6, 197, 1]" = torch.ops.aten.amax.default(mul_196, [-1], True)
    sub_77: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_196, amax_21);  mul_196 = amax_21 = None
    exp_21: "f32[8, 6, 197, 197]" = torch.ops.aten.exp.default(sub_77);  sub_77 = None
    sum_22: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
    div_21: "f32[8, 6, 197, 197]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
    alias_21: "f32[8, 6, 197, 197]" = torch.ops.aten.alias.default(div_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_87: "f32[8, 6, 197, 197]" = torch.ops.aten.expand.default(div_21, [8, 6, 197, 197]);  div_21 = None
    view_447: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(expand_87, [48, 197, 197]);  expand_87 = None
    expand_88: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(permute_207, [8, 6, 197, 64]);  permute_207 = None
    clone_165: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
    view_448: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_165, [48, 197, 64]);  clone_165 = None
    bmm_43: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_447, view_448)
    view_449: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_43, [8, 6, 197, 64]);  bmm_43 = None
    permute_209: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_449, [0, 2, 1, 3]);  view_449 = None
    clone_166: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_209, memory_format = torch.contiguous_format);  permute_209 = None
    view_450: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_166, [8, 197, 384]);  clone_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_451: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_450, [1576, 384]);  view_450 = None
    permute_210: "f32[384, 384]" = torch.ops.aten.permute.default(primals_312, [1, 0]);  primals_312 = None
    addmm_75: "f32[1576, 384]" = torch.ops.aten.addmm.default(primals_313, view_451, permute_210);  primals_313 = None
    view_452: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(addmm_75, [8, 197, 384]);  addmm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    add_190: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_11, view_452);  view_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    var_mean_56 = torch.ops.aten.var_mean.correction(add_190, [2], correction = 0, keepdim = True)
    getitem_156: "f32[8, 197, 1]" = var_mean_56[0]
    getitem_157: "f32[8, 197, 1]" = var_mean_56[1];  var_mean_56 = None
    add_191: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_156, 1e-05);  getitem_156 = None
    rsqrt_56: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
    sub_78: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_190, getitem_157);  getitem_157 = None
    mul_197: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_56);  sub_78 = None
    mul_198: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_197, primals_314)
    add_192: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_198, primals_315);  mul_198 = primals_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_453: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_192, [1576, 384]);  add_192 = None
    permute_211: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_316, [1, 0]);  primals_316 = None
    addmm_76: "f32[1576, 1536]" = torch.ops.aten.addmm.default(primals_317, view_453, permute_211);  primals_317 = None
    view_454: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(addmm_76, [8, 197, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_199: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_454, 0.5)
    mul_200: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_454, 0.7071067811865476);  view_454 = None
    erf_21: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_200);  mul_200 = None
    add_193: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_201: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_199, add_193);  mul_199 = add_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_455: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_201, [1576, 1536]);  mul_201 = None
    permute_212: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_318, [1, 0]);  primals_318 = None
    addmm_77: "f32[1576, 384]" = torch.ops.aten.addmm.default(primals_319, view_455, permute_212);  primals_319 = None
    view_456: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(addmm_77, [8, 197, 384]);  addmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    add_194: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_190, view_456);  add_190 = view_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    mul_203: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_192, primals_320)
    add_196: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_203, primals_321);  mul_203 = primals_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_213: "f32[24, 48]" = torch.ops.aten.permute.default(primals_322, [1, 0]);  primals_322 = None
    view_457: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_196, [25088, 24]);  add_196 = None
    mm_44: "f32[25088, 48]" = torch.ops.aten.mm.default(view_457, permute_213)
    view_458: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(mm_44, [1568, 16, 48]);  mm_44 = None
    view_459: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.reshape.default(view_458, [1568, 16, 2, 4, 6]);  view_458 = None
    permute_214: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_459, [2, 0, 3, 1, 4]);  view_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_22 = torch.ops.aten.unbind.int(permute_214);  permute_214 = None
    getitem_160: "f32[1568, 4, 16, 6]" = unbind_22[0]
    getitem_161: "f32[1568, 4, 16, 6]" = unbind_22[1];  unbind_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_215: "f32[24, 24]" = torch.ops.aten.permute.default(primals_323, [1, 0]);  primals_323 = None
    mm_45: "f32[25088, 24]" = torch.ops.aten.mm.default(view_457, permute_215)
    view_461: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_45, [1568, 16, 24]);  mm_45 = None
    view_462: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_461, [1568, 16, 4, -1]);  view_461 = None
    permute_216: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_462, [0, 2, 1, 3]);  view_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_217: "f32[1568, 4, 6, 16]" = torch.ops.aten.permute.default(getitem_161, [0, 1, 3, 2]);  getitem_161 = None
    expand_89: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(getitem_160, [1568, 4, 16, 6]);  getitem_160 = None
    clone_170: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
    view_463: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_170, [6272, 16, 6]);  clone_170 = None
    expand_90: "f32[1568, 4, 6, 16]" = torch.ops.aten.expand.default(permute_217, [1568, 4, 6, 16]);  permute_217 = None
    clone_171: "f32[1568, 4, 6, 16]" = torch.ops.aten.clone.default(expand_90, memory_format = torch.contiguous_format);  expand_90 = None
    view_464: "f32[6272, 6, 16]" = torch.ops.aten.reshape.default(clone_171, [6272, 6, 16]);  clone_171 = None
    bmm_44: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_463, view_464)
    view_465: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_44, [1568, 4, 16, 16]);  bmm_44 = None
    mul_204: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_465, 0.408248290463863);  view_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_22: "f32[1568, 4, 16, 1]" = torch.ops.aten.amax.default(mul_204, [-1], True)
    sub_80: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_204, amax_22);  mul_204 = amax_22 = None
    exp_22: "f32[1568, 4, 16, 16]" = torch.ops.aten.exp.default(sub_80);  sub_80 = None
    sum_23: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
    div_22: "f32[1568, 4, 16, 16]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
    alias_22: "f32[1568, 4, 16, 16]" = torch.ops.aten.alias.default(div_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_91: "f32[1568, 4, 16, 16]" = torch.ops.aten.expand.default(div_22, [1568, 4, 16, 16]);  div_22 = None
    view_466: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(expand_91, [6272, 16, 16]);  expand_91 = None
    expand_92: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(permute_216, [1568, 4, 16, 6]);  permute_216 = None
    clone_172: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
    view_467: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_172, [6272, 16, 6]);  clone_172 = None
    bmm_45: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_466, view_467)
    view_468: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_45, [1568, 4, 16, 6]);  bmm_45 = None
    permute_218: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_468, [0, 2, 1, 3]);  view_468 = None
    clone_173: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_218, memory_format = torch.contiguous_format);  permute_218 = None
    view_469: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_173, [1568, 16, 24]);  clone_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_470: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_469, [25088, 24]);  view_469 = None
    permute_219: "f32[24, 24]" = torch.ops.aten.permute.default(primals_324, [1, 0]);  primals_324 = None
    addmm_78: "f32[25088, 24]" = torch.ops.aten.addmm.default(primals_325, view_470, permute_219);  primals_325 = None
    view_471: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(addmm_78, [1568, 16, 24]);  addmm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    add_197: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_184, view_471);  add_184 = view_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    clone_174: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_197, memory_format = torch.contiguous_format)
    var_mean_58 = torch.ops.aten.var_mean.correction(clone_174, [2], correction = 0, keepdim = True)
    getitem_162: "f32[1568, 16, 1]" = var_mean_58[0]
    getitem_163: "f32[1568, 16, 1]" = var_mean_58[1];  var_mean_58 = None
    add_198: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_162, 1e-05);  getitem_162 = None
    rsqrt_58: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_198);  add_198 = None
    sub_81: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_174, getitem_163);  clone_174 = getitem_163 = None
    mul_205: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_58);  sub_81 = None
    mul_206: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_205, primals_326)
    add_199: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_206, primals_327);  mul_206 = primals_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_472: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_199, [25088, 24]);  add_199 = None
    permute_220: "f32[24, 96]" = torch.ops.aten.permute.default(primals_328, [1, 0]);  primals_328 = None
    addmm_79: "f32[25088, 96]" = torch.ops.aten.addmm.default(primals_329, view_472, permute_220);  primals_329 = None
    view_473: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(addmm_79, [1568, 16, 96])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_207: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_473, 0.5)
    mul_208: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_473, 0.7071067811865476);  view_473 = None
    erf_22: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_208);  mul_208 = None
    add_200: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_209: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_207, add_200);  mul_207 = add_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_474: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_209, [25088, 96]);  mul_209 = None
    permute_221: "f32[96, 24]" = torch.ops.aten.permute.default(primals_330, [1, 0]);  primals_330 = None
    addmm_80: "f32[25088, 24]" = torch.ops.aten.addmm.default(primals_331, view_474, permute_221);  primals_331 = None
    view_475: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(addmm_80, [1568, 16, 24]);  addmm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    add_201: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_197, view_475);  add_197 = view_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    slice_48: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_194, 1, 0, 1)
    slice_50: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_194, 1, 1, 9223372036854775807);  add_194 = None
    clone_177: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_201, memory_format = torch.contiguous_format);  add_201 = None
    var_mean_59 = torch.ops.aten.var_mean.correction(clone_177, [2], correction = 0, keepdim = True)
    getitem_164: "f32[1568, 16, 1]" = var_mean_59[0]
    getitem_165: "f32[1568, 16, 1]" = var_mean_59[1];  var_mean_59 = None
    add_202: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_164, 1e-05);  getitem_164 = None
    rsqrt_59: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_202);  add_202 = None
    sub_82: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_177, getitem_165);  clone_177 = getitem_165 = None
    mul_210: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_59);  sub_82 = None
    mul_211: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_210, primals_332)
    add_203: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_211, primals_333);  mul_211 = primals_333 = None
    view_476: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_203, [8, 196, -1]);  add_203 = None
    view_477: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_476, [1568, 384]);  view_476 = None
    permute_222: "f32[384, 384]" = torch.ops.aten.permute.default(primals_334, [1, 0]);  primals_334 = None
    addmm_81: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_335, view_477, permute_222);  primals_335 = None
    view_478: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(addmm_81, [8, 196, 384]);  addmm_81 = None
    add_204: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(slice_50, view_478);  slice_50 = view_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    cat_12: "f32[8, 197, 384]" = torch.ops.aten.cat.default([slice_48, add_204], 1);  slice_48 = add_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    var_mean_60 = torch.ops.aten.var_mean.correction(cat_12, [2], correction = 0, keepdim = True)
    getitem_166: "f32[8, 197, 1]" = var_mean_60[0]
    getitem_167: "f32[8, 197, 1]" = var_mean_60[1];  var_mean_60 = None
    add_205: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_166, 1e-05);  getitem_166 = None
    rsqrt_60: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_205);  add_205 = None
    sub_83: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_12, getitem_167)
    mul_212: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_60);  sub_83 = None
    mul_213: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_212, primals_336);  mul_212 = None
    add_206: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_213, primals_337);  mul_213 = primals_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_223: "f32[384, 768]" = torch.ops.aten.permute.default(primals_338, [1, 0]);  primals_338 = None
    view_479: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_206, [1576, 384]);  add_206 = None
    mm_46: "f32[1576, 768]" = torch.ops.aten.mm.default(view_479, permute_223)
    view_480: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_46, [8, 197, 768]);  mm_46 = None
    view_481: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.reshape.default(view_480, [8, 197, 2, 6, 64]);  view_480 = None
    permute_224: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.permute.default(view_481, [2, 0, 3, 1, 4]);  view_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_23 = torch.ops.aten.unbind.int(permute_224);  permute_224 = None
    getitem_168: "f32[8, 6, 197, 64]" = unbind_23[0]
    getitem_169: "f32[8, 6, 197, 64]" = unbind_23[1];  unbind_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_225: "f32[384, 384]" = torch.ops.aten.permute.default(primals_339, [1, 0]);  primals_339 = None
    mm_47: "f32[1576, 384]" = torch.ops.aten.mm.default(view_479, permute_225)
    view_483: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_47, [8, 197, 384]);  mm_47 = None
    view_484: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_483, [8, 197, 6, -1]);  view_483 = None
    permute_226: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_484, [0, 2, 1, 3]);  view_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_227: "f32[8, 6, 64, 197]" = torch.ops.aten.permute.default(getitem_169, [0, 1, 3, 2]);  getitem_169 = None
    expand_93: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(getitem_168, [8, 6, 197, 64]);  getitem_168 = None
    clone_178: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_93, memory_format = torch.contiguous_format);  expand_93 = None
    view_485: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_178, [48, 197, 64]);  clone_178 = None
    expand_94: "f32[8, 6, 64, 197]" = torch.ops.aten.expand.default(permute_227, [8, 6, 64, 197]);  permute_227 = None
    clone_179: "f32[8, 6, 64, 197]" = torch.ops.aten.clone.default(expand_94, memory_format = torch.contiguous_format);  expand_94 = None
    view_486: "f32[48, 64, 197]" = torch.ops.aten.reshape.default(clone_179, [48, 64, 197]);  clone_179 = None
    bmm_46: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_485, view_486)
    view_487: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_46, [8, 6, 197, 197]);  bmm_46 = None
    mul_214: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_487, 0.125);  view_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_23: "f32[8, 6, 197, 1]" = torch.ops.aten.amax.default(mul_214, [-1], True)
    sub_84: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_214, amax_23);  mul_214 = amax_23 = None
    exp_23: "f32[8, 6, 197, 197]" = torch.ops.aten.exp.default(sub_84);  sub_84 = None
    sum_24: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
    div_23: "f32[8, 6, 197, 197]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
    alias_23: "f32[8, 6, 197, 197]" = torch.ops.aten.alias.default(div_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_95: "f32[8, 6, 197, 197]" = torch.ops.aten.expand.default(div_23, [8, 6, 197, 197]);  div_23 = None
    view_488: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(expand_95, [48, 197, 197]);  expand_95 = None
    expand_96: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(permute_226, [8, 6, 197, 64]);  permute_226 = None
    clone_180: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_96, memory_format = torch.contiguous_format);  expand_96 = None
    view_489: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_180, [48, 197, 64]);  clone_180 = None
    bmm_47: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_488, view_489)
    view_490: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_47, [8, 6, 197, 64]);  bmm_47 = None
    permute_228: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_490, [0, 2, 1, 3]);  view_490 = None
    clone_181: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_228, memory_format = torch.contiguous_format);  permute_228 = None
    view_491: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_181, [8, 197, 384]);  clone_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_492: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_491, [1576, 384]);  view_491 = None
    permute_229: "f32[384, 384]" = torch.ops.aten.permute.default(primals_340, [1, 0]);  primals_340 = None
    addmm_82: "f32[1576, 384]" = torch.ops.aten.addmm.default(primals_341, view_492, permute_229);  primals_341 = None
    view_493: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(addmm_82, [8, 197, 384]);  addmm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    add_207: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_12, view_493);  view_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    var_mean_61 = torch.ops.aten.var_mean.correction(add_207, [2], correction = 0, keepdim = True)
    getitem_170: "f32[8, 197, 1]" = var_mean_61[0]
    getitem_171: "f32[8, 197, 1]" = var_mean_61[1];  var_mean_61 = None
    add_208: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_170, 1e-05);  getitem_170 = None
    rsqrt_61: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_208);  add_208 = None
    sub_85: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_207, getitem_171);  getitem_171 = None
    mul_215: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_61);  sub_85 = None
    mul_216: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_215, primals_342)
    add_209: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_216, primals_343);  mul_216 = primals_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_494: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_209, [1576, 384]);  add_209 = None
    permute_230: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_344, [1, 0]);  primals_344 = None
    addmm_83: "f32[1576, 1536]" = torch.ops.aten.addmm.default(primals_345, view_494, permute_230);  primals_345 = None
    view_495: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(addmm_83, [8, 197, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_217: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_495, 0.5)
    mul_218: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_495, 0.7071067811865476);  view_495 = None
    erf_23: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_218);  mul_218 = None
    add_210: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_219: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_217, add_210);  mul_217 = add_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_496: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_219, [1576, 1536]);  mul_219 = None
    permute_231: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_346, [1, 0]);  primals_346 = None
    addmm_84: "f32[1576, 384]" = torch.ops.aten.addmm.default(primals_347, view_496, permute_231);  primals_347 = None
    view_497: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(addmm_84, [8, 197, 384]);  addmm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    add_211: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_207, view_497);  add_207 = view_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:323, code: patch_embed = self.norm(patch_embed)
    var_mean_62 = torch.ops.aten.var_mean.correction(add_211, [2], correction = 0, keepdim = True)
    getitem_172: "f32[8, 197, 1]" = var_mean_62[0]
    getitem_173: "f32[8, 197, 1]" = var_mean_62[1];  var_mean_62 = None
    add_212: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_172, 1e-05);  getitem_172 = None
    rsqrt_62: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_212);  add_212 = None
    sub_86: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_211, getitem_173);  add_211 = getitem_173 = None
    mul_220: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_62);  sub_86 = None
    mul_221: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_220, primals_348)
    add_213: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_221, primals_349);  mul_221 = primals_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:328, code: x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    select: "f32[8, 384]" = torch.ops.aten.select.int(add_213, 1, 0);  add_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:329, code: x = self.head_drop(x)
    clone_184: "f32[8, 384]" = torch.ops.aten.clone.default(select);  select = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:330, code: return x if pre_logits else self.head(x)
    permute_232: "f32[384, 1000]" = torch.ops.aten.permute.default(primals_350, [1, 0]);  primals_350 = None
    addmm_85: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_351, clone_184, permute_232);  primals_351 = None
    permute_233: "f32[1000, 384]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:323, code: patch_embed = self.norm(patch_embed)
    div_24: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_62, 384);  rsqrt_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_237: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_231, [1, 0]);  permute_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_241: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_230, [1, 0]);  permute_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    div_25: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_61, 384);  rsqrt_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    permute_245: "f32[384, 384]" = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    permute_250: "f32[48, 197, 197]" = torch.ops.aten.permute.default(view_488, [0, 2, 1]);  view_488 = None
    permute_251: "f32[48, 64, 197]" = torch.ops.aten.permute.default(view_489, [0, 2, 1]);  view_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    alias_24: "f32[8, 6, 197, 197]" = torch.ops.aten.alias.default(alias_23);  alias_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_252: "f32[48, 64, 197]" = torch.ops.aten.permute.default(view_485, [0, 2, 1]);  view_485 = None
    permute_253: "f32[48, 197, 64]" = torch.ops.aten.permute.default(view_486, [0, 2, 1]);  view_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_258: "f32[384, 384]" = torch.ops.aten.permute.default(permute_225, [1, 0]);  permute_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_263: "f32[768, 384]" = torch.ops.aten.permute.default(permute_223, [1, 0]);  permute_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    permute_265: "f32[384, 384]" = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None
    div_27: "f32[1568, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_59, 24);  rsqrt_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_269: "f32[24, 96]" = torch.ops.aten.permute.default(permute_221, [1, 0]);  permute_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_273: "f32[96, 24]" = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    div_28: "f32[1568, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_58, 24);  rsqrt_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    permute_277: "f32[24, 24]" = torch.ops.aten.permute.default(permute_219, [1, 0]);  permute_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    permute_282: "f32[6272, 16, 16]" = torch.ops.aten.permute.default(view_466, [0, 2, 1]);  view_466 = None
    permute_283: "f32[6272, 6, 16]" = torch.ops.aten.permute.default(view_467, [0, 2, 1]);  view_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    alias_25: "f32[1568, 4, 16, 16]" = torch.ops.aten.alias.default(alias_22);  alias_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_284: "f32[6272, 6, 16]" = torch.ops.aten.permute.default(view_463, [0, 2, 1]);  view_463 = None
    permute_285: "f32[6272, 16, 6]" = torch.ops.aten.permute.default(view_464, [0, 2, 1]);  view_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_290: "f32[24, 24]" = torch.ops.aten.permute.default(permute_215, [1, 0]);  permute_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_295: "f32[48, 24]" = torch.ops.aten.permute.default(permute_213, [1, 0]);  permute_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    div_29: "f32[1568, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_54, 24);  rsqrt_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_297: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_212, [1, 0]);  permute_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_301: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    div_30: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_56, 384);  rsqrt_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    permute_305: "f32[384, 384]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    permute_310: "f32[48, 197, 197]" = torch.ops.aten.permute.default(view_447, [0, 2, 1]);  view_447 = None
    permute_311: "f32[48, 64, 197]" = torch.ops.aten.permute.default(view_448, [0, 2, 1]);  view_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    alias_26: "f32[8, 6, 197, 197]" = torch.ops.aten.alias.default(alias_21);  alias_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_312: "f32[48, 64, 197]" = torch.ops.aten.permute.default(view_444, [0, 2, 1]);  view_444 = None
    permute_313: "f32[48, 197, 64]" = torch.ops.aten.permute.default(view_445, [0, 2, 1]);  view_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_318: "f32[384, 384]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_323: "f32[768, 384]" = torch.ops.aten.permute.default(permute_204, [1, 0]);  permute_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    permute_325: "f32[384, 384]" = torch.ops.aten.permute.default(permute_203, [1, 0]);  permute_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_329: "f32[24, 96]" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_333: "f32[96, 24]" = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    div_33: "f32[1568, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_53, 24);  rsqrt_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    permute_337: "f32[24, 24]" = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    permute_342: "f32[6272, 16, 16]" = torch.ops.aten.permute.default(view_425, [0, 2, 1]);  view_425 = None
    permute_343: "f32[6272, 6, 16]" = torch.ops.aten.permute.default(view_426, [0, 2, 1]);  view_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    alias_27: "f32[1568, 4, 16, 16]" = torch.ops.aten.alias.default(alias_20);  alias_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_344: "f32[6272, 6, 16]" = torch.ops.aten.permute.default(view_422, [0, 2, 1]);  view_422 = None
    permute_345: "f32[6272, 16, 6]" = torch.ops.aten.permute.default(view_423, [0, 2, 1]);  view_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_350: "f32[24, 24]" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_355: "f32[48, 24]" = torch.ops.aten.permute.default(permute_194, [1, 0]);  permute_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    div_34: "f32[1568, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_49, 24);  rsqrt_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_357: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_193, [1, 0]);  permute_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_361: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_192, [1, 0]);  permute_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    div_35: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_51, 384);  rsqrt_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    permute_365: "f32[384, 384]" = torch.ops.aten.permute.default(permute_191, [1, 0]);  permute_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    permute_370: "f32[48, 197, 197]" = torch.ops.aten.permute.default(view_406, [0, 2, 1]);  view_406 = None
    permute_371: "f32[48, 64, 197]" = torch.ops.aten.permute.default(view_407, [0, 2, 1]);  view_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    alias_28: "f32[8, 6, 197, 197]" = torch.ops.aten.alias.default(alias_19);  alias_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_372: "f32[48, 64, 197]" = torch.ops.aten.permute.default(view_403, [0, 2, 1]);  view_403 = None
    permute_373: "f32[48, 197, 64]" = torch.ops.aten.permute.default(view_404, [0, 2, 1]);  view_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_378: "f32[384, 384]" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_383: "f32[768, 384]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    permute_385: "f32[384, 384]" = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_389: "f32[24, 96]" = torch.ops.aten.permute.default(permute_183, [1, 0]);  permute_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_393: "f32[96, 24]" = torch.ops.aten.permute.default(permute_182, [1, 0]);  permute_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    div_38: "f32[1568, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_48, 24);  rsqrt_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    permute_397: "f32[24, 24]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    permute_402: "f32[6272, 16, 16]" = torch.ops.aten.permute.default(view_384, [0, 2, 1]);  view_384 = None
    permute_403: "f32[6272, 6, 16]" = torch.ops.aten.permute.default(view_385, [0, 2, 1]);  view_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    alias_29: "f32[1568, 4, 16, 16]" = torch.ops.aten.alias.default(alias_18);  alias_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_404: "f32[6272, 6, 16]" = torch.ops.aten.permute.default(view_381, [0, 2, 1]);  view_381 = None
    permute_405: "f32[6272, 16, 6]" = torch.ops.aten.permute.default(view_382, [0, 2, 1]);  view_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_410: "f32[24, 24]" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_415: "f32[48, 24]" = torch.ops.aten.permute.default(permute_175, [1, 0]);  permute_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    div_39: "f32[1568, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 24);  rsqrt_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_417: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_421: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    div_40: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_46, 384);  rsqrt_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    permute_425: "f32[384, 384]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    permute_430: "f32[48, 197, 197]" = torch.ops.aten.permute.default(view_365, [0, 2, 1]);  view_365 = None
    permute_431: "f32[48, 64, 197]" = torch.ops.aten.permute.default(view_366, [0, 2, 1]);  view_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    alias_30: "f32[8, 6, 197, 197]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_432: "f32[48, 64, 197]" = torch.ops.aten.permute.default(view_362, [0, 2, 1]);  view_362 = None
    permute_433: "f32[48, 197, 64]" = torch.ops.aten.permute.default(view_363, [0, 2, 1]);  view_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_438: "f32[384, 384]" = torch.ops.aten.permute.default(permute_168, [1, 0]);  permute_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_443: "f32[768, 384]" = torch.ops.aten.permute.default(permute_166, [1, 0]);  permute_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    permute_445: "f32[384, 384]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_449: "f32[24, 96]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_453: "f32[96, 24]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    div_43: "f32[1568, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 24);  rsqrt_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    permute_457: "f32[24, 24]" = torch.ops.aten.permute.default(permute_162, [1, 0]);  permute_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    permute_462: "f32[6272, 16, 16]" = torch.ops.aten.permute.default(view_343, [0, 2, 1]);  view_343 = None
    permute_463: "f32[6272, 6, 16]" = torch.ops.aten.permute.default(view_344, [0, 2, 1]);  view_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    alias_31: "f32[1568, 4, 16, 16]" = torch.ops.aten.alias.default(alias_16);  alias_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_464: "f32[6272, 6, 16]" = torch.ops.aten.permute.default(view_340, [0, 2, 1]);  view_340 = None
    permute_465: "f32[6272, 16, 6]" = torch.ops.aten.permute.default(view_341, [0, 2, 1]);  view_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_470: "f32[24, 24]" = torch.ops.aten.permute.default(permute_158, [1, 0]);  permute_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_475: "f32[48, 24]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    div_44: "f32[1568, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 24);  rsqrt_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_477: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_155, [1, 0]);  permute_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_481: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    div_45: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_41, 384);  rsqrt_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    permute_485: "f32[384, 384]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    permute_490: "f32[48, 197, 197]" = torch.ops.aten.permute.default(view_324, [0, 2, 1]);  view_324 = None
    permute_491: "f32[48, 64, 197]" = torch.ops.aten.permute.default(view_325, [0, 2, 1]);  view_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    alias_32: "f32[8, 6, 197, 197]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_492: "f32[48, 64, 197]" = torch.ops.aten.permute.default(view_321, [0, 2, 1]);  view_321 = None
    permute_493: "f32[48, 197, 64]" = torch.ops.aten.permute.default(view_322, [0, 2, 1]);  view_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_498: "f32[384, 384]" = torch.ops.aten.permute.default(permute_149, [1, 0]);  permute_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_503: "f32[768, 384]" = torch.ops.aten.permute.default(permute_147, [1, 0]);  permute_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    permute_505: "f32[384, 384]" = torch.ops.aten.permute.default(permute_146, [1, 0]);  permute_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_509: "f32[24, 96]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_513: "f32[96, 24]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    div_48: "f32[1568, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 24);  rsqrt_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    permute_517: "f32[24, 24]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    permute_522: "f32[6272, 16, 16]" = torch.ops.aten.permute.default(view_302, [0, 2, 1]);  view_302 = None
    permute_523: "f32[6272, 6, 16]" = torch.ops.aten.permute.default(view_303, [0, 2, 1]);  view_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    alias_33: "f32[1568, 4, 16, 16]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_524: "f32[6272, 6, 16]" = torch.ops.aten.permute.default(view_299, [0, 2, 1]);  view_299 = None
    permute_525: "f32[6272, 16, 6]" = torch.ops.aten.permute.default(view_300, [0, 2, 1]);  view_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_530: "f32[24, 24]" = torch.ops.aten.permute.default(permute_139, [1, 0]);  permute_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_535: "f32[48, 24]" = torch.ops.aten.permute.default(permute_137, [1, 0]);  permute_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    div_49: "f32[1568, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 24);  rsqrt_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_537: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_541: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_135, [1, 0]);  permute_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    div_50: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 384);  rsqrt_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    permute_545: "f32[384, 384]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    permute_550: "f32[48, 197, 197]" = torch.ops.aten.permute.default(view_283, [0, 2, 1]);  view_283 = None
    permute_551: "f32[48, 64, 197]" = torch.ops.aten.permute.default(view_284, [0, 2, 1]);  view_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    alias_34: "f32[8, 6, 197, 197]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_552: "f32[48, 64, 197]" = torch.ops.aten.permute.default(view_280, [0, 2, 1]);  view_280 = None
    permute_553: "f32[48, 197, 64]" = torch.ops.aten.permute.default(view_281, [0, 2, 1]);  view_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_558: "f32[384, 384]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_563: "f32[768, 384]" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    permute_565: "f32[384, 384]" = torch.ops.aten.permute.default(permute_127, [1, 0]);  permute_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_569: "f32[24, 96]" = torch.ops.aten.permute.default(permute_126, [1, 0]);  permute_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_573: "f32[96, 24]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    div_53: "f32[1568, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 24);  rsqrt_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    permute_577: "f32[24, 24]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    permute_582: "f32[6272, 16, 16]" = torch.ops.aten.permute.default(view_261, [0, 2, 1]);  view_261 = None
    permute_583: "f32[6272, 6, 16]" = torch.ops.aten.permute.default(view_262, [0, 2, 1]);  view_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    alias_35: "f32[1568, 4, 16, 16]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_584: "f32[6272, 6, 16]" = torch.ops.aten.permute.default(view_258, [0, 2, 1]);  view_258 = None
    permute_585: "f32[6272, 16, 6]" = torch.ops.aten.permute.default(view_259, [0, 2, 1]);  view_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_590: "f32[24, 24]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_595: "f32[48, 24]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    div_54: "f32[1568, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 24);  rsqrt_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_597: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_117, [1, 0]);  permute_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_601: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    div_55: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 384);  rsqrt_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    permute_605: "f32[384, 384]" = torch.ops.aten.permute.default(permute_115, [1, 0]);  permute_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    permute_610: "f32[48, 197, 197]" = torch.ops.aten.permute.default(view_242, [0, 2, 1]);  view_242 = None
    permute_611: "f32[48, 64, 197]" = torch.ops.aten.permute.default(view_243, [0, 2, 1]);  view_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    alias_36: "f32[8, 6, 197, 197]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_612: "f32[48, 64, 197]" = torch.ops.aten.permute.default(view_239, [0, 2, 1]);  view_239 = None
    permute_613: "f32[48, 197, 64]" = torch.ops.aten.permute.default(view_240, [0, 2, 1]);  view_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_618: "f32[384, 384]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_623: "f32[768, 384]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    permute_625: "f32[384, 384]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_629: "f32[24, 96]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_633: "f32[96, 24]" = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    div_58: "f32[1568, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 24);  rsqrt_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    permute_637: "f32[24, 24]" = torch.ops.aten.permute.default(permute_105, [1, 0]);  permute_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    permute_642: "f32[6272, 16, 16]" = torch.ops.aten.permute.default(view_220, [0, 2, 1]);  view_220 = None
    permute_643: "f32[6272, 6, 16]" = torch.ops.aten.permute.default(view_221, [0, 2, 1]);  view_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    alias_37: "f32[1568, 4, 16, 16]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_644: "f32[6272, 6, 16]" = torch.ops.aten.permute.default(view_217, [0, 2, 1]);  view_217 = None
    permute_645: "f32[6272, 16, 6]" = torch.ops.aten.permute.default(view_218, [0, 2, 1]);  view_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_650: "f32[24, 24]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_655: "f32[48, 24]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    div_59: "f32[1568, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 24);  rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_657: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_661: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    div_60: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 384);  rsqrt_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    permute_665: "f32[384, 384]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    permute_670: "f32[48, 197, 197]" = torch.ops.aten.permute.default(view_201, [0, 2, 1]);  view_201 = None
    permute_671: "f32[48, 64, 197]" = torch.ops.aten.permute.default(view_202, [0, 2, 1]);  view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    alias_38: "f32[8, 6, 197, 197]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_672: "f32[48, 64, 197]" = torch.ops.aten.permute.default(view_198, [0, 2, 1]);  view_198 = None
    permute_673: "f32[48, 197, 64]" = torch.ops.aten.permute.default(view_199, [0, 2, 1]);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_678: "f32[384, 384]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_683: "f32[768, 384]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    permute_685: "f32[384, 384]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_689: "f32[24, 96]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_693: "f32[96, 24]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    div_63: "f32[1568, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 24);  rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    permute_697: "f32[24, 24]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    permute_702: "f32[6272, 16, 16]" = torch.ops.aten.permute.default(view_179, [0, 2, 1]);  view_179 = None
    permute_703: "f32[6272, 6, 16]" = torch.ops.aten.permute.default(view_180, [0, 2, 1]);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    alias_39: "f32[1568, 4, 16, 16]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_704: "f32[6272, 6, 16]" = torch.ops.aten.permute.default(view_176, [0, 2, 1]);  view_176 = None
    permute_705: "f32[6272, 16, 6]" = torch.ops.aten.permute.default(view_177, [0, 2, 1]);  view_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_710: "f32[24, 24]" = torch.ops.aten.permute.default(permute_82, [1, 0]);  permute_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_715: "f32[48, 24]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    div_64: "f32[1568, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 24);  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_717: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_721: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    div_65: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 384);  rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    permute_725: "f32[384, 384]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    permute_730: "f32[48, 197, 197]" = torch.ops.aten.permute.default(view_160, [0, 2, 1]);  view_160 = None
    permute_731: "f32[48, 64, 197]" = torch.ops.aten.permute.default(view_161, [0, 2, 1]);  view_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    alias_40: "f32[8, 6, 197, 197]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_732: "f32[48, 64, 197]" = torch.ops.aten.permute.default(view_157, [0, 2, 1]);  view_157 = None
    permute_733: "f32[48, 197, 64]" = torch.ops.aten.permute.default(view_158, [0, 2, 1]);  view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_738: "f32[384, 384]" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_743: "f32[768, 384]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    permute_745: "f32[384, 384]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_749: "f32[24, 96]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_753: "f32[96, 24]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    div_68: "f32[1568, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 24);  rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    permute_757: "f32[24, 24]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    permute_762: "f32[6272, 16, 16]" = torch.ops.aten.permute.default(view_138, [0, 2, 1]);  view_138 = None
    permute_763: "f32[6272, 6, 16]" = torch.ops.aten.permute.default(view_139, [0, 2, 1]);  view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    alias_41: "f32[1568, 4, 16, 16]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_764: "f32[6272, 6, 16]" = torch.ops.aten.permute.default(view_135, [0, 2, 1]);  view_135 = None
    permute_765: "f32[6272, 16, 6]" = torch.ops.aten.permute.default(view_136, [0, 2, 1]);  view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_770: "f32[24, 24]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_775: "f32[48, 24]" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    div_69: "f32[1568, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 24);  rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_777: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_781: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    div_70: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 384);  rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    permute_785: "f32[384, 384]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    permute_790: "f32[48, 197, 197]" = torch.ops.aten.permute.default(view_119, [0, 2, 1]);  view_119 = None
    permute_791: "f32[48, 64, 197]" = torch.ops.aten.permute.default(view_120, [0, 2, 1]);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    alias_42: "f32[8, 6, 197, 197]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_792: "f32[48, 64, 197]" = torch.ops.aten.permute.default(view_116, [0, 2, 1]);  view_116 = None
    permute_793: "f32[48, 197, 64]" = torch.ops.aten.permute.default(view_117, [0, 2, 1]);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_798: "f32[384, 384]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_803: "f32[768, 384]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    permute_805: "f32[384, 384]" = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_809: "f32[24, 96]" = torch.ops.aten.permute.default(permute_50, [1, 0]);  permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_813: "f32[96, 24]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    div_73: "f32[1568, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 24);  rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    permute_817: "f32[24, 24]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    permute_822: "f32[6272, 16, 16]" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
    permute_823: "f32[6272, 6, 16]" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    alias_43: "f32[1568, 4, 16, 16]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_824: "f32[6272, 6, 16]" = torch.ops.aten.permute.default(view_94, [0, 2, 1]);  view_94 = None
    permute_825: "f32[6272, 16, 6]" = torch.ops.aten.permute.default(view_95, [0, 2, 1]);  view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_830: "f32[24, 24]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_835: "f32[48, 24]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    div_74: "f32[1568, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 24);  rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_837: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_841: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    div_75: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 384);  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    permute_845: "f32[384, 384]" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    permute_850: "f32[48, 197, 197]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
    permute_851: "f32[48, 64, 197]" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    alias_44: "f32[8, 6, 197, 197]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_852: "f32[48, 64, 197]" = torch.ops.aten.permute.default(view_75, [0, 2, 1]);  view_75 = None
    permute_853: "f32[48, 197, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1]);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_858: "f32[384, 384]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_863: "f32[768, 384]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    permute_865: "f32[384, 384]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_869: "f32[24, 96]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_873: "f32[96, 24]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    div_78: "f32[1568, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 24);  rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    permute_877: "f32[24, 24]" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    permute_882: "f32[6272, 16, 16]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
    permute_883: "f32[6272, 6, 16]" = torch.ops.aten.permute.default(view_57, [0, 2, 1]);  view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    alias_45: "f32[1568, 4, 16, 16]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_884: "f32[6272, 6, 16]" = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None
    permute_885: "f32[6272, 16, 6]" = torch.ops.aten.permute.default(view_54, [0, 2, 1]);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_890: "f32[24, 24]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_895: "f32[48, 24]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    div_79: "f32[1568, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 24);  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_897: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_901: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    div_80: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 384);  rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    permute_905: "f32[384, 384]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    permute_910: "f32[48, 197, 197]" = torch.ops.aten.permute.default(view_37, [0, 2, 1]);  view_37 = None
    permute_911: "f32[48, 64, 197]" = torch.ops.aten.permute.default(view_38, [0, 2, 1]);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    alias_46: "f32[8, 6, 197, 197]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_912: "f32[48, 64, 197]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
    permute_913: "f32[48, 197, 64]" = torch.ops.aten.permute.default(view_35, [0, 2, 1]);  view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_918: "f32[384, 384]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_923: "f32[768, 384]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    permute_925: "f32[384, 384]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_929: "f32[24, 96]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_933: "f32[96, 24]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    div_83: "f32[1568, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 24);  rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    permute_937: "f32[24, 24]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    permute_942: "f32[6272, 16, 16]" = torch.ops.aten.permute.default(view_15, [0, 2, 1]);  view_15 = None
    permute_943: "f32[6272, 6, 16]" = torch.ops.aten.permute.default(view_16, [0, 2, 1]);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    alias_47: "f32[1568, 4, 16, 16]" = torch.ops.aten.alias.default(alias);  alias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_944: "f32[6272, 6, 16]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
    permute_945: "f32[6272, 16, 6]" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_950: "f32[24, 24]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_955: "f32[48, 24]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:311, code: patch_embed = self.norm2_proj(self.proj(self.norm1_proj(pixel_embed.reshape(B, self.num_patches, -1))))
    permute_957: "f32[384, 384]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    return [addmm_85, primals_4, primals_6, primals_10, primals_12, primals_18, primals_24, primals_28, primals_34, primals_40, primals_46, primals_52, primals_56, primals_62, primals_68, primals_74, primals_80, primals_84, primals_90, primals_96, primals_102, primals_108, primals_112, primals_118, primals_124, primals_130, primals_136, primals_140, primals_146, primals_152, primals_158, primals_164, primals_168, primals_174, primals_180, primals_186, primals_192, primals_196, primals_202, primals_208, primals_214, primals_220, primals_224, primals_230, primals_236, primals_242, primals_248, primals_252, primals_258, primals_264, primals_270, primals_276, primals_280, primals_286, primals_292, primals_298, primals_304, primals_308, primals_314, primals_320, primals_326, primals_332, primals_336, primals_342, primals_348, primals_352, add, unsqueeze_5, clone_2, getitem_1, rsqrt, view_4, addmm, getitem_3, rsqrt_1, getitem_5, rsqrt_2, view_6, view_19, mul_7, view_21, addmm_2, view_23, mul_12, view_26, cat_1, getitem_13, rsqrt_5, view_28, view_41, mul_17, view_43, addmm_6, view_45, view_47, view_60, mul_25, view_62, addmm_9, view_64, mul_30, view_67, cat_2, getitem_27, rsqrt_10, view_69, view_82, mul_35, view_84, addmm_13, view_86, view_88, view_101, mul_43, view_103, addmm_16, view_105, mul_48, view_108, cat_3, getitem_41, rsqrt_15, view_110, view_123, mul_53, view_125, addmm_20, view_127, view_129, view_142, mul_61, view_144, addmm_23, view_146, mul_66, view_149, cat_4, getitem_55, rsqrt_20, view_151, view_164, mul_71, view_166, addmm_27, view_168, view_170, view_183, mul_79, view_185, addmm_30, view_187, mul_84, view_190, cat_5, getitem_69, rsqrt_25, view_192, view_205, mul_89, view_207, addmm_34, view_209, view_211, view_224, mul_97, view_226, addmm_37, view_228, mul_102, view_231, cat_6, getitem_83, rsqrt_30, view_233, view_246, mul_107, view_248, addmm_41, view_250, view_252, view_265, mul_115, view_267, addmm_44, view_269, mul_120, view_272, cat_7, getitem_97, rsqrt_35, view_274, view_287, mul_125, view_289, addmm_48, view_291, view_293, view_306, mul_133, view_308, addmm_51, view_310, mul_138, view_313, cat_8, getitem_111, rsqrt_40, view_315, view_328, mul_143, view_330, addmm_55, view_332, view_334, view_347, mul_151, view_349, addmm_58, view_351, mul_156, view_354, cat_9, getitem_125, rsqrt_45, view_356, view_369, mul_161, view_371, addmm_62, view_373, view_375, view_388, mul_169, view_390, addmm_65, view_392, mul_174, view_395, cat_10, getitem_139, rsqrt_50, view_397, view_410, mul_179, view_412, addmm_69, view_414, view_416, view_429, mul_187, view_431, addmm_72, view_433, mul_192, view_436, cat_11, getitem_153, rsqrt_55, view_438, view_451, mul_197, view_453, addmm_76, view_455, view_457, view_470, mul_205, view_472, addmm_79, view_474, mul_210, view_477, cat_12, getitem_167, rsqrt_60, view_479, view_492, mul_215, view_494, addmm_83, view_496, mul_220, clone_184, permute_233, div_24, permute_237, permute_241, div_25, permute_245, permute_250, permute_251, alias_24, permute_252, permute_253, permute_258, permute_263, permute_265, div_27, permute_269, permute_273, div_28, permute_277, permute_282, permute_283, alias_25, permute_284, permute_285, permute_290, permute_295, div_29, permute_297, permute_301, div_30, permute_305, permute_310, permute_311, alias_26, permute_312, permute_313, permute_318, permute_323, permute_325, permute_329, permute_333, div_33, permute_337, permute_342, permute_343, alias_27, permute_344, permute_345, permute_350, permute_355, div_34, permute_357, permute_361, div_35, permute_365, permute_370, permute_371, alias_28, permute_372, permute_373, permute_378, permute_383, permute_385, permute_389, permute_393, div_38, permute_397, permute_402, permute_403, alias_29, permute_404, permute_405, permute_410, permute_415, div_39, permute_417, permute_421, div_40, permute_425, permute_430, permute_431, alias_30, permute_432, permute_433, permute_438, permute_443, permute_445, permute_449, permute_453, div_43, permute_457, permute_462, permute_463, alias_31, permute_464, permute_465, permute_470, permute_475, div_44, permute_477, permute_481, div_45, permute_485, permute_490, permute_491, alias_32, permute_492, permute_493, permute_498, permute_503, permute_505, permute_509, permute_513, div_48, permute_517, permute_522, permute_523, alias_33, permute_524, permute_525, permute_530, permute_535, div_49, permute_537, permute_541, div_50, permute_545, permute_550, permute_551, alias_34, permute_552, permute_553, permute_558, permute_563, permute_565, permute_569, permute_573, div_53, permute_577, permute_582, permute_583, alias_35, permute_584, permute_585, permute_590, permute_595, div_54, permute_597, permute_601, div_55, permute_605, permute_610, permute_611, alias_36, permute_612, permute_613, permute_618, permute_623, permute_625, permute_629, permute_633, div_58, permute_637, permute_642, permute_643, alias_37, permute_644, permute_645, permute_650, permute_655, div_59, permute_657, permute_661, div_60, permute_665, permute_670, permute_671, alias_38, permute_672, permute_673, permute_678, permute_683, permute_685, permute_689, permute_693, div_63, permute_697, permute_702, permute_703, alias_39, permute_704, permute_705, permute_710, permute_715, div_64, permute_717, permute_721, div_65, permute_725, permute_730, permute_731, alias_40, permute_732, permute_733, permute_738, permute_743, permute_745, permute_749, permute_753, div_68, permute_757, permute_762, permute_763, alias_41, permute_764, permute_765, permute_770, permute_775, div_69, permute_777, permute_781, div_70, permute_785, permute_790, permute_791, alias_42, permute_792, permute_793, permute_798, permute_803, permute_805, permute_809, permute_813, div_73, permute_817, permute_822, permute_823, alias_43, permute_824, permute_825, permute_830, permute_835, div_74, permute_837, permute_841, div_75, permute_845, permute_850, permute_851, alias_44, permute_852, permute_853, permute_858, permute_863, permute_865, permute_869, permute_873, div_78, permute_877, permute_882, permute_883, alias_45, permute_884, permute_885, permute_890, permute_895, div_79, permute_897, permute_901, div_80, permute_905, permute_910, permute_911, alias_46, permute_912, permute_913, permute_918, permute_923, permute_925, permute_929, permute_933, div_83, permute_937, permute_942, permute_943, alias_47, permute_944, permute_945, permute_950, permute_955, permute_957]
    