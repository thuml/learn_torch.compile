from __future__ import annotations



def forward(self, primals_1: "f32[169, 4]", primals_2: "f32[169, 4]", primals_3: "f32[169, 8]", primals_4: "f32[169, 8]", primals_5: "f32[169, 16]", primals_6: "f32[169, 16]", primals_7: "f32[169, 16]", primals_8: "f32[169, 16]", primals_9: "f32[169, 16]", primals_10: "f32[169, 16]", primals_11: "f32[169, 16]", primals_12: "f32[169, 16]", primals_13: "f32[169, 16]", primals_14: "f32[169, 16]", primals_15: "f32[169, 16]", primals_16: "f32[169, 16]", primals_17: "f32[169, 16]", primals_18: "f32[169, 16]", primals_19: "f32[169, 16]", primals_20: "f32[169, 16]", primals_21: "f32[169, 16]", primals_22: "f32[169, 16]", primals_23: "f32[169, 32]", primals_24: "f32[169, 32]", primals_25: "f32[128, 3, 4, 4]", primals_26: "f32[128]", primals_27: "f32[128]", primals_28: "f32[128]", primals_29: "f32[128]", primals_30: "f32[128]", primals_31: "f32[384, 128]", primals_32: "f32[384]", primals_33: "f32[128, 128]", primals_34: "f32[128]", primals_35: "f32[128]", primals_36: "f32[128]", primals_37: "f32[512, 128]", primals_38: "f32[512]", primals_39: "f32[128, 512]", primals_40: "f32[128]", primals_41: "f32[128]", primals_42: "f32[128]", primals_43: "f32[384, 128]", primals_44: "f32[384]", primals_45: "f32[128, 128]", primals_46: "f32[128]", primals_47: "f32[128]", primals_48: "f32[128]", primals_49: "f32[512, 128]", primals_50: "f32[512]", primals_51: "f32[128, 512]", primals_52: "f32[128]", primals_53: "f32[512]", primals_54: "f32[512]", primals_55: "f32[256, 512]", primals_56: "f32[256]", primals_57: "f32[256]", primals_58: "f32[768, 256]", primals_59: "f32[768]", primals_60: "f32[256, 256]", primals_61: "f32[256]", primals_62: "f32[256]", primals_63: "f32[256]", primals_64: "f32[1024, 256]", primals_65: "f32[1024]", primals_66: "f32[256, 1024]", primals_67: "f32[256]", primals_68: "f32[256]", primals_69: "f32[256]", primals_70: "f32[768, 256]", primals_71: "f32[768]", primals_72: "f32[256, 256]", primals_73: "f32[256]", primals_74: "f32[256]", primals_75: "f32[256]", primals_76: "f32[1024, 256]", primals_77: "f32[1024]", primals_78: "f32[256, 1024]", primals_79: "f32[256]", primals_80: "f32[1024]", primals_81: "f32[1024]", primals_82: "f32[512, 1024]", primals_83: "f32[512]", primals_84: "f32[512]", primals_85: "f32[1536, 512]", primals_86: "f32[1536]", primals_87: "f32[512, 512]", primals_88: "f32[512]", primals_89: "f32[512]", primals_90: "f32[512]", primals_91: "f32[2048, 512]", primals_92: "f32[2048]", primals_93: "f32[512, 2048]", primals_94: "f32[512]", primals_95: "f32[512]", primals_96: "f32[512]", primals_97: "f32[1536, 512]", primals_98: "f32[1536]", primals_99: "f32[512, 512]", primals_100: "f32[512]", primals_101: "f32[512]", primals_102: "f32[512]", primals_103: "f32[2048, 512]", primals_104: "f32[2048]", primals_105: "f32[512, 2048]", primals_106: "f32[512]", primals_107: "f32[512]", primals_108: "f32[512]", primals_109: "f32[1536, 512]", primals_110: "f32[1536]", primals_111: "f32[512, 512]", primals_112: "f32[512]", primals_113: "f32[512]", primals_114: "f32[512]", primals_115: "f32[2048, 512]", primals_116: "f32[2048]", primals_117: "f32[512, 2048]", primals_118: "f32[512]", primals_119: "f32[512]", primals_120: "f32[512]", primals_121: "f32[1536, 512]", primals_122: "f32[1536]", primals_123: "f32[512, 512]", primals_124: "f32[512]", primals_125: "f32[512]", primals_126: "f32[512]", primals_127: "f32[2048, 512]", primals_128: "f32[2048]", primals_129: "f32[512, 2048]", primals_130: "f32[512]", primals_131: "f32[512]", primals_132: "f32[512]", primals_133: "f32[1536, 512]", primals_134: "f32[1536]", primals_135: "f32[512, 512]", primals_136: "f32[512]", primals_137: "f32[512]", primals_138: "f32[512]", primals_139: "f32[2048, 512]", primals_140: "f32[2048]", primals_141: "f32[512, 2048]", primals_142: "f32[512]", primals_143: "f32[512]", primals_144: "f32[512]", primals_145: "f32[1536, 512]", primals_146: "f32[1536]", primals_147: "f32[512, 512]", primals_148: "f32[512]", primals_149: "f32[512]", primals_150: "f32[512]", primals_151: "f32[2048, 512]", primals_152: "f32[2048]", primals_153: "f32[512, 2048]", primals_154: "f32[512]", primals_155: "f32[512]", primals_156: "f32[512]", primals_157: "f32[1536, 512]", primals_158: "f32[1536]", primals_159: "f32[512, 512]", primals_160: "f32[512]", primals_161: "f32[512]", primals_162: "f32[512]", primals_163: "f32[2048, 512]", primals_164: "f32[2048]", primals_165: "f32[512, 2048]", primals_166: "f32[512]", primals_167: "f32[512]", primals_168: "f32[512]", primals_169: "f32[1536, 512]", primals_170: "f32[1536]", primals_171: "f32[512, 512]", primals_172: "f32[512]", primals_173: "f32[512]", primals_174: "f32[512]", primals_175: "f32[2048, 512]", primals_176: "f32[2048]", primals_177: "f32[512, 2048]", primals_178: "f32[512]", primals_179: "f32[512]", primals_180: "f32[512]", primals_181: "f32[1536, 512]", primals_182: "f32[1536]", primals_183: "f32[512, 512]", primals_184: "f32[512]", primals_185: "f32[512]", primals_186: "f32[512]", primals_187: "f32[2048, 512]", primals_188: "f32[2048]", primals_189: "f32[512, 2048]", primals_190: "f32[512]", primals_191: "f32[512]", primals_192: "f32[512]", primals_193: "f32[1536, 512]", primals_194: "f32[1536]", primals_195: "f32[512, 512]", primals_196: "f32[512]", primals_197: "f32[512]", primals_198: "f32[512]", primals_199: "f32[2048, 512]", primals_200: "f32[2048]", primals_201: "f32[512, 2048]", primals_202: "f32[512]", primals_203: "f32[512]", primals_204: "f32[512]", primals_205: "f32[1536, 512]", primals_206: "f32[1536]", primals_207: "f32[512, 512]", primals_208: "f32[512]", primals_209: "f32[512]", primals_210: "f32[512]", primals_211: "f32[2048, 512]", primals_212: "f32[2048]", primals_213: "f32[512, 2048]", primals_214: "f32[512]", primals_215: "f32[512]", primals_216: "f32[512]", primals_217: "f32[1536, 512]", primals_218: "f32[1536]", primals_219: "f32[512, 512]", primals_220: "f32[512]", primals_221: "f32[512]", primals_222: "f32[512]", primals_223: "f32[2048, 512]", primals_224: "f32[2048]", primals_225: "f32[512, 2048]", primals_226: "f32[512]", primals_227: "f32[512]", primals_228: "f32[512]", primals_229: "f32[1536, 512]", primals_230: "f32[1536]", primals_231: "f32[512, 512]", primals_232: "f32[512]", primals_233: "f32[512]", primals_234: "f32[512]", primals_235: "f32[2048, 512]", primals_236: "f32[2048]", primals_237: "f32[512, 2048]", primals_238: "f32[512]", primals_239: "f32[512]", primals_240: "f32[512]", primals_241: "f32[1536, 512]", primals_242: "f32[1536]", primals_243: "f32[512, 512]", primals_244: "f32[512]", primals_245: "f32[512]", primals_246: "f32[512]", primals_247: "f32[2048, 512]", primals_248: "f32[2048]", primals_249: "f32[512, 2048]", primals_250: "f32[512]", primals_251: "f32[512]", primals_252: "f32[512]", primals_253: "f32[1536, 512]", primals_254: "f32[1536]", primals_255: "f32[512, 512]", primals_256: "f32[512]", primals_257: "f32[512]", primals_258: "f32[512]", primals_259: "f32[2048, 512]", primals_260: "f32[2048]", primals_261: "f32[512, 2048]", primals_262: "f32[512]", primals_263: "f32[512]", primals_264: "f32[512]", primals_265: "f32[1536, 512]", primals_266: "f32[1536]", primals_267: "f32[512, 512]", primals_268: "f32[512]", primals_269: "f32[512]", primals_270: "f32[512]", primals_271: "f32[2048, 512]", primals_272: "f32[2048]", primals_273: "f32[512, 2048]", primals_274: "f32[512]", primals_275: "f32[512]", primals_276: "f32[512]", primals_277: "f32[1536, 512]", primals_278: "f32[1536]", primals_279: "f32[512, 512]", primals_280: "f32[512]", primals_281: "f32[512]", primals_282: "f32[512]", primals_283: "f32[2048, 512]", primals_284: "f32[2048]", primals_285: "f32[512, 2048]", primals_286: "f32[512]", primals_287: "f32[512]", primals_288: "f32[512]", primals_289: "f32[1536, 512]", primals_290: "f32[1536]", primals_291: "f32[512, 512]", primals_292: "f32[512]", primals_293: "f32[512]", primals_294: "f32[512]", primals_295: "f32[2048, 512]", primals_296: "f32[2048]", primals_297: "f32[512, 2048]", primals_298: "f32[512]", primals_299: "f32[2048]", primals_300: "f32[2048]", primals_301: "f32[1024, 2048]", primals_302: "f32[1024]", primals_303: "f32[1024]", primals_304: "f32[3072, 1024]", primals_305: "f32[3072]", primals_306: "f32[1024, 1024]", primals_307: "f32[1024]", primals_308: "f32[1024]", primals_309: "f32[1024]", primals_310: "f32[4096, 1024]", primals_311: "f32[4096]", primals_312: "f32[1024, 4096]", primals_313: "f32[1024]", primals_314: "f32[1024]", primals_315: "f32[1024]", primals_316: "f32[3072, 1024]", primals_317: "f32[3072]", primals_318: "f32[1024, 1024]", primals_319: "f32[1024]", primals_320: "f32[1024]", primals_321: "f32[1024]", primals_322: "f32[4096, 1024]", primals_323: "f32[4096]", primals_324: "f32[1024, 4096]", primals_325: "f32[1024]", primals_326: "f32[1024]", primals_327: "f32[1024]", primals_328: "f32[1000, 1024]", primals_329: "f32[1000]", primals_330: "i64[49, 49]", primals_331: "f32[64, 49, 49]", primals_332: "i64[49, 49]", primals_333: "i64[49, 49]", primals_334: "f32[16, 49, 49]", primals_335: "i64[49, 49]", primals_336: "i64[49, 49]", primals_337: "f32[4, 49, 49]", primals_338: "i64[49, 49]", primals_339: "i64[49, 49]", primals_340: "f32[4, 49, 49]", primals_341: "i64[49, 49]", primals_342: "i64[49, 49]", primals_343: "f32[4, 49, 49]", primals_344: "i64[49, 49]", primals_345: "i64[49, 49]", primals_346: "f32[4, 49, 49]", primals_347: "i64[49, 49]", primals_348: "i64[49, 49]", primals_349: "f32[4, 49, 49]", primals_350: "i64[49, 49]", primals_351: "i64[49, 49]", primals_352: "f32[4, 49, 49]", primals_353: "i64[49, 49]", primals_354: "i64[49, 49]", primals_355: "f32[4, 49, 49]", primals_356: "i64[49, 49]", primals_357: "i64[49, 49]", primals_358: "f32[4, 49, 49]", primals_359: "i64[49, 49]", primals_360: "i64[49, 49]", primals_361: "f32[4, 49, 49]", primals_362: "i64[49, 49]", primals_363: "i64[49, 49]", primals_364: "i64[49, 49]", primals_365: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution: "f32[8, 128, 56, 56]" = torch.ops.aten.convolution.default(primals_365, primals_25, primals_26, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  primals_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/format.py:43, code: x = x.permute(0, 2, 3, 1)
    permute: "f32[8, 56, 56, 128]" = torch.ops.aten.permute.default(convolution, [0, 2, 3, 1]);  convolution = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    native_layer_norm = torch.ops.aten.native_layer_norm.default(permute, [128], primals_27, primals_28, 1e-05)
    getitem: "f32[8, 56, 56, 128]" = native_layer_norm[0]
    getitem_1: "f32[8, 56, 56, 1]" = native_layer_norm[1]
    getitem_2: "f32[8, 56, 56, 1]" = native_layer_norm[2];  native_layer_norm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_1 = torch.ops.aten.native_layer_norm.default(getitem, [128], primals_29, primals_30, 1e-05)
    getitem_3: "f32[8, 56, 56, 128]" = native_layer_norm_1[0]
    getitem_4: "f32[8, 56, 56, 1]" = native_layer_norm_1[1]
    getitem_5: "f32[8, 56, 56, 1]" = native_layer_norm_1[2];  native_layer_norm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd: "f32[8, 56, 56, 128]" = torch.ops.aten.constant_pad_nd.default(getitem_3, [0, 0, 0, 0, 0, 0], 0.0);  getitem_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    view: "f32[8, 8, 7, 8, 7, 128]" = torch.ops.aten.view.default(constant_pad_nd, [8, 8, 7, 8, 7, 128]);  constant_pad_nd = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_1: "f32[8, 8, 8, 7, 7, 128]" = torch.ops.aten.permute.default(view, [0, 1, 3, 2, 4, 5]);  view = None
    clone: "f32[8, 8, 8, 7, 7, 128]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
    view_1: "f32[512, 7, 7, 128]" = torch.ops.aten.view.default(clone, [-1, 7, 7, 128]);  clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_2: "f32[512, 49, 128]" = torch.ops.aten.view.default(view_1, [-1, 49, 128]);  view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_3: "f32[25088, 128]" = torch.ops.aten.view.default(view_2, [25088, 128]);  view_2 = None
    t: "f32[128, 384]" = torch.ops.aten.t.default(primals_31);  primals_31 = None
    addmm: "f32[25088, 384]" = torch.ops.aten.addmm.default(primals_32, view_3, t);  primals_32 = None
    view_4: "f32[512, 49, 384]" = torch.ops.aten.view.default(addmm, [512, 49, 384]);  addmm = None
    view_5: "f32[512, 49, 3, 4, 32]" = torch.ops.aten.view.default(view_4, [512, 49, 3, 4, -1]);  view_4 = None
    permute_2: "f32[3, 512, 4, 49, 32]" = torch.ops.aten.permute.default(view_5, [2, 0, 3, 1, 4]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind = torch.ops.aten.unbind.int(permute_2);  permute_2 = None
    getitem_6: "f32[512, 4, 49, 32]" = unbind[0]
    getitem_7: "f32[512, 4, 49, 32]" = unbind[1]
    getitem_8: "f32[512, 4, 49, 32]" = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul: "f32[512, 4, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_6, 0.1767766952966369);  getitem_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose: "f32[512, 4, 32, 49]" = torch.ops.aten.transpose.int(getitem_7, -2, -1);  getitem_7 = None
    expand: "f32[512, 4, 49, 32]" = torch.ops.aten.expand.default(mul, [512, 4, 49, 32]);  mul = None
    clone_1: "f32[512, 4, 49, 32]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
    _unsafe_view: "f32[2048, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_1, [2048, 49, 32]);  clone_1 = None
    expand_1: "f32[512, 4, 32, 49]" = torch.ops.aten.expand.default(transpose, [512, 4, 32, 49]);  transpose = None
    clone_2: "f32[512, 4, 32, 49]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    _unsafe_view_1: "f32[2048, 32, 49]" = torch.ops.aten._unsafe_view.default(clone_2, [2048, 32, 49]);  clone_2 = None
    bmm: "f32[2048, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view, _unsafe_view_1)
    view_6: "f32[512, 4, 49, 49]" = torch.ops.aten.view.default(bmm, [512, 4, 49, 49]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_7: "i64[2401]" = torch.ops.aten.view.default(primals_330, [-1]);  primals_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index: "f32[2401, 4]" = torch.ops.aten.index.Tensor(primals_1, [view_7]);  primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_8: "f32[49, 49, 4]" = torch.ops.aten.view.default(index, [49, 49, -1]);  index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_3: "f32[4, 49, 49]" = torch.ops.aten.permute.default(view_8, [2, 0, 1]);  view_8 = None
    clone_3: "f32[4, 49, 49]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze: "f32[1, 4, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_3, 0);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    add: "f32[512, 4, 49, 49]" = torch.ops.aten.add.Tensor(view_6, unsqueeze);  view_6 = unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax: "f32[512, 4, 49, 49]" = torch.ops.aten._softmax.default(add, -1, False);  add = None
    detach: "f32[512, 4, 49, 49]" = torch.ops.aten.detach.default(_softmax)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    clone_4: "f32[512, 4, 49, 49]" = torch.ops.aten.clone.default(_softmax);  _softmax = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    expand_2: "f32[512, 4, 49, 49]" = torch.ops.aten.expand.default(clone_4, [512, 4, 49, 49]);  clone_4 = None
    view_9: "f32[2048, 49, 49]" = torch.ops.aten.view.default(expand_2, [2048, 49, 49]);  expand_2 = None
    expand_3: "f32[512, 4, 49, 32]" = torch.ops.aten.expand.default(getitem_8, [512, 4, 49, 32]);  getitem_8 = None
    clone_5: "f32[512, 4, 49, 32]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
    _unsafe_view_2: "f32[2048, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_5, [2048, 49, 32]);  clone_5 = None
    bmm_1: "f32[2048, 49, 32]" = torch.ops.aten.bmm.default(view_9, _unsafe_view_2)
    view_10: "f32[512, 4, 49, 32]" = torch.ops.aten.view.default(bmm_1, [512, 4, 49, 32]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_1: "f32[512, 49, 4, 32]" = torch.ops.aten.transpose.int(view_10, 1, 2);  view_10 = None
    clone_6: "f32[512, 49, 4, 32]" = torch.ops.aten.clone.default(transpose_1, memory_format = torch.contiguous_format);  transpose_1 = None
    _unsafe_view_3: "f32[512, 49, 128]" = torch.ops.aten._unsafe_view.default(clone_6, [512, 49, 128]);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_11: "f32[25088, 128]" = torch.ops.aten.view.default(_unsafe_view_3, [25088, 128]);  _unsafe_view_3 = None
    t_1: "f32[128, 128]" = torch.ops.aten.t.default(primals_33);  primals_33 = None
    addmm_1: "f32[25088, 128]" = torch.ops.aten.addmm.default(primals_34, view_11, t_1);  primals_34 = None
    view_12: "f32[512, 49, 128]" = torch.ops.aten.view.default(addmm_1, [512, 49, 128]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    clone_7: "f32[512, 49, 128]" = torch.ops.aten.clone.default(view_12);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_13: "f32[512, 7, 7, 128]" = torch.ops.aten.view.default(clone_7, [-1, 7, 7, 128]);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    view_14: "f32[8, 8, 8, 7, 7, 128]" = torch.ops.aten.view.default(view_13, [-1, 8, 8, 7, 7, 128]);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_4: "f32[8, 8, 7, 8, 7, 128]" = torch.ops.aten.permute.default(view_14, [0, 1, 3, 2, 4, 5]);  view_14 = None
    clone_8: "f32[8, 8, 7, 8, 7, 128]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    view_15: "f32[8, 56, 56, 128]" = torch.ops.aten.view.default(clone_8, [-1, 56, 56, 128]);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_1: "f32[8, 56, 56, 128]" = torch.ops.aten.slice.Tensor(view_15, 0, 0, 9223372036854775807);  view_15 = None
    slice_2: "f32[8, 56, 56, 128]" = torch.ops.aten.slice.Tensor(slice_1, 3, 0, 9223372036854775807);  slice_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_1: "f32[8, 56, 56, 128]" = torch.ops.aten.add.Tensor(getitem, slice_2);  slice_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_16: "f32[8, 3136, 128]" = torch.ops.aten.view.default(add_1, [8, -1, 128]);  add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_2 = torch.ops.aten.native_layer_norm.default(view_16, [128], primals_35, primals_36, 1e-05)
    getitem_9: "f32[8, 3136, 128]" = native_layer_norm_2[0]
    getitem_10: "f32[8, 3136, 1]" = native_layer_norm_2[1]
    getitem_11: "f32[8, 3136, 1]" = native_layer_norm_2[2];  native_layer_norm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_17: "f32[25088, 128]" = torch.ops.aten.view.default(getitem_9, [25088, 128]);  getitem_9 = None
    t_2: "f32[128, 512]" = torch.ops.aten.t.default(primals_37);  primals_37 = None
    addmm_2: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_38, view_17, t_2);  primals_38 = None
    view_18: "f32[8, 3136, 512]" = torch.ops.aten.view.default(addmm_2, [8, 3136, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu: "f32[8, 3136, 512]" = torch.ops.aten.gelu.default(view_18);  view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_9: "f32[8, 3136, 512]" = torch.ops.aten.clone.default(gelu);  gelu = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_19: "f32[25088, 512]" = torch.ops.aten.view.default(clone_9, [25088, 512]);  clone_9 = None
    t_3: "f32[512, 128]" = torch.ops.aten.t.default(primals_39);  primals_39 = None
    addmm_3: "f32[25088, 128]" = torch.ops.aten.addmm.default(primals_40, view_19, t_3);  primals_40 = None
    view_20: "f32[8, 3136, 128]" = torch.ops.aten.view.default(addmm_3, [8, 3136, 128]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_10: "f32[8, 3136, 128]" = torch.ops.aten.clone.default(view_20);  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_2: "f32[8, 3136, 128]" = torch.ops.aten.add.Tensor(view_16, clone_10);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_21: "f32[8, 56, 56, 128]" = torch.ops.aten.view.default(add_2, [8, 56, 56, 128]);  add_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_3 = torch.ops.aten.native_layer_norm.default(view_21, [128], primals_41, primals_42, 1e-05)
    getitem_12: "f32[8, 56, 56, 128]" = native_layer_norm_3[0]
    getitem_13: "f32[8, 56, 56, 1]" = native_layer_norm_3[1]
    getitem_14: "f32[8, 56, 56, 1]" = native_layer_norm_3[2];  native_layer_norm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    roll: "f32[8, 56, 56, 128]" = torch.ops.aten.roll.default(getitem_12, [-3, -3], [1, 2]);  getitem_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_1: "f32[8, 56, 56, 128]" = torch.ops.aten.constant_pad_nd.default(roll, [0, 0, 0, 0, 0, 0], 0.0);  roll = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    view_22: "f32[8, 8, 7, 8, 7, 128]" = torch.ops.aten.view.default(constant_pad_nd_1, [8, 8, 7, 8, 7, 128]);  constant_pad_nd_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_5: "f32[8, 8, 8, 7, 7, 128]" = torch.ops.aten.permute.default(view_22, [0, 1, 3, 2, 4, 5]);  view_22 = None
    clone_11: "f32[8, 8, 8, 7, 7, 128]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    view_23: "f32[512, 7, 7, 128]" = torch.ops.aten.view.default(clone_11, [-1, 7, 7, 128]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_24: "f32[512, 49, 128]" = torch.ops.aten.view.default(view_23, [-1, 49, 128]);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_25: "f32[25088, 128]" = torch.ops.aten.view.default(view_24, [25088, 128]);  view_24 = None
    t_4: "f32[128, 384]" = torch.ops.aten.t.default(primals_43);  primals_43 = None
    addmm_4: "f32[25088, 384]" = torch.ops.aten.addmm.default(primals_44, view_25, t_4);  primals_44 = None
    view_26: "f32[512, 49, 384]" = torch.ops.aten.view.default(addmm_4, [512, 49, 384]);  addmm_4 = None
    view_27: "f32[512, 49, 3, 4, 32]" = torch.ops.aten.view.default(view_26, [512, 49, 3, 4, -1]);  view_26 = None
    permute_6: "f32[3, 512, 4, 49, 32]" = torch.ops.aten.permute.default(view_27, [2, 0, 3, 1, 4]);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_1 = torch.ops.aten.unbind.int(permute_6);  permute_6 = None
    getitem_15: "f32[512, 4, 49, 32]" = unbind_1[0]
    getitem_16: "f32[512, 4, 49, 32]" = unbind_1[1]
    getitem_17: "f32[512, 4, 49, 32]" = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_1: "f32[512, 4, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_15, 0.1767766952966369);  getitem_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_2: "f32[512, 4, 32, 49]" = torch.ops.aten.transpose.int(getitem_16, -2, -1);  getitem_16 = None
    expand_4: "f32[512, 4, 49, 32]" = torch.ops.aten.expand.default(mul_1, [512, 4, 49, 32]);  mul_1 = None
    clone_12: "f32[512, 4, 49, 32]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
    _unsafe_view_4: "f32[2048, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_12, [2048, 49, 32]);  clone_12 = None
    expand_5: "f32[512, 4, 32, 49]" = torch.ops.aten.expand.default(transpose_2, [512, 4, 32, 49]);  transpose_2 = None
    clone_13: "f32[512, 4, 32, 49]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    _unsafe_view_5: "f32[2048, 32, 49]" = torch.ops.aten._unsafe_view.default(clone_13, [2048, 32, 49]);  clone_13 = None
    bmm_2: "f32[2048, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_4, _unsafe_view_5)
    view_28: "f32[512, 4, 49, 49]" = torch.ops.aten.view.default(bmm_2, [512, 4, 49, 49]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_29: "i64[2401]" = torch.ops.aten.view.default(primals_332, [-1]);  primals_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_1: "f32[2401, 4]" = torch.ops.aten.index.Tensor(primals_2, [view_29]);  primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_30: "f32[49, 49, 4]" = torch.ops.aten.view.default(index_1, [49, 49, -1]);  index_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_7: "f32[4, 49, 49]" = torch.ops.aten.permute.default(view_30, [2, 0, 1]);  view_30 = None
    clone_14: "f32[4, 49, 49]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_1: "f32[1, 4, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_14, 0);  clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    add_3: "f32[512, 4, 49, 49]" = torch.ops.aten.add.Tensor(view_28, unsqueeze_1);  view_28 = unsqueeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_31: "f32[8, 64, 4, 49, 49]" = torch.ops.aten.view.default(add_3, [-1, 64, 4, 49, 49]);  add_3 = None
    unsqueeze_2: "f32[64, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(primals_331, 1);  primals_331 = None
    unsqueeze_3: "f32[1, 64, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 0);  unsqueeze_2 = None
    add_4: "f32[8, 64, 4, 49, 49]" = torch.ops.aten.add.Tensor(view_31, unsqueeze_3);  view_31 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    view_32: "f32[512, 4, 49, 49]" = torch.ops.aten.view.default(add_4, [-1, 4, 49, 49]);  add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_1: "f32[512, 4, 49, 49]" = torch.ops.aten._softmax.default(view_32, -1, False);  view_32 = None
    detach_1: "f32[512, 4, 49, 49]" = torch.ops.aten.detach.default(_softmax_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    clone_15: "f32[512, 4, 49, 49]" = torch.ops.aten.clone.default(_softmax_1);  _softmax_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    expand_6: "f32[512, 4, 49, 49]" = torch.ops.aten.expand.default(clone_15, [512, 4, 49, 49]);  clone_15 = None
    view_33: "f32[2048, 49, 49]" = torch.ops.aten.view.default(expand_6, [2048, 49, 49]);  expand_6 = None
    expand_7: "f32[512, 4, 49, 32]" = torch.ops.aten.expand.default(getitem_17, [512, 4, 49, 32]);  getitem_17 = None
    clone_16: "f32[512, 4, 49, 32]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
    _unsafe_view_6: "f32[2048, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_16, [2048, 49, 32]);  clone_16 = None
    bmm_3: "f32[2048, 49, 32]" = torch.ops.aten.bmm.default(view_33, _unsafe_view_6)
    view_34: "f32[512, 4, 49, 32]" = torch.ops.aten.view.default(bmm_3, [512, 4, 49, 32]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_3: "f32[512, 49, 4, 32]" = torch.ops.aten.transpose.int(view_34, 1, 2);  view_34 = None
    clone_17: "f32[512, 49, 4, 32]" = torch.ops.aten.clone.default(transpose_3, memory_format = torch.contiguous_format);  transpose_3 = None
    _unsafe_view_7: "f32[512, 49, 128]" = torch.ops.aten._unsafe_view.default(clone_17, [512, 49, 128]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_35: "f32[25088, 128]" = torch.ops.aten.view.default(_unsafe_view_7, [25088, 128]);  _unsafe_view_7 = None
    t_5: "f32[128, 128]" = torch.ops.aten.t.default(primals_45);  primals_45 = None
    addmm_5: "f32[25088, 128]" = torch.ops.aten.addmm.default(primals_46, view_35, t_5);  primals_46 = None
    view_36: "f32[512, 49, 128]" = torch.ops.aten.view.default(addmm_5, [512, 49, 128]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    clone_18: "f32[512, 49, 128]" = torch.ops.aten.clone.default(view_36);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_37: "f32[512, 7, 7, 128]" = torch.ops.aten.view.default(clone_18, [-1, 7, 7, 128]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    view_38: "f32[8, 8, 8, 7, 7, 128]" = torch.ops.aten.view.default(view_37, [-1, 8, 8, 7, 7, 128]);  view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_8: "f32[8, 8, 7, 8, 7, 128]" = torch.ops.aten.permute.default(view_38, [0, 1, 3, 2, 4, 5]);  view_38 = None
    clone_19: "f32[8, 8, 7, 8, 7, 128]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
    view_39: "f32[8, 56, 56, 128]" = torch.ops.aten.view.default(clone_19, [-1, 56, 56, 128]);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_3: "f32[8, 56, 56, 128]" = torch.ops.aten.slice.Tensor(view_39, 0, 0, 9223372036854775807);  view_39 = None
    slice_4: "f32[8, 56, 56, 128]" = torch.ops.aten.slice.Tensor(slice_3, 3, 0, 9223372036854775807);  slice_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    roll_1: "f32[8, 56, 56, 128]" = torch.ops.aten.roll.default(slice_4, [3, 3], [1, 2]);  slice_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty: "f32[8, 1, 1, 1]" = torch.ops.aten.new_empty.default(roll_1, [8, 1, 1, 1], pin_memory = False)
    bernoulli: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty, 0.9956521736457944);  new_empty = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli, 0.9956521736457944)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_2: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(roll_1, div);  roll_1 = div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_5: "f32[8, 56, 56, 128]" = torch.ops.aten.add.Tensor(view_21, mul_2);  mul_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_40: "f32[8, 3136, 128]" = torch.ops.aten.view.default(add_5, [8, -1, 128]);  add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_4 = torch.ops.aten.native_layer_norm.default(view_40, [128], primals_47, primals_48, 1e-05)
    getitem_18: "f32[8, 3136, 128]" = native_layer_norm_4[0]
    getitem_19: "f32[8, 3136, 1]" = native_layer_norm_4[1]
    getitem_20: "f32[8, 3136, 1]" = native_layer_norm_4[2];  native_layer_norm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_41: "f32[25088, 128]" = torch.ops.aten.view.default(getitem_18, [25088, 128]);  getitem_18 = None
    t_6: "f32[128, 512]" = torch.ops.aten.t.default(primals_49);  primals_49 = None
    addmm_6: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_50, view_41, t_6);  primals_50 = None
    view_42: "f32[8, 3136, 512]" = torch.ops.aten.view.default(addmm_6, [8, 3136, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_1: "f32[8, 3136, 512]" = torch.ops.aten.gelu.default(view_42);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_20: "f32[8, 3136, 512]" = torch.ops.aten.clone.default(gelu_1);  gelu_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_43: "f32[25088, 512]" = torch.ops.aten.view.default(clone_20, [25088, 512]);  clone_20 = None
    t_7: "f32[512, 128]" = torch.ops.aten.t.default(primals_51);  primals_51 = None
    addmm_7: "f32[25088, 128]" = torch.ops.aten.addmm.default(primals_52, view_43, t_7);  primals_52 = None
    view_44: "f32[8, 3136, 128]" = torch.ops.aten.view.default(addmm_7, [8, 3136, 128]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_21: "f32[8, 3136, 128]" = torch.ops.aten.clone.default(view_44);  view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_1: "f32[8, 1, 1]" = torch.ops.aten.new_empty.default(clone_21, [8, 1, 1], pin_memory = False)
    bernoulli_1: "f32[8, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_1, 0.9956521736457944);  new_empty_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_1: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_1, 0.9956521736457944)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_3: "f32[8, 3136, 128]" = torch.ops.aten.mul.Tensor(clone_21, div_1);  clone_21 = div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_6: "f32[8, 3136, 128]" = torch.ops.aten.add.Tensor(view_40, mul_3);  mul_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_45: "f32[8, 56, 56, 128]" = torch.ops.aten.view.default(add_6, [8, 56, 56, 128]);  add_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:356, code: x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
    view_46: "f32[8, 28, 2, 28, 2, 128]" = torch.ops.aten.view.default(view_45, [8, 28, 2, 28, 2, 128]);  view_45 = None
    permute_9: "f32[8, 28, 28, 2, 2, 128]" = torch.ops.aten.permute.default(view_46, [0, 1, 3, 4, 2, 5]);  view_46 = None
    clone_22: "f32[8, 28, 28, 2, 2, 128]" = torch.ops.aten.clone.default(permute_9, memory_format = torch.contiguous_format);  permute_9 = None
    _unsafe_view_8: "f32[8, 28, 28, 512]" = torch.ops.aten._unsafe_view.default(clone_22, [8, 28, 28, 512]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:357, code: x = self.norm(x)
    native_layer_norm_5 = torch.ops.aten.native_layer_norm.default(_unsafe_view_8, [512], primals_53, primals_54, 1e-05)
    getitem_21: "f32[8, 28, 28, 512]" = native_layer_norm_5[0]
    getitem_22: "f32[8, 28, 28, 1]" = native_layer_norm_5[1]
    getitem_23: "f32[8, 28, 28, 1]" = native_layer_norm_5[2];  native_layer_norm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:358, code: x = self.reduction(x)
    t_8: "f32[512, 256]" = torch.ops.aten.t.default(primals_55);  primals_55 = None
    view_47: "f32[6272, 512]" = torch.ops.aten.view.default(getitem_21, [6272, 512]);  getitem_21 = None
    mm: "f32[6272, 256]" = torch.ops.aten.mm.default(view_47, t_8)
    view_48: "f32[8, 28, 28, 256]" = torch.ops.aten.view.default(mm, [8, 28, 28, 256]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_6 = torch.ops.aten.native_layer_norm.default(view_48, [256], primals_56, primals_57, 1e-05)
    getitem_24: "f32[8, 28, 28, 256]" = native_layer_norm_6[0]
    getitem_25: "f32[8, 28, 28, 1]" = native_layer_norm_6[1]
    getitem_26: "f32[8, 28, 28, 1]" = native_layer_norm_6[2];  native_layer_norm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_2: "f32[8, 28, 28, 256]" = torch.ops.aten.constant_pad_nd.default(getitem_24, [0, 0, 0, 0, 0, 0], 0.0);  getitem_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    view_49: "f32[8, 4, 7, 4, 7, 256]" = torch.ops.aten.view.default(constant_pad_nd_2, [8, 4, 7, 4, 7, 256]);  constant_pad_nd_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_10: "f32[8, 4, 4, 7, 7, 256]" = torch.ops.aten.permute.default(view_49, [0, 1, 3, 2, 4, 5]);  view_49 = None
    clone_23: "f32[8, 4, 4, 7, 7, 256]" = torch.ops.aten.clone.default(permute_10, memory_format = torch.contiguous_format);  permute_10 = None
    view_50: "f32[128, 7, 7, 256]" = torch.ops.aten.view.default(clone_23, [-1, 7, 7, 256]);  clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_51: "f32[128, 49, 256]" = torch.ops.aten.view.default(view_50, [-1, 49, 256]);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_52: "f32[6272, 256]" = torch.ops.aten.view.default(view_51, [6272, 256]);  view_51 = None
    t_9: "f32[256, 768]" = torch.ops.aten.t.default(primals_58);  primals_58 = None
    addmm_8: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_59, view_52, t_9);  primals_59 = None
    view_53: "f32[128, 49, 768]" = torch.ops.aten.view.default(addmm_8, [128, 49, 768]);  addmm_8 = None
    view_54: "f32[128, 49, 3, 8, 32]" = torch.ops.aten.view.default(view_53, [128, 49, 3, 8, -1]);  view_53 = None
    permute_11: "f32[3, 128, 8, 49, 32]" = torch.ops.aten.permute.default(view_54, [2, 0, 3, 1, 4]);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_2 = torch.ops.aten.unbind.int(permute_11);  permute_11 = None
    getitem_27: "f32[128, 8, 49, 32]" = unbind_2[0]
    getitem_28: "f32[128, 8, 49, 32]" = unbind_2[1]
    getitem_29: "f32[128, 8, 49, 32]" = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_4: "f32[128, 8, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_27, 0.1767766952966369);  getitem_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_4: "f32[128, 8, 32, 49]" = torch.ops.aten.transpose.int(getitem_28, -2, -1);  getitem_28 = None
    expand_8: "f32[128, 8, 49, 32]" = torch.ops.aten.expand.default(mul_4, [128, 8, 49, 32]);  mul_4 = None
    clone_24: "f32[128, 8, 49, 32]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
    _unsafe_view_9: "f32[1024, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_24, [1024, 49, 32]);  clone_24 = None
    expand_9: "f32[128, 8, 32, 49]" = torch.ops.aten.expand.default(transpose_4, [128, 8, 32, 49]);  transpose_4 = None
    clone_25: "f32[128, 8, 32, 49]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    _unsafe_view_10: "f32[1024, 32, 49]" = torch.ops.aten._unsafe_view.default(clone_25, [1024, 32, 49]);  clone_25 = None
    bmm_4: "f32[1024, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_9, _unsafe_view_10)
    view_55: "f32[128, 8, 49, 49]" = torch.ops.aten.view.default(bmm_4, [128, 8, 49, 49]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_56: "i64[2401]" = torch.ops.aten.view.default(primals_333, [-1]);  primals_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_2: "f32[2401, 8]" = torch.ops.aten.index.Tensor(primals_3, [view_56]);  primals_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_57: "f32[49, 49, 8]" = torch.ops.aten.view.default(index_2, [49, 49, -1]);  index_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_12: "f32[8, 49, 49]" = torch.ops.aten.permute.default(view_57, [2, 0, 1]);  view_57 = None
    clone_26: "f32[8, 49, 49]" = torch.ops.aten.clone.default(permute_12, memory_format = torch.contiguous_format);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_4: "f32[1, 8, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_26, 0);  clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    add_7: "f32[128, 8, 49, 49]" = torch.ops.aten.add.Tensor(view_55, unsqueeze_4);  view_55 = unsqueeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_2: "f32[128, 8, 49, 49]" = torch.ops.aten._softmax.default(add_7, -1, False);  add_7 = None
    detach_2: "f32[128, 8, 49, 49]" = torch.ops.aten.detach.default(_softmax_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    clone_27: "f32[128, 8, 49, 49]" = torch.ops.aten.clone.default(_softmax_2);  _softmax_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    expand_10: "f32[128, 8, 49, 49]" = torch.ops.aten.expand.default(clone_27, [128, 8, 49, 49]);  clone_27 = None
    view_58: "f32[1024, 49, 49]" = torch.ops.aten.view.default(expand_10, [1024, 49, 49]);  expand_10 = None
    expand_11: "f32[128, 8, 49, 32]" = torch.ops.aten.expand.default(getitem_29, [128, 8, 49, 32]);  getitem_29 = None
    clone_28: "f32[128, 8, 49, 32]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
    _unsafe_view_11: "f32[1024, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_28, [1024, 49, 32]);  clone_28 = None
    bmm_5: "f32[1024, 49, 32]" = torch.ops.aten.bmm.default(view_58, _unsafe_view_11)
    view_59: "f32[128, 8, 49, 32]" = torch.ops.aten.view.default(bmm_5, [128, 8, 49, 32]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_5: "f32[128, 49, 8, 32]" = torch.ops.aten.transpose.int(view_59, 1, 2);  view_59 = None
    clone_29: "f32[128, 49, 8, 32]" = torch.ops.aten.clone.default(transpose_5, memory_format = torch.contiguous_format);  transpose_5 = None
    _unsafe_view_12: "f32[128, 49, 256]" = torch.ops.aten._unsafe_view.default(clone_29, [128, 49, 256]);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_60: "f32[6272, 256]" = torch.ops.aten.view.default(_unsafe_view_12, [6272, 256]);  _unsafe_view_12 = None
    t_10: "f32[256, 256]" = torch.ops.aten.t.default(primals_60);  primals_60 = None
    addmm_9: "f32[6272, 256]" = torch.ops.aten.addmm.default(primals_61, view_60, t_10);  primals_61 = None
    view_61: "f32[128, 49, 256]" = torch.ops.aten.view.default(addmm_9, [128, 49, 256]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    clone_30: "f32[128, 49, 256]" = torch.ops.aten.clone.default(view_61);  view_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_62: "f32[128, 7, 7, 256]" = torch.ops.aten.view.default(clone_30, [-1, 7, 7, 256]);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    view_63: "f32[8, 4, 4, 7, 7, 256]" = torch.ops.aten.view.default(view_62, [-1, 4, 4, 7, 7, 256]);  view_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_13: "f32[8, 4, 7, 4, 7, 256]" = torch.ops.aten.permute.default(view_63, [0, 1, 3, 2, 4, 5]);  view_63 = None
    clone_31: "f32[8, 4, 7, 4, 7, 256]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
    view_64: "f32[8, 28, 28, 256]" = torch.ops.aten.view.default(clone_31, [-1, 28, 28, 256]);  clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_5: "f32[8, 28, 28, 256]" = torch.ops.aten.slice.Tensor(view_64, 0, 0, 9223372036854775807);  view_64 = None
    slice_6: "f32[8, 28, 28, 256]" = torch.ops.aten.slice.Tensor(slice_5, 3, 0, 9223372036854775807);  slice_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_2: "f32[8, 1, 1, 1]" = torch.ops.aten.new_empty.default(slice_6, [8, 1, 1, 1], pin_memory = False)
    bernoulli_2: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_2, 0.9913043472915888);  new_empty_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_2: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_2, 0.9913043472915888)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_5: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(slice_6, div_2);  slice_6 = div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_8: "f32[8, 28, 28, 256]" = torch.ops.aten.add.Tensor(view_48, mul_5);  mul_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_65: "f32[8, 784, 256]" = torch.ops.aten.view.default(add_8, [8, -1, 256]);  add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_7 = torch.ops.aten.native_layer_norm.default(view_65, [256], primals_62, primals_63, 1e-05)
    getitem_30: "f32[8, 784, 256]" = native_layer_norm_7[0]
    getitem_31: "f32[8, 784, 1]" = native_layer_norm_7[1]
    getitem_32: "f32[8, 784, 1]" = native_layer_norm_7[2];  native_layer_norm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_66: "f32[6272, 256]" = torch.ops.aten.view.default(getitem_30, [6272, 256]);  getitem_30 = None
    t_11: "f32[256, 1024]" = torch.ops.aten.t.default(primals_64);  primals_64 = None
    addmm_10: "f32[6272, 1024]" = torch.ops.aten.addmm.default(primals_65, view_66, t_11);  primals_65 = None
    view_67: "f32[8, 784, 1024]" = torch.ops.aten.view.default(addmm_10, [8, 784, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_2: "f32[8, 784, 1024]" = torch.ops.aten.gelu.default(view_67);  view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_32: "f32[8, 784, 1024]" = torch.ops.aten.clone.default(gelu_2);  gelu_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_68: "f32[6272, 1024]" = torch.ops.aten.view.default(clone_32, [6272, 1024]);  clone_32 = None
    t_12: "f32[1024, 256]" = torch.ops.aten.t.default(primals_66);  primals_66 = None
    addmm_11: "f32[6272, 256]" = torch.ops.aten.addmm.default(primals_67, view_68, t_12);  primals_67 = None
    view_69: "f32[8, 784, 256]" = torch.ops.aten.view.default(addmm_11, [8, 784, 256]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_33: "f32[8, 784, 256]" = torch.ops.aten.clone.default(view_69);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_3: "f32[8, 1, 1]" = torch.ops.aten.new_empty.default(clone_33, [8, 1, 1], pin_memory = False)
    bernoulli_3: "f32[8, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_3, 0.9913043472915888);  new_empty_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_3: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_3, 0.9913043472915888)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_6: "f32[8, 784, 256]" = torch.ops.aten.mul.Tensor(clone_33, div_3);  clone_33 = div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_9: "f32[8, 784, 256]" = torch.ops.aten.add.Tensor(view_65, mul_6);  mul_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_70: "f32[8, 28, 28, 256]" = torch.ops.aten.view.default(add_9, [8, 28, 28, 256]);  add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_8 = torch.ops.aten.native_layer_norm.default(view_70, [256], primals_68, primals_69, 1e-05)
    getitem_33: "f32[8, 28, 28, 256]" = native_layer_norm_8[0]
    getitem_34: "f32[8, 28, 28, 1]" = native_layer_norm_8[1]
    getitem_35: "f32[8, 28, 28, 1]" = native_layer_norm_8[2];  native_layer_norm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    roll_2: "f32[8, 28, 28, 256]" = torch.ops.aten.roll.default(getitem_33, [-3, -3], [1, 2]);  getitem_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_3: "f32[8, 28, 28, 256]" = torch.ops.aten.constant_pad_nd.default(roll_2, [0, 0, 0, 0, 0, 0], 0.0);  roll_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    view_71: "f32[8, 4, 7, 4, 7, 256]" = torch.ops.aten.view.default(constant_pad_nd_3, [8, 4, 7, 4, 7, 256]);  constant_pad_nd_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_14: "f32[8, 4, 4, 7, 7, 256]" = torch.ops.aten.permute.default(view_71, [0, 1, 3, 2, 4, 5]);  view_71 = None
    clone_34: "f32[8, 4, 4, 7, 7, 256]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
    view_72: "f32[128, 7, 7, 256]" = torch.ops.aten.view.default(clone_34, [-1, 7, 7, 256]);  clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_73: "f32[128, 49, 256]" = torch.ops.aten.view.default(view_72, [-1, 49, 256]);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_74: "f32[6272, 256]" = torch.ops.aten.view.default(view_73, [6272, 256]);  view_73 = None
    t_13: "f32[256, 768]" = torch.ops.aten.t.default(primals_70);  primals_70 = None
    addmm_12: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_71, view_74, t_13);  primals_71 = None
    view_75: "f32[128, 49, 768]" = torch.ops.aten.view.default(addmm_12, [128, 49, 768]);  addmm_12 = None
    view_76: "f32[128, 49, 3, 8, 32]" = torch.ops.aten.view.default(view_75, [128, 49, 3, 8, -1]);  view_75 = None
    permute_15: "f32[3, 128, 8, 49, 32]" = torch.ops.aten.permute.default(view_76, [2, 0, 3, 1, 4]);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_3 = torch.ops.aten.unbind.int(permute_15);  permute_15 = None
    getitem_36: "f32[128, 8, 49, 32]" = unbind_3[0]
    getitem_37: "f32[128, 8, 49, 32]" = unbind_3[1]
    getitem_38: "f32[128, 8, 49, 32]" = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_7: "f32[128, 8, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_36, 0.1767766952966369);  getitem_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_6: "f32[128, 8, 32, 49]" = torch.ops.aten.transpose.int(getitem_37, -2, -1);  getitem_37 = None
    expand_12: "f32[128, 8, 49, 32]" = torch.ops.aten.expand.default(mul_7, [128, 8, 49, 32]);  mul_7 = None
    clone_35: "f32[128, 8, 49, 32]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
    _unsafe_view_13: "f32[1024, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_35, [1024, 49, 32]);  clone_35 = None
    expand_13: "f32[128, 8, 32, 49]" = torch.ops.aten.expand.default(transpose_6, [128, 8, 32, 49]);  transpose_6 = None
    clone_36: "f32[128, 8, 32, 49]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
    _unsafe_view_14: "f32[1024, 32, 49]" = torch.ops.aten._unsafe_view.default(clone_36, [1024, 32, 49]);  clone_36 = None
    bmm_6: "f32[1024, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_13, _unsafe_view_14)
    view_77: "f32[128, 8, 49, 49]" = torch.ops.aten.view.default(bmm_6, [128, 8, 49, 49]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_78: "i64[2401]" = torch.ops.aten.view.default(primals_335, [-1]);  primals_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_3: "f32[2401, 8]" = torch.ops.aten.index.Tensor(primals_4, [view_78]);  primals_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_79: "f32[49, 49, 8]" = torch.ops.aten.view.default(index_3, [49, 49, -1]);  index_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_16: "f32[8, 49, 49]" = torch.ops.aten.permute.default(view_79, [2, 0, 1]);  view_79 = None
    clone_37: "f32[8, 49, 49]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_5: "f32[1, 8, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_37, 0);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    add_10: "f32[128, 8, 49, 49]" = torch.ops.aten.add.Tensor(view_77, unsqueeze_5);  view_77 = unsqueeze_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_80: "f32[8, 16, 8, 49, 49]" = torch.ops.aten.view.default(add_10, [-1, 16, 8, 49, 49]);  add_10 = None
    unsqueeze_6: "f32[16, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(primals_334, 1);  primals_334 = None
    unsqueeze_7: "f32[1, 16, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 0);  unsqueeze_6 = None
    add_11: "f32[8, 16, 8, 49, 49]" = torch.ops.aten.add.Tensor(view_80, unsqueeze_7);  view_80 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    view_81: "f32[128, 8, 49, 49]" = torch.ops.aten.view.default(add_11, [-1, 8, 49, 49]);  add_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_3: "f32[128, 8, 49, 49]" = torch.ops.aten._softmax.default(view_81, -1, False);  view_81 = None
    detach_3: "f32[128, 8, 49, 49]" = torch.ops.aten.detach.default(_softmax_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    clone_38: "f32[128, 8, 49, 49]" = torch.ops.aten.clone.default(_softmax_3);  _softmax_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    expand_14: "f32[128, 8, 49, 49]" = torch.ops.aten.expand.default(clone_38, [128, 8, 49, 49]);  clone_38 = None
    view_82: "f32[1024, 49, 49]" = torch.ops.aten.view.default(expand_14, [1024, 49, 49]);  expand_14 = None
    expand_15: "f32[128, 8, 49, 32]" = torch.ops.aten.expand.default(getitem_38, [128, 8, 49, 32]);  getitem_38 = None
    clone_39: "f32[128, 8, 49, 32]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
    _unsafe_view_15: "f32[1024, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_39, [1024, 49, 32]);  clone_39 = None
    bmm_7: "f32[1024, 49, 32]" = torch.ops.aten.bmm.default(view_82, _unsafe_view_15)
    view_83: "f32[128, 8, 49, 32]" = torch.ops.aten.view.default(bmm_7, [128, 8, 49, 32]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_7: "f32[128, 49, 8, 32]" = torch.ops.aten.transpose.int(view_83, 1, 2);  view_83 = None
    clone_40: "f32[128, 49, 8, 32]" = torch.ops.aten.clone.default(transpose_7, memory_format = torch.contiguous_format);  transpose_7 = None
    _unsafe_view_16: "f32[128, 49, 256]" = torch.ops.aten._unsafe_view.default(clone_40, [128, 49, 256]);  clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_84: "f32[6272, 256]" = torch.ops.aten.view.default(_unsafe_view_16, [6272, 256]);  _unsafe_view_16 = None
    t_14: "f32[256, 256]" = torch.ops.aten.t.default(primals_72);  primals_72 = None
    addmm_13: "f32[6272, 256]" = torch.ops.aten.addmm.default(primals_73, view_84, t_14);  primals_73 = None
    view_85: "f32[128, 49, 256]" = torch.ops.aten.view.default(addmm_13, [128, 49, 256]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    clone_41: "f32[128, 49, 256]" = torch.ops.aten.clone.default(view_85);  view_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_86: "f32[128, 7, 7, 256]" = torch.ops.aten.view.default(clone_41, [-1, 7, 7, 256]);  clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    view_87: "f32[8, 4, 4, 7, 7, 256]" = torch.ops.aten.view.default(view_86, [-1, 4, 4, 7, 7, 256]);  view_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_17: "f32[8, 4, 7, 4, 7, 256]" = torch.ops.aten.permute.default(view_87, [0, 1, 3, 2, 4, 5]);  view_87 = None
    clone_42: "f32[8, 4, 7, 4, 7, 256]" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
    view_88: "f32[8, 28, 28, 256]" = torch.ops.aten.view.default(clone_42, [-1, 28, 28, 256]);  clone_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_7: "f32[8, 28, 28, 256]" = torch.ops.aten.slice.Tensor(view_88, 0, 0, 9223372036854775807);  view_88 = None
    slice_8: "f32[8, 28, 28, 256]" = torch.ops.aten.slice.Tensor(slice_7, 3, 0, 9223372036854775807);  slice_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    roll_3: "f32[8, 28, 28, 256]" = torch.ops.aten.roll.default(slice_8, [3, 3], [1, 2]);  slice_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_4: "f32[8, 1, 1, 1]" = torch.ops.aten.new_empty.default(roll_3, [8, 1, 1, 1], pin_memory = False)
    bernoulli_4: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_4, 0.9869565209373832);  new_empty_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_4: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_4, 0.9869565209373832)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_8: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(roll_3, div_4);  roll_3 = div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_12: "f32[8, 28, 28, 256]" = torch.ops.aten.add.Tensor(view_70, mul_8);  mul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_89: "f32[8, 784, 256]" = torch.ops.aten.view.default(add_12, [8, -1, 256]);  add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_9 = torch.ops.aten.native_layer_norm.default(view_89, [256], primals_74, primals_75, 1e-05)
    getitem_39: "f32[8, 784, 256]" = native_layer_norm_9[0]
    getitem_40: "f32[8, 784, 1]" = native_layer_norm_9[1]
    getitem_41: "f32[8, 784, 1]" = native_layer_norm_9[2];  native_layer_norm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_90: "f32[6272, 256]" = torch.ops.aten.view.default(getitem_39, [6272, 256]);  getitem_39 = None
    t_15: "f32[256, 1024]" = torch.ops.aten.t.default(primals_76);  primals_76 = None
    addmm_14: "f32[6272, 1024]" = torch.ops.aten.addmm.default(primals_77, view_90, t_15);  primals_77 = None
    view_91: "f32[8, 784, 1024]" = torch.ops.aten.view.default(addmm_14, [8, 784, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_3: "f32[8, 784, 1024]" = torch.ops.aten.gelu.default(view_91);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_43: "f32[8, 784, 1024]" = torch.ops.aten.clone.default(gelu_3);  gelu_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_92: "f32[6272, 1024]" = torch.ops.aten.view.default(clone_43, [6272, 1024]);  clone_43 = None
    t_16: "f32[1024, 256]" = torch.ops.aten.t.default(primals_78);  primals_78 = None
    addmm_15: "f32[6272, 256]" = torch.ops.aten.addmm.default(primals_79, view_92, t_16);  primals_79 = None
    view_93: "f32[8, 784, 256]" = torch.ops.aten.view.default(addmm_15, [8, 784, 256]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_44: "f32[8, 784, 256]" = torch.ops.aten.clone.default(view_93);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_5: "f32[8, 1, 1]" = torch.ops.aten.new_empty.default(clone_44, [8, 1, 1], pin_memory = False)
    bernoulli_5: "f32[8, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_5, 0.9869565209373832);  new_empty_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_5: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_5, 0.9869565209373832)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_9: "f32[8, 784, 256]" = torch.ops.aten.mul.Tensor(clone_44, div_5);  clone_44 = div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_13: "f32[8, 784, 256]" = torch.ops.aten.add.Tensor(view_89, mul_9);  mul_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_94: "f32[8, 28, 28, 256]" = torch.ops.aten.view.default(add_13, [8, 28, 28, 256]);  add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:356, code: x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
    view_95: "f32[8, 14, 2, 14, 2, 256]" = torch.ops.aten.view.default(view_94, [8, 14, 2, 14, 2, 256]);  view_94 = None
    permute_18: "f32[8, 14, 14, 2, 2, 256]" = torch.ops.aten.permute.default(view_95, [0, 1, 3, 4, 2, 5]);  view_95 = None
    clone_45: "f32[8, 14, 14, 2, 2, 256]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    _unsafe_view_17: "f32[8, 14, 14, 1024]" = torch.ops.aten._unsafe_view.default(clone_45, [8, 14, 14, 1024]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:357, code: x = self.norm(x)
    native_layer_norm_10 = torch.ops.aten.native_layer_norm.default(_unsafe_view_17, [1024], primals_80, primals_81, 1e-05)
    getitem_42: "f32[8, 14, 14, 1024]" = native_layer_norm_10[0]
    getitem_43: "f32[8, 14, 14, 1]" = native_layer_norm_10[1]
    getitem_44: "f32[8, 14, 14, 1]" = native_layer_norm_10[2];  native_layer_norm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:358, code: x = self.reduction(x)
    t_17: "f32[1024, 512]" = torch.ops.aten.t.default(primals_82);  primals_82 = None
    view_96: "f32[1568, 1024]" = torch.ops.aten.view.default(getitem_42, [1568, 1024]);  getitem_42 = None
    mm_1: "f32[1568, 512]" = torch.ops.aten.mm.default(view_96, t_17)
    view_97: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(mm_1, [8, 14, 14, 512]);  mm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_11 = torch.ops.aten.native_layer_norm.default(view_97, [512], primals_83, primals_84, 1e-05)
    getitem_45: "f32[8, 14, 14, 512]" = native_layer_norm_11[0]
    getitem_46: "f32[8, 14, 14, 1]" = native_layer_norm_11[1]
    getitem_47: "f32[8, 14, 14, 1]" = native_layer_norm_11[2];  native_layer_norm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_4: "f32[8, 14, 14, 512]" = torch.ops.aten.constant_pad_nd.default(getitem_45, [0, 0, 0, 0, 0, 0], 0.0);  getitem_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    view_98: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(constant_pad_nd_4, [8, 2, 7, 2, 7, 512]);  constant_pad_nd_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_19: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_98, [0, 1, 3, 2, 4, 5]);  view_98 = None
    clone_46: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
    view_99: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_46, [-1, 7, 7, 512]);  clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_100: "f32[32, 49, 512]" = torch.ops.aten.view.default(view_99, [-1, 49, 512]);  view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_101: "f32[1568, 512]" = torch.ops.aten.view.default(view_100, [1568, 512]);  view_100 = None
    t_18: "f32[512, 1536]" = torch.ops.aten.t.default(primals_85);  primals_85 = None
    addmm_16: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_86, view_101, t_18);  primals_86 = None
    view_102: "f32[32, 49, 1536]" = torch.ops.aten.view.default(addmm_16, [32, 49, 1536]);  addmm_16 = None
    view_103: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_102, [32, 49, 3, 16, -1]);  view_102 = None
    permute_20: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.permute.default(view_103, [2, 0, 3, 1, 4]);  view_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_4 = torch.ops.aten.unbind.int(permute_20);  permute_20 = None
    getitem_48: "f32[32, 16, 49, 32]" = unbind_4[0]
    getitem_49: "f32[32, 16, 49, 32]" = unbind_4[1]
    getitem_50: "f32[32, 16, 49, 32]" = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_10: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_48, 0.1767766952966369);  getitem_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_8: "f32[32, 16, 32, 49]" = torch.ops.aten.transpose.int(getitem_49, -2, -1);  getitem_49 = None
    expand_16: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(mul_10, [32, 16, 49, 32]);  mul_10 = None
    clone_47: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
    _unsafe_view_18: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_47, [512, 49, 32]);  clone_47 = None
    expand_17: "f32[32, 16, 32, 49]" = torch.ops.aten.expand.default(transpose_8, [32, 16, 32, 49]);  transpose_8 = None
    clone_48: "f32[32, 16, 32, 49]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    _unsafe_view_19: "f32[512, 32, 49]" = torch.ops.aten._unsafe_view.default(clone_48, [512, 32, 49]);  clone_48 = None
    bmm_8: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_18, _unsafe_view_19)
    view_104: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_8, [32, 16, 49, 49]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_105: "i64[2401]" = torch.ops.aten.view.default(primals_336, [-1]);  primals_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_4: "f32[2401, 16]" = torch.ops.aten.index.Tensor(primals_5, [view_105]);  primals_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_106: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_4, [49, 49, -1]);  index_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_21: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_106, [2, 0, 1]);  view_106 = None
    clone_49: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_21, memory_format = torch.contiguous_format);  permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_8: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_49, 0);  clone_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    add_14: "f32[32, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_104, unsqueeze_8);  view_104 = unsqueeze_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_4: "f32[32, 16, 49, 49]" = torch.ops.aten._softmax.default(add_14, -1, False);  add_14 = None
    detach_4: "f32[32, 16, 49, 49]" = torch.ops.aten.detach.default(_softmax_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    clone_50: "f32[32, 16, 49, 49]" = torch.ops.aten.clone.default(_softmax_4);  _softmax_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    expand_18: "f32[32, 16, 49, 49]" = torch.ops.aten.expand.default(clone_50, [32, 16, 49, 49]);  clone_50 = None
    view_107: "f32[512, 49, 49]" = torch.ops.aten.view.default(expand_18, [512, 49, 49]);  expand_18 = None
    expand_19: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_50, [32, 16, 49, 32]);  getitem_50 = None
    clone_51: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
    _unsafe_view_20: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_51, [512, 49, 32]);  clone_51 = None
    bmm_9: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_107, _unsafe_view_20)
    view_108: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_9, [32, 16, 49, 32]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_9: "f32[32, 49, 16, 32]" = torch.ops.aten.transpose.int(view_108, 1, 2);  view_108 = None
    clone_52: "f32[32, 49, 16, 32]" = torch.ops.aten.clone.default(transpose_9, memory_format = torch.contiguous_format);  transpose_9 = None
    _unsafe_view_21: "f32[32, 49, 512]" = torch.ops.aten._unsafe_view.default(clone_52, [32, 49, 512]);  clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_109: "f32[1568, 512]" = torch.ops.aten.view.default(_unsafe_view_21, [1568, 512]);  _unsafe_view_21 = None
    t_19: "f32[512, 512]" = torch.ops.aten.t.default(primals_87);  primals_87 = None
    addmm_17: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_88, view_109, t_19);  primals_88 = None
    view_110: "f32[32, 49, 512]" = torch.ops.aten.view.default(addmm_17, [32, 49, 512]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    clone_53: "f32[32, 49, 512]" = torch.ops.aten.clone.default(view_110);  view_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_111: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_53, [-1, 7, 7, 512]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    view_112: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_111, [-1, 2, 2, 7, 7, 512]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_22: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_112, [0, 1, 3, 2, 4, 5]);  view_112 = None
    clone_54: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_22, memory_format = torch.contiguous_format);  permute_22 = None
    view_113: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(clone_54, [-1, 14, 14, 512]);  clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_9: "f32[8, 14, 14, 512]" = torch.ops.aten.slice.Tensor(view_113, 0, 0, 9223372036854775807);  view_113 = None
    slice_10: "f32[8, 14, 14, 512]" = torch.ops.aten.slice.Tensor(slice_9, 3, 0, 9223372036854775807);  slice_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_6: "f32[8, 1, 1, 1]" = torch.ops.aten.new_empty.default(slice_10, [8, 1, 1, 1], pin_memory = False)
    bernoulli_6: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_6, 0.9826086945831776);  new_empty_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_6: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_6, 0.9826086945831776)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_11: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(slice_10, div_6);  slice_10 = div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_15: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_97, mul_11);  mul_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_114: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_15, [8, -1, 512]);  add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_12 = torch.ops.aten.native_layer_norm.default(view_114, [512], primals_89, primals_90, 1e-05)
    getitem_51: "f32[8, 196, 512]" = native_layer_norm_12[0]
    getitem_52: "f32[8, 196, 1]" = native_layer_norm_12[1]
    getitem_53: "f32[8, 196, 1]" = native_layer_norm_12[2];  native_layer_norm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_115: "f32[1568, 512]" = torch.ops.aten.view.default(getitem_51, [1568, 512]);  getitem_51 = None
    t_20: "f32[512, 2048]" = torch.ops.aten.t.default(primals_91);  primals_91 = None
    addmm_18: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_92, view_115, t_20);  primals_92 = None
    view_116: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_18, [8, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_4: "f32[8, 196, 2048]" = torch.ops.aten.gelu.default(view_116);  view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_55: "f32[8, 196, 2048]" = torch.ops.aten.clone.default(gelu_4);  gelu_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_117: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_55, [1568, 2048]);  clone_55 = None
    t_21: "f32[2048, 512]" = torch.ops.aten.t.default(primals_93);  primals_93 = None
    addmm_19: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_94, view_117, t_21);  primals_94 = None
    view_118: "f32[8, 196, 512]" = torch.ops.aten.view.default(addmm_19, [8, 196, 512]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_56: "f32[8, 196, 512]" = torch.ops.aten.clone.default(view_118);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_7: "f32[8, 1, 1]" = torch.ops.aten.new_empty.default(clone_56, [8, 1, 1], pin_memory = False)
    bernoulli_7: "f32[8, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_7, 0.9826086945831776);  new_empty_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_7: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_7, 0.9826086945831776)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_12: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(clone_56, div_7);  clone_56 = div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_16: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_114, mul_12);  mul_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_119: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_16, [8, 14, 14, 512]);  add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_13 = torch.ops.aten.native_layer_norm.default(view_119, [512], primals_95, primals_96, 1e-05)
    getitem_54: "f32[8, 14, 14, 512]" = native_layer_norm_13[0]
    getitem_55: "f32[8, 14, 14, 1]" = native_layer_norm_13[1]
    getitem_56: "f32[8, 14, 14, 1]" = native_layer_norm_13[2];  native_layer_norm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    roll_4: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(getitem_54, [-3, -3], [1, 2]);  getitem_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_5: "f32[8, 14, 14, 512]" = torch.ops.aten.constant_pad_nd.default(roll_4, [0, 0, 0, 0, 0, 0], 0.0);  roll_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    view_120: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(constant_pad_nd_5, [8, 2, 7, 2, 7, 512]);  constant_pad_nd_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_23: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_120, [0, 1, 3, 2, 4, 5]);  view_120 = None
    clone_57: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
    view_121: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_57, [-1, 7, 7, 512]);  clone_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_122: "f32[32, 49, 512]" = torch.ops.aten.view.default(view_121, [-1, 49, 512]);  view_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_123: "f32[1568, 512]" = torch.ops.aten.view.default(view_122, [1568, 512]);  view_122 = None
    t_22: "f32[512, 1536]" = torch.ops.aten.t.default(primals_97);  primals_97 = None
    addmm_20: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_98, view_123, t_22);  primals_98 = None
    view_124: "f32[32, 49, 1536]" = torch.ops.aten.view.default(addmm_20, [32, 49, 1536]);  addmm_20 = None
    view_125: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_124, [32, 49, 3, 16, -1]);  view_124 = None
    permute_24: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.permute.default(view_125, [2, 0, 3, 1, 4]);  view_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_5 = torch.ops.aten.unbind.int(permute_24);  permute_24 = None
    getitem_57: "f32[32, 16, 49, 32]" = unbind_5[0]
    getitem_58: "f32[32, 16, 49, 32]" = unbind_5[1]
    getitem_59: "f32[32, 16, 49, 32]" = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_13: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_57, 0.1767766952966369);  getitem_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_10: "f32[32, 16, 32, 49]" = torch.ops.aten.transpose.int(getitem_58, -2, -1);  getitem_58 = None
    expand_20: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(mul_13, [32, 16, 49, 32]);  mul_13 = None
    clone_58: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
    _unsafe_view_22: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_58, [512, 49, 32]);  clone_58 = None
    expand_21: "f32[32, 16, 32, 49]" = torch.ops.aten.expand.default(transpose_10, [32, 16, 32, 49]);  transpose_10 = None
    clone_59: "f32[32, 16, 32, 49]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    _unsafe_view_23: "f32[512, 32, 49]" = torch.ops.aten._unsafe_view.default(clone_59, [512, 32, 49]);  clone_59 = None
    bmm_10: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_22, _unsafe_view_23)
    view_126: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_10, [32, 16, 49, 49]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_127: "i64[2401]" = torch.ops.aten.view.default(primals_338, [-1]);  primals_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_5: "f32[2401, 16]" = torch.ops.aten.index.Tensor(primals_6, [view_127]);  primals_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_128: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_5, [49, 49, -1]);  index_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_25: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_128, [2, 0, 1]);  view_128 = None
    clone_60: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_25, memory_format = torch.contiguous_format);  permute_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_9: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_60, 0);  clone_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    add_17: "f32[32, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_126, unsqueeze_9);  view_126 = unsqueeze_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_129: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_17, [-1, 4, 16, 49, 49]);  add_17 = None
    unsqueeze_10: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(primals_337, 1);  primals_337 = None
    unsqueeze_11: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 0);  unsqueeze_10 = None
    add_18: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_129, unsqueeze_11);  view_129 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    view_130: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(add_18, [-1, 16, 49, 49]);  add_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_5: "f32[32, 16, 49, 49]" = torch.ops.aten._softmax.default(view_130, -1, False);  view_130 = None
    detach_5: "f32[32, 16, 49, 49]" = torch.ops.aten.detach.default(_softmax_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    clone_61: "f32[32, 16, 49, 49]" = torch.ops.aten.clone.default(_softmax_5);  _softmax_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    expand_22: "f32[32, 16, 49, 49]" = torch.ops.aten.expand.default(clone_61, [32, 16, 49, 49]);  clone_61 = None
    view_131: "f32[512, 49, 49]" = torch.ops.aten.view.default(expand_22, [512, 49, 49]);  expand_22 = None
    expand_23: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_59, [32, 16, 49, 32]);  getitem_59 = None
    clone_62: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
    _unsafe_view_24: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_62, [512, 49, 32]);  clone_62 = None
    bmm_11: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_131, _unsafe_view_24)
    view_132: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_11, [32, 16, 49, 32]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_11: "f32[32, 49, 16, 32]" = torch.ops.aten.transpose.int(view_132, 1, 2);  view_132 = None
    clone_63: "f32[32, 49, 16, 32]" = torch.ops.aten.clone.default(transpose_11, memory_format = torch.contiguous_format);  transpose_11 = None
    _unsafe_view_25: "f32[32, 49, 512]" = torch.ops.aten._unsafe_view.default(clone_63, [32, 49, 512]);  clone_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_133: "f32[1568, 512]" = torch.ops.aten.view.default(_unsafe_view_25, [1568, 512]);  _unsafe_view_25 = None
    t_23: "f32[512, 512]" = torch.ops.aten.t.default(primals_99);  primals_99 = None
    addmm_21: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_100, view_133, t_23);  primals_100 = None
    view_134: "f32[32, 49, 512]" = torch.ops.aten.view.default(addmm_21, [32, 49, 512]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    clone_64: "f32[32, 49, 512]" = torch.ops.aten.clone.default(view_134);  view_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_135: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_64, [-1, 7, 7, 512]);  clone_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    view_136: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_135, [-1, 2, 2, 7, 7, 512]);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_26: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_136, [0, 1, 3, 2, 4, 5]);  view_136 = None
    clone_65: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
    view_137: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(clone_65, [-1, 14, 14, 512]);  clone_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_11: "f32[8, 14, 14, 512]" = torch.ops.aten.slice.Tensor(view_137, 0, 0, 9223372036854775807);  view_137 = None
    slice_12: "f32[8, 14, 14, 512]" = torch.ops.aten.slice.Tensor(slice_11, 3, 0, 9223372036854775807);  slice_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    roll_5: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(slice_12, [3, 3], [1, 2]);  slice_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_8: "f32[8, 1, 1, 1]" = torch.ops.aten.new_empty.default(roll_5, [8, 1, 1, 1], pin_memory = False)
    bernoulli_8: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_8, 0.9782608672976494);  new_empty_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_8: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_8, 0.9782608672976494)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_14: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(roll_5, div_8);  roll_5 = div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_19: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_119, mul_14);  mul_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_138: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_19, [8, -1, 512]);  add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_14 = torch.ops.aten.native_layer_norm.default(view_138, [512], primals_101, primals_102, 1e-05)
    getitem_60: "f32[8, 196, 512]" = native_layer_norm_14[0]
    getitem_61: "f32[8, 196, 1]" = native_layer_norm_14[1]
    getitem_62: "f32[8, 196, 1]" = native_layer_norm_14[2];  native_layer_norm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_139: "f32[1568, 512]" = torch.ops.aten.view.default(getitem_60, [1568, 512]);  getitem_60 = None
    t_24: "f32[512, 2048]" = torch.ops.aten.t.default(primals_103);  primals_103 = None
    addmm_22: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_104, view_139, t_24);  primals_104 = None
    view_140: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_22, [8, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_5: "f32[8, 196, 2048]" = torch.ops.aten.gelu.default(view_140);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_66: "f32[8, 196, 2048]" = torch.ops.aten.clone.default(gelu_5);  gelu_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_141: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_66, [1568, 2048]);  clone_66 = None
    t_25: "f32[2048, 512]" = torch.ops.aten.t.default(primals_105);  primals_105 = None
    addmm_23: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_106, view_141, t_25);  primals_106 = None
    view_142: "f32[8, 196, 512]" = torch.ops.aten.view.default(addmm_23, [8, 196, 512]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_67: "f32[8, 196, 512]" = torch.ops.aten.clone.default(view_142);  view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_9: "f32[8, 1, 1]" = torch.ops.aten.new_empty.default(clone_67, [8, 1, 1], pin_memory = False)
    bernoulli_9: "f32[8, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_9, 0.9782608672976494);  new_empty_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_9: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_9, 0.9782608672976494)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_15: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(clone_67, div_9);  clone_67 = div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_20: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_138, mul_15);  mul_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_143: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_20, [8, 14, 14, 512]);  add_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_15 = torch.ops.aten.native_layer_norm.default(view_143, [512], primals_107, primals_108, 1e-05)
    getitem_63: "f32[8, 14, 14, 512]" = native_layer_norm_15[0]
    getitem_64: "f32[8, 14, 14, 1]" = native_layer_norm_15[1]
    getitem_65: "f32[8, 14, 14, 1]" = native_layer_norm_15[2];  native_layer_norm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_6: "f32[8, 14, 14, 512]" = torch.ops.aten.constant_pad_nd.default(getitem_63, [0, 0, 0, 0, 0, 0], 0.0);  getitem_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    view_144: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(constant_pad_nd_6, [8, 2, 7, 2, 7, 512]);  constant_pad_nd_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_27: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_144, [0, 1, 3, 2, 4, 5]);  view_144 = None
    clone_68: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
    view_145: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_68, [-1, 7, 7, 512]);  clone_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_146: "f32[32, 49, 512]" = torch.ops.aten.view.default(view_145, [-1, 49, 512]);  view_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_147: "f32[1568, 512]" = torch.ops.aten.view.default(view_146, [1568, 512]);  view_146 = None
    t_26: "f32[512, 1536]" = torch.ops.aten.t.default(primals_109);  primals_109 = None
    addmm_24: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_110, view_147, t_26);  primals_110 = None
    view_148: "f32[32, 49, 1536]" = torch.ops.aten.view.default(addmm_24, [32, 49, 1536]);  addmm_24 = None
    view_149: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_148, [32, 49, 3, 16, -1]);  view_148 = None
    permute_28: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.permute.default(view_149, [2, 0, 3, 1, 4]);  view_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_6 = torch.ops.aten.unbind.int(permute_28);  permute_28 = None
    getitem_66: "f32[32, 16, 49, 32]" = unbind_6[0]
    getitem_67: "f32[32, 16, 49, 32]" = unbind_6[1]
    getitem_68: "f32[32, 16, 49, 32]" = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_16: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_66, 0.1767766952966369);  getitem_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_12: "f32[32, 16, 32, 49]" = torch.ops.aten.transpose.int(getitem_67, -2, -1);  getitem_67 = None
    expand_24: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(mul_16, [32, 16, 49, 32]);  mul_16 = None
    clone_69: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
    _unsafe_view_26: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_69, [512, 49, 32]);  clone_69 = None
    expand_25: "f32[32, 16, 32, 49]" = torch.ops.aten.expand.default(transpose_12, [32, 16, 32, 49]);  transpose_12 = None
    clone_70: "f32[32, 16, 32, 49]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    _unsafe_view_27: "f32[512, 32, 49]" = torch.ops.aten._unsafe_view.default(clone_70, [512, 32, 49]);  clone_70 = None
    bmm_12: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_26, _unsafe_view_27)
    view_150: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_12, [32, 16, 49, 49]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_151: "i64[2401]" = torch.ops.aten.view.default(primals_339, [-1]);  primals_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_6: "f32[2401, 16]" = torch.ops.aten.index.Tensor(primals_7, [view_151]);  primals_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_152: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_6, [49, 49, -1]);  index_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_29: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_152, [2, 0, 1]);  view_152 = None
    clone_71: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_12: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_71, 0);  clone_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    add_21: "f32[32, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_150, unsqueeze_12);  view_150 = unsqueeze_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_6: "f32[32, 16, 49, 49]" = torch.ops.aten._softmax.default(add_21, -1, False);  add_21 = None
    detach_6: "f32[32, 16, 49, 49]" = torch.ops.aten.detach.default(_softmax_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    clone_72: "f32[32, 16, 49, 49]" = torch.ops.aten.clone.default(_softmax_6);  _softmax_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    expand_26: "f32[32, 16, 49, 49]" = torch.ops.aten.expand.default(clone_72, [32, 16, 49, 49]);  clone_72 = None
    view_153: "f32[512, 49, 49]" = torch.ops.aten.view.default(expand_26, [512, 49, 49]);  expand_26 = None
    expand_27: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_68, [32, 16, 49, 32]);  getitem_68 = None
    clone_73: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
    _unsafe_view_28: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_73, [512, 49, 32]);  clone_73 = None
    bmm_13: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_153, _unsafe_view_28)
    view_154: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_13, [32, 16, 49, 32]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_13: "f32[32, 49, 16, 32]" = torch.ops.aten.transpose.int(view_154, 1, 2);  view_154 = None
    clone_74: "f32[32, 49, 16, 32]" = torch.ops.aten.clone.default(transpose_13, memory_format = torch.contiguous_format);  transpose_13 = None
    _unsafe_view_29: "f32[32, 49, 512]" = torch.ops.aten._unsafe_view.default(clone_74, [32, 49, 512]);  clone_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_155: "f32[1568, 512]" = torch.ops.aten.view.default(_unsafe_view_29, [1568, 512]);  _unsafe_view_29 = None
    t_27: "f32[512, 512]" = torch.ops.aten.t.default(primals_111);  primals_111 = None
    addmm_25: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_112, view_155, t_27);  primals_112 = None
    view_156: "f32[32, 49, 512]" = torch.ops.aten.view.default(addmm_25, [32, 49, 512]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    clone_75: "f32[32, 49, 512]" = torch.ops.aten.clone.default(view_156);  view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_157: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_75, [-1, 7, 7, 512]);  clone_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    view_158: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_157, [-1, 2, 2, 7, 7, 512]);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_30: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_158, [0, 1, 3, 2, 4, 5]);  view_158 = None
    clone_76: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
    view_159: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(clone_76, [-1, 14, 14, 512]);  clone_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_13: "f32[8, 14, 14, 512]" = torch.ops.aten.slice.Tensor(view_159, 0, 0, 9223372036854775807);  view_159 = None
    slice_14: "f32[8, 14, 14, 512]" = torch.ops.aten.slice.Tensor(slice_13, 3, 0, 9223372036854775807);  slice_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_10: "f32[8, 1, 1, 1]" = torch.ops.aten.new_empty.default(slice_14, [8, 1, 1, 1], pin_memory = False)
    bernoulli_10: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_10, 0.9739130418747663);  new_empty_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_10: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_10, 0.9739130418747663)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_17: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(slice_14, div_10);  slice_14 = div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_22: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_143, mul_17);  mul_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_160: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_22, [8, -1, 512]);  add_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_16 = torch.ops.aten.native_layer_norm.default(view_160, [512], primals_113, primals_114, 1e-05)
    getitem_69: "f32[8, 196, 512]" = native_layer_norm_16[0]
    getitem_70: "f32[8, 196, 1]" = native_layer_norm_16[1]
    getitem_71: "f32[8, 196, 1]" = native_layer_norm_16[2];  native_layer_norm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_161: "f32[1568, 512]" = torch.ops.aten.view.default(getitem_69, [1568, 512]);  getitem_69 = None
    t_28: "f32[512, 2048]" = torch.ops.aten.t.default(primals_115);  primals_115 = None
    addmm_26: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_116, view_161, t_28);  primals_116 = None
    view_162: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_26, [8, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_6: "f32[8, 196, 2048]" = torch.ops.aten.gelu.default(view_162);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_77: "f32[8, 196, 2048]" = torch.ops.aten.clone.default(gelu_6);  gelu_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_163: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_77, [1568, 2048]);  clone_77 = None
    t_29: "f32[2048, 512]" = torch.ops.aten.t.default(primals_117);  primals_117 = None
    addmm_27: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_118, view_163, t_29);  primals_118 = None
    view_164: "f32[8, 196, 512]" = torch.ops.aten.view.default(addmm_27, [8, 196, 512]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_78: "f32[8, 196, 512]" = torch.ops.aten.clone.default(view_164);  view_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_11: "f32[8, 1, 1]" = torch.ops.aten.new_empty.default(clone_78, [8, 1, 1], pin_memory = False)
    bernoulli_11: "f32[8, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_11, 0.9739130418747663);  new_empty_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_11: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_11, 0.9739130418747663)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_18: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(clone_78, div_11);  clone_78 = div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_23: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_160, mul_18);  mul_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_165: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_23, [8, 14, 14, 512]);  add_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_17 = torch.ops.aten.native_layer_norm.default(view_165, [512], primals_119, primals_120, 1e-05)
    getitem_72: "f32[8, 14, 14, 512]" = native_layer_norm_17[0]
    getitem_73: "f32[8, 14, 14, 1]" = native_layer_norm_17[1]
    getitem_74: "f32[8, 14, 14, 1]" = native_layer_norm_17[2];  native_layer_norm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    roll_6: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(getitem_72, [-3, -3], [1, 2]);  getitem_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_7: "f32[8, 14, 14, 512]" = torch.ops.aten.constant_pad_nd.default(roll_6, [0, 0, 0, 0, 0, 0], 0.0);  roll_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    view_166: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(constant_pad_nd_7, [8, 2, 7, 2, 7, 512]);  constant_pad_nd_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_31: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_166, [0, 1, 3, 2, 4, 5]);  view_166 = None
    clone_79: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_31, memory_format = torch.contiguous_format);  permute_31 = None
    view_167: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_79, [-1, 7, 7, 512]);  clone_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_168: "f32[32, 49, 512]" = torch.ops.aten.view.default(view_167, [-1, 49, 512]);  view_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_169: "f32[1568, 512]" = torch.ops.aten.view.default(view_168, [1568, 512]);  view_168 = None
    t_30: "f32[512, 1536]" = torch.ops.aten.t.default(primals_121);  primals_121 = None
    addmm_28: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_122, view_169, t_30);  primals_122 = None
    view_170: "f32[32, 49, 1536]" = torch.ops.aten.view.default(addmm_28, [32, 49, 1536]);  addmm_28 = None
    view_171: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_170, [32, 49, 3, 16, -1]);  view_170 = None
    permute_32: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.permute.default(view_171, [2, 0, 3, 1, 4]);  view_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_7 = torch.ops.aten.unbind.int(permute_32);  permute_32 = None
    getitem_75: "f32[32, 16, 49, 32]" = unbind_7[0]
    getitem_76: "f32[32, 16, 49, 32]" = unbind_7[1]
    getitem_77: "f32[32, 16, 49, 32]" = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_19: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_75, 0.1767766952966369);  getitem_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_14: "f32[32, 16, 32, 49]" = torch.ops.aten.transpose.int(getitem_76, -2, -1);  getitem_76 = None
    expand_28: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(mul_19, [32, 16, 49, 32]);  mul_19 = None
    clone_80: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
    _unsafe_view_30: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_80, [512, 49, 32]);  clone_80 = None
    expand_29: "f32[32, 16, 32, 49]" = torch.ops.aten.expand.default(transpose_14, [32, 16, 32, 49]);  transpose_14 = None
    clone_81: "f32[32, 16, 32, 49]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
    _unsafe_view_31: "f32[512, 32, 49]" = torch.ops.aten._unsafe_view.default(clone_81, [512, 32, 49]);  clone_81 = None
    bmm_14: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_30, _unsafe_view_31)
    view_172: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_14, [32, 16, 49, 49]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_173: "i64[2401]" = torch.ops.aten.view.default(primals_341, [-1]);  primals_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_7: "f32[2401, 16]" = torch.ops.aten.index.Tensor(primals_8, [view_173]);  primals_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_174: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_7, [49, 49, -1]);  index_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_33: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_174, [2, 0, 1]);  view_174 = None
    clone_82: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_33, memory_format = torch.contiguous_format);  permute_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_13: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_82, 0);  clone_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    add_24: "f32[32, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_172, unsqueeze_13);  view_172 = unsqueeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_175: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_24, [-1, 4, 16, 49, 49]);  add_24 = None
    unsqueeze_14: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(primals_340, 1);  primals_340 = None
    unsqueeze_15: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, 0);  unsqueeze_14 = None
    add_25: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_175, unsqueeze_15);  view_175 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    view_176: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(add_25, [-1, 16, 49, 49]);  add_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_7: "f32[32, 16, 49, 49]" = torch.ops.aten._softmax.default(view_176, -1, False);  view_176 = None
    detach_7: "f32[32, 16, 49, 49]" = torch.ops.aten.detach.default(_softmax_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    clone_83: "f32[32, 16, 49, 49]" = torch.ops.aten.clone.default(_softmax_7);  _softmax_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    expand_30: "f32[32, 16, 49, 49]" = torch.ops.aten.expand.default(clone_83, [32, 16, 49, 49]);  clone_83 = None
    view_177: "f32[512, 49, 49]" = torch.ops.aten.view.default(expand_30, [512, 49, 49]);  expand_30 = None
    expand_31: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_77, [32, 16, 49, 32]);  getitem_77 = None
    clone_84: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
    _unsafe_view_32: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_84, [512, 49, 32]);  clone_84 = None
    bmm_15: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_177, _unsafe_view_32)
    view_178: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_15, [32, 16, 49, 32]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_15: "f32[32, 49, 16, 32]" = torch.ops.aten.transpose.int(view_178, 1, 2);  view_178 = None
    clone_85: "f32[32, 49, 16, 32]" = torch.ops.aten.clone.default(transpose_15, memory_format = torch.contiguous_format);  transpose_15 = None
    _unsafe_view_33: "f32[32, 49, 512]" = torch.ops.aten._unsafe_view.default(clone_85, [32, 49, 512]);  clone_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_179: "f32[1568, 512]" = torch.ops.aten.view.default(_unsafe_view_33, [1568, 512]);  _unsafe_view_33 = None
    t_31: "f32[512, 512]" = torch.ops.aten.t.default(primals_123);  primals_123 = None
    addmm_29: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_124, view_179, t_31);  primals_124 = None
    view_180: "f32[32, 49, 512]" = torch.ops.aten.view.default(addmm_29, [32, 49, 512]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    clone_86: "f32[32, 49, 512]" = torch.ops.aten.clone.default(view_180);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_181: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_86, [-1, 7, 7, 512]);  clone_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    view_182: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_181, [-1, 2, 2, 7, 7, 512]);  view_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_34: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_182, [0, 1, 3, 2, 4, 5]);  view_182 = None
    clone_87: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
    view_183: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(clone_87, [-1, 14, 14, 512]);  clone_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_15: "f32[8, 14, 14, 512]" = torch.ops.aten.slice.Tensor(view_183, 0, 0, 9223372036854775807);  view_183 = None
    slice_16: "f32[8, 14, 14, 512]" = torch.ops.aten.slice.Tensor(slice_15, 3, 0, 9223372036854775807);  slice_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    roll_7: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(slice_16, [3, 3], [1, 2]);  slice_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_12: "f32[8, 1, 1, 1]" = torch.ops.aten.new_empty.default(roll_7, [8, 1, 1, 1], pin_memory = False)
    bernoulli_12: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_12, 0.9695652164518833);  new_empty_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_12: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_12, 0.9695652164518833)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_20: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(roll_7, div_12);  roll_7 = div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_26: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_165, mul_20);  mul_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_184: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_26, [8, -1, 512]);  add_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_18 = torch.ops.aten.native_layer_norm.default(view_184, [512], primals_125, primals_126, 1e-05)
    getitem_78: "f32[8, 196, 512]" = native_layer_norm_18[0]
    getitem_79: "f32[8, 196, 1]" = native_layer_norm_18[1]
    getitem_80: "f32[8, 196, 1]" = native_layer_norm_18[2];  native_layer_norm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_185: "f32[1568, 512]" = torch.ops.aten.view.default(getitem_78, [1568, 512]);  getitem_78 = None
    t_32: "f32[512, 2048]" = torch.ops.aten.t.default(primals_127);  primals_127 = None
    addmm_30: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_128, view_185, t_32);  primals_128 = None
    view_186: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_30, [8, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_7: "f32[8, 196, 2048]" = torch.ops.aten.gelu.default(view_186);  view_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_88: "f32[8, 196, 2048]" = torch.ops.aten.clone.default(gelu_7);  gelu_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_187: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_88, [1568, 2048]);  clone_88 = None
    t_33: "f32[2048, 512]" = torch.ops.aten.t.default(primals_129);  primals_129 = None
    addmm_31: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_130, view_187, t_33);  primals_130 = None
    view_188: "f32[8, 196, 512]" = torch.ops.aten.view.default(addmm_31, [8, 196, 512]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_89: "f32[8, 196, 512]" = torch.ops.aten.clone.default(view_188);  view_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_13: "f32[8, 1, 1]" = torch.ops.aten.new_empty.default(clone_89, [8, 1, 1], pin_memory = False)
    bernoulli_13: "f32[8, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_13, 0.9695652164518833);  new_empty_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_13: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_13, 0.9695652164518833)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_21: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(clone_89, div_13);  clone_89 = div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_27: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_184, mul_21);  mul_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_189: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_27, [8, 14, 14, 512]);  add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_19 = torch.ops.aten.native_layer_norm.default(view_189, [512], primals_131, primals_132, 1e-05)
    getitem_81: "f32[8, 14, 14, 512]" = native_layer_norm_19[0]
    getitem_82: "f32[8, 14, 14, 1]" = native_layer_norm_19[1]
    getitem_83: "f32[8, 14, 14, 1]" = native_layer_norm_19[2];  native_layer_norm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_8: "f32[8, 14, 14, 512]" = torch.ops.aten.constant_pad_nd.default(getitem_81, [0, 0, 0, 0, 0, 0], 0.0);  getitem_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    view_190: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(constant_pad_nd_8, [8, 2, 7, 2, 7, 512]);  constant_pad_nd_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_35: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_190, [0, 1, 3, 2, 4, 5]);  view_190 = None
    clone_90: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
    view_191: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_90, [-1, 7, 7, 512]);  clone_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_192: "f32[32, 49, 512]" = torch.ops.aten.view.default(view_191, [-1, 49, 512]);  view_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_193: "f32[1568, 512]" = torch.ops.aten.view.default(view_192, [1568, 512]);  view_192 = None
    t_34: "f32[512, 1536]" = torch.ops.aten.t.default(primals_133);  primals_133 = None
    addmm_32: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_134, view_193, t_34);  primals_134 = None
    view_194: "f32[32, 49, 1536]" = torch.ops.aten.view.default(addmm_32, [32, 49, 1536]);  addmm_32 = None
    view_195: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_194, [32, 49, 3, 16, -1]);  view_194 = None
    permute_36: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.permute.default(view_195, [2, 0, 3, 1, 4]);  view_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_8 = torch.ops.aten.unbind.int(permute_36);  permute_36 = None
    getitem_84: "f32[32, 16, 49, 32]" = unbind_8[0]
    getitem_85: "f32[32, 16, 49, 32]" = unbind_8[1]
    getitem_86: "f32[32, 16, 49, 32]" = unbind_8[2];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_22: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_84, 0.1767766952966369);  getitem_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_16: "f32[32, 16, 32, 49]" = torch.ops.aten.transpose.int(getitem_85, -2, -1);  getitem_85 = None
    expand_32: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(mul_22, [32, 16, 49, 32]);  mul_22 = None
    clone_91: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
    _unsafe_view_34: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_91, [512, 49, 32]);  clone_91 = None
    expand_33: "f32[32, 16, 32, 49]" = torch.ops.aten.expand.default(transpose_16, [32, 16, 32, 49]);  transpose_16 = None
    clone_92: "f32[32, 16, 32, 49]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
    _unsafe_view_35: "f32[512, 32, 49]" = torch.ops.aten._unsafe_view.default(clone_92, [512, 32, 49]);  clone_92 = None
    bmm_16: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_34, _unsafe_view_35)
    view_196: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_16, [32, 16, 49, 49]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_197: "i64[2401]" = torch.ops.aten.view.default(primals_342, [-1]);  primals_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_8: "f32[2401, 16]" = torch.ops.aten.index.Tensor(primals_9, [view_197]);  primals_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_198: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_8, [49, 49, -1]);  index_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_37: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_198, [2, 0, 1]);  view_198 = None
    clone_93: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_16: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_93, 0);  clone_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    add_28: "f32[32, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_196, unsqueeze_16);  view_196 = unsqueeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_8: "f32[32, 16, 49, 49]" = torch.ops.aten._softmax.default(add_28, -1, False);  add_28 = None
    detach_8: "f32[32, 16, 49, 49]" = torch.ops.aten.detach.default(_softmax_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    clone_94: "f32[32, 16, 49, 49]" = torch.ops.aten.clone.default(_softmax_8);  _softmax_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    expand_34: "f32[32, 16, 49, 49]" = torch.ops.aten.expand.default(clone_94, [32, 16, 49, 49]);  clone_94 = None
    view_199: "f32[512, 49, 49]" = torch.ops.aten.view.default(expand_34, [512, 49, 49]);  expand_34 = None
    expand_35: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_86, [32, 16, 49, 32]);  getitem_86 = None
    clone_95: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
    _unsafe_view_36: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_95, [512, 49, 32]);  clone_95 = None
    bmm_17: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_199, _unsafe_view_36)
    view_200: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_17, [32, 16, 49, 32]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_17: "f32[32, 49, 16, 32]" = torch.ops.aten.transpose.int(view_200, 1, 2);  view_200 = None
    clone_96: "f32[32, 49, 16, 32]" = torch.ops.aten.clone.default(transpose_17, memory_format = torch.contiguous_format);  transpose_17 = None
    _unsafe_view_37: "f32[32, 49, 512]" = torch.ops.aten._unsafe_view.default(clone_96, [32, 49, 512]);  clone_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_201: "f32[1568, 512]" = torch.ops.aten.view.default(_unsafe_view_37, [1568, 512]);  _unsafe_view_37 = None
    t_35: "f32[512, 512]" = torch.ops.aten.t.default(primals_135);  primals_135 = None
    addmm_33: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_136, view_201, t_35);  primals_136 = None
    view_202: "f32[32, 49, 512]" = torch.ops.aten.view.default(addmm_33, [32, 49, 512]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    clone_97: "f32[32, 49, 512]" = torch.ops.aten.clone.default(view_202);  view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_203: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_97, [-1, 7, 7, 512]);  clone_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    view_204: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_203, [-1, 2, 2, 7, 7, 512]);  view_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_38: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_204, [0, 1, 3, 2, 4, 5]);  view_204 = None
    clone_98: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    view_205: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(clone_98, [-1, 14, 14, 512]);  clone_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_17: "f32[8, 14, 14, 512]" = torch.ops.aten.slice.Tensor(view_205, 0, 0, 9223372036854775807);  view_205 = None
    slice_18: "f32[8, 14, 14, 512]" = torch.ops.aten.slice.Tensor(slice_17, 3, 0, 9223372036854775807);  slice_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_14: "f32[8, 1, 1, 1]" = torch.ops.aten.new_empty.default(slice_18, [8, 1, 1, 1], pin_memory = False)
    bernoulli_14: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_14, 0.9652173891663551);  new_empty_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_14: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_14, 0.9652173891663551)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_23: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(slice_18, div_14);  slice_18 = div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_29: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_189, mul_23);  mul_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_206: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_29, [8, -1, 512]);  add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_20 = torch.ops.aten.native_layer_norm.default(view_206, [512], primals_137, primals_138, 1e-05)
    getitem_87: "f32[8, 196, 512]" = native_layer_norm_20[0]
    getitem_88: "f32[8, 196, 1]" = native_layer_norm_20[1]
    getitem_89: "f32[8, 196, 1]" = native_layer_norm_20[2];  native_layer_norm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_207: "f32[1568, 512]" = torch.ops.aten.view.default(getitem_87, [1568, 512]);  getitem_87 = None
    t_36: "f32[512, 2048]" = torch.ops.aten.t.default(primals_139);  primals_139 = None
    addmm_34: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_140, view_207, t_36);  primals_140 = None
    view_208: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_34, [8, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_8: "f32[8, 196, 2048]" = torch.ops.aten.gelu.default(view_208);  view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_99: "f32[8, 196, 2048]" = torch.ops.aten.clone.default(gelu_8);  gelu_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_209: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_99, [1568, 2048]);  clone_99 = None
    t_37: "f32[2048, 512]" = torch.ops.aten.t.default(primals_141);  primals_141 = None
    addmm_35: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_142, view_209, t_37);  primals_142 = None
    view_210: "f32[8, 196, 512]" = torch.ops.aten.view.default(addmm_35, [8, 196, 512]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_100: "f32[8, 196, 512]" = torch.ops.aten.clone.default(view_210);  view_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_15: "f32[8, 1, 1]" = torch.ops.aten.new_empty.default(clone_100, [8, 1, 1], pin_memory = False)
    bernoulli_15: "f32[8, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_15, 0.9652173891663551);  new_empty_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_15: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_15, 0.9652173891663551)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_24: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(clone_100, div_15);  clone_100 = div_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_30: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_206, mul_24);  mul_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_211: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_30, [8, 14, 14, 512]);  add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_21 = torch.ops.aten.native_layer_norm.default(view_211, [512], primals_143, primals_144, 1e-05)
    getitem_90: "f32[8, 14, 14, 512]" = native_layer_norm_21[0]
    getitem_91: "f32[8, 14, 14, 1]" = native_layer_norm_21[1]
    getitem_92: "f32[8, 14, 14, 1]" = native_layer_norm_21[2];  native_layer_norm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    roll_8: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(getitem_90, [-3, -3], [1, 2]);  getitem_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_9: "f32[8, 14, 14, 512]" = torch.ops.aten.constant_pad_nd.default(roll_8, [0, 0, 0, 0, 0, 0], 0.0);  roll_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    view_212: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(constant_pad_nd_9, [8, 2, 7, 2, 7, 512]);  constant_pad_nd_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_39: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_212, [0, 1, 3, 2, 4, 5]);  view_212 = None
    clone_101: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_39, memory_format = torch.contiguous_format);  permute_39 = None
    view_213: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_101, [-1, 7, 7, 512]);  clone_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_214: "f32[32, 49, 512]" = torch.ops.aten.view.default(view_213, [-1, 49, 512]);  view_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_215: "f32[1568, 512]" = torch.ops.aten.view.default(view_214, [1568, 512]);  view_214 = None
    t_38: "f32[512, 1536]" = torch.ops.aten.t.default(primals_145);  primals_145 = None
    addmm_36: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_146, view_215, t_38);  primals_146 = None
    view_216: "f32[32, 49, 1536]" = torch.ops.aten.view.default(addmm_36, [32, 49, 1536]);  addmm_36 = None
    view_217: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_216, [32, 49, 3, 16, -1]);  view_216 = None
    permute_40: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.permute.default(view_217, [2, 0, 3, 1, 4]);  view_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_9 = torch.ops.aten.unbind.int(permute_40);  permute_40 = None
    getitem_93: "f32[32, 16, 49, 32]" = unbind_9[0]
    getitem_94: "f32[32, 16, 49, 32]" = unbind_9[1]
    getitem_95: "f32[32, 16, 49, 32]" = unbind_9[2];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_25: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_93, 0.1767766952966369);  getitem_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_18: "f32[32, 16, 32, 49]" = torch.ops.aten.transpose.int(getitem_94, -2, -1);  getitem_94 = None
    expand_36: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(mul_25, [32, 16, 49, 32]);  mul_25 = None
    clone_102: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
    _unsafe_view_38: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_102, [512, 49, 32]);  clone_102 = None
    expand_37: "f32[32, 16, 32, 49]" = torch.ops.aten.expand.default(transpose_18, [32, 16, 32, 49]);  transpose_18 = None
    clone_103: "f32[32, 16, 32, 49]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
    _unsafe_view_39: "f32[512, 32, 49]" = torch.ops.aten._unsafe_view.default(clone_103, [512, 32, 49]);  clone_103 = None
    bmm_18: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_38, _unsafe_view_39)
    view_218: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_18, [32, 16, 49, 49]);  bmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_219: "i64[2401]" = torch.ops.aten.view.default(primals_344, [-1]);  primals_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_9: "f32[2401, 16]" = torch.ops.aten.index.Tensor(primals_10, [view_219]);  primals_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_220: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_9, [49, 49, -1]);  index_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_41: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_220, [2, 0, 1]);  view_220 = None
    clone_104: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_17: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_104, 0);  clone_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    add_31: "f32[32, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_218, unsqueeze_17);  view_218 = unsqueeze_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_221: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_31, [-1, 4, 16, 49, 49]);  add_31 = None
    unsqueeze_18: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(primals_343, 1);  primals_343 = None
    unsqueeze_19: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, 0);  unsqueeze_18 = None
    add_32: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_221, unsqueeze_19);  view_221 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    view_222: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(add_32, [-1, 16, 49, 49]);  add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_9: "f32[32, 16, 49, 49]" = torch.ops.aten._softmax.default(view_222, -1, False);  view_222 = None
    detach_9: "f32[32, 16, 49, 49]" = torch.ops.aten.detach.default(_softmax_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    clone_105: "f32[32, 16, 49, 49]" = torch.ops.aten.clone.default(_softmax_9);  _softmax_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    expand_38: "f32[32, 16, 49, 49]" = torch.ops.aten.expand.default(clone_105, [32, 16, 49, 49]);  clone_105 = None
    view_223: "f32[512, 49, 49]" = torch.ops.aten.view.default(expand_38, [512, 49, 49]);  expand_38 = None
    expand_39: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_95, [32, 16, 49, 32]);  getitem_95 = None
    clone_106: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
    _unsafe_view_40: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_106, [512, 49, 32]);  clone_106 = None
    bmm_19: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_223, _unsafe_view_40)
    view_224: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_19, [32, 16, 49, 32]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_19: "f32[32, 49, 16, 32]" = torch.ops.aten.transpose.int(view_224, 1, 2);  view_224 = None
    clone_107: "f32[32, 49, 16, 32]" = torch.ops.aten.clone.default(transpose_19, memory_format = torch.contiguous_format);  transpose_19 = None
    _unsafe_view_41: "f32[32, 49, 512]" = torch.ops.aten._unsafe_view.default(clone_107, [32, 49, 512]);  clone_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_225: "f32[1568, 512]" = torch.ops.aten.view.default(_unsafe_view_41, [1568, 512]);  _unsafe_view_41 = None
    t_39: "f32[512, 512]" = torch.ops.aten.t.default(primals_147);  primals_147 = None
    addmm_37: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_148, view_225, t_39);  primals_148 = None
    view_226: "f32[32, 49, 512]" = torch.ops.aten.view.default(addmm_37, [32, 49, 512]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    clone_108: "f32[32, 49, 512]" = torch.ops.aten.clone.default(view_226);  view_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_227: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_108, [-1, 7, 7, 512]);  clone_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    view_228: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_227, [-1, 2, 2, 7, 7, 512]);  view_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_42: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_228, [0, 1, 3, 2, 4, 5]);  view_228 = None
    clone_109: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_42, memory_format = torch.contiguous_format);  permute_42 = None
    view_229: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(clone_109, [-1, 14, 14, 512]);  clone_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_19: "f32[8, 14, 14, 512]" = torch.ops.aten.slice.Tensor(view_229, 0, 0, 9223372036854775807);  view_229 = None
    slice_20: "f32[8, 14, 14, 512]" = torch.ops.aten.slice.Tensor(slice_19, 3, 0, 9223372036854775807);  slice_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    roll_9: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(slice_20, [3, 3], [1, 2]);  slice_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_16: "f32[8, 1, 1, 1]" = torch.ops.aten.new_empty.default(roll_9, [8, 1, 1, 1], pin_memory = False)
    bernoulli_16: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_16, 0.960869561880827);  new_empty_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_16: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_16, 0.960869561880827)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_26: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(roll_9, div_16);  roll_9 = div_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_33: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_211, mul_26);  mul_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_230: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_33, [8, -1, 512]);  add_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_22 = torch.ops.aten.native_layer_norm.default(view_230, [512], primals_149, primals_150, 1e-05)
    getitem_96: "f32[8, 196, 512]" = native_layer_norm_22[0]
    getitem_97: "f32[8, 196, 1]" = native_layer_norm_22[1]
    getitem_98: "f32[8, 196, 1]" = native_layer_norm_22[2];  native_layer_norm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_231: "f32[1568, 512]" = torch.ops.aten.view.default(getitem_96, [1568, 512]);  getitem_96 = None
    t_40: "f32[512, 2048]" = torch.ops.aten.t.default(primals_151);  primals_151 = None
    addmm_38: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_152, view_231, t_40);  primals_152 = None
    view_232: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_38, [8, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_9: "f32[8, 196, 2048]" = torch.ops.aten.gelu.default(view_232);  view_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_110: "f32[8, 196, 2048]" = torch.ops.aten.clone.default(gelu_9);  gelu_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_233: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_110, [1568, 2048]);  clone_110 = None
    t_41: "f32[2048, 512]" = torch.ops.aten.t.default(primals_153);  primals_153 = None
    addmm_39: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_154, view_233, t_41);  primals_154 = None
    view_234: "f32[8, 196, 512]" = torch.ops.aten.view.default(addmm_39, [8, 196, 512]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_111: "f32[8, 196, 512]" = torch.ops.aten.clone.default(view_234);  view_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_17: "f32[8, 1, 1]" = torch.ops.aten.new_empty.default(clone_111, [8, 1, 1], pin_memory = False)
    bernoulli_17: "f32[8, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_17, 0.960869561880827);  new_empty_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_17: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_17, 0.960869561880827)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_27: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(clone_111, div_17);  clone_111 = div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_34: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_230, mul_27);  mul_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_235: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_34, [8, 14, 14, 512]);  add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_23 = torch.ops.aten.native_layer_norm.default(view_235, [512], primals_155, primals_156, 1e-05)
    getitem_99: "f32[8, 14, 14, 512]" = native_layer_norm_23[0]
    getitem_100: "f32[8, 14, 14, 1]" = native_layer_norm_23[1]
    getitem_101: "f32[8, 14, 14, 1]" = native_layer_norm_23[2];  native_layer_norm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_10: "f32[8, 14, 14, 512]" = torch.ops.aten.constant_pad_nd.default(getitem_99, [0, 0, 0, 0, 0, 0], 0.0);  getitem_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    view_236: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(constant_pad_nd_10, [8, 2, 7, 2, 7, 512]);  constant_pad_nd_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_43: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_236, [0, 1, 3, 2, 4, 5]);  view_236 = None
    clone_112: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_43, memory_format = torch.contiguous_format);  permute_43 = None
    view_237: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_112, [-1, 7, 7, 512]);  clone_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_238: "f32[32, 49, 512]" = torch.ops.aten.view.default(view_237, [-1, 49, 512]);  view_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_239: "f32[1568, 512]" = torch.ops.aten.view.default(view_238, [1568, 512]);  view_238 = None
    t_42: "f32[512, 1536]" = torch.ops.aten.t.default(primals_157);  primals_157 = None
    addmm_40: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_158, view_239, t_42);  primals_158 = None
    view_240: "f32[32, 49, 1536]" = torch.ops.aten.view.default(addmm_40, [32, 49, 1536]);  addmm_40 = None
    view_241: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_240, [32, 49, 3, 16, -1]);  view_240 = None
    permute_44: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.permute.default(view_241, [2, 0, 3, 1, 4]);  view_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_10 = torch.ops.aten.unbind.int(permute_44);  permute_44 = None
    getitem_102: "f32[32, 16, 49, 32]" = unbind_10[0]
    getitem_103: "f32[32, 16, 49, 32]" = unbind_10[1]
    getitem_104: "f32[32, 16, 49, 32]" = unbind_10[2];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_28: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_102, 0.1767766952966369);  getitem_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_20: "f32[32, 16, 32, 49]" = torch.ops.aten.transpose.int(getitem_103, -2, -1);  getitem_103 = None
    expand_40: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(mul_28, [32, 16, 49, 32]);  mul_28 = None
    clone_113: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
    _unsafe_view_42: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_113, [512, 49, 32]);  clone_113 = None
    expand_41: "f32[32, 16, 32, 49]" = torch.ops.aten.expand.default(transpose_20, [32, 16, 32, 49]);  transpose_20 = None
    clone_114: "f32[32, 16, 32, 49]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
    _unsafe_view_43: "f32[512, 32, 49]" = torch.ops.aten._unsafe_view.default(clone_114, [512, 32, 49]);  clone_114 = None
    bmm_20: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_42, _unsafe_view_43)
    view_242: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_20, [32, 16, 49, 49]);  bmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_243: "i64[2401]" = torch.ops.aten.view.default(primals_345, [-1]);  primals_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_10: "f32[2401, 16]" = torch.ops.aten.index.Tensor(primals_11, [view_243]);  primals_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_244: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_10, [49, 49, -1]);  index_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_45: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_244, [2, 0, 1]);  view_244 = None
    clone_115: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_20: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_115, 0);  clone_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    add_35: "f32[32, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_242, unsqueeze_20);  view_242 = unsqueeze_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_10: "f32[32, 16, 49, 49]" = torch.ops.aten._softmax.default(add_35, -1, False);  add_35 = None
    detach_10: "f32[32, 16, 49, 49]" = torch.ops.aten.detach.default(_softmax_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    clone_116: "f32[32, 16, 49, 49]" = torch.ops.aten.clone.default(_softmax_10);  _softmax_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    expand_42: "f32[32, 16, 49, 49]" = torch.ops.aten.expand.default(clone_116, [32, 16, 49, 49]);  clone_116 = None
    view_245: "f32[512, 49, 49]" = torch.ops.aten.view.default(expand_42, [512, 49, 49]);  expand_42 = None
    expand_43: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_104, [32, 16, 49, 32]);  getitem_104 = None
    clone_117: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
    _unsafe_view_44: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_117, [512, 49, 32]);  clone_117 = None
    bmm_21: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_245, _unsafe_view_44)
    view_246: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_21, [32, 16, 49, 32]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_21: "f32[32, 49, 16, 32]" = torch.ops.aten.transpose.int(view_246, 1, 2);  view_246 = None
    clone_118: "f32[32, 49, 16, 32]" = torch.ops.aten.clone.default(transpose_21, memory_format = torch.contiguous_format);  transpose_21 = None
    _unsafe_view_45: "f32[32, 49, 512]" = torch.ops.aten._unsafe_view.default(clone_118, [32, 49, 512]);  clone_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_247: "f32[1568, 512]" = torch.ops.aten.view.default(_unsafe_view_45, [1568, 512]);  _unsafe_view_45 = None
    t_43: "f32[512, 512]" = torch.ops.aten.t.default(primals_159);  primals_159 = None
    addmm_41: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_160, view_247, t_43);  primals_160 = None
    view_248: "f32[32, 49, 512]" = torch.ops.aten.view.default(addmm_41, [32, 49, 512]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    clone_119: "f32[32, 49, 512]" = torch.ops.aten.clone.default(view_248);  view_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_249: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_119, [-1, 7, 7, 512]);  clone_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    view_250: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_249, [-1, 2, 2, 7, 7, 512]);  view_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_46: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_250, [0, 1, 3, 2, 4, 5]);  view_250 = None
    clone_120: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
    view_251: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(clone_120, [-1, 14, 14, 512]);  clone_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_21: "f32[8, 14, 14, 512]" = torch.ops.aten.slice.Tensor(view_251, 0, 0, 9223372036854775807);  view_251 = None
    slice_22: "f32[8, 14, 14, 512]" = torch.ops.aten.slice.Tensor(slice_21, 3, 0, 9223372036854775807);  slice_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_18: "f32[8, 1, 1, 1]" = torch.ops.aten.new_empty.default(slice_22, [8, 1, 1, 1], pin_memory = False)
    bernoulli_18: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_18, 0.9565217345952988);  new_empty_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_18: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_18, 0.9565217345952988)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_29: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(slice_22, div_18);  slice_22 = div_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_36: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_235, mul_29);  mul_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_252: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_36, [8, -1, 512]);  add_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_24 = torch.ops.aten.native_layer_norm.default(view_252, [512], primals_161, primals_162, 1e-05)
    getitem_105: "f32[8, 196, 512]" = native_layer_norm_24[0]
    getitem_106: "f32[8, 196, 1]" = native_layer_norm_24[1]
    getitem_107: "f32[8, 196, 1]" = native_layer_norm_24[2];  native_layer_norm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_253: "f32[1568, 512]" = torch.ops.aten.view.default(getitem_105, [1568, 512]);  getitem_105 = None
    t_44: "f32[512, 2048]" = torch.ops.aten.t.default(primals_163);  primals_163 = None
    addmm_42: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_164, view_253, t_44);  primals_164 = None
    view_254: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_42, [8, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_10: "f32[8, 196, 2048]" = torch.ops.aten.gelu.default(view_254);  view_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_121: "f32[8, 196, 2048]" = torch.ops.aten.clone.default(gelu_10);  gelu_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_255: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_121, [1568, 2048]);  clone_121 = None
    t_45: "f32[2048, 512]" = torch.ops.aten.t.default(primals_165);  primals_165 = None
    addmm_43: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_166, view_255, t_45);  primals_166 = None
    view_256: "f32[8, 196, 512]" = torch.ops.aten.view.default(addmm_43, [8, 196, 512]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_122: "f32[8, 196, 512]" = torch.ops.aten.clone.default(view_256);  view_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_19: "f32[8, 1, 1]" = torch.ops.aten.new_empty.default(clone_122, [8, 1, 1], pin_memory = False)
    bernoulli_19: "f32[8, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_19, 0.9565217345952988);  new_empty_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_19: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_19, 0.9565217345952988)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_30: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(clone_122, div_19);  clone_122 = div_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_37: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_252, mul_30);  mul_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_257: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_37, [8, 14, 14, 512]);  add_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_25 = torch.ops.aten.native_layer_norm.default(view_257, [512], primals_167, primals_168, 1e-05)
    getitem_108: "f32[8, 14, 14, 512]" = native_layer_norm_25[0]
    getitem_109: "f32[8, 14, 14, 1]" = native_layer_norm_25[1]
    getitem_110: "f32[8, 14, 14, 1]" = native_layer_norm_25[2];  native_layer_norm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    roll_10: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(getitem_108, [-3, -3], [1, 2]);  getitem_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_11: "f32[8, 14, 14, 512]" = torch.ops.aten.constant_pad_nd.default(roll_10, [0, 0, 0, 0, 0, 0], 0.0);  roll_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    view_258: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(constant_pad_nd_11, [8, 2, 7, 2, 7, 512]);  constant_pad_nd_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_47: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_258, [0, 1, 3, 2, 4, 5]);  view_258 = None
    clone_123: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
    view_259: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_123, [-1, 7, 7, 512]);  clone_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_260: "f32[32, 49, 512]" = torch.ops.aten.view.default(view_259, [-1, 49, 512]);  view_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_261: "f32[1568, 512]" = torch.ops.aten.view.default(view_260, [1568, 512]);  view_260 = None
    t_46: "f32[512, 1536]" = torch.ops.aten.t.default(primals_169);  primals_169 = None
    addmm_44: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_170, view_261, t_46);  primals_170 = None
    view_262: "f32[32, 49, 1536]" = torch.ops.aten.view.default(addmm_44, [32, 49, 1536]);  addmm_44 = None
    view_263: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_262, [32, 49, 3, 16, -1]);  view_262 = None
    permute_48: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.permute.default(view_263, [2, 0, 3, 1, 4]);  view_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_11 = torch.ops.aten.unbind.int(permute_48);  permute_48 = None
    getitem_111: "f32[32, 16, 49, 32]" = unbind_11[0]
    getitem_112: "f32[32, 16, 49, 32]" = unbind_11[1]
    getitem_113: "f32[32, 16, 49, 32]" = unbind_11[2];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_31: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_111, 0.1767766952966369);  getitem_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_22: "f32[32, 16, 32, 49]" = torch.ops.aten.transpose.int(getitem_112, -2, -1);  getitem_112 = None
    expand_44: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(mul_31, [32, 16, 49, 32]);  mul_31 = None
    clone_124: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
    _unsafe_view_46: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_124, [512, 49, 32]);  clone_124 = None
    expand_45: "f32[32, 16, 32, 49]" = torch.ops.aten.expand.default(transpose_22, [32, 16, 32, 49]);  transpose_22 = None
    clone_125: "f32[32, 16, 32, 49]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
    _unsafe_view_47: "f32[512, 32, 49]" = torch.ops.aten._unsafe_view.default(clone_125, [512, 32, 49]);  clone_125 = None
    bmm_22: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_46, _unsafe_view_47)
    view_264: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_22, [32, 16, 49, 49]);  bmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_265: "i64[2401]" = torch.ops.aten.view.default(primals_347, [-1]);  primals_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_11: "f32[2401, 16]" = torch.ops.aten.index.Tensor(primals_12, [view_265]);  primals_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_266: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_11, [49, 49, -1]);  index_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_49: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_266, [2, 0, 1]);  view_266 = None
    clone_126: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_21: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_126, 0);  clone_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    add_38: "f32[32, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_264, unsqueeze_21);  view_264 = unsqueeze_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_267: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_38, [-1, 4, 16, 49, 49]);  add_38 = None
    unsqueeze_22: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(primals_346, 1);  primals_346 = None
    unsqueeze_23: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, 0);  unsqueeze_22 = None
    add_39: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_267, unsqueeze_23);  view_267 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    view_268: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(add_39, [-1, 16, 49, 49]);  add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_11: "f32[32, 16, 49, 49]" = torch.ops.aten._softmax.default(view_268, -1, False);  view_268 = None
    detach_11: "f32[32, 16, 49, 49]" = torch.ops.aten.detach.default(_softmax_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    clone_127: "f32[32, 16, 49, 49]" = torch.ops.aten.clone.default(_softmax_11);  _softmax_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    expand_46: "f32[32, 16, 49, 49]" = torch.ops.aten.expand.default(clone_127, [32, 16, 49, 49]);  clone_127 = None
    view_269: "f32[512, 49, 49]" = torch.ops.aten.view.default(expand_46, [512, 49, 49]);  expand_46 = None
    expand_47: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_113, [32, 16, 49, 32]);  getitem_113 = None
    clone_128: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
    _unsafe_view_48: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_128, [512, 49, 32]);  clone_128 = None
    bmm_23: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_269, _unsafe_view_48)
    view_270: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_23, [32, 16, 49, 32]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_23: "f32[32, 49, 16, 32]" = torch.ops.aten.transpose.int(view_270, 1, 2);  view_270 = None
    clone_129: "f32[32, 49, 16, 32]" = torch.ops.aten.clone.default(transpose_23, memory_format = torch.contiguous_format);  transpose_23 = None
    _unsafe_view_49: "f32[32, 49, 512]" = torch.ops.aten._unsafe_view.default(clone_129, [32, 49, 512]);  clone_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_271: "f32[1568, 512]" = torch.ops.aten.view.default(_unsafe_view_49, [1568, 512]);  _unsafe_view_49 = None
    t_47: "f32[512, 512]" = torch.ops.aten.t.default(primals_171);  primals_171 = None
    addmm_45: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_172, view_271, t_47);  primals_172 = None
    view_272: "f32[32, 49, 512]" = torch.ops.aten.view.default(addmm_45, [32, 49, 512]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    clone_130: "f32[32, 49, 512]" = torch.ops.aten.clone.default(view_272);  view_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_273: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_130, [-1, 7, 7, 512]);  clone_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    view_274: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_273, [-1, 2, 2, 7, 7, 512]);  view_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_50: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_274, [0, 1, 3, 2, 4, 5]);  view_274 = None
    clone_131: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_50, memory_format = torch.contiguous_format);  permute_50 = None
    view_275: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(clone_131, [-1, 14, 14, 512]);  clone_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_23: "f32[8, 14, 14, 512]" = torch.ops.aten.slice.Tensor(view_275, 0, 0, 9223372036854775807);  view_275 = None
    slice_24: "f32[8, 14, 14, 512]" = torch.ops.aten.slice.Tensor(slice_23, 3, 0, 9223372036854775807);  slice_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    roll_11: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(slice_24, [3, 3], [1, 2]);  slice_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_20: "f32[8, 1, 1, 1]" = torch.ops.aten.new_empty.default(roll_11, [8, 1, 1, 1], pin_memory = False)
    bernoulli_20: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_20, 0.9521739110350609);  new_empty_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_20: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_20, 0.9521739110350609)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_32: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(roll_11, div_20);  roll_11 = div_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_40: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_257, mul_32);  mul_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_276: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_40, [8, -1, 512]);  add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_26 = torch.ops.aten.native_layer_norm.default(view_276, [512], primals_173, primals_174, 1e-05)
    getitem_114: "f32[8, 196, 512]" = native_layer_norm_26[0]
    getitem_115: "f32[8, 196, 1]" = native_layer_norm_26[1]
    getitem_116: "f32[8, 196, 1]" = native_layer_norm_26[2];  native_layer_norm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_277: "f32[1568, 512]" = torch.ops.aten.view.default(getitem_114, [1568, 512]);  getitem_114 = None
    t_48: "f32[512, 2048]" = torch.ops.aten.t.default(primals_175);  primals_175 = None
    addmm_46: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_176, view_277, t_48);  primals_176 = None
    view_278: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_46, [8, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_11: "f32[8, 196, 2048]" = torch.ops.aten.gelu.default(view_278);  view_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_132: "f32[8, 196, 2048]" = torch.ops.aten.clone.default(gelu_11);  gelu_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_279: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_132, [1568, 2048]);  clone_132 = None
    t_49: "f32[2048, 512]" = torch.ops.aten.t.default(primals_177);  primals_177 = None
    addmm_47: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_178, view_279, t_49);  primals_178 = None
    view_280: "f32[8, 196, 512]" = torch.ops.aten.view.default(addmm_47, [8, 196, 512]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_133: "f32[8, 196, 512]" = torch.ops.aten.clone.default(view_280);  view_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_21: "f32[8, 1, 1]" = torch.ops.aten.new_empty.default(clone_133, [8, 1, 1], pin_memory = False)
    bernoulli_21: "f32[8, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_21, 0.9521739110350609);  new_empty_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_21: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_21, 0.9521739110350609)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_33: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(clone_133, div_21);  clone_133 = div_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_41: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_276, mul_33);  mul_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_281: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_41, [8, 14, 14, 512]);  add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_27 = torch.ops.aten.native_layer_norm.default(view_281, [512], primals_179, primals_180, 1e-05)
    getitem_117: "f32[8, 14, 14, 512]" = native_layer_norm_27[0]
    getitem_118: "f32[8, 14, 14, 1]" = native_layer_norm_27[1]
    getitem_119: "f32[8, 14, 14, 1]" = native_layer_norm_27[2];  native_layer_norm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_12: "f32[8, 14, 14, 512]" = torch.ops.aten.constant_pad_nd.default(getitem_117, [0, 0, 0, 0, 0, 0], 0.0);  getitem_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    view_282: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(constant_pad_nd_12, [8, 2, 7, 2, 7, 512]);  constant_pad_nd_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_51: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_282, [0, 1, 3, 2, 4, 5]);  view_282 = None
    clone_134: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    view_283: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_134, [-1, 7, 7, 512]);  clone_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_284: "f32[32, 49, 512]" = torch.ops.aten.view.default(view_283, [-1, 49, 512]);  view_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_285: "f32[1568, 512]" = torch.ops.aten.view.default(view_284, [1568, 512]);  view_284 = None
    t_50: "f32[512, 1536]" = torch.ops.aten.t.default(primals_181);  primals_181 = None
    addmm_48: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_182, view_285, t_50);  primals_182 = None
    view_286: "f32[32, 49, 1536]" = torch.ops.aten.view.default(addmm_48, [32, 49, 1536]);  addmm_48 = None
    view_287: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_286, [32, 49, 3, 16, -1]);  view_286 = None
    permute_52: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.permute.default(view_287, [2, 0, 3, 1, 4]);  view_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_12 = torch.ops.aten.unbind.int(permute_52);  permute_52 = None
    getitem_120: "f32[32, 16, 49, 32]" = unbind_12[0]
    getitem_121: "f32[32, 16, 49, 32]" = unbind_12[1]
    getitem_122: "f32[32, 16, 49, 32]" = unbind_12[2];  unbind_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_34: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_120, 0.1767766952966369);  getitem_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_24: "f32[32, 16, 32, 49]" = torch.ops.aten.transpose.int(getitem_121, -2, -1);  getitem_121 = None
    expand_48: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(mul_34, [32, 16, 49, 32]);  mul_34 = None
    clone_135: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
    _unsafe_view_50: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_135, [512, 49, 32]);  clone_135 = None
    expand_49: "f32[32, 16, 32, 49]" = torch.ops.aten.expand.default(transpose_24, [32, 16, 32, 49]);  transpose_24 = None
    clone_136: "f32[32, 16, 32, 49]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
    _unsafe_view_51: "f32[512, 32, 49]" = torch.ops.aten._unsafe_view.default(clone_136, [512, 32, 49]);  clone_136 = None
    bmm_24: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_50, _unsafe_view_51)
    view_288: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_24, [32, 16, 49, 49]);  bmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_289: "i64[2401]" = torch.ops.aten.view.default(primals_348, [-1]);  primals_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_12: "f32[2401, 16]" = torch.ops.aten.index.Tensor(primals_13, [view_289]);  primals_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_290: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_12, [49, 49, -1]);  index_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_53: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_290, [2, 0, 1]);  view_290 = None
    clone_137: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_53, memory_format = torch.contiguous_format);  permute_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_24: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_137, 0);  clone_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    add_42: "f32[32, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_288, unsqueeze_24);  view_288 = unsqueeze_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_12: "f32[32, 16, 49, 49]" = torch.ops.aten._softmax.default(add_42, -1, False);  add_42 = None
    detach_12: "f32[32, 16, 49, 49]" = torch.ops.aten.detach.default(_softmax_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    clone_138: "f32[32, 16, 49, 49]" = torch.ops.aten.clone.default(_softmax_12);  _softmax_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    expand_50: "f32[32, 16, 49, 49]" = torch.ops.aten.expand.default(clone_138, [32, 16, 49, 49]);  clone_138 = None
    view_291: "f32[512, 49, 49]" = torch.ops.aten.view.default(expand_50, [512, 49, 49]);  expand_50 = None
    expand_51: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_122, [32, 16, 49, 32]);  getitem_122 = None
    clone_139: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
    _unsafe_view_52: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_139, [512, 49, 32]);  clone_139 = None
    bmm_25: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_291, _unsafe_view_52)
    view_292: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_25, [32, 16, 49, 32]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_25: "f32[32, 49, 16, 32]" = torch.ops.aten.transpose.int(view_292, 1, 2);  view_292 = None
    clone_140: "f32[32, 49, 16, 32]" = torch.ops.aten.clone.default(transpose_25, memory_format = torch.contiguous_format);  transpose_25 = None
    _unsafe_view_53: "f32[32, 49, 512]" = torch.ops.aten._unsafe_view.default(clone_140, [32, 49, 512]);  clone_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_293: "f32[1568, 512]" = torch.ops.aten.view.default(_unsafe_view_53, [1568, 512]);  _unsafe_view_53 = None
    t_51: "f32[512, 512]" = torch.ops.aten.t.default(primals_183);  primals_183 = None
    addmm_49: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_184, view_293, t_51);  primals_184 = None
    view_294: "f32[32, 49, 512]" = torch.ops.aten.view.default(addmm_49, [32, 49, 512]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    clone_141: "f32[32, 49, 512]" = torch.ops.aten.clone.default(view_294);  view_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_295: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_141, [-1, 7, 7, 512]);  clone_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    view_296: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_295, [-1, 2, 2, 7, 7, 512]);  view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_54: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_296, [0, 1, 3, 2, 4, 5]);  view_296 = None
    clone_142: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_54, memory_format = torch.contiguous_format);  permute_54 = None
    view_297: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(clone_142, [-1, 14, 14, 512]);  clone_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_25: "f32[8, 14, 14, 512]" = torch.ops.aten.slice.Tensor(view_297, 0, 0, 9223372036854775807);  view_297 = None
    slice_26: "f32[8, 14, 14, 512]" = torch.ops.aten.slice.Tensor(slice_25, 3, 0, 9223372036854775807);  slice_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_22: "f32[8, 1, 1, 1]" = torch.ops.aten.new_empty.default(slice_26, [8, 1, 1, 1], pin_memory = False)
    bernoulli_22: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_22, 0.947826087474823);  new_empty_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_22: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_22, 0.947826087474823)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_35: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(slice_26, div_22);  slice_26 = div_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_43: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_281, mul_35);  mul_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_298: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_43, [8, -1, 512]);  add_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_28 = torch.ops.aten.native_layer_norm.default(view_298, [512], primals_185, primals_186, 1e-05)
    getitem_123: "f32[8, 196, 512]" = native_layer_norm_28[0]
    getitem_124: "f32[8, 196, 1]" = native_layer_norm_28[1]
    getitem_125: "f32[8, 196, 1]" = native_layer_norm_28[2];  native_layer_norm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_299: "f32[1568, 512]" = torch.ops.aten.view.default(getitem_123, [1568, 512]);  getitem_123 = None
    t_52: "f32[512, 2048]" = torch.ops.aten.t.default(primals_187);  primals_187 = None
    addmm_50: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_188, view_299, t_52);  primals_188 = None
    view_300: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_50, [8, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_12: "f32[8, 196, 2048]" = torch.ops.aten.gelu.default(view_300);  view_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_143: "f32[8, 196, 2048]" = torch.ops.aten.clone.default(gelu_12);  gelu_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_301: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_143, [1568, 2048]);  clone_143 = None
    t_53: "f32[2048, 512]" = torch.ops.aten.t.default(primals_189);  primals_189 = None
    addmm_51: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_190, view_301, t_53);  primals_190 = None
    view_302: "f32[8, 196, 512]" = torch.ops.aten.view.default(addmm_51, [8, 196, 512]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_144: "f32[8, 196, 512]" = torch.ops.aten.clone.default(view_302);  view_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_23: "f32[8, 1, 1]" = torch.ops.aten.new_empty.default(clone_144, [8, 1, 1], pin_memory = False)
    bernoulli_23: "f32[8, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_23, 0.947826087474823);  new_empty_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_23: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_23, 0.947826087474823)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_36: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(clone_144, div_23);  clone_144 = div_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_44: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_298, mul_36);  mul_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_303: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_44, [8, 14, 14, 512]);  add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_29 = torch.ops.aten.native_layer_norm.default(view_303, [512], primals_191, primals_192, 1e-05)
    getitem_126: "f32[8, 14, 14, 512]" = native_layer_norm_29[0]
    getitem_127: "f32[8, 14, 14, 1]" = native_layer_norm_29[1]
    getitem_128: "f32[8, 14, 14, 1]" = native_layer_norm_29[2];  native_layer_norm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    roll_12: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(getitem_126, [-3, -3], [1, 2]);  getitem_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_13: "f32[8, 14, 14, 512]" = torch.ops.aten.constant_pad_nd.default(roll_12, [0, 0, 0, 0, 0, 0], 0.0);  roll_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    view_304: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(constant_pad_nd_13, [8, 2, 7, 2, 7, 512]);  constant_pad_nd_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_55: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_304, [0, 1, 3, 2, 4, 5]);  view_304 = None
    clone_145: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_55, memory_format = torch.contiguous_format);  permute_55 = None
    view_305: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_145, [-1, 7, 7, 512]);  clone_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_306: "f32[32, 49, 512]" = torch.ops.aten.view.default(view_305, [-1, 49, 512]);  view_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_307: "f32[1568, 512]" = torch.ops.aten.view.default(view_306, [1568, 512]);  view_306 = None
    t_54: "f32[512, 1536]" = torch.ops.aten.t.default(primals_193);  primals_193 = None
    addmm_52: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_194, view_307, t_54);  primals_194 = None
    view_308: "f32[32, 49, 1536]" = torch.ops.aten.view.default(addmm_52, [32, 49, 1536]);  addmm_52 = None
    view_309: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_308, [32, 49, 3, 16, -1]);  view_308 = None
    permute_56: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.permute.default(view_309, [2, 0, 3, 1, 4]);  view_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_13 = torch.ops.aten.unbind.int(permute_56);  permute_56 = None
    getitem_129: "f32[32, 16, 49, 32]" = unbind_13[0]
    getitem_130: "f32[32, 16, 49, 32]" = unbind_13[1]
    getitem_131: "f32[32, 16, 49, 32]" = unbind_13[2];  unbind_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_37: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_129, 0.1767766952966369);  getitem_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_26: "f32[32, 16, 32, 49]" = torch.ops.aten.transpose.int(getitem_130, -2, -1);  getitem_130 = None
    expand_52: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(mul_37, [32, 16, 49, 32]);  mul_37 = None
    clone_146: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
    _unsafe_view_54: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_146, [512, 49, 32]);  clone_146 = None
    expand_53: "f32[32, 16, 32, 49]" = torch.ops.aten.expand.default(transpose_26, [32, 16, 32, 49]);  transpose_26 = None
    clone_147: "f32[32, 16, 32, 49]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
    _unsafe_view_55: "f32[512, 32, 49]" = torch.ops.aten._unsafe_view.default(clone_147, [512, 32, 49]);  clone_147 = None
    bmm_26: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_54, _unsafe_view_55)
    view_310: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_26, [32, 16, 49, 49]);  bmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_311: "i64[2401]" = torch.ops.aten.view.default(primals_350, [-1]);  primals_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_13: "f32[2401, 16]" = torch.ops.aten.index.Tensor(primals_14, [view_311]);  primals_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_312: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_13, [49, 49, -1]);  index_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_57: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_312, [2, 0, 1]);  view_312 = None
    clone_148: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_25: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_148, 0);  clone_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    add_45: "f32[32, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_310, unsqueeze_25);  view_310 = unsqueeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_313: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_45, [-1, 4, 16, 49, 49]);  add_45 = None
    unsqueeze_26: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(primals_349, 1);  primals_349 = None
    unsqueeze_27: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, 0);  unsqueeze_26 = None
    add_46: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_313, unsqueeze_27);  view_313 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    view_314: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(add_46, [-1, 16, 49, 49]);  add_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_13: "f32[32, 16, 49, 49]" = torch.ops.aten._softmax.default(view_314, -1, False);  view_314 = None
    detach_13: "f32[32, 16, 49, 49]" = torch.ops.aten.detach.default(_softmax_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    clone_149: "f32[32, 16, 49, 49]" = torch.ops.aten.clone.default(_softmax_13);  _softmax_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    expand_54: "f32[32, 16, 49, 49]" = torch.ops.aten.expand.default(clone_149, [32, 16, 49, 49]);  clone_149 = None
    view_315: "f32[512, 49, 49]" = torch.ops.aten.view.default(expand_54, [512, 49, 49]);  expand_54 = None
    expand_55: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_131, [32, 16, 49, 32]);  getitem_131 = None
    clone_150: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
    _unsafe_view_56: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_150, [512, 49, 32]);  clone_150 = None
    bmm_27: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_315, _unsafe_view_56)
    view_316: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_27, [32, 16, 49, 32]);  bmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_27: "f32[32, 49, 16, 32]" = torch.ops.aten.transpose.int(view_316, 1, 2);  view_316 = None
    clone_151: "f32[32, 49, 16, 32]" = torch.ops.aten.clone.default(transpose_27, memory_format = torch.contiguous_format);  transpose_27 = None
    _unsafe_view_57: "f32[32, 49, 512]" = torch.ops.aten._unsafe_view.default(clone_151, [32, 49, 512]);  clone_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_317: "f32[1568, 512]" = torch.ops.aten.view.default(_unsafe_view_57, [1568, 512]);  _unsafe_view_57 = None
    t_55: "f32[512, 512]" = torch.ops.aten.t.default(primals_195);  primals_195 = None
    addmm_53: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_196, view_317, t_55);  primals_196 = None
    view_318: "f32[32, 49, 512]" = torch.ops.aten.view.default(addmm_53, [32, 49, 512]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    clone_152: "f32[32, 49, 512]" = torch.ops.aten.clone.default(view_318);  view_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_319: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_152, [-1, 7, 7, 512]);  clone_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    view_320: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_319, [-1, 2, 2, 7, 7, 512]);  view_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_58: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_320, [0, 1, 3, 2, 4, 5]);  view_320 = None
    clone_153: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_58, memory_format = torch.contiguous_format);  permute_58 = None
    view_321: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(clone_153, [-1, 14, 14, 512]);  clone_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_27: "f32[8, 14, 14, 512]" = torch.ops.aten.slice.Tensor(view_321, 0, 0, 9223372036854775807);  view_321 = None
    slice_28: "f32[8, 14, 14, 512]" = torch.ops.aten.slice.Tensor(slice_27, 3, 0, 9223372036854775807);  slice_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    roll_13: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(slice_28, [3, 3], [1, 2]);  slice_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_24: "f32[8, 1, 1, 1]" = torch.ops.aten.new_empty.default(roll_13, [8, 1, 1, 1], pin_memory = False)
    bernoulli_24: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_24, 0.9434782639145851);  new_empty_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_24: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_24, 0.9434782639145851)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_38: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(roll_13, div_24);  roll_13 = div_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_47: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_303, mul_38);  mul_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_322: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_47, [8, -1, 512]);  add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_30 = torch.ops.aten.native_layer_norm.default(view_322, [512], primals_197, primals_198, 1e-05)
    getitem_132: "f32[8, 196, 512]" = native_layer_norm_30[0]
    getitem_133: "f32[8, 196, 1]" = native_layer_norm_30[1]
    getitem_134: "f32[8, 196, 1]" = native_layer_norm_30[2];  native_layer_norm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_323: "f32[1568, 512]" = torch.ops.aten.view.default(getitem_132, [1568, 512]);  getitem_132 = None
    t_56: "f32[512, 2048]" = torch.ops.aten.t.default(primals_199);  primals_199 = None
    addmm_54: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_200, view_323, t_56);  primals_200 = None
    view_324: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_54, [8, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_13: "f32[8, 196, 2048]" = torch.ops.aten.gelu.default(view_324);  view_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_154: "f32[8, 196, 2048]" = torch.ops.aten.clone.default(gelu_13);  gelu_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_325: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_154, [1568, 2048]);  clone_154 = None
    t_57: "f32[2048, 512]" = torch.ops.aten.t.default(primals_201);  primals_201 = None
    addmm_55: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_202, view_325, t_57);  primals_202 = None
    view_326: "f32[8, 196, 512]" = torch.ops.aten.view.default(addmm_55, [8, 196, 512]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_155: "f32[8, 196, 512]" = torch.ops.aten.clone.default(view_326);  view_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_25: "f32[8, 1, 1]" = torch.ops.aten.new_empty.default(clone_155, [8, 1, 1], pin_memory = False)
    bernoulli_25: "f32[8, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_25, 0.9434782639145851);  new_empty_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_25: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_25, 0.9434782639145851)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_39: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(clone_155, div_25);  clone_155 = div_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_48: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_322, mul_39);  mul_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_327: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_48, [8, 14, 14, 512]);  add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_31 = torch.ops.aten.native_layer_norm.default(view_327, [512], primals_203, primals_204, 1e-05)
    getitem_135: "f32[8, 14, 14, 512]" = native_layer_norm_31[0]
    getitem_136: "f32[8, 14, 14, 1]" = native_layer_norm_31[1]
    getitem_137: "f32[8, 14, 14, 1]" = native_layer_norm_31[2];  native_layer_norm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_14: "f32[8, 14, 14, 512]" = torch.ops.aten.constant_pad_nd.default(getitem_135, [0, 0, 0, 0, 0, 0], 0.0);  getitem_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    view_328: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(constant_pad_nd_14, [8, 2, 7, 2, 7, 512]);  constant_pad_nd_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_59: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_328, [0, 1, 3, 2, 4, 5]);  view_328 = None
    clone_156: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    view_329: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_156, [-1, 7, 7, 512]);  clone_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_330: "f32[32, 49, 512]" = torch.ops.aten.view.default(view_329, [-1, 49, 512]);  view_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_331: "f32[1568, 512]" = torch.ops.aten.view.default(view_330, [1568, 512]);  view_330 = None
    t_58: "f32[512, 1536]" = torch.ops.aten.t.default(primals_205);  primals_205 = None
    addmm_56: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_206, view_331, t_58);  primals_206 = None
    view_332: "f32[32, 49, 1536]" = torch.ops.aten.view.default(addmm_56, [32, 49, 1536]);  addmm_56 = None
    view_333: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_332, [32, 49, 3, 16, -1]);  view_332 = None
    permute_60: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.permute.default(view_333, [2, 0, 3, 1, 4]);  view_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_14 = torch.ops.aten.unbind.int(permute_60);  permute_60 = None
    getitem_138: "f32[32, 16, 49, 32]" = unbind_14[0]
    getitem_139: "f32[32, 16, 49, 32]" = unbind_14[1]
    getitem_140: "f32[32, 16, 49, 32]" = unbind_14[2];  unbind_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_40: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_138, 0.1767766952966369);  getitem_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_28: "f32[32, 16, 32, 49]" = torch.ops.aten.transpose.int(getitem_139, -2, -1);  getitem_139 = None
    expand_56: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(mul_40, [32, 16, 49, 32]);  mul_40 = None
    clone_157: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
    _unsafe_view_58: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_157, [512, 49, 32]);  clone_157 = None
    expand_57: "f32[32, 16, 32, 49]" = torch.ops.aten.expand.default(transpose_28, [32, 16, 32, 49]);  transpose_28 = None
    clone_158: "f32[32, 16, 32, 49]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
    _unsafe_view_59: "f32[512, 32, 49]" = torch.ops.aten._unsafe_view.default(clone_158, [512, 32, 49]);  clone_158 = None
    bmm_28: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_58, _unsafe_view_59)
    view_334: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_28, [32, 16, 49, 49]);  bmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_335: "i64[2401]" = torch.ops.aten.view.default(primals_351, [-1]);  primals_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_14: "f32[2401, 16]" = torch.ops.aten.index.Tensor(primals_15, [view_335]);  primals_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_336: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_14, [49, 49, -1]);  index_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_61: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_336, [2, 0, 1]);  view_336 = None
    clone_159: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_61, memory_format = torch.contiguous_format);  permute_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_28: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_159, 0);  clone_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    add_49: "f32[32, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_334, unsqueeze_28);  view_334 = unsqueeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_14: "f32[32, 16, 49, 49]" = torch.ops.aten._softmax.default(add_49, -1, False);  add_49 = None
    detach_14: "f32[32, 16, 49, 49]" = torch.ops.aten.detach.default(_softmax_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    clone_160: "f32[32, 16, 49, 49]" = torch.ops.aten.clone.default(_softmax_14);  _softmax_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    expand_58: "f32[32, 16, 49, 49]" = torch.ops.aten.expand.default(clone_160, [32, 16, 49, 49]);  clone_160 = None
    view_337: "f32[512, 49, 49]" = torch.ops.aten.view.default(expand_58, [512, 49, 49]);  expand_58 = None
    expand_59: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_140, [32, 16, 49, 32]);  getitem_140 = None
    clone_161: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
    _unsafe_view_60: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_161, [512, 49, 32]);  clone_161 = None
    bmm_29: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_337, _unsafe_view_60)
    view_338: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_29, [32, 16, 49, 32]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_29: "f32[32, 49, 16, 32]" = torch.ops.aten.transpose.int(view_338, 1, 2);  view_338 = None
    clone_162: "f32[32, 49, 16, 32]" = torch.ops.aten.clone.default(transpose_29, memory_format = torch.contiguous_format);  transpose_29 = None
    _unsafe_view_61: "f32[32, 49, 512]" = torch.ops.aten._unsafe_view.default(clone_162, [32, 49, 512]);  clone_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_339: "f32[1568, 512]" = torch.ops.aten.view.default(_unsafe_view_61, [1568, 512]);  _unsafe_view_61 = None
    t_59: "f32[512, 512]" = torch.ops.aten.t.default(primals_207);  primals_207 = None
    addmm_57: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_208, view_339, t_59);  primals_208 = None
    view_340: "f32[32, 49, 512]" = torch.ops.aten.view.default(addmm_57, [32, 49, 512]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    clone_163: "f32[32, 49, 512]" = torch.ops.aten.clone.default(view_340);  view_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_341: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_163, [-1, 7, 7, 512]);  clone_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    view_342: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_341, [-1, 2, 2, 7, 7, 512]);  view_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_62: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_342, [0, 1, 3, 2, 4, 5]);  view_342 = None
    clone_164: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    view_343: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(clone_164, [-1, 14, 14, 512]);  clone_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_29: "f32[8, 14, 14, 512]" = torch.ops.aten.slice.Tensor(view_343, 0, 0, 9223372036854775807);  view_343 = None
    slice_30: "f32[8, 14, 14, 512]" = torch.ops.aten.slice.Tensor(slice_29, 3, 0, 9223372036854775807);  slice_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_26: "f32[8, 1, 1, 1]" = torch.ops.aten.new_empty.default(slice_30, [8, 1, 1, 1], pin_memory = False)
    bernoulli_26: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_26, 0.9391304366290569);  new_empty_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_26: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_26, 0.9391304366290569)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_41: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(slice_30, div_26);  slice_30 = div_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_50: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_327, mul_41);  mul_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_344: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_50, [8, -1, 512]);  add_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_32 = torch.ops.aten.native_layer_norm.default(view_344, [512], primals_209, primals_210, 1e-05)
    getitem_141: "f32[8, 196, 512]" = native_layer_norm_32[0]
    getitem_142: "f32[8, 196, 1]" = native_layer_norm_32[1]
    getitem_143: "f32[8, 196, 1]" = native_layer_norm_32[2];  native_layer_norm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_345: "f32[1568, 512]" = torch.ops.aten.view.default(getitem_141, [1568, 512]);  getitem_141 = None
    t_60: "f32[512, 2048]" = torch.ops.aten.t.default(primals_211);  primals_211 = None
    addmm_58: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_212, view_345, t_60);  primals_212 = None
    view_346: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_58, [8, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_14: "f32[8, 196, 2048]" = torch.ops.aten.gelu.default(view_346);  view_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_165: "f32[8, 196, 2048]" = torch.ops.aten.clone.default(gelu_14);  gelu_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_347: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_165, [1568, 2048]);  clone_165 = None
    t_61: "f32[2048, 512]" = torch.ops.aten.t.default(primals_213);  primals_213 = None
    addmm_59: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_214, view_347, t_61);  primals_214 = None
    view_348: "f32[8, 196, 512]" = torch.ops.aten.view.default(addmm_59, [8, 196, 512]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_166: "f32[8, 196, 512]" = torch.ops.aten.clone.default(view_348);  view_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_27: "f32[8, 1, 1]" = torch.ops.aten.new_empty.default(clone_166, [8, 1, 1], pin_memory = False)
    bernoulli_27: "f32[8, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_27, 0.9391304366290569);  new_empty_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_27: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_27, 0.9391304366290569)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_42: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(clone_166, div_27);  clone_166 = div_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_51: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_344, mul_42);  mul_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_349: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_51, [8, 14, 14, 512]);  add_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_33 = torch.ops.aten.native_layer_norm.default(view_349, [512], primals_215, primals_216, 1e-05)
    getitem_144: "f32[8, 14, 14, 512]" = native_layer_norm_33[0]
    getitem_145: "f32[8, 14, 14, 1]" = native_layer_norm_33[1]
    getitem_146: "f32[8, 14, 14, 1]" = native_layer_norm_33[2];  native_layer_norm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    roll_14: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(getitem_144, [-3, -3], [1, 2]);  getitem_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_15: "f32[8, 14, 14, 512]" = torch.ops.aten.constant_pad_nd.default(roll_14, [0, 0, 0, 0, 0, 0], 0.0);  roll_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    view_350: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(constant_pad_nd_15, [8, 2, 7, 2, 7, 512]);  constant_pad_nd_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_63: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_350, [0, 1, 3, 2, 4, 5]);  view_350 = None
    clone_167: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None
    view_351: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_167, [-1, 7, 7, 512]);  clone_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_352: "f32[32, 49, 512]" = torch.ops.aten.view.default(view_351, [-1, 49, 512]);  view_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_353: "f32[1568, 512]" = torch.ops.aten.view.default(view_352, [1568, 512]);  view_352 = None
    t_62: "f32[512, 1536]" = torch.ops.aten.t.default(primals_217);  primals_217 = None
    addmm_60: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_218, view_353, t_62);  primals_218 = None
    view_354: "f32[32, 49, 1536]" = torch.ops.aten.view.default(addmm_60, [32, 49, 1536]);  addmm_60 = None
    view_355: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_354, [32, 49, 3, 16, -1]);  view_354 = None
    permute_64: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.permute.default(view_355, [2, 0, 3, 1, 4]);  view_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_15 = torch.ops.aten.unbind.int(permute_64);  permute_64 = None
    getitem_147: "f32[32, 16, 49, 32]" = unbind_15[0]
    getitem_148: "f32[32, 16, 49, 32]" = unbind_15[1]
    getitem_149: "f32[32, 16, 49, 32]" = unbind_15[2];  unbind_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_43: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_147, 0.1767766952966369);  getitem_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_30: "f32[32, 16, 32, 49]" = torch.ops.aten.transpose.int(getitem_148, -2, -1);  getitem_148 = None
    expand_60: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(mul_43, [32, 16, 49, 32]);  mul_43 = None
    clone_168: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
    _unsafe_view_62: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_168, [512, 49, 32]);  clone_168 = None
    expand_61: "f32[32, 16, 32, 49]" = torch.ops.aten.expand.default(transpose_30, [32, 16, 32, 49]);  transpose_30 = None
    clone_169: "f32[32, 16, 32, 49]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
    _unsafe_view_63: "f32[512, 32, 49]" = torch.ops.aten._unsafe_view.default(clone_169, [512, 32, 49]);  clone_169 = None
    bmm_30: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_62, _unsafe_view_63)
    view_356: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_30, [32, 16, 49, 49]);  bmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_357: "i64[2401]" = torch.ops.aten.view.default(primals_353, [-1]);  primals_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_15: "f32[2401, 16]" = torch.ops.aten.index.Tensor(primals_16, [view_357]);  primals_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_358: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_15, [49, 49, -1]);  index_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_65: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_358, [2, 0, 1]);  view_358 = None
    clone_170: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_65, memory_format = torch.contiguous_format);  permute_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_29: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_170, 0);  clone_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    add_52: "f32[32, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_356, unsqueeze_29);  view_356 = unsqueeze_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_359: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_52, [-1, 4, 16, 49, 49]);  add_52 = None
    unsqueeze_30: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(primals_352, 1);  primals_352 = None
    unsqueeze_31: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, 0);  unsqueeze_30 = None
    add_53: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_359, unsqueeze_31);  view_359 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    view_360: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(add_53, [-1, 16, 49, 49]);  add_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_15: "f32[32, 16, 49, 49]" = torch.ops.aten._softmax.default(view_360, -1, False);  view_360 = None
    detach_15: "f32[32, 16, 49, 49]" = torch.ops.aten.detach.default(_softmax_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    clone_171: "f32[32, 16, 49, 49]" = torch.ops.aten.clone.default(_softmax_15);  _softmax_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    expand_62: "f32[32, 16, 49, 49]" = torch.ops.aten.expand.default(clone_171, [32, 16, 49, 49]);  clone_171 = None
    view_361: "f32[512, 49, 49]" = torch.ops.aten.view.default(expand_62, [512, 49, 49]);  expand_62 = None
    expand_63: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_149, [32, 16, 49, 32]);  getitem_149 = None
    clone_172: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
    _unsafe_view_64: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_172, [512, 49, 32]);  clone_172 = None
    bmm_31: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_361, _unsafe_view_64)
    view_362: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_31, [32, 16, 49, 32]);  bmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_31: "f32[32, 49, 16, 32]" = torch.ops.aten.transpose.int(view_362, 1, 2);  view_362 = None
    clone_173: "f32[32, 49, 16, 32]" = torch.ops.aten.clone.default(transpose_31, memory_format = torch.contiguous_format);  transpose_31 = None
    _unsafe_view_65: "f32[32, 49, 512]" = torch.ops.aten._unsafe_view.default(clone_173, [32, 49, 512]);  clone_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_363: "f32[1568, 512]" = torch.ops.aten.view.default(_unsafe_view_65, [1568, 512]);  _unsafe_view_65 = None
    t_63: "f32[512, 512]" = torch.ops.aten.t.default(primals_219);  primals_219 = None
    addmm_61: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_220, view_363, t_63);  primals_220 = None
    view_364: "f32[32, 49, 512]" = torch.ops.aten.view.default(addmm_61, [32, 49, 512]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    clone_174: "f32[32, 49, 512]" = torch.ops.aten.clone.default(view_364);  view_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_365: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_174, [-1, 7, 7, 512]);  clone_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    view_366: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_365, [-1, 2, 2, 7, 7, 512]);  view_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_66: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_366, [0, 1, 3, 2, 4, 5]);  view_366 = None
    clone_175: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_66, memory_format = torch.contiguous_format);  permute_66 = None
    view_367: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(clone_175, [-1, 14, 14, 512]);  clone_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_31: "f32[8, 14, 14, 512]" = torch.ops.aten.slice.Tensor(view_367, 0, 0, 9223372036854775807);  view_367 = None
    slice_32: "f32[8, 14, 14, 512]" = torch.ops.aten.slice.Tensor(slice_31, 3, 0, 9223372036854775807);  slice_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    roll_15: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(slice_32, [3, 3], [1, 2]);  slice_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_28: "f32[8, 1, 1, 1]" = torch.ops.aten.new_empty.default(roll_15, [8, 1, 1, 1], pin_memory = False)
    bernoulli_28: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_28, 0.9347826093435287);  new_empty_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_28: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_28, 0.9347826093435287)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_44: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(roll_15, div_28);  roll_15 = div_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_54: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_349, mul_44);  mul_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_368: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_54, [8, -1, 512]);  add_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_34 = torch.ops.aten.native_layer_norm.default(view_368, [512], primals_221, primals_222, 1e-05)
    getitem_150: "f32[8, 196, 512]" = native_layer_norm_34[0]
    getitem_151: "f32[8, 196, 1]" = native_layer_norm_34[1]
    getitem_152: "f32[8, 196, 1]" = native_layer_norm_34[2];  native_layer_norm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_369: "f32[1568, 512]" = torch.ops.aten.view.default(getitem_150, [1568, 512]);  getitem_150 = None
    t_64: "f32[512, 2048]" = torch.ops.aten.t.default(primals_223);  primals_223 = None
    addmm_62: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_224, view_369, t_64);  primals_224 = None
    view_370: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_62, [8, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_15: "f32[8, 196, 2048]" = torch.ops.aten.gelu.default(view_370);  view_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_176: "f32[8, 196, 2048]" = torch.ops.aten.clone.default(gelu_15);  gelu_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_371: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_176, [1568, 2048]);  clone_176 = None
    t_65: "f32[2048, 512]" = torch.ops.aten.t.default(primals_225);  primals_225 = None
    addmm_63: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_226, view_371, t_65);  primals_226 = None
    view_372: "f32[8, 196, 512]" = torch.ops.aten.view.default(addmm_63, [8, 196, 512]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_177: "f32[8, 196, 512]" = torch.ops.aten.clone.default(view_372);  view_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_29: "f32[8, 1, 1]" = torch.ops.aten.new_empty.default(clone_177, [8, 1, 1], pin_memory = False)
    bernoulli_29: "f32[8, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_29, 0.9347826093435287);  new_empty_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_29: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_29, 0.9347826093435287)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_45: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(clone_177, div_29);  clone_177 = div_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_55: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_368, mul_45);  mul_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_373: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_55, [8, 14, 14, 512]);  add_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_35 = torch.ops.aten.native_layer_norm.default(view_373, [512], primals_227, primals_228, 1e-05)
    getitem_153: "f32[8, 14, 14, 512]" = native_layer_norm_35[0]
    getitem_154: "f32[8, 14, 14, 1]" = native_layer_norm_35[1]
    getitem_155: "f32[8, 14, 14, 1]" = native_layer_norm_35[2];  native_layer_norm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_16: "f32[8, 14, 14, 512]" = torch.ops.aten.constant_pad_nd.default(getitem_153, [0, 0, 0, 0, 0, 0], 0.0);  getitem_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    view_374: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(constant_pad_nd_16, [8, 2, 7, 2, 7, 512]);  constant_pad_nd_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_67: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_374, [0, 1, 3, 2, 4, 5]);  view_374 = None
    clone_178: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None
    view_375: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_178, [-1, 7, 7, 512]);  clone_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_376: "f32[32, 49, 512]" = torch.ops.aten.view.default(view_375, [-1, 49, 512]);  view_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_377: "f32[1568, 512]" = torch.ops.aten.view.default(view_376, [1568, 512]);  view_376 = None
    t_66: "f32[512, 1536]" = torch.ops.aten.t.default(primals_229);  primals_229 = None
    addmm_64: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_230, view_377, t_66);  primals_230 = None
    view_378: "f32[32, 49, 1536]" = torch.ops.aten.view.default(addmm_64, [32, 49, 1536]);  addmm_64 = None
    view_379: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_378, [32, 49, 3, 16, -1]);  view_378 = None
    permute_68: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.permute.default(view_379, [2, 0, 3, 1, 4]);  view_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_16 = torch.ops.aten.unbind.int(permute_68);  permute_68 = None
    getitem_156: "f32[32, 16, 49, 32]" = unbind_16[0]
    getitem_157: "f32[32, 16, 49, 32]" = unbind_16[1]
    getitem_158: "f32[32, 16, 49, 32]" = unbind_16[2];  unbind_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_46: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_156, 0.1767766952966369);  getitem_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_32: "f32[32, 16, 32, 49]" = torch.ops.aten.transpose.int(getitem_157, -2, -1);  getitem_157 = None
    expand_64: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(mul_46, [32, 16, 49, 32]);  mul_46 = None
    clone_179: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
    _unsafe_view_66: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_179, [512, 49, 32]);  clone_179 = None
    expand_65: "f32[32, 16, 32, 49]" = torch.ops.aten.expand.default(transpose_32, [32, 16, 32, 49]);  transpose_32 = None
    clone_180: "f32[32, 16, 32, 49]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
    _unsafe_view_67: "f32[512, 32, 49]" = torch.ops.aten._unsafe_view.default(clone_180, [512, 32, 49]);  clone_180 = None
    bmm_32: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_66, _unsafe_view_67)
    view_380: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_32, [32, 16, 49, 49]);  bmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_381: "i64[2401]" = torch.ops.aten.view.default(primals_354, [-1]);  primals_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_16: "f32[2401, 16]" = torch.ops.aten.index.Tensor(primals_17, [view_381]);  primals_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_382: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_16, [49, 49, -1]);  index_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_69: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_382, [2, 0, 1]);  view_382 = None
    clone_181: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_69, memory_format = torch.contiguous_format);  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_32: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_181, 0);  clone_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    add_56: "f32[32, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_380, unsqueeze_32);  view_380 = unsqueeze_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_16: "f32[32, 16, 49, 49]" = torch.ops.aten._softmax.default(add_56, -1, False);  add_56 = None
    detach_16: "f32[32, 16, 49, 49]" = torch.ops.aten.detach.default(_softmax_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    clone_182: "f32[32, 16, 49, 49]" = torch.ops.aten.clone.default(_softmax_16);  _softmax_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    expand_66: "f32[32, 16, 49, 49]" = torch.ops.aten.expand.default(clone_182, [32, 16, 49, 49]);  clone_182 = None
    view_383: "f32[512, 49, 49]" = torch.ops.aten.view.default(expand_66, [512, 49, 49]);  expand_66 = None
    expand_67: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_158, [32, 16, 49, 32]);  getitem_158 = None
    clone_183: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
    _unsafe_view_68: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_183, [512, 49, 32]);  clone_183 = None
    bmm_33: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_383, _unsafe_view_68)
    view_384: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_33, [32, 16, 49, 32]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_33: "f32[32, 49, 16, 32]" = torch.ops.aten.transpose.int(view_384, 1, 2);  view_384 = None
    clone_184: "f32[32, 49, 16, 32]" = torch.ops.aten.clone.default(transpose_33, memory_format = torch.contiguous_format);  transpose_33 = None
    _unsafe_view_69: "f32[32, 49, 512]" = torch.ops.aten._unsafe_view.default(clone_184, [32, 49, 512]);  clone_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_385: "f32[1568, 512]" = torch.ops.aten.view.default(_unsafe_view_69, [1568, 512]);  _unsafe_view_69 = None
    t_67: "f32[512, 512]" = torch.ops.aten.t.default(primals_231);  primals_231 = None
    addmm_65: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_232, view_385, t_67);  primals_232 = None
    view_386: "f32[32, 49, 512]" = torch.ops.aten.view.default(addmm_65, [32, 49, 512]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    clone_185: "f32[32, 49, 512]" = torch.ops.aten.clone.default(view_386);  view_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_387: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_185, [-1, 7, 7, 512]);  clone_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    view_388: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_387, [-1, 2, 2, 7, 7, 512]);  view_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_70: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_388, [0, 1, 3, 2, 4, 5]);  view_388 = None
    clone_186: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
    view_389: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(clone_186, [-1, 14, 14, 512]);  clone_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_33: "f32[8, 14, 14, 512]" = torch.ops.aten.slice.Tensor(view_389, 0, 0, 9223372036854775807);  view_389 = None
    slice_34: "f32[8, 14, 14, 512]" = torch.ops.aten.slice.Tensor(slice_33, 3, 0, 9223372036854775807);  slice_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_30: "f32[8, 1, 1, 1]" = torch.ops.aten.new_empty.default(slice_34, [8, 1, 1, 1], pin_memory = False)
    bernoulli_30: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_30, 0.9304347857832909);  new_empty_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_30: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_30, 0.9304347857832909)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_47: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(slice_34, div_30);  slice_34 = div_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_57: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_373, mul_47);  mul_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_390: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_57, [8, -1, 512]);  add_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_36 = torch.ops.aten.native_layer_norm.default(view_390, [512], primals_233, primals_234, 1e-05)
    getitem_159: "f32[8, 196, 512]" = native_layer_norm_36[0]
    getitem_160: "f32[8, 196, 1]" = native_layer_norm_36[1]
    getitem_161: "f32[8, 196, 1]" = native_layer_norm_36[2];  native_layer_norm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_391: "f32[1568, 512]" = torch.ops.aten.view.default(getitem_159, [1568, 512]);  getitem_159 = None
    t_68: "f32[512, 2048]" = torch.ops.aten.t.default(primals_235);  primals_235 = None
    addmm_66: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_236, view_391, t_68);  primals_236 = None
    view_392: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_66, [8, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_16: "f32[8, 196, 2048]" = torch.ops.aten.gelu.default(view_392);  view_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_187: "f32[8, 196, 2048]" = torch.ops.aten.clone.default(gelu_16);  gelu_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_393: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_187, [1568, 2048]);  clone_187 = None
    t_69: "f32[2048, 512]" = torch.ops.aten.t.default(primals_237);  primals_237 = None
    addmm_67: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_238, view_393, t_69);  primals_238 = None
    view_394: "f32[8, 196, 512]" = torch.ops.aten.view.default(addmm_67, [8, 196, 512]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_188: "f32[8, 196, 512]" = torch.ops.aten.clone.default(view_394);  view_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_31: "f32[8, 1, 1]" = torch.ops.aten.new_empty.default(clone_188, [8, 1, 1], pin_memory = False)
    bernoulli_31: "f32[8, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_31, 0.9304347857832909);  new_empty_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_31: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_31, 0.9304347857832909)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_48: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(clone_188, div_31);  clone_188 = div_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_58: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_390, mul_48);  mul_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_395: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_58, [8, 14, 14, 512]);  add_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_37 = torch.ops.aten.native_layer_norm.default(view_395, [512], primals_239, primals_240, 1e-05)
    getitem_162: "f32[8, 14, 14, 512]" = native_layer_norm_37[0]
    getitem_163: "f32[8, 14, 14, 1]" = native_layer_norm_37[1]
    getitem_164: "f32[8, 14, 14, 1]" = native_layer_norm_37[2];  native_layer_norm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    roll_16: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(getitem_162, [-3, -3], [1, 2]);  getitem_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_17: "f32[8, 14, 14, 512]" = torch.ops.aten.constant_pad_nd.default(roll_16, [0, 0, 0, 0, 0, 0], 0.0);  roll_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    view_396: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(constant_pad_nd_17, [8, 2, 7, 2, 7, 512]);  constant_pad_nd_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_71: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_396, [0, 1, 3, 2, 4, 5]);  view_396 = None
    clone_189: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
    view_397: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_189, [-1, 7, 7, 512]);  clone_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_398: "f32[32, 49, 512]" = torch.ops.aten.view.default(view_397, [-1, 49, 512]);  view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_399: "f32[1568, 512]" = torch.ops.aten.view.default(view_398, [1568, 512]);  view_398 = None
    t_70: "f32[512, 1536]" = torch.ops.aten.t.default(primals_241);  primals_241 = None
    addmm_68: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_242, view_399, t_70);  primals_242 = None
    view_400: "f32[32, 49, 1536]" = torch.ops.aten.view.default(addmm_68, [32, 49, 1536]);  addmm_68 = None
    view_401: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_400, [32, 49, 3, 16, -1]);  view_400 = None
    permute_72: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.permute.default(view_401, [2, 0, 3, 1, 4]);  view_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_17 = torch.ops.aten.unbind.int(permute_72);  permute_72 = None
    getitem_165: "f32[32, 16, 49, 32]" = unbind_17[0]
    getitem_166: "f32[32, 16, 49, 32]" = unbind_17[1]
    getitem_167: "f32[32, 16, 49, 32]" = unbind_17[2];  unbind_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_49: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_165, 0.1767766952966369);  getitem_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_34: "f32[32, 16, 32, 49]" = torch.ops.aten.transpose.int(getitem_166, -2, -1);  getitem_166 = None
    expand_68: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(mul_49, [32, 16, 49, 32]);  mul_49 = None
    clone_190: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
    _unsafe_view_70: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_190, [512, 49, 32]);  clone_190 = None
    expand_69: "f32[32, 16, 32, 49]" = torch.ops.aten.expand.default(transpose_34, [32, 16, 32, 49]);  transpose_34 = None
    clone_191: "f32[32, 16, 32, 49]" = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
    _unsafe_view_71: "f32[512, 32, 49]" = torch.ops.aten._unsafe_view.default(clone_191, [512, 32, 49]);  clone_191 = None
    bmm_34: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_70, _unsafe_view_71)
    view_402: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_34, [32, 16, 49, 49]);  bmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_403: "i64[2401]" = torch.ops.aten.view.default(primals_356, [-1]);  primals_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_17: "f32[2401, 16]" = torch.ops.aten.index.Tensor(primals_18, [view_403]);  primals_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_404: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_17, [49, 49, -1]);  index_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_73: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_404, [2, 0, 1]);  view_404 = None
    clone_192: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_33: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_192, 0);  clone_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    add_59: "f32[32, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_402, unsqueeze_33);  view_402 = unsqueeze_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_405: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_59, [-1, 4, 16, 49, 49]);  add_59 = None
    unsqueeze_34: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(primals_355, 1);  primals_355 = None
    unsqueeze_35: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, 0);  unsqueeze_34 = None
    add_60: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_405, unsqueeze_35);  view_405 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    view_406: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(add_60, [-1, 16, 49, 49]);  add_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_17: "f32[32, 16, 49, 49]" = torch.ops.aten._softmax.default(view_406, -1, False);  view_406 = None
    detach_17: "f32[32, 16, 49, 49]" = torch.ops.aten.detach.default(_softmax_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    clone_193: "f32[32, 16, 49, 49]" = torch.ops.aten.clone.default(_softmax_17);  _softmax_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    expand_70: "f32[32, 16, 49, 49]" = torch.ops.aten.expand.default(clone_193, [32, 16, 49, 49]);  clone_193 = None
    view_407: "f32[512, 49, 49]" = torch.ops.aten.view.default(expand_70, [512, 49, 49]);  expand_70 = None
    expand_71: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_167, [32, 16, 49, 32]);  getitem_167 = None
    clone_194: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
    _unsafe_view_72: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_194, [512, 49, 32]);  clone_194 = None
    bmm_35: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_407, _unsafe_view_72)
    view_408: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_35, [32, 16, 49, 32]);  bmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_35: "f32[32, 49, 16, 32]" = torch.ops.aten.transpose.int(view_408, 1, 2);  view_408 = None
    clone_195: "f32[32, 49, 16, 32]" = torch.ops.aten.clone.default(transpose_35, memory_format = torch.contiguous_format);  transpose_35 = None
    _unsafe_view_73: "f32[32, 49, 512]" = torch.ops.aten._unsafe_view.default(clone_195, [32, 49, 512]);  clone_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_409: "f32[1568, 512]" = torch.ops.aten.view.default(_unsafe_view_73, [1568, 512]);  _unsafe_view_73 = None
    t_71: "f32[512, 512]" = torch.ops.aten.t.default(primals_243);  primals_243 = None
    addmm_69: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_244, view_409, t_71);  primals_244 = None
    view_410: "f32[32, 49, 512]" = torch.ops.aten.view.default(addmm_69, [32, 49, 512]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    clone_196: "f32[32, 49, 512]" = torch.ops.aten.clone.default(view_410);  view_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_411: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_196, [-1, 7, 7, 512]);  clone_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    view_412: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_411, [-1, 2, 2, 7, 7, 512]);  view_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_74: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_412, [0, 1, 3, 2, 4, 5]);  view_412 = None
    clone_197: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format);  permute_74 = None
    view_413: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(clone_197, [-1, 14, 14, 512]);  clone_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_35: "f32[8, 14, 14, 512]" = torch.ops.aten.slice.Tensor(view_413, 0, 0, 9223372036854775807);  view_413 = None
    slice_36: "f32[8, 14, 14, 512]" = torch.ops.aten.slice.Tensor(slice_35, 3, 0, 9223372036854775807);  slice_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    roll_17: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(slice_36, [3, 3], [1, 2]);  slice_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_32: "f32[8, 1, 1, 1]" = torch.ops.aten.new_empty.default(roll_17, [8, 1, 1, 1], pin_memory = False)
    bernoulli_32: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_32, 0.9260869547724724);  new_empty_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_32: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_32, 0.9260869547724724)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_50: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(roll_17, div_32);  roll_17 = div_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_61: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_395, mul_50);  mul_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_414: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_61, [8, -1, 512]);  add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_38 = torch.ops.aten.native_layer_norm.default(view_414, [512], primals_245, primals_246, 1e-05)
    getitem_168: "f32[8, 196, 512]" = native_layer_norm_38[0]
    getitem_169: "f32[8, 196, 1]" = native_layer_norm_38[1]
    getitem_170: "f32[8, 196, 1]" = native_layer_norm_38[2];  native_layer_norm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_415: "f32[1568, 512]" = torch.ops.aten.view.default(getitem_168, [1568, 512]);  getitem_168 = None
    t_72: "f32[512, 2048]" = torch.ops.aten.t.default(primals_247);  primals_247 = None
    addmm_70: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_248, view_415, t_72);  primals_248 = None
    view_416: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_70, [8, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_17: "f32[8, 196, 2048]" = torch.ops.aten.gelu.default(view_416);  view_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_198: "f32[8, 196, 2048]" = torch.ops.aten.clone.default(gelu_17);  gelu_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_417: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_198, [1568, 2048]);  clone_198 = None
    t_73: "f32[2048, 512]" = torch.ops.aten.t.default(primals_249);  primals_249 = None
    addmm_71: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_250, view_417, t_73);  primals_250 = None
    view_418: "f32[8, 196, 512]" = torch.ops.aten.view.default(addmm_71, [8, 196, 512]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_199: "f32[8, 196, 512]" = torch.ops.aten.clone.default(view_418);  view_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_33: "f32[8, 1, 1]" = torch.ops.aten.new_empty.default(clone_199, [8, 1, 1], pin_memory = False)
    bernoulli_33: "f32[8, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_33, 0.9260869547724724);  new_empty_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_33: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_33, 0.9260869547724724)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_51: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(clone_199, div_33);  clone_199 = div_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_62: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_414, mul_51);  mul_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_419: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_62, [8, 14, 14, 512]);  add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_39 = torch.ops.aten.native_layer_norm.default(view_419, [512], primals_251, primals_252, 1e-05)
    getitem_171: "f32[8, 14, 14, 512]" = native_layer_norm_39[0]
    getitem_172: "f32[8, 14, 14, 1]" = native_layer_norm_39[1]
    getitem_173: "f32[8, 14, 14, 1]" = native_layer_norm_39[2];  native_layer_norm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_18: "f32[8, 14, 14, 512]" = torch.ops.aten.constant_pad_nd.default(getitem_171, [0, 0, 0, 0, 0, 0], 0.0);  getitem_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    view_420: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(constant_pad_nd_18, [8, 2, 7, 2, 7, 512]);  constant_pad_nd_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_75: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_420, [0, 1, 3, 2, 4, 5]);  view_420 = None
    clone_200: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_75, memory_format = torch.contiguous_format);  permute_75 = None
    view_421: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_200, [-1, 7, 7, 512]);  clone_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_422: "f32[32, 49, 512]" = torch.ops.aten.view.default(view_421, [-1, 49, 512]);  view_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_423: "f32[1568, 512]" = torch.ops.aten.view.default(view_422, [1568, 512]);  view_422 = None
    t_74: "f32[512, 1536]" = torch.ops.aten.t.default(primals_253);  primals_253 = None
    addmm_72: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_254, view_423, t_74);  primals_254 = None
    view_424: "f32[32, 49, 1536]" = torch.ops.aten.view.default(addmm_72, [32, 49, 1536]);  addmm_72 = None
    view_425: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_424, [32, 49, 3, 16, -1]);  view_424 = None
    permute_76: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.permute.default(view_425, [2, 0, 3, 1, 4]);  view_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_18 = torch.ops.aten.unbind.int(permute_76);  permute_76 = None
    getitem_174: "f32[32, 16, 49, 32]" = unbind_18[0]
    getitem_175: "f32[32, 16, 49, 32]" = unbind_18[1]
    getitem_176: "f32[32, 16, 49, 32]" = unbind_18[2];  unbind_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_52: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_174, 0.1767766952966369);  getitem_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_36: "f32[32, 16, 32, 49]" = torch.ops.aten.transpose.int(getitem_175, -2, -1);  getitem_175 = None
    expand_72: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(mul_52, [32, 16, 49, 32]);  mul_52 = None
    clone_201: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
    _unsafe_view_74: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_201, [512, 49, 32]);  clone_201 = None
    expand_73: "f32[32, 16, 32, 49]" = torch.ops.aten.expand.default(transpose_36, [32, 16, 32, 49]);  transpose_36 = None
    clone_202: "f32[32, 16, 32, 49]" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
    _unsafe_view_75: "f32[512, 32, 49]" = torch.ops.aten._unsafe_view.default(clone_202, [512, 32, 49]);  clone_202 = None
    bmm_36: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_74, _unsafe_view_75)
    view_426: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_36, [32, 16, 49, 49]);  bmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_427: "i64[2401]" = torch.ops.aten.view.default(primals_357, [-1]);  primals_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_18: "f32[2401, 16]" = torch.ops.aten.index.Tensor(primals_19, [view_427]);  primals_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_428: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_18, [49, 49, -1]);  index_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_77: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_428, [2, 0, 1]);  view_428 = None
    clone_203: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_36: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_203, 0);  clone_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    add_63: "f32[32, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_426, unsqueeze_36);  view_426 = unsqueeze_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_18: "f32[32, 16, 49, 49]" = torch.ops.aten._softmax.default(add_63, -1, False);  add_63 = None
    detach_18: "f32[32, 16, 49, 49]" = torch.ops.aten.detach.default(_softmax_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    clone_204: "f32[32, 16, 49, 49]" = torch.ops.aten.clone.default(_softmax_18);  _softmax_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    expand_74: "f32[32, 16, 49, 49]" = torch.ops.aten.expand.default(clone_204, [32, 16, 49, 49]);  clone_204 = None
    view_429: "f32[512, 49, 49]" = torch.ops.aten.view.default(expand_74, [512, 49, 49]);  expand_74 = None
    expand_75: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_176, [32, 16, 49, 32]);  getitem_176 = None
    clone_205: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
    _unsafe_view_76: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_205, [512, 49, 32]);  clone_205 = None
    bmm_37: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_429, _unsafe_view_76)
    view_430: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_37, [32, 16, 49, 32]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_37: "f32[32, 49, 16, 32]" = torch.ops.aten.transpose.int(view_430, 1, 2);  view_430 = None
    clone_206: "f32[32, 49, 16, 32]" = torch.ops.aten.clone.default(transpose_37, memory_format = torch.contiguous_format);  transpose_37 = None
    _unsafe_view_77: "f32[32, 49, 512]" = torch.ops.aten._unsafe_view.default(clone_206, [32, 49, 512]);  clone_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_431: "f32[1568, 512]" = torch.ops.aten.view.default(_unsafe_view_77, [1568, 512]);  _unsafe_view_77 = None
    t_75: "f32[512, 512]" = torch.ops.aten.t.default(primals_255);  primals_255 = None
    addmm_73: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_256, view_431, t_75);  primals_256 = None
    view_432: "f32[32, 49, 512]" = torch.ops.aten.view.default(addmm_73, [32, 49, 512]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    clone_207: "f32[32, 49, 512]" = torch.ops.aten.clone.default(view_432);  view_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_433: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_207, [-1, 7, 7, 512]);  clone_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    view_434: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_433, [-1, 2, 2, 7, 7, 512]);  view_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_78: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_434, [0, 1, 3, 2, 4, 5]);  view_434 = None
    clone_208: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_78, memory_format = torch.contiguous_format);  permute_78 = None
    view_435: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(clone_208, [-1, 14, 14, 512]);  clone_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_37: "f32[8, 14, 14, 512]" = torch.ops.aten.slice.Tensor(view_435, 0, 0, 9223372036854775807);  view_435 = None
    slice_38: "f32[8, 14, 14, 512]" = torch.ops.aten.slice.Tensor(slice_37, 3, 0, 9223372036854775807);  slice_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_34: "f32[8, 1, 1, 1]" = torch.ops.aten.new_empty.default(slice_38, [8, 1, 1, 1], pin_memory = False)
    bernoulli_34: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_34, 0.9217391312122345);  new_empty_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_34: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_34, 0.9217391312122345)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_53: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(slice_38, div_34);  slice_38 = div_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_64: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_419, mul_53);  mul_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_436: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_64, [8, -1, 512]);  add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_40 = torch.ops.aten.native_layer_norm.default(view_436, [512], primals_257, primals_258, 1e-05)
    getitem_177: "f32[8, 196, 512]" = native_layer_norm_40[0]
    getitem_178: "f32[8, 196, 1]" = native_layer_norm_40[1]
    getitem_179: "f32[8, 196, 1]" = native_layer_norm_40[2];  native_layer_norm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_437: "f32[1568, 512]" = torch.ops.aten.view.default(getitem_177, [1568, 512]);  getitem_177 = None
    t_76: "f32[512, 2048]" = torch.ops.aten.t.default(primals_259);  primals_259 = None
    addmm_74: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_260, view_437, t_76);  primals_260 = None
    view_438: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_74, [8, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_18: "f32[8, 196, 2048]" = torch.ops.aten.gelu.default(view_438);  view_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_209: "f32[8, 196, 2048]" = torch.ops.aten.clone.default(gelu_18);  gelu_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_439: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_209, [1568, 2048]);  clone_209 = None
    t_77: "f32[2048, 512]" = torch.ops.aten.t.default(primals_261);  primals_261 = None
    addmm_75: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_262, view_439, t_77);  primals_262 = None
    view_440: "f32[8, 196, 512]" = torch.ops.aten.view.default(addmm_75, [8, 196, 512]);  addmm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_210: "f32[8, 196, 512]" = torch.ops.aten.clone.default(view_440);  view_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_35: "f32[8, 1, 1]" = torch.ops.aten.new_empty.default(clone_210, [8, 1, 1], pin_memory = False)
    bernoulli_35: "f32[8, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_35, 0.9217391312122345);  new_empty_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_35: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_35, 0.9217391312122345)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_54: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(clone_210, div_35);  clone_210 = div_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_65: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_436, mul_54);  mul_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_441: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_65, [8, 14, 14, 512]);  add_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_41 = torch.ops.aten.native_layer_norm.default(view_441, [512], primals_263, primals_264, 1e-05)
    getitem_180: "f32[8, 14, 14, 512]" = native_layer_norm_41[0]
    getitem_181: "f32[8, 14, 14, 1]" = native_layer_norm_41[1]
    getitem_182: "f32[8, 14, 14, 1]" = native_layer_norm_41[2];  native_layer_norm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    roll_18: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(getitem_180, [-3, -3], [1, 2]);  getitem_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_19: "f32[8, 14, 14, 512]" = torch.ops.aten.constant_pad_nd.default(roll_18, [0, 0, 0, 0, 0, 0], 0.0);  roll_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    view_442: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(constant_pad_nd_19, [8, 2, 7, 2, 7, 512]);  constant_pad_nd_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_79: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_442, [0, 1, 3, 2, 4, 5]);  view_442 = None
    clone_211: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
    view_443: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_211, [-1, 7, 7, 512]);  clone_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_444: "f32[32, 49, 512]" = torch.ops.aten.view.default(view_443, [-1, 49, 512]);  view_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_445: "f32[1568, 512]" = torch.ops.aten.view.default(view_444, [1568, 512]);  view_444 = None
    t_78: "f32[512, 1536]" = torch.ops.aten.t.default(primals_265);  primals_265 = None
    addmm_76: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_266, view_445, t_78);  primals_266 = None
    view_446: "f32[32, 49, 1536]" = torch.ops.aten.view.default(addmm_76, [32, 49, 1536]);  addmm_76 = None
    view_447: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_446, [32, 49, 3, 16, -1]);  view_446 = None
    permute_80: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.permute.default(view_447, [2, 0, 3, 1, 4]);  view_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_19 = torch.ops.aten.unbind.int(permute_80);  permute_80 = None
    getitem_183: "f32[32, 16, 49, 32]" = unbind_19[0]
    getitem_184: "f32[32, 16, 49, 32]" = unbind_19[1]
    getitem_185: "f32[32, 16, 49, 32]" = unbind_19[2];  unbind_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_55: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_183, 0.1767766952966369);  getitem_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_38: "f32[32, 16, 32, 49]" = torch.ops.aten.transpose.int(getitem_184, -2, -1);  getitem_184 = None
    expand_76: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(mul_55, [32, 16, 49, 32]);  mul_55 = None
    clone_212: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
    _unsafe_view_78: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_212, [512, 49, 32]);  clone_212 = None
    expand_77: "f32[32, 16, 32, 49]" = torch.ops.aten.expand.default(transpose_38, [32, 16, 32, 49]);  transpose_38 = None
    clone_213: "f32[32, 16, 32, 49]" = torch.ops.aten.clone.default(expand_77, memory_format = torch.contiguous_format);  expand_77 = None
    _unsafe_view_79: "f32[512, 32, 49]" = torch.ops.aten._unsafe_view.default(clone_213, [512, 32, 49]);  clone_213 = None
    bmm_38: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_78, _unsafe_view_79)
    view_448: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_38, [32, 16, 49, 49]);  bmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_449: "i64[2401]" = torch.ops.aten.view.default(primals_359, [-1]);  primals_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_19: "f32[2401, 16]" = torch.ops.aten.index.Tensor(primals_20, [view_449]);  primals_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_450: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_19, [49, 49, -1]);  index_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_81: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_450, [2, 0, 1]);  view_450 = None
    clone_214: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_37: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_214, 0);  clone_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    add_66: "f32[32, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_448, unsqueeze_37);  view_448 = unsqueeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_451: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_66, [-1, 4, 16, 49, 49]);  add_66 = None
    unsqueeze_38: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(primals_358, 1);  primals_358 = None
    unsqueeze_39: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, 0);  unsqueeze_38 = None
    add_67: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_451, unsqueeze_39);  view_451 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    view_452: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(add_67, [-1, 16, 49, 49]);  add_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_19: "f32[32, 16, 49, 49]" = torch.ops.aten._softmax.default(view_452, -1, False);  view_452 = None
    detach_19: "f32[32, 16, 49, 49]" = torch.ops.aten.detach.default(_softmax_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    clone_215: "f32[32, 16, 49, 49]" = torch.ops.aten.clone.default(_softmax_19);  _softmax_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    expand_78: "f32[32, 16, 49, 49]" = torch.ops.aten.expand.default(clone_215, [32, 16, 49, 49]);  clone_215 = None
    view_453: "f32[512, 49, 49]" = torch.ops.aten.view.default(expand_78, [512, 49, 49]);  expand_78 = None
    expand_79: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_185, [32, 16, 49, 32]);  getitem_185 = None
    clone_216: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_79, memory_format = torch.contiguous_format);  expand_79 = None
    _unsafe_view_80: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_216, [512, 49, 32]);  clone_216 = None
    bmm_39: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_453, _unsafe_view_80)
    view_454: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_39, [32, 16, 49, 32]);  bmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_39: "f32[32, 49, 16, 32]" = torch.ops.aten.transpose.int(view_454, 1, 2);  view_454 = None
    clone_217: "f32[32, 49, 16, 32]" = torch.ops.aten.clone.default(transpose_39, memory_format = torch.contiguous_format);  transpose_39 = None
    _unsafe_view_81: "f32[32, 49, 512]" = torch.ops.aten._unsafe_view.default(clone_217, [32, 49, 512]);  clone_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_455: "f32[1568, 512]" = torch.ops.aten.view.default(_unsafe_view_81, [1568, 512]);  _unsafe_view_81 = None
    t_79: "f32[512, 512]" = torch.ops.aten.t.default(primals_267);  primals_267 = None
    addmm_77: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_268, view_455, t_79);  primals_268 = None
    view_456: "f32[32, 49, 512]" = torch.ops.aten.view.default(addmm_77, [32, 49, 512]);  addmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    clone_218: "f32[32, 49, 512]" = torch.ops.aten.clone.default(view_456);  view_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_457: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_218, [-1, 7, 7, 512]);  clone_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    view_458: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_457, [-1, 2, 2, 7, 7, 512]);  view_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_82: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_458, [0, 1, 3, 2, 4, 5]);  view_458 = None
    clone_219: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
    view_459: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(clone_219, [-1, 14, 14, 512]);  clone_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_39: "f32[8, 14, 14, 512]" = torch.ops.aten.slice.Tensor(view_459, 0, 0, 9223372036854775807);  view_459 = None
    slice_40: "f32[8, 14, 14, 512]" = torch.ops.aten.slice.Tensor(slice_39, 3, 0, 9223372036854775807);  slice_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    roll_19: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(slice_40, [3, 3], [1, 2]);  slice_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_36: "f32[8, 1, 1, 1]" = torch.ops.aten.new_empty.default(roll_19, [8, 1, 1, 1], pin_memory = False)
    bernoulli_36: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_36, 0.917391300201416);  new_empty_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_36: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_36, 0.917391300201416)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_56: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(roll_19, div_36);  roll_19 = div_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_68: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_441, mul_56);  mul_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_460: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_68, [8, -1, 512]);  add_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_42 = torch.ops.aten.native_layer_norm.default(view_460, [512], primals_269, primals_270, 1e-05)
    getitem_186: "f32[8, 196, 512]" = native_layer_norm_42[0]
    getitem_187: "f32[8, 196, 1]" = native_layer_norm_42[1]
    getitem_188: "f32[8, 196, 1]" = native_layer_norm_42[2];  native_layer_norm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_461: "f32[1568, 512]" = torch.ops.aten.view.default(getitem_186, [1568, 512]);  getitem_186 = None
    t_80: "f32[512, 2048]" = torch.ops.aten.t.default(primals_271);  primals_271 = None
    addmm_78: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_272, view_461, t_80);  primals_272 = None
    view_462: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_78, [8, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_19: "f32[8, 196, 2048]" = torch.ops.aten.gelu.default(view_462);  view_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_220: "f32[8, 196, 2048]" = torch.ops.aten.clone.default(gelu_19);  gelu_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_463: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_220, [1568, 2048]);  clone_220 = None
    t_81: "f32[2048, 512]" = torch.ops.aten.t.default(primals_273);  primals_273 = None
    addmm_79: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_274, view_463, t_81);  primals_274 = None
    view_464: "f32[8, 196, 512]" = torch.ops.aten.view.default(addmm_79, [8, 196, 512]);  addmm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_221: "f32[8, 196, 512]" = torch.ops.aten.clone.default(view_464);  view_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_37: "f32[8, 1, 1]" = torch.ops.aten.new_empty.default(clone_221, [8, 1, 1], pin_memory = False)
    bernoulli_37: "f32[8, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_37, 0.917391300201416);  new_empty_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_37: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_37, 0.917391300201416)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_57: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(clone_221, div_37);  clone_221 = div_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_69: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_460, mul_57);  mul_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_465: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_69, [8, 14, 14, 512]);  add_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_43 = torch.ops.aten.native_layer_norm.default(view_465, [512], primals_275, primals_276, 1e-05)
    getitem_189: "f32[8, 14, 14, 512]" = native_layer_norm_43[0]
    getitem_190: "f32[8, 14, 14, 1]" = native_layer_norm_43[1]
    getitem_191: "f32[8, 14, 14, 1]" = native_layer_norm_43[2];  native_layer_norm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_20: "f32[8, 14, 14, 512]" = torch.ops.aten.constant_pad_nd.default(getitem_189, [0, 0, 0, 0, 0, 0], 0.0);  getitem_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    view_466: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(constant_pad_nd_20, [8, 2, 7, 2, 7, 512]);  constant_pad_nd_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_83: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_466, [0, 1, 3, 2, 4, 5]);  view_466 = None
    clone_222: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_83, memory_format = torch.contiguous_format);  permute_83 = None
    view_467: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_222, [-1, 7, 7, 512]);  clone_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_468: "f32[32, 49, 512]" = torch.ops.aten.view.default(view_467, [-1, 49, 512]);  view_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_469: "f32[1568, 512]" = torch.ops.aten.view.default(view_468, [1568, 512]);  view_468 = None
    t_82: "f32[512, 1536]" = torch.ops.aten.t.default(primals_277);  primals_277 = None
    addmm_80: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_278, view_469, t_82);  primals_278 = None
    view_470: "f32[32, 49, 1536]" = torch.ops.aten.view.default(addmm_80, [32, 49, 1536]);  addmm_80 = None
    view_471: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_470, [32, 49, 3, 16, -1]);  view_470 = None
    permute_84: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.permute.default(view_471, [2, 0, 3, 1, 4]);  view_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_20 = torch.ops.aten.unbind.int(permute_84);  permute_84 = None
    getitem_192: "f32[32, 16, 49, 32]" = unbind_20[0]
    getitem_193: "f32[32, 16, 49, 32]" = unbind_20[1]
    getitem_194: "f32[32, 16, 49, 32]" = unbind_20[2];  unbind_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_58: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_192, 0.1767766952966369);  getitem_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_40: "f32[32, 16, 32, 49]" = torch.ops.aten.transpose.int(getitem_193, -2, -1);  getitem_193 = None
    expand_80: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(mul_58, [32, 16, 49, 32]);  mul_58 = None
    clone_223: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
    _unsafe_view_82: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_223, [512, 49, 32]);  clone_223 = None
    expand_81: "f32[32, 16, 32, 49]" = torch.ops.aten.expand.default(transpose_40, [32, 16, 32, 49]);  transpose_40 = None
    clone_224: "f32[32, 16, 32, 49]" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
    _unsafe_view_83: "f32[512, 32, 49]" = torch.ops.aten._unsafe_view.default(clone_224, [512, 32, 49]);  clone_224 = None
    bmm_40: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_82, _unsafe_view_83)
    view_472: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_40, [32, 16, 49, 49]);  bmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_473: "i64[2401]" = torch.ops.aten.view.default(primals_360, [-1]);  primals_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_20: "f32[2401, 16]" = torch.ops.aten.index.Tensor(primals_21, [view_473]);  primals_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_474: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_20, [49, 49, -1]);  index_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_85: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_474, [2, 0, 1]);  view_474 = None
    clone_225: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_40: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_225, 0);  clone_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    add_70: "f32[32, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_472, unsqueeze_40);  view_472 = unsqueeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_20: "f32[32, 16, 49, 49]" = torch.ops.aten._softmax.default(add_70, -1, False);  add_70 = None
    detach_20: "f32[32, 16, 49, 49]" = torch.ops.aten.detach.default(_softmax_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    clone_226: "f32[32, 16, 49, 49]" = torch.ops.aten.clone.default(_softmax_20);  _softmax_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    expand_82: "f32[32, 16, 49, 49]" = torch.ops.aten.expand.default(clone_226, [32, 16, 49, 49]);  clone_226 = None
    view_475: "f32[512, 49, 49]" = torch.ops.aten.view.default(expand_82, [512, 49, 49]);  expand_82 = None
    expand_83: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_194, [32, 16, 49, 32]);  getitem_194 = None
    clone_227: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_83, memory_format = torch.contiguous_format);  expand_83 = None
    _unsafe_view_84: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_227, [512, 49, 32]);  clone_227 = None
    bmm_41: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_475, _unsafe_view_84)
    view_476: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_41, [32, 16, 49, 32]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_41: "f32[32, 49, 16, 32]" = torch.ops.aten.transpose.int(view_476, 1, 2);  view_476 = None
    clone_228: "f32[32, 49, 16, 32]" = torch.ops.aten.clone.default(transpose_41, memory_format = torch.contiguous_format);  transpose_41 = None
    _unsafe_view_85: "f32[32, 49, 512]" = torch.ops.aten._unsafe_view.default(clone_228, [32, 49, 512]);  clone_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_477: "f32[1568, 512]" = torch.ops.aten.view.default(_unsafe_view_85, [1568, 512]);  _unsafe_view_85 = None
    t_83: "f32[512, 512]" = torch.ops.aten.t.default(primals_279);  primals_279 = None
    addmm_81: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_280, view_477, t_83);  primals_280 = None
    view_478: "f32[32, 49, 512]" = torch.ops.aten.view.default(addmm_81, [32, 49, 512]);  addmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    clone_229: "f32[32, 49, 512]" = torch.ops.aten.clone.default(view_478);  view_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_479: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_229, [-1, 7, 7, 512]);  clone_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    view_480: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_479, [-1, 2, 2, 7, 7, 512]);  view_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_86: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_480, [0, 1, 3, 2, 4, 5]);  view_480 = None
    clone_230: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_86, memory_format = torch.contiguous_format);  permute_86 = None
    view_481: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(clone_230, [-1, 14, 14, 512]);  clone_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_41: "f32[8, 14, 14, 512]" = torch.ops.aten.slice.Tensor(view_481, 0, 0, 9223372036854775807);  view_481 = None
    slice_42: "f32[8, 14, 14, 512]" = torch.ops.aten.slice.Tensor(slice_41, 3, 0, 9223372036854775807);  slice_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_38: "f32[8, 1, 1, 1]" = torch.ops.aten.new_empty.default(slice_42, [8, 1, 1, 1], pin_memory = False)
    bernoulli_38: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_38, 0.9130434766411781);  new_empty_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_38: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_38, 0.9130434766411781)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_59: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(slice_42, div_38);  slice_42 = div_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_71: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_465, mul_59);  mul_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_482: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_71, [8, -1, 512]);  add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_44 = torch.ops.aten.native_layer_norm.default(view_482, [512], primals_281, primals_282, 1e-05)
    getitem_195: "f32[8, 196, 512]" = native_layer_norm_44[0]
    getitem_196: "f32[8, 196, 1]" = native_layer_norm_44[1]
    getitem_197: "f32[8, 196, 1]" = native_layer_norm_44[2];  native_layer_norm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_483: "f32[1568, 512]" = torch.ops.aten.view.default(getitem_195, [1568, 512]);  getitem_195 = None
    t_84: "f32[512, 2048]" = torch.ops.aten.t.default(primals_283);  primals_283 = None
    addmm_82: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_284, view_483, t_84);  primals_284 = None
    view_484: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_82, [8, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_20: "f32[8, 196, 2048]" = torch.ops.aten.gelu.default(view_484);  view_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_231: "f32[8, 196, 2048]" = torch.ops.aten.clone.default(gelu_20);  gelu_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_485: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_231, [1568, 2048]);  clone_231 = None
    t_85: "f32[2048, 512]" = torch.ops.aten.t.default(primals_285);  primals_285 = None
    addmm_83: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_286, view_485, t_85);  primals_286 = None
    view_486: "f32[8, 196, 512]" = torch.ops.aten.view.default(addmm_83, [8, 196, 512]);  addmm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_232: "f32[8, 196, 512]" = torch.ops.aten.clone.default(view_486);  view_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_39: "f32[8, 1, 1]" = torch.ops.aten.new_empty.default(clone_232, [8, 1, 1], pin_memory = False)
    bernoulli_39: "f32[8, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_39, 0.9130434766411781);  new_empty_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_39: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_39, 0.9130434766411781)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_60: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(clone_232, div_39);  clone_232 = div_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_72: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_482, mul_60);  mul_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_487: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_72, [8, 14, 14, 512]);  add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_45 = torch.ops.aten.native_layer_norm.default(view_487, [512], primals_287, primals_288, 1e-05)
    getitem_198: "f32[8, 14, 14, 512]" = native_layer_norm_45[0]
    getitem_199: "f32[8, 14, 14, 1]" = native_layer_norm_45[1]
    getitem_200: "f32[8, 14, 14, 1]" = native_layer_norm_45[2];  native_layer_norm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    roll_20: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(getitem_198, [-3, -3], [1, 2]);  getitem_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_21: "f32[8, 14, 14, 512]" = torch.ops.aten.constant_pad_nd.default(roll_20, [0, 0, 0, 0, 0, 0], 0.0);  roll_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    view_488: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(constant_pad_nd_21, [8, 2, 7, 2, 7, 512]);  constant_pad_nd_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_87: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_488, [0, 1, 3, 2, 4, 5]);  view_488 = None
    clone_233: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_87, memory_format = torch.contiguous_format);  permute_87 = None
    view_489: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_233, [-1, 7, 7, 512]);  clone_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_490: "f32[32, 49, 512]" = torch.ops.aten.view.default(view_489, [-1, 49, 512]);  view_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_491: "f32[1568, 512]" = torch.ops.aten.view.default(view_490, [1568, 512]);  view_490 = None
    t_86: "f32[512, 1536]" = torch.ops.aten.t.default(primals_289);  primals_289 = None
    addmm_84: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_290, view_491, t_86);  primals_290 = None
    view_492: "f32[32, 49, 1536]" = torch.ops.aten.view.default(addmm_84, [32, 49, 1536]);  addmm_84 = None
    view_493: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_492, [32, 49, 3, 16, -1]);  view_492 = None
    permute_88: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.permute.default(view_493, [2, 0, 3, 1, 4]);  view_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_21 = torch.ops.aten.unbind.int(permute_88);  permute_88 = None
    getitem_201: "f32[32, 16, 49, 32]" = unbind_21[0]
    getitem_202: "f32[32, 16, 49, 32]" = unbind_21[1]
    getitem_203: "f32[32, 16, 49, 32]" = unbind_21[2];  unbind_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_61: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_201, 0.1767766952966369);  getitem_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_42: "f32[32, 16, 32, 49]" = torch.ops.aten.transpose.int(getitem_202, -2, -1);  getitem_202 = None
    expand_84: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(mul_61, [32, 16, 49, 32]);  mul_61 = None
    clone_234: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_84, memory_format = torch.contiguous_format);  expand_84 = None
    _unsafe_view_86: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_234, [512, 49, 32]);  clone_234 = None
    expand_85: "f32[32, 16, 32, 49]" = torch.ops.aten.expand.default(transpose_42, [32, 16, 32, 49]);  transpose_42 = None
    clone_235: "f32[32, 16, 32, 49]" = torch.ops.aten.clone.default(expand_85, memory_format = torch.contiguous_format);  expand_85 = None
    _unsafe_view_87: "f32[512, 32, 49]" = torch.ops.aten._unsafe_view.default(clone_235, [512, 32, 49]);  clone_235 = None
    bmm_42: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_86, _unsafe_view_87)
    view_494: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_42, [32, 16, 49, 49]);  bmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_495: "i64[2401]" = torch.ops.aten.view.default(primals_362, [-1]);  primals_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_21: "f32[2401, 16]" = torch.ops.aten.index.Tensor(primals_22, [view_495]);  primals_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_496: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_21, [49, 49, -1]);  index_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_89: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_496, [2, 0, 1]);  view_496 = None
    clone_236: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_89, memory_format = torch.contiguous_format);  permute_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_41: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_236, 0);  clone_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    add_73: "f32[32, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_494, unsqueeze_41);  view_494 = unsqueeze_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_497: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_73, [-1, 4, 16, 49, 49]);  add_73 = None
    unsqueeze_42: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(primals_361, 1);  primals_361 = None
    unsqueeze_43: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, 0);  unsqueeze_42 = None
    add_74: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_497, unsqueeze_43);  view_497 = unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    view_498: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(add_74, [-1, 16, 49, 49]);  add_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_21: "f32[32, 16, 49, 49]" = torch.ops.aten._softmax.default(view_498, -1, False);  view_498 = None
    detach_21: "f32[32, 16, 49, 49]" = torch.ops.aten.detach.default(_softmax_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    clone_237: "f32[32, 16, 49, 49]" = torch.ops.aten.clone.default(_softmax_21);  _softmax_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    expand_86: "f32[32, 16, 49, 49]" = torch.ops.aten.expand.default(clone_237, [32, 16, 49, 49]);  clone_237 = None
    view_499: "f32[512, 49, 49]" = torch.ops.aten.view.default(expand_86, [512, 49, 49]);  expand_86 = None
    expand_87: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_203, [32, 16, 49, 32]);  getitem_203 = None
    clone_238: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_87, memory_format = torch.contiguous_format);  expand_87 = None
    _unsafe_view_88: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_238, [512, 49, 32]);  clone_238 = None
    bmm_43: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_499, _unsafe_view_88)
    view_500: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_43, [32, 16, 49, 32]);  bmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_43: "f32[32, 49, 16, 32]" = torch.ops.aten.transpose.int(view_500, 1, 2);  view_500 = None
    clone_239: "f32[32, 49, 16, 32]" = torch.ops.aten.clone.default(transpose_43, memory_format = torch.contiguous_format);  transpose_43 = None
    _unsafe_view_89: "f32[32, 49, 512]" = torch.ops.aten._unsafe_view.default(clone_239, [32, 49, 512]);  clone_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_501: "f32[1568, 512]" = torch.ops.aten.view.default(_unsafe_view_89, [1568, 512]);  _unsafe_view_89 = None
    t_87: "f32[512, 512]" = torch.ops.aten.t.default(primals_291);  primals_291 = None
    addmm_85: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_292, view_501, t_87);  primals_292 = None
    view_502: "f32[32, 49, 512]" = torch.ops.aten.view.default(addmm_85, [32, 49, 512]);  addmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    clone_240: "f32[32, 49, 512]" = torch.ops.aten.clone.default(view_502);  view_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_503: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_240, [-1, 7, 7, 512]);  clone_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    view_504: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_503, [-1, 2, 2, 7, 7, 512]);  view_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_90: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_504, [0, 1, 3, 2, 4, 5]);  view_504 = None
    clone_241: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
    view_505: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(clone_241, [-1, 14, 14, 512]);  clone_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_43: "f32[8, 14, 14, 512]" = torch.ops.aten.slice.Tensor(view_505, 0, 0, 9223372036854775807);  view_505 = None
    slice_44: "f32[8, 14, 14, 512]" = torch.ops.aten.slice.Tensor(slice_43, 3, 0, 9223372036854775807);  slice_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    roll_21: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(slice_44, [3, 3], [1, 2]);  slice_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_40: "f32[8, 1, 1, 1]" = torch.ops.aten.new_empty.default(roll_21, [8, 1, 1, 1], pin_memory = False)
    bernoulli_40: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_40, 0.9086956530809402);  new_empty_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_40: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_40, 0.9086956530809402)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_62: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(roll_21, div_40);  roll_21 = div_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_75: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_487, mul_62);  mul_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_506: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_75, [8, -1, 512]);  add_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_46 = torch.ops.aten.native_layer_norm.default(view_506, [512], primals_293, primals_294, 1e-05)
    getitem_204: "f32[8, 196, 512]" = native_layer_norm_46[0]
    getitem_205: "f32[8, 196, 1]" = native_layer_norm_46[1]
    getitem_206: "f32[8, 196, 1]" = native_layer_norm_46[2];  native_layer_norm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_507: "f32[1568, 512]" = torch.ops.aten.view.default(getitem_204, [1568, 512]);  getitem_204 = None
    t_88: "f32[512, 2048]" = torch.ops.aten.t.default(primals_295);  primals_295 = None
    addmm_86: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_296, view_507, t_88);  primals_296 = None
    view_508: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_86, [8, 196, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_21: "f32[8, 196, 2048]" = torch.ops.aten.gelu.default(view_508);  view_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_242: "f32[8, 196, 2048]" = torch.ops.aten.clone.default(gelu_21);  gelu_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_509: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_242, [1568, 2048]);  clone_242 = None
    t_89: "f32[2048, 512]" = torch.ops.aten.t.default(primals_297);  primals_297 = None
    addmm_87: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_298, view_509, t_89);  primals_298 = None
    view_510: "f32[8, 196, 512]" = torch.ops.aten.view.default(addmm_87, [8, 196, 512]);  addmm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_243: "f32[8, 196, 512]" = torch.ops.aten.clone.default(view_510);  view_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_41: "f32[8, 1, 1]" = torch.ops.aten.new_empty.default(clone_243, [8, 1, 1], pin_memory = False)
    bernoulli_41: "f32[8, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_41, 0.9086956530809402);  new_empty_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_41: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_41, 0.9086956530809402)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_63: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(clone_243, div_41);  clone_243 = div_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_76: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_506, mul_63);  mul_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_511: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_76, [8, 14, 14, 512]);  add_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:356, code: x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
    view_512: "f32[8, 7, 2, 7, 2, 512]" = torch.ops.aten.view.default(view_511, [8, 7, 2, 7, 2, 512]);  view_511 = None
    permute_91: "f32[8, 7, 7, 2, 2, 512]" = torch.ops.aten.permute.default(view_512, [0, 1, 3, 4, 2, 5]);  view_512 = None
    clone_244: "f32[8, 7, 7, 2, 2, 512]" = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format);  permute_91 = None
    _unsafe_view_90: "f32[8, 7, 7, 2048]" = torch.ops.aten._unsafe_view.default(clone_244, [8, 7, 7, 2048]);  clone_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:357, code: x = self.norm(x)
    native_layer_norm_47 = torch.ops.aten.native_layer_norm.default(_unsafe_view_90, [2048], primals_299, primals_300, 1e-05)
    getitem_207: "f32[8, 7, 7, 2048]" = native_layer_norm_47[0]
    getitem_208: "f32[8, 7, 7, 1]" = native_layer_norm_47[1]
    getitem_209: "f32[8, 7, 7, 1]" = native_layer_norm_47[2];  native_layer_norm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:358, code: x = self.reduction(x)
    t_90: "f32[2048, 1024]" = torch.ops.aten.t.default(primals_301);  primals_301 = None
    view_513: "f32[392, 2048]" = torch.ops.aten.view.default(getitem_207, [392, 2048]);  getitem_207 = None
    mm_2: "f32[392, 1024]" = torch.ops.aten.mm.default(view_513, t_90)
    view_514: "f32[8, 7, 7, 1024]" = torch.ops.aten.view.default(mm_2, [8, 7, 7, 1024]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_48 = torch.ops.aten.native_layer_norm.default(view_514, [1024], primals_302, primals_303, 1e-05)
    getitem_210: "f32[8, 7, 7, 1024]" = native_layer_norm_48[0]
    getitem_211: "f32[8, 7, 7, 1]" = native_layer_norm_48[1]
    getitem_212: "f32[8, 7, 7, 1]" = native_layer_norm_48[2];  native_layer_norm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_22: "f32[8, 7, 7, 1024]" = torch.ops.aten.constant_pad_nd.default(getitem_210, [0, 0, 0, 0, 0, 0], 0.0);  getitem_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    view_515: "f32[8, 1, 7, 1, 7, 1024]" = torch.ops.aten.view.default(constant_pad_nd_22, [8, 1, 7, 1, 7, 1024]);  constant_pad_nd_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_92: "f32[8, 1, 1, 7, 7, 1024]" = torch.ops.aten.permute.default(view_515, [0, 1, 3, 2, 4, 5]);  view_515 = None
    view_516: "f32[8, 7, 7, 1024]" = torch.ops.aten.view.default(permute_92, [-1, 7, 7, 1024]);  permute_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_517: "f32[8, 49, 1024]" = torch.ops.aten.view.default(view_516, [-1, 49, 1024]);  view_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_518: "f32[392, 1024]" = torch.ops.aten.view.default(view_517, [392, 1024]);  view_517 = None
    t_91: "f32[1024, 3072]" = torch.ops.aten.t.default(primals_304);  primals_304 = None
    addmm_88: "f32[392, 3072]" = torch.ops.aten.addmm.default(primals_305, view_518, t_91);  primals_305 = None
    view_519: "f32[8, 49, 3072]" = torch.ops.aten.view.default(addmm_88, [8, 49, 3072]);  addmm_88 = None
    view_520: "f32[8, 49, 3, 32, 32]" = torch.ops.aten.view.default(view_519, [8, 49, 3, 32, -1]);  view_519 = None
    permute_93: "f32[3, 8, 32, 49, 32]" = torch.ops.aten.permute.default(view_520, [2, 0, 3, 1, 4]);  view_520 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_22 = torch.ops.aten.unbind.int(permute_93);  permute_93 = None
    getitem_213: "f32[8, 32, 49, 32]" = unbind_22[0]
    getitem_214: "f32[8, 32, 49, 32]" = unbind_22[1]
    getitem_215: "f32[8, 32, 49, 32]" = unbind_22[2];  unbind_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_64: "f32[8, 32, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_213, 0.1767766952966369);  getitem_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_44: "f32[8, 32, 32, 49]" = torch.ops.aten.transpose.int(getitem_214, -2, -1);  getitem_214 = None
    expand_88: "f32[8, 32, 49, 32]" = torch.ops.aten.expand.default(mul_64, [8, 32, 49, 32]);  mul_64 = None
    clone_245: "f32[8, 32, 49, 32]" = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
    _unsafe_view_91: "f32[256, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_245, [256, 49, 32]);  clone_245 = None
    expand_89: "f32[8, 32, 32, 49]" = torch.ops.aten.expand.default(transpose_44, [8, 32, 32, 49]);  transpose_44 = None
    clone_246: "f32[8, 32, 32, 49]" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
    _unsafe_view_92: "f32[256, 32, 49]" = torch.ops.aten._unsafe_view.default(clone_246, [256, 32, 49]);  clone_246 = None
    bmm_44: "f32[256, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_91, _unsafe_view_92)
    view_521: "f32[8, 32, 49, 49]" = torch.ops.aten.view.default(bmm_44, [8, 32, 49, 49]);  bmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_522: "i64[2401]" = torch.ops.aten.view.default(primals_363, [-1]);  primals_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_22: "f32[2401, 32]" = torch.ops.aten.index.Tensor(primals_23, [view_522]);  primals_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_523: "f32[49, 49, 32]" = torch.ops.aten.view.default(index_22, [49, 49, -1]);  index_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_94: "f32[32, 49, 49]" = torch.ops.aten.permute.default(view_523, [2, 0, 1]);  view_523 = None
    clone_247: "f32[32, 49, 49]" = torch.ops.aten.clone.default(permute_94, memory_format = torch.contiguous_format);  permute_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_44: "f32[1, 32, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_247, 0);  clone_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    add_77: "f32[8, 32, 49, 49]" = torch.ops.aten.add.Tensor(view_521, unsqueeze_44);  view_521 = unsqueeze_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_22: "f32[8, 32, 49, 49]" = torch.ops.aten._softmax.default(add_77, -1, False);  add_77 = None
    detach_22: "f32[8, 32, 49, 49]" = torch.ops.aten.detach.default(_softmax_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    clone_248: "f32[8, 32, 49, 49]" = torch.ops.aten.clone.default(_softmax_22);  _softmax_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    expand_90: "f32[8, 32, 49, 49]" = torch.ops.aten.expand.default(clone_248, [8, 32, 49, 49]);  clone_248 = None
    view_524: "f32[256, 49, 49]" = torch.ops.aten.view.default(expand_90, [256, 49, 49]);  expand_90 = None
    expand_91: "f32[8, 32, 49, 32]" = torch.ops.aten.expand.default(getitem_215, [8, 32, 49, 32]);  getitem_215 = None
    clone_249: "f32[8, 32, 49, 32]" = torch.ops.aten.clone.default(expand_91, memory_format = torch.contiguous_format);  expand_91 = None
    _unsafe_view_93: "f32[256, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_249, [256, 49, 32]);  clone_249 = None
    bmm_45: "f32[256, 49, 32]" = torch.ops.aten.bmm.default(view_524, _unsafe_view_93)
    view_525: "f32[8, 32, 49, 32]" = torch.ops.aten.view.default(bmm_45, [8, 32, 49, 32]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_45: "f32[8, 49, 32, 32]" = torch.ops.aten.transpose.int(view_525, 1, 2);  view_525 = None
    clone_250: "f32[8, 49, 32, 32]" = torch.ops.aten.clone.default(transpose_45, memory_format = torch.contiguous_format);  transpose_45 = None
    _unsafe_view_94: "f32[8, 49, 1024]" = torch.ops.aten._unsafe_view.default(clone_250, [8, 49, 1024]);  clone_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_526: "f32[392, 1024]" = torch.ops.aten.view.default(_unsafe_view_94, [392, 1024]);  _unsafe_view_94 = None
    t_92: "f32[1024, 1024]" = torch.ops.aten.t.default(primals_306);  primals_306 = None
    addmm_89: "f32[392, 1024]" = torch.ops.aten.addmm.default(primals_307, view_526, t_92);  primals_307 = None
    view_527: "f32[8, 49, 1024]" = torch.ops.aten.view.default(addmm_89, [8, 49, 1024]);  addmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    clone_251: "f32[8, 49, 1024]" = torch.ops.aten.clone.default(view_527);  view_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_528: "f32[8, 7, 7, 1024]" = torch.ops.aten.view.default(clone_251, [-1, 7, 7, 1024]);  clone_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    view_529: "f32[8, 1, 1, 7, 7, 1024]" = torch.ops.aten.view.default(view_528, [-1, 1, 1, 7, 7, 1024]);  view_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_95: "f32[8, 1, 7, 1, 7, 1024]" = torch.ops.aten.permute.default(view_529, [0, 1, 3, 2, 4, 5]);  view_529 = None
    view_530: "f32[8, 7, 7, 1024]" = torch.ops.aten.view.default(permute_95, [-1, 7, 7, 1024]);  permute_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_45: "f32[8, 7, 7, 1024]" = torch.ops.aten.slice.Tensor(view_530, 0, 0, 9223372036854775807);  view_530 = None
    slice_46: "f32[8, 7, 7, 1024]" = torch.ops.aten.slice.Tensor(slice_45, 3, 0, 9223372036854775807);  slice_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_42: "f32[8, 1, 1, 1]" = torch.ops.aten.new_empty.default(slice_46, [8, 1, 1, 1], pin_memory = False)
    bernoulli_42: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_42, 0.9043478220701218);  new_empty_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_42: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_42, 0.9043478220701218)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_65: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(slice_46, div_42);  slice_46 = div_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_78: "f32[8, 7, 7, 1024]" = torch.ops.aten.add.Tensor(view_514, mul_65);  mul_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_531: "f32[8, 49, 1024]" = torch.ops.aten.view.default(add_78, [8, -1, 1024]);  add_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_49 = torch.ops.aten.native_layer_norm.default(view_531, [1024], primals_308, primals_309, 1e-05)
    getitem_216: "f32[8, 49, 1024]" = native_layer_norm_49[0]
    getitem_217: "f32[8, 49, 1]" = native_layer_norm_49[1]
    getitem_218: "f32[8, 49, 1]" = native_layer_norm_49[2];  native_layer_norm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_532: "f32[392, 1024]" = torch.ops.aten.view.default(getitem_216, [392, 1024]);  getitem_216 = None
    t_93: "f32[1024, 4096]" = torch.ops.aten.t.default(primals_310);  primals_310 = None
    addmm_90: "f32[392, 4096]" = torch.ops.aten.addmm.default(primals_311, view_532, t_93);  primals_311 = None
    view_533: "f32[8, 49, 4096]" = torch.ops.aten.view.default(addmm_90, [8, 49, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_22: "f32[8, 49, 4096]" = torch.ops.aten.gelu.default(view_533);  view_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_252: "f32[8, 49, 4096]" = torch.ops.aten.clone.default(gelu_22);  gelu_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_534: "f32[392, 4096]" = torch.ops.aten.view.default(clone_252, [392, 4096]);  clone_252 = None
    t_94: "f32[4096, 1024]" = torch.ops.aten.t.default(primals_312);  primals_312 = None
    addmm_91: "f32[392, 1024]" = torch.ops.aten.addmm.default(primals_313, view_534, t_94);  primals_313 = None
    view_535: "f32[8, 49, 1024]" = torch.ops.aten.view.default(addmm_91, [8, 49, 1024]);  addmm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_253: "f32[8, 49, 1024]" = torch.ops.aten.clone.default(view_535);  view_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_43: "f32[8, 1, 1]" = torch.ops.aten.new_empty.default(clone_253, [8, 1, 1], pin_memory = False)
    bernoulli_43: "f32[8, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_43, 0.9043478220701218);  new_empty_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_43: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_43, 0.9043478220701218)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_66: "f32[8, 49, 1024]" = torch.ops.aten.mul.Tensor(clone_253, div_43);  clone_253 = div_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_79: "f32[8, 49, 1024]" = torch.ops.aten.add.Tensor(view_531, mul_66);  mul_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_536: "f32[8, 7, 7, 1024]" = torch.ops.aten.view.default(add_79, [8, 7, 7, 1024]);  add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_50 = torch.ops.aten.native_layer_norm.default(view_536, [1024], primals_314, primals_315, 1e-05)
    getitem_219: "f32[8, 7, 7, 1024]" = native_layer_norm_50[0]
    getitem_220: "f32[8, 7, 7, 1]" = native_layer_norm_50[1]
    getitem_221: "f32[8, 7, 7, 1]" = native_layer_norm_50[2];  native_layer_norm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_23: "f32[8, 7, 7, 1024]" = torch.ops.aten.constant_pad_nd.default(getitem_219, [0, 0, 0, 0, 0, 0], 0.0);  getitem_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    view_537: "f32[8, 1, 7, 1, 7, 1024]" = torch.ops.aten.view.default(constant_pad_nd_23, [8, 1, 7, 1, 7, 1024]);  constant_pad_nd_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_96: "f32[8, 1, 1, 7, 7, 1024]" = torch.ops.aten.permute.default(view_537, [0, 1, 3, 2, 4, 5]);  view_537 = None
    view_538: "f32[8, 7, 7, 1024]" = torch.ops.aten.view.default(permute_96, [-1, 7, 7, 1024]);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_539: "f32[8, 49, 1024]" = torch.ops.aten.view.default(view_538, [-1, 49, 1024]);  view_538 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_540: "f32[392, 1024]" = torch.ops.aten.view.default(view_539, [392, 1024]);  view_539 = None
    t_95: "f32[1024, 3072]" = torch.ops.aten.t.default(primals_316);  primals_316 = None
    addmm_92: "f32[392, 3072]" = torch.ops.aten.addmm.default(primals_317, view_540, t_95);  primals_317 = None
    view_541: "f32[8, 49, 3072]" = torch.ops.aten.view.default(addmm_92, [8, 49, 3072]);  addmm_92 = None
    view_542: "f32[8, 49, 3, 32, 32]" = torch.ops.aten.view.default(view_541, [8, 49, 3, 32, -1]);  view_541 = None
    permute_97: "f32[3, 8, 32, 49, 32]" = torch.ops.aten.permute.default(view_542, [2, 0, 3, 1, 4]);  view_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_23 = torch.ops.aten.unbind.int(permute_97);  permute_97 = None
    getitem_222: "f32[8, 32, 49, 32]" = unbind_23[0]
    getitem_223: "f32[8, 32, 49, 32]" = unbind_23[1]
    getitem_224: "f32[8, 32, 49, 32]" = unbind_23[2];  unbind_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_67: "f32[8, 32, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_222, 0.1767766952966369);  getitem_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_46: "f32[8, 32, 32, 49]" = torch.ops.aten.transpose.int(getitem_223, -2, -1);  getitem_223 = None
    expand_92: "f32[8, 32, 49, 32]" = torch.ops.aten.expand.default(mul_67, [8, 32, 49, 32]);  mul_67 = None
    clone_254: "f32[8, 32, 49, 32]" = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
    _unsafe_view_95: "f32[256, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_254, [256, 49, 32]);  clone_254 = None
    expand_93: "f32[8, 32, 32, 49]" = torch.ops.aten.expand.default(transpose_46, [8, 32, 32, 49]);  transpose_46 = None
    clone_255: "f32[8, 32, 32, 49]" = torch.ops.aten.clone.default(expand_93, memory_format = torch.contiguous_format);  expand_93 = None
    _unsafe_view_96: "f32[256, 32, 49]" = torch.ops.aten._unsafe_view.default(clone_255, [256, 32, 49]);  clone_255 = None
    bmm_46: "f32[256, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_95, _unsafe_view_96)
    view_543: "f32[8, 32, 49, 49]" = torch.ops.aten.view.default(bmm_46, [8, 32, 49, 49]);  bmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_544: "i64[2401]" = torch.ops.aten.view.default(primals_364, [-1]);  primals_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_23: "f32[2401, 32]" = torch.ops.aten.index.Tensor(primals_24, [view_544]);  primals_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_545: "f32[49, 49, 32]" = torch.ops.aten.view.default(index_23, [49, 49, -1]);  index_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_98: "f32[32, 49, 49]" = torch.ops.aten.permute.default(view_545, [2, 0, 1]);  view_545 = None
    clone_256: "f32[32, 49, 49]" = torch.ops.aten.clone.default(permute_98, memory_format = torch.contiguous_format);  permute_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_45: "f32[1, 32, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_256, 0);  clone_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    add_80: "f32[8, 32, 49, 49]" = torch.ops.aten.add.Tensor(view_543, unsqueeze_45);  view_543 = unsqueeze_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_23: "f32[8, 32, 49, 49]" = torch.ops.aten._softmax.default(add_80, -1, False);  add_80 = None
    detach_23: "f32[8, 32, 49, 49]" = torch.ops.aten.detach.default(_softmax_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    clone_257: "f32[8, 32, 49, 49]" = torch.ops.aten.clone.default(_softmax_23);  _softmax_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    expand_94: "f32[8, 32, 49, 49]" = torch.ops.aten.expand.default(clone_257, [8, 32, 49, 49]);  clone_257 = None
    view_546: "f32[256, 49, 49]" = torch.ops.aten.view.default(expand_94, [256, 49, 49]);  expand_94 = None
    expand_95: "f32[8, 32, 49, 32]" = torch.ops.aten.expand.default(getitem_224, [8, 32, 49, 32]);  getitem_224 = None
    clone_258: "f32[8, 32, 49, 32]" = torch.ops.aten.clone.default(expand_95, memory_format = torch.contiguous_format);  expand_95 = None
    _unsafe_view_97: "f32[256, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_258, [256, 49, 32]);  clone_258 = None
    bmm_47: "f32[256, 49, 32]" = torch.ops.aten.bmm.default(view_546, _unsafe_view_97)
    view_547: "f32[8, 32, 49, 32]" = torch.ops.aten.view.default(bmm_47, [8, 32, 49, 32]);  bmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_47: "f32[8, 49, 32, 32]" = torch.ops.aten.transpose.int(view_547, 1, 2);  view_547 = None
    clone_259: "f32[8, 49, 32, 32]" = torch.ops.aten.clone.default(transpose_47, memory_format = torch.contiguous_format);  transpose_47 = None
    _unsafe_view_98: "f32[8, 49, 1024]" = torch.ops.aten._unsafe_view.default(clone_259, [8, 49, 1024]);  clone_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_548: "f32[392, 1024]" = torch.ops.aten.view.default(_unsafe_view_98, [392, 1024]);  _unsafe_view_98 = None
    t_96: "f32[1024, 1024]" = torch.ops.aten.t.default(primals_318);  primals_318 = None
    addmm_93: "f32[392, 1024]" = torch.ops.aten.addmm.default(primals_319, view_548, t_96);  primals_319 = None
    view_549: "f32[8, 49, 1024]" = torch.ops.aten.view.default(addmm_93, [8, 49, 1024]);  addmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    clone_260: "f32[8, 49, 1024]" = torch.ops.aten.clone.default(view_549);  view_549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_550: "f32[8, 7, 7, 1024]" = torch.ops.aten.view.default(clone_260, [-1, 7, 7, 1024]);  clone_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    view_551: "f32[8, 1, 1, 7, 7, 1024]" = torch.ops.aten.view.default(view_550, [-1, 1, 1, 7, 7, 1024]);  view_550 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_99: "f32[8, 1, 7, 1, 7, 1024]" = torch.ops.aten.permute.default(view_551, [0, 1, 3, 2, 4, 5]);  view_551 = None
    view_552: "f32[8, 7, 7, 1024]" = torch.ops.aten.view.default(permute_99, [-1, 7, 7, 1024]);  permute_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_47: "f32[8, 7, 7, 1024]" = torch.ops.aten.slice.Tensor(view_552, 0, 0, 9223372036854775807);  view_552 = None
    slice_48: "f32[8, 7, 7, 1024]" = torch.ops.aten.slice.Tensor(slice_47, 3, 0, 9223372036854775807);  slice_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_44: "f32[8, 1, 1, 1]" = torch.ops.aten.new_empty.default(slice_48, [8, 1, 1, 1], pin_memory = False)
    bernoulli_44: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_44, 0.8999999985098839);  new_empty_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_44: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_44, 0.8999999985098839)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_68: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(slice_48, div_44);  slice_48 = div_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_81: "f32[8, 7, 7, 1024]" = torch.ops.aten.add.Tensor(view_536, mul_68);  mul_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_553: "f32[8, 49, 1024]" = torch.ops.aten.view.default(add_81, [8, -1, 1024]);  add_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_51 = torch.ops.aten.native_layer_norm.default(view_553, [1024], primals_320, primals_321, 1e-05)
    getitem_225: "f32[8, 49, 1024]" = native_layer_norm_51[0]
    getitem_226: "f32[8, 49, 1]" = native_layer_norm_51[1]
    getitem_227: "f32[8, 49, 1]" = native_layer_norm_51[2];  native_layer_norm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_554: "f32[392, 1024]" = torch.ops.aten.view.default(getitem_225, [392, 1024]);  getitem_225 = None
    t_97: "f32[1024, 4096]" = torch.ops.aten.t.default(primals_322);  primals_322 = None
    addmm_94: "f32[392, 4096]" = torch.ops.aten.addmm.default(primals_323, view_554, t_97);  primals_323 = None
    view_555: "f32[8, 49, 4096]" = torch.ops.aten.view.default(addmm_94, [8, 49, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_23: "f32[8, 49, 4096]" = torch.ops.aten.gelu.default(view_555);  view_555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_261: "f32[8, 49, 4096]" = torch.ops.aten.clone.default(gelu_23);  gelu_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_556: "f32[392, 4096]" = torch.ops.aten.view.default(clone_261, [392, 4096]);  clone_261 = None
    t_98: "f32[4096, 1024]" = torch.ops.aten.t.default(primals_324);  primals_324 = None
    addmm_95: "f32[392, 1024]" = torch.ops.aten.addmm.default(primals_325, view_556, t_98);  primals_325 = None
    view_557: "f32[8, 49, 1024]" = torch.ops.aten.view.default(addmm_95, [8, 49, 1024]);  addmm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_262: "f32[8, 49, 1024]" = torch.ops.aten.clone.default(view_557);  view_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    new_empty_45: "f32[8, 1, 1]" = torch.ops.aten.new_empty.default(clone_262, [8, 1, 1], pin_memory = False)
    bernoulli_45: "f32[8, 1, 1]" = torch.ops.aten.bernoulli.p(new_empty_45, 0.8999999985098839);  new_empty_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_45: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_45, 0.8999999985098839)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_69: "f32[8, 49, 1024]" = torch.ops.aten.mul.Tensor(clone_262, div_45);  clone_262 = div_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_82: "f32[8, 49, 1024]" = torch.ops.aten.add.Tensor(view_553, mul_69);  mul_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_558: "f32[8, 7, 7, 1024]" = torch.ops.aten.view.default(add_82, [8, 7, 7, 1024]);  add_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:610, code: x = self.norm(x)
    native_layer_norm_52 = torch.ops.aten.native_layer_norm.default(view_558, [1024], primals_326, primals_327, 1e-05)
    getitem_228: "f32[8, 7, 7, 1024]" = native_layer_norm_52[0]
    getitem_229: "f32[8, 7, 7, 1]" = native_layer_norm_52[1]
    getitem_230: "f32[8, 7, 7, 1]" = native_layer_norm_52[2];  native_layer_norm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:65, code: return x.mean(self.dim, keepdim=not self.flatten)
    mean: "f32[8, 1024]" = torch.ops.aten.mean.dim(getitem_228, [1, 2]);  getitem_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:131, code: x = self.drop(x)
    clone_263: "f32[8, 1024]" = torch.ops.aten.clone.default(mean);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    t_99: "f32[1024, 1000]" = torch.ops.aten.t.default(primals_328);  primals_328 = None
    addmm_96: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_329, clone_263, t_99);  primals_329 = None
    t_100: "f32[1000, 1024]" = torch.ops.aten.t.default(t_99);  t_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_104: "f32[1024, 4096]" = torch.ops.aten.t.default(t_98);  t_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_108: "f32[4096, 1024]" = torch.ops.aten.t.default(t_97);  t_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    t_112: "f32[1024, 1024]" = torch.ops.aten.t.default(t_96);  t_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    transpose_49: "f32[256, 49, 49]" = torch.ops.aten.transpose.int(view_546, 1, 2);  view_546 = None
    transpose_50: "f32[256, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_97, 1, 2);  _unsafe_view_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    detach_24: "f32[8, 32, 49, 49]" = torch.ops.aten.detach.default(detach_23);  detach_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_51: "f32[256, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_95, 1, 2);  _unsafe_view_95 = None
    transpose_52: "f32[256, 49, 32]" = torch.ops.aten.transpose.int(_unsafe_view_96, 1, 2);  _unsafe_view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    t_116: "f32[3072, 1024]" = torch.ops.aten.t.default(t_95);  t_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_120: "f32[1024, 4096]" = torch.ops.aten.t.default(t_94);  t_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_124: "f32[4096, 1024]" = torch.ops.aten.t.default(t_93);  t_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    t_128: "f32[1024, 1024]" = torch.ops.aten.t.default(t_92);  t_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    transpose_55: "f32[256, 49, 49]" = torch.ops.aten.transpose.int(view_524, 1, 2);  view_524 = None
    transpose_56: "f32[256, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_93, 1, 2);  _unsafe_view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    detach_25: "f32[8, 32, 49, 49]" = torch.ops.aten.detach.default(detach_22);  detach_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_57: "f32[256, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_91, 1, 2);  _unsafe_view_91 = None
    transpose_58: "f32[256, 49, 32]" = torch.ops.aten.transpose.int(_unsafe_view_92, 1, 2);  _unsafe_view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    t_132: "f32[3072, 1024]" = torch.ops.aten.t.default(t_91);  t_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:358, code: x = self.reduction(x)
    t_138: "f32[1024, 2048]" = torch.ops.aten.t.default(t_90);  t_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_140: "f32[512, 2048]" = torch.ops.aten.t.default(t_89);  t_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_144: "f32[2048, 512]" = torch.ops.aten.t.default(t_88);  t_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    t_148: "f32[512, 512]" = torch.ops.aten.t.default(t_87);  t_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    transpose_61: "f32[512, 49, 49]" = torch.ops.aten.transpose.int(view_499, 1, 2);  view_499 = None
    transpose_62: "f32[512, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_88, 1, 2);  _unsafe_view_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    detach_26: "f32[32, 16, 49, 49]" = torch.ops.aten.detach.default(detach_21);  detach_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_63: "f32[512, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_86, 1, 2);  _unsafe_view_86 = None
    transpose_64: "f32[512, 49, 32]" = torch.ops.aten.transpose.int(_unsafe_view_87, 1, 2);  _unsafe_view_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    t_152: "f32[1536, 512]" = torch.ops.aten.t.default(t_86);  t_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_156: "f32[512, 2048]" = torch.ops.aten.t.default(t_85);  t_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_160: "f32[2048, 512]" = torch.ops.aten.t.default(t_84);  t_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    t_164: "f32[512, 512]" = torch.ops.aten.t.default(t_83);  t_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    transpose_67: "f32[512, 49, 49]" = torch.ops.aten.transpose.int(view_475, 1, 2);  view_475 = None
    transpose_68: "f32[512, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_84, 1, 2);  _unsafe_view_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    detach_27: "f32[32, 16, 49, 49]" = torch.ops.aten.detach.default(detach_20);  detach_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_69: "f32[512, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_82, 1, 2);  _unsafe_view_82 = None
    transpose_70: "f32[512, 49, 32]" = torch.ops.aten.transpose.int(_unsafe_view_83, 1, 2);  _unsafe_view_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    t_168: "f32[1536, 512]" = torch.ops.aten.t.default(t_82);  t_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_172: "f32[512, 2048]" = torch.ops.aten.t.default(t_81);  t_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_176: "f32[2048, 512]" = torch.ops.aten.t.default(t_80);  t_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    t_180: "f32[512, 512]" = torch.ops.aten.t.default(t_79);  t_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    transpose_73: "f32[512, 49, 49]" = torch.ops.aten.transpose.int(view_453, 1, 2);  view_453 = None
    transpose_74: "f32[512, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_80, 1, 2);  _unsafe_view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    detach_28: "f32[32, 16, 49, 49]" = torch.ops.aten.detach.default(detach_19);  detach_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_75: "f32[512, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_78, 1, 2);  _unsafe_view_78 = None
    transpose_76: "f32[512, 49, 32]" = torch.ops.aten.transpose.int(_unsafe_view_79, 1, 2);  _unsafe_view_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    t_184: "f32[1536, 512]" = torch.ops.aten.t.default(t_78);  t_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_188: "f32[512, 2048]" = torch.ops.aten.t.default(t_77);  t_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_192: "f32[2048, 512]" = torch.ops.aten.t.default(t_76);  t_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    t_196: "f32[512, 512]" = torch.ops.aten.t.default(t_75);  t_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    transpose_79: "f32[512, 49, 49]" = torch.ops.aten.transpose.int(view_429, 1, 2);  view_429 = None
    transpose_80: "f32[512, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_76, 1, 2);  _unsafe_view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    detach_29: "f32[32, 16, 49, 49]" = torch.ops.aten.detach.default(detach_18);  detach_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_81: "f32[512, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_74, 1, 2);  _unsafe_view_74 = None
    transpose_82: "f32[512, 49, 32]" = torch.ops.aten.transpose.int(_unsafe_view_75, 1, 2);  _unsafe_view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    t_200: "f32[1536, 512]" = torch.ops.aten.t.default(t_74);  t_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_204: "f32[512, 2048]" = torch.ops.aten.t.default(t_73);  t_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_208: "f32[2048, 512]" = torch.ops.aten.t.default(t_72);  t_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    t_212: "f32[512, 512]" = torch.ops.aten.t.default(t_71);  t_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    transpose_85: "f32[512, 49, 49]" = torch.ops.aten.transpose.int(view_407, 1, 2);  view_407 = None
    transpose_86: "f32[512, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_72, 1, 2);  _unsafe_view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    detach_30: "f32[32, 16, 49, 49]" = torch.ops.aten.detach.default(detach_17);  detach_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_87: "f32[512, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_70, 1, 2);  _unsafe_view_70 = None
    transpose_88: "f32[512, 49, 32]" = torch.ops.aten.transpose.int(_unsafe_view_71, 1, 2);  _unsafe_view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    t_216: "f32[1536, 512]" = torch.ops.aten.t.default(t_70);  t_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_220: "f32[512, 2048]" = torch.ops.aten.t.default(t_69);  t_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_224: "f32[2048, 512]" = torch.ops.aten.t.default(t_68);  t_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    t_228: "f32[512, 512]" = torch.ops.aten.t.default(t_67);  t_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    transpose_91: "f32[512, 49, 49]" = torch.ops.aten.transpose.int(view_383, 1, 2);  view_383 = None
    transpose_92: "f32[512, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_68, 1, 2);  _unsafe_view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    detach_31: "f32[32, 16, 49, 49]" = torch.ops.aten.detach.default(detach_16);  detach_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_93: "f32[512, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_66, 1, 2);  _unsafe_view_66 = None
    transpose_94: "f32[512, 49, 32]" = torch.ops.aten.transpose.int(_unsafe_view_67, 1, 2);  _unsafe_view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    t_232: "f32[1536, 512]" = torch.ops.aten.t.default(t_66);  t_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_236: "f32[512, 2048]" = torch.ops.aten.t.default(t_65);  t_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_240: "f32[2048, 512]" = torch.ops.aten.t.default(t_64);  t_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    t_244: "f32[512, 512]" = torch.ops.aten.t.default(t_63);  t_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    transpose_97: "f32[512, 49, 49]" = torch.ops.aten.transpose.int(view_361, 1, 2);  view_361 = None
    transpose_98: "f32[512, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_64, 1, 2);  _unsafe_view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    detach_32: "f32[32, 16, 49, 49]" = torch.ops.aten.detach.default(detach_15);  detach_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_99: "f32[512, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_62, 1, 2);  _unsafe_view_62 = None
    transpose_100: "f32[512, 49, 32]" = torch.ops.aten.transpose.int(_unsafe_view_63, 1, 2);  _unsafe_view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    t_248: "f32[1536, 512]" = torch.ops.aten.t.default(t_62);  t_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_252: "f32[512, 2048]" = torch.ops.aten.t.default(t_61);  t_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_256: "f32[2048, 512]" = torch.ops.aten.t.default(t_60);  t_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    t_260: "f32[512, 512]" = torch.ops.aten.t.default(t_59);  t_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    transpose_103: "f32[512, 49, 49]" = torch.ops.aten.transpose.int(view_337, 1, 2);  view_337 = None
    transpose_104: "f32[512, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_60, 1, 2);  _unsafe_view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    detach_33: "f32[32, 16, 49, 49]" = torch.ops.aten.detach.default(detach_14);  detach_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_105: "f32[512, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_58, 1, 2);  _unsafe_view_58 = None
    transpose_106: "f32[512, 49, 32]" = torch.ops.aten.transpose.int(_unsafe_view_59, 1, 2);  _unsafe_view_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    t_264: "f32[1536, 512]" = torch.ops.aten.t.default(t_58);  t_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_268: "f32[512, 2048]" = torch.ops.aten.t.default(t_57);  t_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_272: "f32[2048, 512]" = torch.ops.aten.t.default(t_56);  t_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    t_276: "f32[512, 512]" = torch.ops.aten.t.default(t_55);  t_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    transpose_109: "f32[512, 49, 49]" = torch.ops.aten.transpose.int(view_315, 1, 2);  view_315 = None
    transpose_110: "f32[512, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_56, 1, 2);  _unsafe_view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    detach_34: "f32[32, 16, 49, 49]" = torch.ops.aten.detach.default(detach_13);  detach_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_111: "f32[512, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_54, 1, 2);  _unsafe_view_54 = None
    transpose_112: "f32[512, 49, 32]" = torch.ops.aten.transpose.int(_unsafe_view_55, 1, 2);  _unsafe_view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    t_280: "f32[1536, 512]" = torch.ops.aten.t.default(t_54);  t_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_284: "f32[512, 2048]" = torch.ops.aten.t.default(t_53);  t_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_288: "f32[2048, 512]" = torch.ops.aten.t.default(t_52);  t_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    t_292: "f32[512, 512]" = torch.ops.aten.t.default(t_51);  t_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    transpose_115: "f32[512, 49, 49]" = torch.ops.aten.transpose.int(view_291, 1, 2);  view_291 = None
    transpose_116: "f32[512, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_52, 1, 2);  _unsafe_view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    detach_35: "f32[32, 16, 49, 49]" = torch.ops.aten.detach.default(detach_12);  detach_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_117: "f32[512, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_50, 1, 2);  _unsafe_view_50 = None
    transpose_118: "f32[512, 49, 32]" = torch.ops.aten.transpose.int(_unsafe_view_51, 1, 2);  _unsafe_view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    t_296: "f32[1536, 512]" = torch.ops.aten.t.default(t_50);  t_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_300: "f32[512, 2048]" = torch.ops.aten.t.default(t_49);  t_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_304: "f32[2048, 512]" = torch.ops.aten.t.default(t_48);  t_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    t_308: "f32[512, 512]" = torch.ops.aten.t.default(t_47);  t_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    transpose_121: "f32[512, 49, 49]" = torch.ops.aten.transpose.int(view_269, 1, 2);  view_269 = None
    transpose_122: "f32[512, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_48, 1, 2);  _unsafe_view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    detach_36: "f32[32, 16, 49, 49]" = torch.ops.aten.detach.default(detach_11);  detach_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_123: "f32[512, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_46, 1, 2);  _unsafe_view_46 = None
    transpose_124: "f32[512, 49, 32]" = torch.ops.aten.transpose.int(_unsafe_view_47, 1, 2);  _unsafe_view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    t_312: "f32[1536, 512]" = torch.ops.aten.t.default(t_46);  t_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_316: "f32[512, 2048]" = torch.ops.aten.t.default(t_45);  t_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_320: "f32[2048, 512]" = torch.ops.aten.t.default(t_44);  t_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    t_324: "f32[512, 512]" = torch.ops.aten.t.default(t_43);  t_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    transpose_127: "f32[512, 49, 49]" = torch.ops.aten.transpose.int(view_245, 1, 2);  view_245 = None
    transpose_128: "f32[512, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_44, 1, 2);  _unsafe_view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    detach_37: "f32[32, 16, 49, 49]" = torch.ops.aten.detach.default(detach_10);  detach_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_129: "f32[512, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_42, 1, 2);  _unsafe_view_42 = None
    transpose_130: "f32[512, 49, 32]" = torch.ops.aten.transpose.int(_unsafe_view_43, 1, 2);  _unsafe_view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    t_328: "f32[1536, 512]" = torch.ops.aten.t.default(t_42);  t_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_332: "f32[512, 2048]" = torch.ops.aten.t.default(t_41);  t_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_336: "f32[2048, 512]" = torch.ops.aten.t.default(t_40);  t_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    t_340: "f32[512, 512]" = torch.ops.aten.t.default(t_39);  t_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    transpose_133: "f32[512, 49, 49]" = torch.ops.aten.transpose.int(view_223, 1, 2);  view_223 = None
    transpose_134: "f32[512, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_40, 1, 2);  _unsafe_view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    detach_38: "f32[32, 16, 49, 49]" = torch.ops.aten.detach.default(detach_9);  detach_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_135: "f32[512, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_38, 1, 2);  _unsafe_view_38 = None
    transpose_136: "f32[512, 49, 32]" = torch.ops.aten.transpose.int(_unsafe_view_39, 1, 2);  _unsafe_view_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    t_344: "f32[1536, 512]" = torch.ops.aten.t.default(t_38);  t_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_348: "f32[512, 2048]" = torch.ops.aten.t.default(t_37);  t_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_352: "f32[2048, 512]" = torch.ops.aten.t.default(t_36);  t_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    t_356: "f32[512, 512]" = torch.ops.aten.t.default(t_35);  t_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    transpose_139: "f32[512, 49, 49]" = torch.ops.aten.transpose.int(view_199, 1, 2);  view_199 = None
    transpose_140: "f32[512, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_36, 1, 2);  _unsafe_view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    detach_39: "f32[32, 16, 49, 49]" = torch.ops.aten.detach.default(detach_8);  detach_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_141: "f32[512, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_34, 1, 2);  _unsafe_view_34 = None
    transpose_142: "f32[512, 49, 32]" = torch.ops.aten.transpose.int(_unsafe_view_35, 1, 2);  _unsafe_view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    t_360: "f32[1536, 512]" = torch.ops.aten.t.default(t_34);  t_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_364: "f32[512, 2048]" = torch.ops.aten.t.default(t_33);  t_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_368: "f32[2048, 512]" = torch.ops.aten.t.default(t_32);  t_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    t_372: "f32[512, 512]" = torch.ops.aten.t.default(t_31);  t_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    transpose_145: "f32[512, 49, 49]" = torch.ops.aten.transpose.int(view_177, 1, 2);  view_177 = None
    transpose_146: "f32[512, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_32, 1, 2);  _unsafe_view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    detach_40: "f32[32, 16, 49, 49]" = torch.ops.aten.detach.default(detach_7);  detach_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_147: "f32[512, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_30, 1, 2);  _unsafe_view_30 = None
    transpose_148: "f32[512, 49, 32]" = torch.ops.aten.transpose.int(_unsafe_view_31, 1, 2);  _unsafe_view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    t_376: "f32[1536, 512]" = torch.ops.aten.t.default(t_30);  t_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_380: "f32[512, 2048]" = torch.ops.aten.t.default(t_29);  t_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_384: "f32[2048, 512]" = torch.ops.aten.t.default(t_28);  t_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    t_388: "f32[512, 512]" = torch.ops.aten.t.default(t_27);  t_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    transpose_151: "f32[512, 49, 49]" = torch.ops.aten.transpose.int(view_153, 1, 2);  view_153 = None
    transpose_152: "f32[512, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_28, 1, 2);  _unsafe_view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    detach_41: "f32[32, 16, 49, 49]" = torch.ops.aten.detach.default(detach_6);  detach_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_153: "f32[512, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_26, 1, 2);  _unsafe_view_26 = None
    transpose_154: "f32[512, 49, 32]" = torch.ops.aten.transpose.int(_unsafe_view_27, 1, 2);  _unsafe_view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    t_392: "f32[1536, 512]" = torch.ops.aten.t.default(t_26);  t_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_396: "f32[512, 2048]" = torch.ops.aten.t.default(t_25);  t_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_400: "f32[2048, 512]" = torch.ops.aten.t.default(t_24);  t_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    t_404: "f32[512, 512]" = torch.ops.aten.t.default(t_23);  t_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    transpose_157: "f32[512, 49, 49]" = torch.ops.aten.transpose.int(view_131, 1, 2);  view_131 = None
    transpose_158: "f32[512, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_24, 1, 2);  _unsafe_view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    detach_42: "f32[32, 16, 49, 49]" = torch.ops.aten.detach.default(detach_5);  detach_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_159: "f32[512, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_22, 1, 2);  _unsafe_view_22 = None
    transpose_160: "f32[512, 49, 32]" = torch.ops.aten.transpose.int(_unsafe_view_23, 1, 2);  _unsafe_view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    t_408: "f32[1536, 512]" = torch.ops.aten.t.default(t_22);  t_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_412: "f32[512, 2048]" = torch.ops.aten.t.default(t_21);  t_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_416: "f32[2048, 512]" = torch.ops.aten.t.default(t_20);  t_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    t_420: "f32[512, 512]" = torch.ops.aten.t.default(t_19);  t_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    transpose_163: "f32[512, 49, 49]" = torch.ops.aten.transpose.int(view_107, 1, 2);  view_107 = None
    transpose_164: "f32[512, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_20, 1, 2);  _unsafe_view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    detach_43: "f32[32, 16, 49, 49]" = torch.ops.aten.detach.default(detach_4);  detach_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_165: "f32[512, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_18, 1, 2);  _unsafe_view_18 = None
    transpose_166: "f32[512, 49, 32]" = torch.ops.aten.transpose.int(_unsafe_view_19, 1, 2);  _unsafe_view_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    t_424: "f32[1536, 512]" = torch.ops.aten.t.default(t_18);  t_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:358, code: x = self.reduction(x)
    t_430: "f32[512, 1024]" = torch.ops.aten.t.default(t_17);  t_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_432: "f32[256, 1024]" = torch.ops.aten.t.default(t_16);  t_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_436: "f32[1024, 256]" = torch.ops.aten.t.default(t_15);  t_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    t_440: "f32[256, 256]" = torch.ops.aten.t.default(t_14);  t_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    transpose_169: "f32[1024, 49, 49]" = torch.ops.aten.transpose.int(view_82, 1, 2);  view_82 = None
    transpose_170: "f32[1024, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_15, 1, 2);  _unsafe_view_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    detach_44: "f32[128, 8, 49, 49]" = torch.ops.aten.detach.default(detach_3);  detach_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_171: "f32[1024, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_13, 1, 2);  _unsafe_view_13 = None
    transpose_172: "f32[1024, 49, 32]" = torch.ops.aten.transpose.int(_unsafe_view_14, 1, 2);  _unsafe_view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    t_444: "f32[768, 256]" = torch.ops.aten.t.default(t_13);  t_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_448: "f32[256, 1024]" = torch.ops.aten.t.default(t_12);  t_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_452: "f32[1024, 256]" = torch.ops.aten.t.default(t_11);  t_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    t_456: "f32[256, 256]" = torch.ops.aten.t.default(t_10);  t_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    transpose_175: "f32[1024, 49, 49]" = torch.ops.aten.transpose.int(view_58, 1, 2);  view_58 = None
    transpose_176: "f32[1024, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_11, 1, 2);  _unsafe_view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    detach_45: "f32[128, 8, 49, 49]" = torch.ops.aten.detach.default(detach_2);  detach_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_177: "f32[1024, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_9, 1, 2);  _unsafe_view_9 = None
    transpose_178: "f32[1024, 49, 32]" = torch.ops.aten.transpose.int(_unsafe_view_10, 1, 2);  _unsafe_view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    t_460: "f32[768, 256]" = torch.ops.aten.t.default(t_9);  t_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:358, code: x = self.reduction(x)
    t_466: "f32[256, 512]" = torch.ops.aten.t.default(t_8);  t_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_468: "f32[128, 512]" = torch.ops.aten.t.default(t_7);  t_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_472: "f32[512, 128]" = torch.ops.aten.t.default(t_6);  t_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    t_476: "f32[128, 128]" = torch.ops.aten.t.default(t_5);  t_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    transpose_181: "f32[2048, 49, 49]" = torch.ops.aten.transpose.int(view_33, 1, 2);  view_33 = None
    transpose_182: "f32[2048, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_6, 1, 2);  _unsafe_view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    detach_46: "f32[512, 4, 49, 49]" = torch.ops.aten.detach.default(detach_1);  detach_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_183: "f32[2048, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_4, 1, 2);  _unsafe_view_4 = None
    transpose_184: "f32[2048, 49, 32]" = torch.ops.aten.transpose.int(_unsafe_view_5, 1, 2);  _unsafe_view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    t_480: "f32[384, 128]" = torch.ops.aten.t.default(t_4);  t_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_484: "f32[128, 512]" = torch.ops.aten.t.default(t_3);  t_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_488: "f32[512, 128]" = torch.ops.aten.t.default(t_2);  t_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    t_492: "f32[128, 128]" = torch.ops.aten.t.default(t_1);  t_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    transpose_187: "f32[2048, 49, 49]" = torch.ops.aten.transpose.int(view_9, 1, 2);  view_9 = None
    transpose_188: "f32[2048, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view_2, 1, 2);  _unsafe_view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    detach_47: "f32[512, 4, 49, 49]" = torch.ops.aten.detach.default(detach);  detach = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_189: "f32[2048, 32, 49]" = torch.ops.aten.transpose.int(_unsafe_view, 1, 2);  _unsafe_view = None
    transpose_190: "f32[2048, 49, 32]" = torch.ops.aten.transpose.int(_unsafe_view_1, 1, 2);  _unsafe_view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    t_496: "f32[384, 128]" = torch.ops.aten.t.default(t);  t = None
    return [addmm_96, primals_25, primals_27, primals_28, primals_29, primals_30, primals_35, primals_36, primals_41, primals_42, primals_47, primals_48, primals_53, primals_54, primals_56, primals_57, primals_62, primals_63, primals_68, primals_69, primals_74, primals_75, primals_80, primals_81, primals_83, primals_84, primals_89, primals_90, primals_95, primals_96, primals_101, primals_102, primals_107, primals_108, primals_113, primals_114, primals_119, primals_120, primals_125, primals_126, primals_131, primals_132, primals_137, primals_138, primals_143, primals_144, primals_149, primals_150, primals_155, primals_156, primals_161, primals_162, primals_167, primals_168, primals_173, primals_174, primals_179, primals_180, primals_185, primals_186, primals_191, primals_192, primals_197, primals_198, primals_203, primals_204, primals_209, primals_210, primals_215, primals_216, primals_221, primals_222, primals_227, primals_228, primals_233, primals_234, primals_239, primals_240, primals_245, primals_246, primals_251, primals_252, primals_257, primals_258, primals_263, primals_264, primals_269, primals_270, primals_275, primals_276, primals_281, primals_282, primals_287, primals_288, primals_293, primals_294, primals_299, primals_300, primals_302, primals_303, primals_308, primals_309, primals_314, primals_315, primals_320, primals_321, primals_326, primals_327, primals_365, permute, getitem, getitem_1, getitem_2, getitem_4, getitem_5, view_3, view_7, view_11, view_16, getitem_10, getitem_11, view_17, addmm_2, view_19, view_21, getitem_13, getitem_14, view_25, view_29, view_35, bernoulli, view_40, getitem_19, getitem_20, view_41, addmm_6, view_43, bernoulli_1, _unsafe_view_8, getitem_22, getitem_23, view_47, view_48, getitem_25, getitem_26, view_52, view_56, view_60, bernoulli_2, view_65, getitem_31, getitem_32, view_66, addmm_10, view_68, bernoulli_3, view_70, getitem_34, getitem_35, view_74, view_78, view_84, bernoulli_4, view_89, getitem_40, getitem_41, view_90, addmm_14, view_92, bernoulli_5, _unsafe_view_17, getitem_43, getitem_44, view_96, view_97, getitem_46, getitem_47, view_101, view_105, view_109, bernoulli_6, view_114, getitem_52, getitem_53, view_115, addmm_18, view_117, bernoulli_7, view_119, getitem_55, getitem_56, view_123, view_127, view_133, bernoulli_8, view_138, getitem_61, getitem_62, view_139, addmm_22, view_141, bernoulli_9, view_143, getitem_64, getitem_65, view_147, view_151, view_155, bernoulli_10, view_160, getitem_70, getitem_71, view_161, addmm_26, view_163, bernoulli_11, view_165, getitem_73, getitem_74, view_169, view_173, view_179, bernoulli_12, view_184, getitem_79, getitem_80, view_185, addmm_30, view_187, bernoulli_13, view_189, getitem_82, getitem_83, view_193, view_197, view_201, bernoulli_14, view_206, getitem_88, getitem_89, view_207, addmm_34, view_209, bernoulli_15, view_211, getitem_91, getitem_92, view_215, view_219, view_225, bernoulli_16, view_230, getitem_97, getitem_98, view_231, addmm_38, view_233, bernoulli_17, view_235, getitem_100, getitem_101, view_239, view_243, view_247, bernoulli_18, view_252, getitem_106, getitem_107, view_253, addmm_42, view_255, bernoulli_19, view_257, getitem_109, getitem_110, view_261, view_265, view_271, bernoulli_20, view_276, getitem_115, getitem_116, view_277, addmm_46, view_279, bernoulli_21, view_281, getitem_118, getitem_119, view_285, view_289, view_293, bernoulli_22, view_298, getitem_124, getitem_125, view_299, addmm_50, view_301, bernoulli_23, view_303, getitem_127, getitem_128, view_307, view_311, view_317, bernoulli_24, view_322, getitem_133, getitem_134, view_323, addmm_54, view_325, bernoulli_25, view_327, getitem_136, getitem_137, view_331, view_335, view_339, bernoulli_26, view_344, getitem_142, getitem_143, view_345, addmm_58, view_347, bernoulli_27, view_349, getitem_145, getitem_146, view_353, view_357, view_363, bernoulli_28, view_368, getitem_151, getitem_152, view_369, addmm_62, view_371, bernoulli_29, view_373, getitem_154, getitem_155, view_377, view_381, view_385, bernoulli_30, view_390, getitem_160, getitem_161, view_391, addmm_66, view_393, bernoulli_31, view_395, getitem_163, getitem_164, view_399, view_403, view_409, bernoulli_32, view_414, getitem_169, getitem_170, view_415, addmm_70, view_417, bernoulli_33, view_419, getitem_172, getitem_173, view_423, view_427, view_431, bernoulli_34, view_436, getitem_178, getitem_179, view_437, addmm_74, view_439, bernoulli_35, view_441, getitem_181, getitem_182, view_445, view_449, view_455, bernoulli_36, view_460, getitem_187, getitem_188, view_461, addmm_78, view_463, bernoulli_37, view_465, getitem_190, getitem_191, view_469, view_473, view_477, bernoulli_38, view_482, getitem_196, getitem_197, view_483, addmm_82, view_485, bernoulli_39, view_487, getitem_199, getitem_200, view_491, view_495, view_501, bernoulli_40, view_506, getitem_205, getitem_206, view_507, addmm_86, view_509, bernoulli_41, _unsafe_view_90, getitem_208, getitem_209, view_513, view_514, getitem_211, getitem_212, view_518, view_522, view_526, bernoulli_42, view_531, getitem_217, getitem_218, view_532, addmm_90, view_534, bernoulli_43, view_536, getitem_220, getitem_221, view_540, view_544, view_548, bernoulli_44, view_553, getitem_226, getitem_227, view_554, addmm_94, view_556, bernoulli_45, view_558, getitem_229, getitem_230, clone_263, t_100, t_104, t_108, t_112, transpose_49, transpose_50, detach_24, transpose_51, transpose_52, t_116, t_120, t_124, t_128, transpose_55, transpose_56, detach_25, transpose_57, transpose_58, t_132, t_138, t_140, t_144, t_148, transpose_61, transpose_62, detach_26, transpose_63, transpose_64, t_152, t_156, t_160, t_164, transpose_67, transpose_68, detach_27, transpose_69, transpose_70, t_168, t_172, t_176, t_180, transpose_73, transpose_74, detach_28, transpose_75, transpose_76, t_184, t_188, t_192, t_196, transpose_79, transpose_80, detach_29, transpose_81, transpose_82, t_200, t_204, t_208, t_212, transpose_85, transpose_86, detach_30, transpose_87, transpose_88, t_216, t_220, t_224, t_228, transpose_91, transpose_92, detach_31, transpose_93, transpose_94, t_232, t_236, t_240, t_244, transpose_97, transpose_98, detach_32, transpose_99, transpose_100, t_248, t_252, t_256, t_260, transpose_103, transpose_104, detach_33, transpose_105, transpose_106, t_264, t_268, t_272, t_276, transpose_109, transpose_110, detach_34, transpose_111, transpose_112, t_280, t_284, t_288, t_292, transpose_115, transpose_116, detach_35, transpose_117, transpose_118, t_296, t_300, t_304, t_308, transpose_121, transpose_122, detach_36, transpose_123, transpose_124, t_312, t_316, t_320, t_324, transpose_127, transpose_128, detach_37, transpose_129, transpose_130, t_328, t_332, t_336, t_340, transpose_133, transpose_134, detach_38, transpose_135, transpose_136, t_344, t_348, t_352, t_356, transpose_139, transpose_140, detach_39, transpose_141, transpose_142, t_360, t_364, t_368, t_372, transpose_145, transpose_146, detach_40, transpose_147, transpose_148, t_376, t_380, t_384, t_388, transpose_151, transpose_152, detach_41, transpose_153, transpose_154, t_392, t_396, t_400, t_404, transpose_157, transpose_158, detach_42, transpose_159, transpose_160, t_408, t_412, t_416, t_420, transpose_163, transpose_164, detach_43, transpose_165, transpose_166, t_424, t_430, t_432, t_436, t_440, transpose_169, transpose_170, detach_44, transpose_171, transpose_172, t_444, t_448, t_452, t_456, transpose_175, transpose_176, detach_45, transpose_177, transpose_178, t_460, t_466, t_468, t_472, t_476, transpose_181, transpose_182, detach_46, transpose_183, transpose_184, t_480, t_484, t_488, t_492, transpose_187, transpose_188, detach_47, transpose_189, transpose_190, t_496]
    