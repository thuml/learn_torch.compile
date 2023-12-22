from __future__ import annotations



def forward(self, primals_1: "f32[96]", primals_2: "f32[96]", primals_3: "f32[96]", primals_4: "f32[96]", primals_5: "f32[96]", primals_6: "f32[96]", primals_7: "f32[96]", primals_8: "f32[96]", primals_9: "f32[96]", primals_10: "f32[96]", primals_11: "f32[96]", primals_12: "f32[96]", primals_13: "f32[96]", primals_14: "f32[96]", primals_15: "f32[96]", primals_16: "f32[96]", primals_17: "f32[96]", primals_18: "f32[96]", primals_19: "f32[96]", primals_20: "f32[96]", primals_21: "f32[96]", primals_22: "f32[96]", primals_23: "f32[96]", primals_24: "f32[96]", primals_25: "f32[96]", primals_26: "f32[96]", primals_27: "f32[96]", primals_28: "f32[96]", primals_29: "f32[96]", primals_30: "f32[96]", primals_31: "f32[96]", primals_32: "f32[96]", primals_33: "f32[96]", primals_34: "f32[96]", primals_35: "f32[96]", primals_36: "f32[96]", primals_37: "f32[192]", primals_38: "f32[192]", primals_39: "f32[192]", primals_40: "f32[192]", primals_41: "f32[192]", primals_42: "f32[192]", primals_43: "f32[192]", primals_44: "f32[192]", primals_45: "f32[192]", primals_46: "f32[192]", primals_47: "f32[192]", primals_48: "f32[192]", primals_49: "f32[192]", primals_50: "f32[192]", primals_51: "f32[192]", primals_52: "f32[192]", primals_53: "f32[192]", primals_54: "f32[192]", primals_55: "f32[192]", primals_56: "f32[192]", primals_57: "f32[192]", primals_58: "f32[192]", primals_59: "f32[192]", primals_60: "f32[192]", primals_61: "f32[192]", primals_62: "f32[192]", primals_63: "f32[192]", primals_64: "f32[192]", primals_65: "f32[192]", primals_66: "f32[192]", primals_67: "f32[192]", primals_68: "f32[192]", primals_69: "f32[192]", primals_70: "f32[192]", primals_71: "f32[192]", primals_72: "f32[192]", primals_73: "f32[384]", primals_74: "f32[384]", primals_75: "f32[384]", primals_76: "f32[384]", primals_77: "f32[384]", primals_78: "f32[384]", primals_79: "f32[384]", primals_80: "f32[384]", primals_81: "f32[384]", primals_82: "f32[384]", primals_83: "f32[384]", primals_84: "f32[384]", primals_85: "f32[384]", primals_86: "f32[384]", primals_87: "f32[384]", primals_88: "f32[384]", primals_89: "f32[384]", primals_90: "f32[384]", primals_91: "f32[384]", primals_92: "f32[384]", primals_93: "f32[384]", primals_94: "f32[384]", primals_95: "f32[384]", primals_96: "f32[384]", primals_97: "f32[384]", primals_98: "f32[384]", primals_99: "f32[384]", primals_100: "f32[384]", primals_101: "f32[384]", primals_102: "f32[384]", primals_103: "f32[384]", primals_104: "f32[384]", primals_105: "f32[384]", primals_106: "f32[384]", primals_107: "f32[384]", primals_108: "f32[384]", primals_109: "f32[384]", primals_110: "f32[384]", primals_111: "f32[384]", primals_112: "f32[384]", primals_113: "f32[384]", primals_114: "f32[384]", primals_115: "f32[384]", primals_116: "f32[384]", primals_117: "f32[384]", primals_118: "f32[384]", primals_119: "f32[384]", primals_120: "f32[384]", primals_121: "f32[384]", primals_122: "f32[384]", primals_123: "f32[384]", primals_124: "f32[384]", primals_125: "f32[384]", primals_126: "f32[384]", primals_127: "f32[384]", primals_128: "f32[384]", primals_129: "f32[384]", primals_130: "f32[384]", primals_131: "f32[384]", primals_132: "f32[384]", primals_133: "f32[384]", primals_134: "f32[384]", primals_135: "f32[384]", primals_136: "f32[384]", primals_137: "f32[384]", primals_138: "f32[384]", primals_139: "f32[384]", primals_140: "f32[384]", primals_141: "f32[384]", primals_142: "f32[384]", primals_143: "f32[384]", primals_144: "f32[384]", primals_145: "f32[384]", primals_146: "f32[384]", primals_147: "f32[384]", primals_148: "f32[384]", primals_149: "f32[384]", primals_150: "f32[384]", primals_151: "f32[384]", primals_152: "f32[384]", primals_153: "f32[384]", primals_154: "f32[384]", primals_155: "f32[384]", primals_156: "f32[384]", primals_157: "f32[384]", primals_158: "f32[384]", primals_159: "f32[384]", primals_160: "f32[384]", primals_161: "f32[384]", primals_162: "f32[384]", primals_163: "f32[384]", primals_164: "f32[384]", primals_165: "f32[384]", primals_166: "f32[384]", primals_167: "f32[384]", primals_168: "f32[384]", primals_169: "f32[384]", primals_170: "f32[384]", primals_171: "f32[384]", primals_172: "f32[384]", primals_173: "f32[384]", primals_174: "f32[384]", primals_175: "f32[384]", primals_176: "f32[384]", primals_177: "f32[384]", primals_178: "f32[384]", primals_179: "f32[384]", primals_180: "f32[384]", primals_181: "f32[768]", primals_182: "f32[768]", primals_183: "f32[768]", primals_184: "f32[768]", primals_185: "f32[768]", primals_186: "f32[768]", primals_187: "f32[768]", primals_188: "f32[768]", primals_189: "f32[768]", primals_190: "f32[768]", primals_191: "f32[768]", primals_192: "f32[768]", primals_193: "f32[768]", primals_194: "f32[768]", primals_195: "f32[768]", primals_196: "f32[768]", primals_197: "f32[768]", primals_198: "f32[768]", primals_199: "f32[768]", primals_200: "f32[768]", primals_201: "f32[768]", primals_202: "f32[768]", primals_203: "f32[768]", primals_204: "f32[768]", primals_205: "f32[768]", primals_206: "f32[768]", primals_207: "f32[768]", primals_208: "f32[768]", primals_209: "f32[768]", primals_210: "f32[768]", primals_211: "f32[768]", primals_212: "f32[768]", primals_213: "f32[768]", primals_214: "f32[768]", primals_215: "f32[768]", primals_216: "f32[768]", primals_217: "f32[768]", primals_218: "f32[768]", primals_219: "f32[96, 3, 7, 7]", primals_220: "f32[96]", primals_221: "f32[384, 96, 1, 1]", primals_222: "f32[384]", primals_223: "f32[96, 384, 1, 1]", primals_224: "f32[96]", primals_225: "f32[384, 96, 1, 1]", primals_226: "f32[384]", primals_227: "f32[96, 384, 1, 1]", primals_228: "f32[96]", primals_229: "f32[384, 96, 1, 1]", primals_230: "f32[384]", primals_231: "f32[96, 384, 1, 1]", primals_232: "f32[96]", primals_233: "f32[384, 96, 1, 1]", primals_234: "f32[384]", primals_235: "f32[96, 384, 1, 1]", primals_236: "f32[96]", primals_237: "f32[384, 96, 1, 1]", primals_238: "f32[384]", primals_239: "f32[96, 384, 1, 1]", primals_240: "f32[96]", primals_241: "f32[384, 96, 1, 1]", primals_242: "f32[384]", primals_243: "f32[96, 384, 1, 1]", primals_244: "f32[96]", primals_245: "f32[192, 96, 3, 3]", primals_246: "f32[192]", primals_247: "f32[768, 192, 1, 1]", primals_248: "f32[768]", primals_249: "f32[192, 768, 1, 1]", primals_250: "f32[192]", primals_251: "f32[768, 192, 1, 1]", primals_252: "f32[768]", primals_253: "f32[192, 768, 1, 1]", primals_254: "f32[192]", primals_255: "f32[768, 192, 1, 1]", primals_256: "f32[768]", primals_257: "f32[192, 768, 1, 1]", primals_258: "f32[192]", primals_259: "f32[768, 192, 1, 1]", primals_260: "f32[768]", primals_261: "f32[192, 768, 1, 1]", primals_262: "f32[192]", primals_263: "f32[768, 192, 1, 1]", primals_264: "f32[768]", primals_265: "f32[192, 768, 1, 1]", primals_266: "f32[192]", primals_267: "f32[768, 192, 1, 1]", primals_268: "f32[768]", primals_269: "f32[192, 768, 1, 1]", primals_270: "f32[192]", primals_271: "f32[384, 192, 3, 3]", primals_272: "f32[384]", primals_273: "f32[1536, 384, 1, 1]", primals_274: "f32[1536]", primals_275: "f32[384, 1536, 1, 1]", primals_276: "f32[384]", primals_277: "f32[1536, 384, 1, 1]", primals_278: "f32[1536]", primals_279: "f32[384, 1536, 1, 1]", primals_280: "f32[384]", primals_281: "f32[1536, 384, 1, 1]", primals_282: "f32[1536]", primals_283: "f32[384, 1536, 1, 1]", primals_284: "f32[384]", primals_285: "f32[1536, 384, 1, 1]", primals_286: "f32[1536]", primals_287: "f32[384, 1536, 1, 1]", primals_288: "f32[384]", primals_289: "f32[1536, 384, 1, 1]", primals_290: "f32[1536]", primals_291: "f32[384, 1536, 1, 1]", primals_292: "f32[384]", primals_293: "f32[1536, 384, 1, 1]", primals_294: "f32[1536]", primals_295: "f32[384, 1536, 1, 1]", primals_296: "f32[384]", primals_297: "f32[1536, 384, 1, 1]", primals_298: "f32[1536]", primals_299: "f32[384, 1536, 1, 1]", primals_300: "f32[384]", primals_301: "f32[1536, 384, 1, 1]", primals_302: "f32[1536]", primals_303: "f32[384, 1536, 1, 1]", primals_304: "f32[384]", primals_305: "f32[1536, 384, 1, 1]", primals_306: "f32[1536]", primals_307: "f32[384, 1536, 1, 1]", primals_308: "f32[384]", primals_309: "f32[1536, 384, 1, 1]", primals_310: "f32[1536]", primals_311: "f32[384, 1536, 1, 1]", primals_312: "f32[384]", primals_313: "f32[1536, 384, 1, 1]", primals_314: "f32[1536]", primals_315: "f32[384, 1536, 1, 1]", primals_316: "f32[384]", primals_317: "f32[1536, 384, 1, 1]", primals_318: "f32[1536]", primals_319: "f32[384, 1536, 1, 1]", primals_320: "f32[384]", primals_321: "f32[1536, 384, 1, 1]", primals_322: "f32[1536]", primals_323: "f32[384, 1536, 1, 1]", primals_324: "f32[384]", primals_325: "f32[1536, 384, 1, 1]", primals_326: "f32[1536]", primals_327: "f32[384, 1536, 1, 1]", primals_328: "f32[384]", primals_329: "f32[1536, 384, 1, 1]", primals_330: "f32[1536]", primals_331: "f32[384, 1536, 1, 1]", primals_332: "f32[384]", primals_333: "f32[1536, 384, 1, 1]", primals_334: "f32[1536]", primals_335: "f32[384, 1536, 1, 1]", primals_336: "f32[384]", primals_337: "f32[1536, 384, 1, 1]", primals_338: "f32[1536]", primals_339: "f32[384, 1536, 1, 1]", primals_340: "f32[384]", primals_341: "f32[1536, 384, 1, 1]", primals_342: "f32[1536]", primals_343: "f32[384, 1536, 1, 1]", primals_344: "f32[384]", primals_345: "f32[768, 384, 3, 3]", primals_346: "f32[768]", primals_347: "f32[3072, 768, 1, 1]", primals_348: "f32[3072]", primals_349: "f32[768, 3072, 1, 1]", primals_350: "f32[768]", primals_351: "f32[3072, 768, 1, 1]", primals_352: "f32[3072]", primals_353: "f32[768, 3072, 1, 1]", primals_354: "f32[768]", primals_355: "f32[3072, 768, 1, 1]", primals_356: "f32[3072]", primals_357: "f32[768, 3072, 1, 1]", primals_358: "f32[768]", primals_359: "f32[3072, 768, 1, 1]", primals_360: "f32[3072]", primals_361: "f32[768, 3072, 1, 1]", primals_362: "f32[768]", primals_363: "f32[3072, 768, 1, 1]", primals_364: "f32[3072]", primals_365: "f32[768, 3072, 1, 1]", primals_366: "f32[768]", primals_367: "f32[3072, 768, 1, 1]", primals_368: "f32[3072]", primals_369: "f32[768, 3072, 1, 1]", primals_370: "f32[768]", primals_371: "f32[1000, 768]", primals_372: "f32[1000]", primals_373: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:72, code: x = self.conv(x)
    convolution: "f32[8, 96, 56, 56]" = torch.ops.aten.convolution.default(primals_373, primals_219, primals_220, [4, 4], [2, 2], [1, 1], False, [0, 0], 1);  primals_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(convolution, [8, 1, 96, 3136])
    var_mean = torch.ops.aten.var_mean.correction(view, [2, 3], correction = 0, keepdim = True)
    getitem: "f32[8, 1, 1, 1]" = var_mean[0]
    getitem_1: "f32[8, 1, 1, 1]" = var_mean[1];  var_mean = None
    add: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
    sub: "f32[8, 1, 96, 3136]" = torch.ops.aten.sub.Tensor(view, getitem_1);  view = None
    mul: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    view_1: "f32[8, 96, 56, 56]" = torch.ops.aten.view.default(mul, [8, 96, 56, 56]);  mul = None
    unsqueeze: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_2, 0);  primals_2 = None
    unsqueeze_1: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
    unsqueeze_2: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
    unsqueeze_3: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_1, 0)
    unsqueeze_4: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
    unsqueeze_5: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
    mul_1: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(view_1, unsqueeze_5);  view_1 = unsqueeze_5 = None
    add_1: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_1, unsqueeze_2);  mul_1 = unsqueeze_2 = None
    squeeze: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_1, [2, 3]);  getitem_1 = None
    squeeze_1: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt, [2, 3]);  rsqrt = None
    alias: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze);  squeeze = None
    alias_1: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_1);  squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d: "f32[8, 96, 56, 56]" = torch.ops.aten.avg_pool2d.default(add_1, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_1: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(avg_pool2d, add_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_2: "f32[96, 1, 1]" = torch.ops.aten.view.default(primals_3, [96, 1, 1])
    mul_2: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_1, view_2);  sub_1 = view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_2: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(convolution, mul_2);  mul_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_3: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_2, [8, 1, 96, 3136])
    var_mean_1 = torch.ops.aten.var_mean.correction(view_3, [2, 3], correction = 0, keepdim = True)
    getitem_2: "f32[8, 1, 1, 1]" = var_mean_1[0]
    getitem_3: "f32[8, 1, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_3: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
    rsqrt_1: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
    sub_2: "f32[8, 1, 96, 3136]" = torch.ops.aten.sub.Tensor(view_3, getitem_3);  view_3 = None
    mul_3: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
    view_4: "f32[8, 96, 56, 56]" = torch.ops.aten.view.default(mul_3, [8, 96, 56, 56]);  mul_3 = None
    unsqueeze_6: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_5, 0);  primals_5 = None
    unsqueeze_7: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 2);  unsqueeze_6 = None
    unsqueeze_8: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None
    unsqueeze_9: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_4, 0)
    unsqueeze_10: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 2);  unsqueeze_9 = None
    unsqueeze_11: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 3);  unsqueeze_10 = None
    mul_4: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(view_4, unsqueeze_11);  view_4 = unsqueeze_11 = None
    add_4: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_4, unsqueeze_8);  mul_4 = unsqueeze_8 = None
    squeeze_2: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_3, [2, 3]);  getitem_3 = None
    squeeze_3: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_1, [2, 3]);  rsqrt_1 = None
    alias_2: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_2);  squeeze_2 = None
    alias_3: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_3);  squeeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_1: "f32[8, 384, 56, 56]" = torch.ops.aten.convolution.default(add_4, primals_221, primals_222, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_5: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_1, 0.5)
    mul_6: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_1, 0.7071067811865476)
    erf: "f32[8, 384, 56, 56]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
    add_5: "f32[8, 384, 56, 56]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_7: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(mul_5, add_5);  mul_5 = add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone: "f32[8, 384, 56, 56]" = torch.ops.aten.clone.default(mul_7);  mul_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_2: "f32[8, 96, 56, 56]" = torch.ops.aten.convolution.default(clone, primals_223, primals_224, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_1: "f32[8, 96, 56, 56]" = torch.ops.aten.clone.default(convolution_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_5: "f32[96, 1, 1]" = torch.ops.aten.view.default(primals_6, [96, 1, 1])
    mul_8: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(clone_1, view_5);  clone_1 = view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_6: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_2, mul_8);  add_2 = mul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_6: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_6, [8, 1, 96, 3136])
    var_mean_2 = torch.ops.aten.var_mean.correction(view_6, [2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[8, 1, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[8, 1, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_7: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_2: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    sub_3: "f32[8, 1, 96, 3136]" = torch.ops.aten.sub.Tensor(view_6, getitem_5);  view_6 = None
    mul_9: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
    view_7: "f32[8, 96, 56, 56]" = torch.ops.aten.view.default(mul_9, [8, 96, 56, 56]);  mul_9 = None
    unsqueeze_12: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_8, 0);  primals_8 = None
    unsqueeze_13: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, 2);  unsqueeze_12 = None
    unsqueeze_14: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 3);  unsqueeze_13 = None
    unsqueeze_15: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_7, 0)
    unsqueeze_16: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_15, 2);  unsqueeze_15 = None
    unsqueeze_17: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, 3);  unsqueeze_16 = None
    mul_10: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(view_7, unsqueeze_17);  view_7 = unsqueeze_17 = None
    add_8: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_10, unsqueeze_14);  mul_10 = unsqueeze_14 = None
    squeeze_4: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_5, [2, 3]);  getitem_5 = None
    squeeze_5: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_2, [2, 3]);  rsqrt_2 = None
    alias_4: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_4);  squeeze_4 = None
    alias_5: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_5);  squeeze_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_1: "f32[8, 96, 56, 56]" = torch.ops.aten.avg_pool2d.default(add_8, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_4: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(avg_pool2d_1, add_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_8: "f32[96, 1, 1]" = torch.ops.aten.view.default(primals_9, [96, 1, 1])
    mul_11: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, view_8);  sub_4 = view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_9: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_6, mul_11);  add_6 = mul_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_9: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_9, [8, 1, 96, 3136])
    var_mean_3 = torch.ops.aten.var_mean.correction(view_9, [2, 3], correction = 0, keepdim = True)
    getitem_6: "f32[8, 1, 1, 1]" = var_mean_3[0]
    getitem_7: "f32[8, 1, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_10: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
    rsqrt_3: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
    sub_5: "f32[8, 1, 96, 3136]" = torch.ops.aten.sub.Tensor(view_9, getitem_7);  view_9 = None
    mul_12: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
    view_10: "f32[8, 96, 56, 56]" = torch.ops.aten.view.default(mul_12, [8, 96, 56, 56]);  mul_12 = None
    unsqueeze_18: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_11, 0);  primals_11 = None
    unsqueeze_19: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, 2);  unsqueeze_18 = None
    unsqueeze_20: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_19, 3);  unsqueeze_19 = None
    unsqueeze_21: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_10, 0)
    unsqueeze_22: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_21, 2);  unsqueeze_21 = None
    unsqueeze_23: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, 3);  unsqueeze_22 = None
    mul_13: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(view_10, unsqueeze_23);  view_10 = unsqueeze_23 = None
    add_11: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_20);  mul_13 = unsqueeze_20 = None
    squeeze_6: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_7, [2, 3]);  getitem_7 = None
    squeeze_7: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_3, [2, 3]);  rsqrt_3 = None
    alias_6: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_6);  squeeze_6 = None
    alias_7: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_7);  squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_3: "f32[8, 384, 56, 56]" = torch.ops.aten.convolution.default(add_11, primals_225, primals_226, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_14: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_3, 0.5)
    mul_15: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_3, 0.7071067811865476)
    erf_1: "f32[8, 384, 56, 56]" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
    add_12: "f32[8, 384, 56, 56]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_16: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(mul_14, add_12);  mul_14 = add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_2: "f32[8, 384, 56, 56]" = torch.ops.aten.clone.default(mul_16);  mul_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_4: "f32[8, 96, 56, 56]" = torch.ops.aten.convolution.default(clone_2, primals_227, primals_228, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_3: "f32[8, 96, 56, 56]" = torch.ops.aten.clone.default(convolution_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_11: "f32[96, 1, 1]" = torch.ops.aten.view.default(primals_12, [96, 1, 1])
    mul_17: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(clone_3, view_11);  clone_3 = view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_13: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_9, mul_17);  add_9 = mul_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_12: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_13, [8, 1, 96, 3136])
    var_mean_4 = torch.ops.aten.var_mean.correction(view_12, [2, 3], correction = 0, keepdim = True)
    getitem_8: "f32[8, 1, 1, 1]" = var_mean_4[0]
    getitem_9: "f32[8, 1, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_14: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_4: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    sub_6: "f32[8, 1, 96, 3136]" = torch.ops.aten.sub.Tensor(view_12, getitem_9);  view_12 = None
    mul_18: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
    view_13: "f32[8, 96, 56, 56]" = torch.ops.aten.view.default(mul_18, [8, 96, 56, 56]);  mul_18 = None
    unsqueeze_24: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_14, 0);  primals_14 = None
    unsqueeze_25: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, 2);  unsqueeze_24 = None
    unsqueeze_26: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_25, 3);  unsqueeze_25 = None
    unsqueeze_27: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_13, 0)
    unsqueeze_28: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_27, 2);  unsqueeze_27 = None
    unsqueeze_29: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, 3);  unsqueeze_28 = None
    mul_19: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(view_13, unsqueeze_29);  view_13 = unsqueeze_29 = None
    add_15: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_19, unsqueeze_26);  mul_19 = unsqueeze_26 = None
    squeeze_8: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_9, [2, 3]);  getitem_9 = None
    squeeze_9: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_4, [2, 3]);  rsqrt_4 = None
    alias_8: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_8);  squeeze_8 = None
    alias_9: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_9);  squeeze_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_2: "f32[8, 96, 56, 56]" = torch.ops.aten.avg_pool2d.default(add_15, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_7: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(avg_pool2d_2, add_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_14: "f32[96, 1, 1]" = torch.ops.aten.view.default(primals_15, [96, 1, 1])
    mul_20: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, view_14);  sub_7 = view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_16: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_13, mul_20);  add_13 = mul_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_15: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_16, [8, 1, 96, 3136])
    var_mean_5 = torch.ops.aten.var_mean.correction(view_15, [2, 3], correction = 0, keepdim = True)
    getitem_10: "f32[8, 1, 1, 1]" = var_mean_5[0]
    getitem_11: "f32[8, 1, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_17: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
    rsqrt_5: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
    sub_8: "f32[8, 1, 96, 3136]" = torch.ops.aten.sub.Tensor(view_15, getitem_11);  view_15 = None
    mul_21: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = None
    view_16: "f32[8, 96, 56, 56]" = torch.ops.aten.view.default(mul_21, [8, 96, 56, 56]);  mul_21 = None
    unsqueeze_30: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_17, 0);  primals_17 = None
    unsqueeze_31: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, 2);  unsqueeze_30 = None
    unsqueeze_32: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_31, 3);  unsqueeze_31 = None
    unsqueeze_33: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_16, 0)
    unsqueeze_34: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_33, 2);  unsqueeze_33 = None
    unsqueeze_35: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, 3);  unsqueeze_34 = None
    mul_22: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(view_16, unsqueeze_35);  view_16 = unsqueeze_35 = None
    add_18: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_22, unsqueeze_32);  mul_22 = unsqueeze_32 = None
    squeeze_10: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_11, [2, 3]);  getitem_11 = None
    squeeze_11: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_5, [2, 3]);  rsqrt_5 = None
    alias_10: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_10);  squeeze_10 = None
    alias_11: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_11);  squeeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_5: "f32[8, 384, 56, 56]" = torch.ops.aten.convolution.default(add_18, primals_229, primals_230, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_23: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_5, 0.5)
    mul_24: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_5, 0.7071067811865476)
    erf_2: "f32[8, 384, 56, 56]" = torch.ops.aten.erf.default(mul_24);  mul_24 = None
    add_19: "f32[8, 384, 56, 56]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_25: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(mul_23, add_19);  mul_23 = add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_4: "f32[8, 384, 56, 56]" = torch.ops.aten.clone.default(mul_25);  mul_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_6: "f32[8, 96, 56, 56]" = torch.ops.aten.convolution.default(clone_4, primals_231, primals_232, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_5: "f32[8, 96, 56, 56]" = torch.ops.aten.clone.default(convolution_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_17: "f32[96, 1, 1]" = torch.ops.aten.view.default(primals_18, [96, 1, 1])
    mul_26: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(clone_5, view_17);  clone_5 = view_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_20: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_16, mul_26);  add_16 = mul_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_18: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_20, [8, 1, 96, 3136])
    var_mean_6 = torch.ops.aten.var_mean.correction(view_18, [2, 3], correction = 0, keepdim = True)
    getitem_12: "f32[8, 1, 1, 1]" = var_mean_6[0]
    getitem_13: "f32[8, 1, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_21: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_6: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_9: "f32[8, 1, 96, 3136]" = torch.ops.aten.sub.Tensor(view_18, getitem_13);  view_18 = None
    mul_27: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = None
    view_19: "f32[8, 96, 56, 56]" = torch.ops.aten.view.default(mul_27, [8, 96, 56, 56]);  mul_27 = None
    unsqueeze_36: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_20, 0);  primals_20 = None
    unsqueeze_37: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, 2);  unsqueeze_36 = None
    unsqueeze_38: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_37, 3);  unsqueeze_37 = None
    unsqueeze_39: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_19, 0)
    unsqueeze_40: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_39, 2);  unsqueeze_39 = None
    unsqueeze_41: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, 3);  unsqueeze_40 = None
    mul_28: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(view_19, unsqueeze_41);  view_19 = unsqueeze_41 = None
    add_22: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_28, unsqueeze_38);  mul_28 = unsqueeze_38 = None
    squeeze_12: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_13, [2, 3]);  getitem_13 = None
    squeeze_13: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_6, [2, 3]);  rsqrt_6 = None
    alias_12: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_12);  squeeze_12 = None
    alias_13: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_13);  squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_3: "f32[8, 96, 56, 56]" = torch.ops.aten.avg_pool2d.default(add_22, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_10: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(avg_pool2d_3, add_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_20: "f32[96, 1, 1]" = torch.ops.aten.view.default(primals_21, [96, 1, 1])
    mul_29: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_10, view_20);  sub_10 = view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_23: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_20, mul_29);  add_20 = mul_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_21: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_23, [8, 1, 96, 3136])
    var_mean_7 = torch.ops.aten.var_mean.correction(view_21, [2, 3], correction = 0, keepdim = True)
    getitem_14: "f32[8, 1, 1, 1]" = var_mean_7[0]
    getitem_15: "f32[8, 1, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_24: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
    rsqrt_7: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
    sub_11: "f32[8, 1, 96, 3136]" = torch.ops.aten.sub.Tensor(view_21, getitem_15);  view_21 = None
    mul_30: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = None
    view_22: "f32[8, 96, 56, 56]" = torch.ops.aten.view.default(mul_30, [8, 96, 56, 56]);  mul_30 = None
    unsqueeze_42: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_23, 0);  primals_23 = None
    unsqueeze_43: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, 2);  unsqueeze_42 = None
    unsqueeze_44: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_43, 3);  unsqueeze_43 = None
    unsqueeze_45: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_22, 0)
    unsqueeze_46: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_45, 2);  unsqueeze_45 = None
    unsqueeze_47: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, 3);  unsqueeze_46 = None
    mul_31: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(view_22, unsqueeze_47);  view_22 = unsqueeze_47 = None
    add_25: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_31, unsqueeze_44);  mul_31 = unsqueeze_44 = None
    squeeze_14: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_15, [2, 3]);  getitem_15 = None
    squeeze_15: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_7, [2, 3]);  rsqrt_7 = None
    alias_14: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_14);  squeeze_14 = None
    alias_15: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_15);  squeeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_7: "f32[8, 384, 56, 56]" = torch.ops.aten.convolution.default(add_25, primals_233, primals_234, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_32: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_7, 0.5)
    mul_33: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_7, 0.7071067811865476)
    erf_3: "f32[8, 384, 56, 56]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
    add_26: "f32[8, 384, 56, 56]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_34: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(mul_32, add_26);  mul_32 = add_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_6: "f32[8, 384, 56, 56]" = torch.ops.aten.clone.default(mul_34);  mul_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_8: "f32[8, 96, 56, 56]" = torch.ops.aten.convolution.default(clone_6, primals_235, primals_236, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_7: "f32[8, 96, 56, 56]" = torch.ops.aten.clone.default(convolution_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_23: "f32[96, 1, 1]" = torch.ops.aten.view.default(primals_24, [96, 1, 1])
    mul_35: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(clone_7, view_23);  clone_7 = view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_27: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_23, mul_35);  add_23 = mul_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_24: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_27, [8, 1, 96, 3136])
    var_mean_8 = torch.ops.aten.var_mean.correction(view_24, [2, 3], correction = 0, keepdim = True)
    getitem_16: "f32[8, 1, 1, 1]" = var_mean_8[0]
    getitem_17: "f32[8, 1, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_28: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_8: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_12: "f32[8, 1, 96, 3136]" = torch.ops.aten.sub.Tensor(view_24, getitem_17);  view_24 = None
    mul_36: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = None
    view_25: "f32[8, 96, 56, 56]" = torch.ops.aten.view.default(mul_36, [8, 96, 56, 56]);  mul_36 = None
    unsqueeze_48: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_26, 0);  primals_26 = None
    unsqueeze_49: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, 2);  unsqueeze_48 = None
    unsqueeze_50: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_49, 3);  unsqueeze_49 = None
    unsqueeze_51: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_25, 0)
    unsqueeze_52: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_51, 2);  unsqueeze_51 = None
    unsqueeze_53: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, 3);  unsqueeze_52 = None
    mul_37: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(view_25, unsqueeze_53);  view_25 = unsqueeze_53 = None
    add_29: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_37, unsqueeze_50);  mul_37 = unsqueeze_50 = None
    squeeze_16: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_17, [2, 3]);  getitem_17 = None
    squeeze_17: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_8, [2, 3]);  rsqrt_8 = None
    alias_16: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_16);  squeeze_16 = None
    alias_17: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_17);  squeeze_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_4: "f32[8, 96, 56, 56]" = torch.ops.aten.avg_pool2d.default(add_29, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_13: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(avg_pool2d_4, add_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_26: "f32[96, 1, 1]" = torch.ops.aten.view.default(primals_27, [96, 1, 1])
    mul_38: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_13, view_26);  sub_13 = view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_30: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_27, mul_38);  add_27 = mul_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_27: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_30, [8, 1, 96, 3136])
    var_mean_9 = torch.ops.aten.var_mean.correction(view_27, [2, 3], correction = 0, keepdim = True)
    getitem_18: "f32[8, 1, 1, 1]" = var_mean_9[0]
    getitem_19: "f32[8, 1, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_31: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
    rsqrt_9: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    sub_14: "f32[8, 1, 96, 3136]" = torch.ops.aten.sub.Tensor(view_27, getitem_19);  view_27 = None
    mul_39: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = None
    view_28: "f32[8, 96, 56, 56]" = torch.ops.aten.view.default(mul_39, [8, 96, 56, 56]);  mul_39 = None
    unsqueeze_54: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_29, 0);  primals_29 = None
    unsqueeze_55: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, 2);  unsqueeze_54 = None
    unsqueeze_56: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_55, 3);  unsqueeze_55 = None
    unsqueeze_57: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_28, 0)
    unsqueeze_58: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_57, 2);  unsqueeze_57 = None
    unsqueeze_59: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, 3);  unsqueeze_58 = None
    mul_40: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(view_28, unsqueeze_59);  view_28 = unsqueeze_59 = None
    add_32: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_40, unsqueeze_56);  mul_40 = unsqueeze_56 = None
    squeeze_18: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_19, [2, 3]);  getitem_19 = None
    squeeze_19: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_9, [2, 3]);  rsqrt_9 = None
    alias_18: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_18);  squeeze_18 = None
    alias_19: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_19);  squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_9: "f32[8, 384, 56, 56]" = torch.ops.aten.convolution.default(add_32, primals_237, primals_238, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_41: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_9, 0.5)
    mul_42: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_9, 0.7071067811865476)
    erf_4: "f32[8, 384, 56, 56]" = torch.ops.aten.erf.default(mul_42);  mul_42 = None
    add_33: "f32[8, 384, 56, 56]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_43: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(mul_41, add_33);  mul_41 = add_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_8: "f32[8, 384, 56, 56]" = torch.ops.aten.clone.default(mul_43);  mul_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_10: "f32[8, 96, 56, 56]" = torch.ops.aten.convolution.default(clone_8, primals_239, primals_240, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_9: "f32[8, 96, 56, 56]" = torch.ops.aten.clone.default(convolution_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_29: "f32[96, 1, 1]" = torch.ops.aten.view.default(primals_30, [96, 1, 1])
    mul_44: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(clone_9, view_29);  clone_9 = view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_34: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_30, mul_44);  add_30 = mul_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_30: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_34, [8, 1, 96, 3136])
    var_mean_10 = torch.ops.aten.var_mean.correction(view_30, [2, 3], correction = 0, keepdim = True)
    getitem_20: "f32[8, 1, 1, 1]" = var_mean_10[0]
    getitem_21: "f32[8, 1, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_35: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
    rsqrt_10: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
    sub_15: "f32[8, 1, 96, 3136]" = torch.ops.aten.sub.Tensor(view_30, getitem_21);  view_30 = None
    mul_45: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = None
    view_31: "f32[8, 96, 56, 56]" = torch.ops.aten.view.default(mul_45, [8, 96, 56, 56]);  mul_45 = None
    unsqueeze_60: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_32, 0);  primals_32 = None
    unsqueeze_61: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, 2);  unsqueeze_60 = None
    unsqueeze_62: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_61, 3);  unsqueeze_61 = None
    unsqueeze_63: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_31, 0)
    unsqueeze_64: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_63, 2);  unsqueeze_63 = None
    unsqueeze_65: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, 3);  unsqueeze_64 = None
    mul_46: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(view_31, unsqueeze_65);  view_31 = unsqueeze_65 = None
    add_36: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_46, unsqueeze_62);  mul_46 = unsqueeze_62 = None
    squeeze_20: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_21, [2, 3]);  getitem_21 = None
    squeeze_21: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_10, [2, 3]);  rsqrt_10 = None
    alias_20: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_20);  squeeze_20 = None
    alias_21: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_21);  squeeze_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_5: "f32[8, 96, 56, 56]" = torch.ops.aten.avg_pool2d.default(add_36, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_16: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(avg_pool2d_5, add_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_32: "f32[96, 1, 1]" = torch.ops.aten.view.default(primals_33, [96, 1, 1])
    mul_47: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_16, view_32);  sub_16 = view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_37: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_34, mul_47);  add_34 = mul_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_33: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_37, [8, 1, 96, 3136])
    var_mean_11 = torch.ops.aten.var_mean.correction(view_33, [2, 3], correction = 0, keepdim = True)
    getitem_22: "f32[8, 1, 1, 1]" = var_mean_11[0]
    getitem_23: "f32[8, 1, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_38: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_11: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_17: "f32[8, 1, 96, 3136]" = torch.ops.aten.sub.Tensor(view_33, getitem_23);  view_33 = None
    mul_48: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = None
    view_34: "f32[8, 96, 56, 56]" = torch.ops.aten.view.default(mul_48, [8, 96, 56, 56]);  mul_48 = None
    unsqueeze_66: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_35, 0);  primals_35 = None
    unsqueeze_67: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, 2);  unsqueeze_66 = None
    unsqueeze_68: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_67, 3);  unsqueeze_67 = None
    unsqueeze_69: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_34, 0)
    unsqueeze_70: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_69, 2);  unsqueeze_69 = None
    unsqueeze_71: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, 3);  unsqueeze_70 = None
    mul_49: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(view_34, unsqueeze_71);  view_34 = unsqueeze_71 = None
    add_39: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_49, unsqueeze_68);  mul_49 = unsqueeze_68 = None
    squeeze_22: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_23, [2, 3]);  getitem_23 = None
    squeeze_23: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_11, [2, 3]);  rsqrt_11 = None
    alias_22: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_22);  squeeze_22 = None
    alias_23: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_23);  squeeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_11: "f32[8, 384, 56, 56]" = torch.ops.aten.convolution.default(add_39, primals_241, primals_242, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_50: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_11, 0.5)
    mul_51: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_11, 0.7071067811865476)
    erf_5: "f32[8, 384, 56, 56]" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
    add_40: "f32[8, 384, 56, 56]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_52: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(mul_50, add_40);  mul_50 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_10: "f32[8, 384, 56, 56]" = torch.ops.aten.clone.default(mul_52);  mul_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_12: "f32[8, 96, 56, 56]" = torch.ops.aten.convolution.default(clone_10, primals_243, primals_244, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_11: "f32[8, 96, 56, 56]" = torch.ops.aten.clone.default(convolution_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_35: "f32[96, 1, 1]" = torch.ops.aten.view.default(primals_36, [96, 1, 1])
    mul_53: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(clone_11, view_35);  clone_11 = view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_41: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_37, mul_53);  add_37 = mul_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:103, code: x = self.conv(x)
    convolution_13: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(add_41, primals_245, primals_246, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  primals_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_36: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(convolution_13, [8, 1, 192, 784])
    var_mean_12 = torch.ops.aten.var_mean.correction(view_36, [2, 3], correction = 0, keepdim = True)
    getitem_24: "f32[8, 1, 1, 1]" = var_mean_12[0]
    getitem_25: "f32[8, 1, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_42: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
    rsqrt_12: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_18: "f32[8, 1, 192, 784]" = torch.ops.aten.sub.Tensor(view_36, getitem_25);  view_36 = None
    mul_54: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = None
    view_37: "f32[8, 192, 28, 28]" = torch.ops.aten.view.default(mul_54, [8, 192, 28, 28]);  mul_54 = None
    unsqueeze_72: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_38, 0);  primals_38 = None
    unsqueeze_73: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, 2);  unsqueeze_72 = None
    unsqueeze_74: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_73, 3);  unsqueeze_73 = None
    unsqueeze_75: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_37, 0)
    unsqueeze_76: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_75, 2);  unsqueeze_75 = None
    unsqueeze_77: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, 3);  unsqueeze_76 = None
    mul_55: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(view_37, unsqueeze_77);  view_37 = unsqueeze_77 = None
    add_43: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_74);  mul_55 = unsqueeze_74 = None
    squeeze_24: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_25, [2, 3]);  getitem_25 = None
    squeeze_25: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_12, [2, 3]);  rsqrt_12 = None
    alias_24: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_24);  squeeze_24 = None
    alias_25: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_25);  squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_6: "f32[8, 192, 28, 28]" = torch.ops.aten.avg_pool2d.default(add_43, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_19: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(avg_pool2d_6, add_43)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_38: "f32[192, 1, 1]" = torch.ops.aten.view.default(primals_39, [192, 1, 1])
    mul_56: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_19, view_38);  sub_19 = view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_44: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(convolution_13, mul_56);  mul_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_39: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_44, [8, 1, 192, 784])
    var_mean_13 = torch.ops.aten.var_mean.correction(view_39, [2, 3], correction = 0, keepdim = True)
    getitem_26: "f32[8, 1, 1, 1]" = var_mean_13[0]
    getitem_27: "f32[8, 1, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_45: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
    rsqrt_13: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
    sub_20: "f32[8, 1, 192, 784]" = torch.ops.aten.sub.Tensor(view_39, getitem_27);  view_39 = None
    mul_57: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = None
    view_40: "f32[8, 192, 28, 28]" = torch.ops.aten.view.default(mul_57, [8, 192, 28, 28]);  mul_57 = None
    unsqueeze_78: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_41, 0);  primals_41 = None
    unsqueeze_79: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, 2);  unsqueeze_78 = None
    unsqueeze_80: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_79, 3);  unsqueeze_79 = None
    unsqueeze_81: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_40, 0)
    unsqueeze_82: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_81, 2);  unsqueeze_81 = None
    unsqueeze_83: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, 3);  unsqueeze_82 = None
    mul_58: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(view_40, unsqueeze_83);  view_40 = unsqueeze_83 = None
    add_46: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_58, unsqueeze_80);  mul_58 = unsqueeze_80 = None
    squeeze_26: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_27, [2, 3]);  getitem_27 = None
    squeeze_27: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_13, [2, 3]);  rsqrt_13 = None
    alias_26: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_26);  squeeze_26 = None
    alias_27: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_27);  squeeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_14: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_46, primals_247, primals_248, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_59: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_14, 0.5)
    mul_60: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_14, 0.7071067811865476)
    erf_6: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_60);  mul_60 = None
    add_47: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_61: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_59, add_47);  mul_59 = add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_12: "f32[8, 768, 28, 28]" = torch.ops.aten.clone.default(mul_61);  mul_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_15: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(clone_12, primals_249, primals_250, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_13: "f32[8, 192, 28, 28]" = torch.ops.aten.clone.default(convolution_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_41: "f32[192, 1, 1]" = torch.ops.aten.view.default(primals_42, [192, 1, 1])
    mul_62: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(clone_13, view_41);  clone_13 = view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_48: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_44, mul_62);  add_44 = mul_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_42: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_48, [8, 1, 192, 784])
    var_mean_14 = torch.ops.aten.var_mean.correction(view_42, [2, 3], correction = 0, keepdim = True)
    getitem_28: "f32[8, 1, 1, 1]" = var_mean_14[0]
    getitem_29: "f32[8, 1, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_49: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_14: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    sub_21: "f32[8, 1, 192, 784]" = torch.ops.aten.sub.Tensor(view_42, getitem_29);  view_42 = None
    mul_63: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = None
    view_43: "f32[8, 192, 28, 28]" = torch.ops.aten.view.default(mul_63, [8, 192, 28, 28]);  mul_63 = None
    unsqueeze_84: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_44, 0);  primals_44 = None
    unsqueeze_85: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, 2);  unsqueeze_84 = None
    unsqueeze_86: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_85, 3);  unsqueeze_85 = None
    unsqueeze_87: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_43, 0)
    unsqueeze_88: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_87, 2);  unsqueeze_87 = None
    unsqueeze_89: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, 3);  unsqueeze_88 = None
    mul_64: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(view_43, unsqueeze_89);  view_43 = unsqueeze_89 = None
    add_50: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_64, unsqueeze_86);  mul_64 = unsqueeze_86 = None
    squeeze_28: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_29, [2, 3]);  getitem_29 = None
    squeeze_29: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_14, [2, 3]);  rsqrt_14 = None
    alias_28: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_28);  squeeze_28 = None
    alias_29: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_29);  squeeze_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_7: "f32[8, 192, 28, 28]" = torch.ops.aten.avg_pool2d.default(add_50, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_22: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(avg_pool2d_7, add_50)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_44: "f32[192, 1, 1]" = torch.ops.aten.view.default(primals_45, [192, 1, 1])
    mul_65: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_22, view_44);  sub_22 = view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_51: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_48, mul_65);  add_48 = mul_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_45: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_51, [8, 1, 192, 784])
    var_mean_15 = torch.ops.aten.var_mean.correction(view_45, [2, 3], correction = 0, keepdim = True)
    getitem_30: "f32[8, 1, 1, 1]" = var_mean_15[0]
    getitem_31: "f32[8, 1, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_52: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
    rsqrt_15: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_23: "f32[8, 1, 192, 784]" = torch.ops.aten.sub.Tensor(view_45, getitem_31);  view_45 = None
    mul_66: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = None
    view_46: "f32[8, 192, 28, 28]" = torch.ops.aten.view.default(mul_66, [8, 192, 28, 28]);  mul_66 = None
    unsqueeze_90: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_47, 0);  primals_47 = None
    unsqueeze_91: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, 2);  unsqueeze_90 = None
    unsqueeze_92: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_91, 3);  unsqueeze_91 = None
    unsqueeze_93: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_46, 0)
    unsqueeze_94: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_93, 2);  unsqueeze_93 = None
    unsqueeze_95: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, 3);  unsqueeze_94 = None
    mul_67: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(view_46, unsqueeze_95);  view_46 = unsqueeze_95 = None
    add_53: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_67, unsqueeze_92);  mul_67 = unsqueeze_92 = None
    squeeze_30: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_31, [2, 3]);  getitem_31 = None
    squeeze_31: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_15, [2, 3]);  rsqrt_15 = None
    alias_30: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_30);  squeeze_30 = None
    alias_31: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_31);  squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_16: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_53, primals_251, primals_252, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_68: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_16, 0.5)
    mul_69: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_16, 0.7071067811865476)
    erf_7: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
    add_54: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_70: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_68, add_54);  mul_68 = add_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_14: "f32[8, 768, 28, 28]" = torch.ops.aten.clone.default(mul_70);  mul_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_17: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(clone_14, primals_253, primals_254, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_15: "f32[8, 192, 28, 28]" = torch.ops.aten.clone.default(convolution_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_47: "f32[192, 1, 1]" = torch.ops.aten.view.default(primals_48, [192, 1, 1])
    mul_71: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(clone_15, view_47);  clone_15 = view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_55: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_51, mul_71);  add_51 = mul_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_48: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_55, [8, 1, 192, 784])
    var_mean_16 = torch.ops.aten.var_mean.correction(view_48, [2, 3], correction = 0, keepdim = True)
    getitem_32: "f32[8, 1, 1, 1]" = var_mean_16[0]
    getitem_33: "f32[8, 1, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_56: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
    rsqrt_16: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_24: "f32[8, 1, 192, 784]" = torch.ops.aten.sub.Tensor(view_48, getitem_33);  view_48 = None
    mul_72: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = None
    view_49: "f32[8, 192, 28, 28]" = torch.ops.aten.view.default(mul_72, [8, 192, 28, 28]);  mul_72 = None
    unsqueeze_96: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_50, 0);  primals_50 = None
    unsqueeze_97: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, 2);  unsqueeze_96 = None
    unsqueeze_98: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_97, 3);  unsqueeze_97 = None
    unsqueeze_99: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_49, 0)
    unsqueeze_100: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_99, 2);  unsqueeze_99 = None
    unsqueeze_101: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, 3);  unsqueeze_100 = None
    mul_73: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(view_49, unsqueeze_101);  view_49 = unsqueeze_101 = None
    add_57: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_73, unsqueeze_98);  mul_73 = unsqueeze_98 = None
    squeeze_32: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_33, [2, 3]);  getitem_33 = None
    squeeze_33: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_16, [2, 3]);  rsqrt_16 = None
    alias_32: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_32);  squeeze_32 = None
    alias_33: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_33);  squeeze_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_8: "f32[8, 192, 28, 28]" = torch.ops.aten.avg_pool2d.default(add_57, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_25: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(avg_pool2d_8, add_57)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_50: "f32[192, 1, 1]" = torch.ops.aten.view.default(primals_51, [192, 1, 1])
    mul_74: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_25, view_50);  sub_25 = view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_58: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_55, mul_74);  add_55 = mul_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_51: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_58, [8, 1, 192, 784])
    var_mean_17 = torch.ops.aten.var_mean.correction(view_51, [2, 3], correction = 0, keepdim = True)
    getitem_34: "f32[8, 1, 1, 1]" = var_mean_17[0]
    getitem_35: "f32[8, 1, 1, 1]" = var_mean_17[1];  var_mean_17 = None
    add_59: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
    rsqrt_17: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
    sub_26: "f32[8, 1, 192, 784]" = torch.ops.aten.sub.Tensor(view_51, getitem_35);  view_51 = None
    mul_75: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = None
    view_52: "f32[8, 192, 28, 28]" = torch.ops.aten.view.default(mul_75, [8, 192, 28, 28]);  mul_75 = None
    unsqueeze_102: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_53, 0);  primals_53 = None
    unsqueeze_103: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, 2);  unsqueeze_102 = None
    unsqueeze_104: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_103, 3);  unsqueeze_103 = None
    unsqueeze_105: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_52, 0)
    unsqueeze_106: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_105, 2);  unsqueeze_105 = None
    unsqueeze_107: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, 3);  unsqueeze_106 = None
    mul_76: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(view_52, unsqueeze_107);  view_52 = unsqueeze_107 = None
    add_60: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_76, unsqueeze_104);  mul_76 = unsqueeze_104 = None
    squeeze_34: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_35, [2, 3]);  getitem_35 = None
    squeeze_35: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_17, [2, 3]);  rsqrt_17 = None
    alias_34: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_34);  squeeze_34 = None
    alias_35: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_35);  squeeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_18: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_60, primals_255, primals_256, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_77: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_18, 0.5)
    mul_78: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_18, 0.7071067811865476)
    erf_8: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_78);  mul_78 = None
    add_61: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_79: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_77, add_61);  mul_77 = add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_16: "f32[8, 768, 28, 28]" = torch.ops.aten.clone.default(mul_79);  mul_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_19: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(clone_16, primals_257, primals_258, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_17: "f32[8, 192, 28, 28]" = torch.ops.aten.clone.default(convolution_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_53: "f32[192, 1, 1]" = torch.ops.aten.view.default(primals_54, [192, 1, 1])
    mul_80: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(clone_17, view_53);  clone_17 = view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_62: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_58, mul_80);  add_58 = mul_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_54: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_62, [8, 1, 192, 784])
    var_mean_18 = torch.ops.aten.var_mean.correction(view_54, [2, 3], correction = 0, keepdim = True)
    getitem_36: "f32[8, 1, 1, 1]" = var_mean_18[0]
    getitem_37: "f32[8, 1, 1, 1]" = var_mean_18[1];  var_mean_18 = None
    add_63: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
    rsqrt_18: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    sub_27: "f32[8, 1, 192, 784]" = torch.ops.aten.sub.Tensor(view_54, getitem_37);  view_54 = None
    mul_81: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = None
    view_55: "f32[8, 192, 28, 28]" = torch.ops.aten.view.default(mul_81, [8, 192, 28, 28]);  mul_81 = None
    unsqueeze_108: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_56, 0);  primals_56 = None
    unsqueeze_109: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, 2);  unsqueeze_108 = None
    unsqueeze_110: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_109, 3);  unsqueeze_109 = None
    unsqueeze_111: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_55, 0)
    unsqueeze_112: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_111, 2);  unsqueeze_111 = None
    unsqueeze_113: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, 3);  unsqueeze_112 = None
    mul_82: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(view_55, unsqueeze_113);  view_55 = unsqueeze_113 = None
    add_64: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_82, unsqueeze_110);  mul_82 = unsqueeze_110 = None
    squeeze_36: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_37, [2, 3]);  getitem_37 = None
    squeeze_37: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_18, [2, 3]);  rsqrt_18 = None
    alias_36: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_36);  squeeze_36 = None
    alias_37: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_37);  squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_9: "f32[8, 192, 28, 28]" = torch.ops.aten.avg_pool2d.default(add_64, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_28: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(avg_pool2d_9, add_64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_56: "f32[192, 1, 1]" = torch.ops.aten.view.default(primals_57, [192, 1, 1])
    mul_83: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_28, view_56);  sub_28 = view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_65: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_62, mul_83);  add_62 = mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_57: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_65, [8, 1, 192, 784])
    var_mean_19 = torch.ops.aten.var_mean.correction(view_57, [2, 3], correction = 0, keepdim = True)
    getitem_38: "f32[8, 1, 1, 1]" = var_mean_19[0]
    getitem_39: "f32[8, 1, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_66: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
    rsqrt_19: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    sub_29: "f32[8, 1, 192, 784]" = torch.ops.aten.sub.Tensor(view_57, getitem_39);  view_57 = None
    mul_84: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = None
    view_58: "f32[8, 192, 28, 28]" = torch.ops.aten.view.default(mul_84, [8, 192, 28, 28]);  mul_84 = None
    unsqueeze_114: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_59, 0);  primals_59 = None
    unsqueeze_115: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, 2);  unsqueeze_114 = None
    unsqueeze_116: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_115, 3);  unsqueeze_115 = None
    unsqueeze_117: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_58, 0)
    unsqueeze_118: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_117, 2);  unsqueeze_117 = None
    unsqueeze_119: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, 3);  unsqueeze_118 = None
    mul_85: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(view_58, unsqueeze_119);  view_58 = unsqueeze_119 = None
    add_67: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_85, unsqueeze_116);  mul_85 = unsqueeze_116 = None
    squeeze_38: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_39, [2, 3]);  getitem_39 = None
    squeeze_39: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_19, [2, 3]);  rsqrt_19 = None
    alias_38: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_38);  squeeze_38 = None
    alias_39: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_39);  squeeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_20: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_67, primals_259, primals_260, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_86: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_20, 0.5)
    mul_87: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_20, 0.7071067811865476)
    erf_9: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
    add_68: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_88: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_86, add_68);  mul_86 = add_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_18: "f32[8, 768, 28, 28]" = torch.ops.aten.clone.default(mul_88);  mul_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_21: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(clone_18, primals_261, primals_262, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_19: "f32[8, 192, 28, 28]" = torch.ops.aten.clone.default(convolution_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_59: "f32[192, 1, 1]" = torch.ops.aten.view.default(primals_60, [192, 1, 1])
    mul_89: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(clone_19, view_59);  clone_19 = view_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_69: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_65, mul_89);  add_65 = mul_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_60: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_69, [8, 1, 192, 784])
    var_mean_20 = torch.ops.aten.var_mean.correction(view_60, [2, 3], correction = 0, keepdim = True)
    getitem_40: "f32[8, 1, 1, 1]" = var_mean_20[0]
    getitem_41: "f32[8, 1, 1, 1]" = var_mean_20[1];  var_mean_20 = None
    add_70: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
    rsqrt_20: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    sub_30: "f32[8, 1, 192, 784]" = torch.ops.aten.sub.Tensor(view_60, getitem_41);  view_60 = None
    mul_90: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = None
    view_61: "f32[8, 192, 28, 28]" = torch.ops.aten.view.default(mul_90, [8, 192, 28, 28]);  mul_90 = None
    unsqueeze_120: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_62, 0);  primals_62 = None
    unsqueeze_121: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, 2);  unsqueeze_120 = None
    unsqueeze_122: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_121, 3);  unsqueeze_121 = None
    unsqueeze_123: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_61, 0)
    unsqueeze_124: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_123, 2);  unsqueeze_123 = None
    unsqueeze_125: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, 3);  unsqueeze_124 = None
    mul_91: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(view_61, unsqueeze_125);  view_61 = unsqueeze_125 = None
    add_71: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_91, unsqueeze_122);  mul_91 = unsqueeze_122 = None
    squeeze_40: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_41, [2, 3]);  getitem_41 = None
    squeeze_41: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_20, [2, 3]);  rsqrt_20 = None
    alias_40: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_40);  squeeze_40 = None
    alias_41: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_41);  squeeze_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_10: "f32[8, 192, 28, 28]" = torch.ops.aten.avg_pool2d.default(add_71, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_31: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(avg_pool2d_10, add_71)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_62: "f32[192, 1, 1]" = torch.ops.aten.view.default(primals_63, [192, 1, 1])
    mul_92: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_31, view_62);  sub_31 = view_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_72: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_69, mul_92);  add_69 = mul_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_63: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_72, [8, 1, 192, 784])
    var_mean_21 = torch.ops.aten.var_mean.correction(view_63, [2, 3], correction = 0, keepdim = True)
    getitem_42: "f32[8, 1, 1, 1]" = var_mean_21[0]
    getitem_43: "f32[8, 1, 1, 1]" = var_mean_21[1];  var_mean_21 = None
    add_73: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
    rsqrt_21: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    sub_32: "f32[8, 1, 192, 784]" = torch.ops.aten.sub.Tensor(view_63, getitem_43);  view_63 = None
    mul_93: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = None
    view_64: "f32[8, 192, 28, 28]" = torch.ops.aten.view.default(mul_93, [8, 192, 28, 28]);  mul_93 = None
    unsqueeze_126: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_65, 0);  primals_65 = None
    unsqueeze_127: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, 2);  unsqueeze_126 = None
    unsqueeze_128: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_127, 3);  unsqueeze_127 = None
    unsqueeze_129: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_64, 0)
    unsqueeze_130: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_129, 2);  unsqueeze_129 = None
    unsqueeze_131: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, 3);  unsqueeze_130 = None
    mul_94: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(view_64, unsqueeze_131);  view_64 = unsqueeze_131 = None
    add_74: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_94, unsqueeze_128);  mul_94 = unsqueeze_128 = None
    squeeze_42: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_43, [2, 3]);  getitem_43 = None
    squeeze_43: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_21, [2, 3]);  rsqrt_21 = None
    alias_42: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_42);  squeeze_42 = None
    alias_43: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_43);  squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_22: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_74, primals_263, primals_264, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_95: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_22, 0.5)
    mul_96: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_22, 0.7071067811865476)
    erf_10: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_96);  mul_96 = None
    add_75: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_97: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_95, add_75);  mul_95 = add_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_20: "f32[8, 768, 28, 28]" = torch.ops.aten.clone.default(mul_97);  mul_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_23: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(clone_20, primals_265, primals_266, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_21: "f32[8, 192, 28, 28]" = torch.ops.aten.clone.default(convolution_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_65: "f32[192, 1, 1]" = torch.ops.aten.view.default(primals_66, [192, 1, 1])
    mul_98: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(clone_21, view_65);  clone_21 = view_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_76: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_72, mul_98);  add_72 = mul_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_66: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_76, [8, 1, 192, 784])
    var_mean_22 = torch.ops.aten.var_mean.correction(view_66, [2, 3], correction = 0, keepdim = True)
    getitem_44: "f32[8, 1, 1, 1]" = var_mean_22[0]
    getitem_45: "f32[8, 1, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_77: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
    rsqrt_22: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
    sub_33: "f32[8, 1, 192, 784]" = torch.ops.aten.sub.Tensor(view_66, getitem_45);  view_66 = None
    mul_99: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = None
    view_67: "f32[8, 192, 28, 28]" = torch.ops.aten.view.default(mul_99, [8, 192, 28, 28]);  mul_99 = None
    unsqueeze_132: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_68, 0);  primals_68 = None
    unsqueeze_133: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, 2);  unsqueeze_132 = None
    unsqueeze_134: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_133, 3);  unsqueeze_133 = None
    unsqueeze_135: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_67, 0)
    unsqueeze_136: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_135, 2);  unsqueeze_135 = None
    unsqueeze_137: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, 3);  unsqueeze_136 = None
    mul_100: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(view_67, unsqueeze_137);  view_67 = unsqueeze_137 = None
    add_78: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_100, unsqueeze_134);  mul_100 = unsqueeze_134 = None
    squeeze_44: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_45, [2, 3]);  getitem_45 = None
    squeeze_45: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_22, [2, 3]);  rsqrt_22 = None
    alias_44: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_44);  squeeze_44 = None
    alias_45: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_45);  squeeze_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_11: "f32[8, 192, 28, 28]" = torch.ops.aten.avg_pool2d.default(add_78, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_34: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(avg_pool2d_11, add_78)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_68: "f32[192, 1, 1]" = torch.ops.aten.view.default(primals_69, [192, 1, 1])
    mul_101: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_34, view_68);  sub_34 = view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_79: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_76, mul_101);  add_76 = mul_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_69: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_79, [8, 1, 192, 784])
    var_mean_23 = torch.ops.aten.var_mean.correction(view_69, [2, 3], correction = 0, keepdim = True)
    getitem_46: "f32[8, 1, 1, 1]" = var_mean_23[0]
    getitem_47: "f32[8, 1, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_80: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
    rsqrt_23: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
    sub_35: "f32[8, 1, 192, 784]" = torch.ops.aten.sub.Tensor(view_69, getitem_47);  view_69 = None
    mul_102: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = None
    view_70: "f32[8, 192, 28, 28]" = torch.ops.aten.view.default(mul_102, [8, 192, 28, 28]);  mul_102 = None
    unsqueeze_138: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_71, 0);  primals_71 = None
    unsqueeze_139: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, 2);  unsqueeze_138 = None
    unsqueeze_140: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_139, 3);  unsqueeze_139 = None
    unsqueeze_141: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_70, 0)
    unsqueeze_142: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_141, 2);  unsqueeze_141 = None
    unsqueeze_143: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, 3);  unsqueeze_142 = None
    mul_103: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(view_70, unsqueeze_143);  view_70 = unsqueeze_143 = None
    add_81: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_103, unsqueeze_140);  mul_103 = unsqueeze_140 = None
    squeeze_46: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_47, [2, 3]);  getitem_47 = None
    squeeze_47: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_23, [2, 3]);  rsqrt_23 = None
    alias_46: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_46);  squeeze_46 = None
    alias_47: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_47);  squeeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_24: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_81, primals_267, primals_268, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_104: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_24, 0.5)
    mul_105: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_24, 0.7071067811865476)
    erf_11: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_105);  mul_105 = None
    add_82: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_106: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_104, add_82);  mul_104 = add_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_22: "f32[8, 768, 28, 28]" = torch.ops.aten.clone.default(mul_106);  mul_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_25: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(clone_22, primals_269, primals_270, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_23: "f32[8, 192, 28, 28]" = torch.ops.aten.clone.default(convolution_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_71: "f32[192, 1, 1]" = torch.ops.aten.view.default(primals_72, [192, 1, 1])
    mul_107: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(clone_23, view_71);  clone_23 = view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_83: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_79, mul_107);  add_79 = mul_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:103, code: x = self.conv(x)
    convolution_26: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(add_83, primals_271, primals_272, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  primals_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_72: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(convolution_26, [8, 1, 384, 196])
    var_mean_24 = torch.ops.aten.var_mean.correction(view_72, [2, 3], correction = 0, keepdim = True)
    getitem_48: "f32[8, 1, 1, 1]" = var_mean_24[0]
    getitem_49: "f32[8, 1, 1, 1]" = var_mean_24[1];  var_mean_24 = None
    add_84: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
    rsqrt_24: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_36: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_72, getitem_49);  view_72 = None
    mul_108: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = None
    view_73: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_108, [8, 384, 14, 14]);  mul_108 = None
    unsqueeze_144: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_74, 0);  primals_74 = None
    unsqueeze_145: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, 2);  unsqueeze_144 = None
    unsqueeze_146: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_145, 3);  unsqueeze_145 = None
    unsqueeze_147: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_73, 0)
    unsqueeze_148: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_147, 2);  unsqueeze_147 = None
    unsqueeze_149: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, 3);  unsqueeze_148 = None
    mul_109: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_73, unsqueeze_149);  view_73 = unsqueeze_149 = None
    add_85: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_109, unsqueeze_146);  mul_109 = unsqueeze_146 = None
    squeeze_48: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_49, [2, 3]);  getitem_49 = None
    squeeze_49: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_24, [2, 3]);  rsqrt_24 = None
    alias_48: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_48);  squeeze_48 = None
    alias_49: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_49);  squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_12: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d.default(add_85, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_37: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_12, add_85)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_74: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_75, [384, 1, 1])
    mul_110: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_37, view_74);  sub_37 = view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_86: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(convolution_26, mul_110);  mul_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_75: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_86, [8, 1, 384, 196])
    var_mean_25 = torch.ops.aten.var_mean.correction(view_75, [2, 3], correction = 0, keepdim = True)
    getitem_50: "f32[8, 1, 1, 1]" = var_mean_25[0]
    getitem_51: "f32[8, 1, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_87: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
    rsqrt_25: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
    sub_38: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_75, getitem_51);  view_75 = None
    mul_111: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_25);  sub_38 = None
    view_76: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_111, [8, 384, 14, 14]);  mul_111 = None
    unsqueeze_150: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_77, 0);  primals_77 = None
    unsqueeze_151: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, 2);  unsqueeze_150 = None
    unsqueeze_152: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_151, 3);  unsqueeze_151 = None
    unsqueeze_153: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_76, 0)
    unsqueeze_154: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_153, 2);  unsqueeze_153 = None
    unsqueeze_155: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, 3);  unsqueeze_154 = None
    mul_112: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_76, unsqueeze_155);  view_76 = unsqueeze_155 = None
    add_88: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_112, unsqueeze_152);  mul_112 = unsqueeze_152 = None
    squeeze_50: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_51, [2, 3]);  getitem_51 = None
    squeeze_51: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_25, [2, 3]);  rsqrt_25 = None
    alias_50: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_50);  squeeze_50 = None
    alias_51: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_51);  squeeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_27: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_88, primals_273, primals_274, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_113: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_27, 0.5)
    mul_114: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_27, 0.7071067811865476)
    erf_12: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_114);  mul_114 = None
    add_89: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_115: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_113, add_89);  mul_113 = add_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_24: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_115);  mul_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_28: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_24, primals_275, primals_276, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_25: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_77: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_78, [384, 1, 1])
    mul_116: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_25, view_77);  clone_25 = view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_90: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_86, mul_116);  add_86 = mul_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_78: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_90, [8, 1, 384, 196])
    var_mean_26 = torch.ops.aten.var_mean.correction(view_78, [2, 3], correction = 0, keepdim = True)
    getitem_52: "f32[8, 1, 1, 1]" = var_mean_26[0]
    getitem_53: "f32[8, 1, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_91: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
    rsqrt_26: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
    sub_39: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_78, getitem_53);  view_78 = None
    mul_117: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_26);  sub_39 = None
    view_79: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_117, [8, 384, 14, 14]);  mul_117 = None
    unsqueeze_156: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_80, 0);  primals_80 = None
    unsqueeze_157: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, 2);  unsqueeze_156 = None
    unsqueeze_158: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_157, 3);  unsqueeze_157 = None
    unsqueeze_159: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_79, 0)
    unsqueeze_160: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_159, 2);  unsqueeze_159 = None
    unsqueeze_161: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, 3);  unsqueeze_160 = None
    mul_118: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_79, unsqueeze_161);  view_79 = unsqueeze_161 = None
    add_92: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_158);  mul_118 = unsqueeze_158 = None
    squeeze_52: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_53, [2, 3]);  getitem_53 = None
    squeeze_53: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_26, [2, 3]);  rsqrt_26 = None
    alias_52: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_52);  squeeze_52 = None
    alias_53: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_53);  squeeze_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_13: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d.default(add_92, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_40: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_13, add_92)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_80: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_81, [384, 1, 1])
    mul_119: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_40, view_80);  sub_40 = view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_93: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_90, mul_119);  add_90 = mul_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_81: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_93, [8, 1, 384, 196])
    var_mean_27 = torch.ops.aten.var_mean.correction(view_81, [2, 3], correction = 0, keepdim = True)
    getitem_54: "f32[8, 1, 1, 1]" = var_mean_27[0]
    getitem_55: "f32[8, 1, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_94: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
    rsqrt_27: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
    sub_41: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_81, getitem_55);  view_81 = None
    mul_120: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_27);  sub_41 = None
    view_82: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_120, [8, 384, 14, 14]);  mul_120 = None
    unsqueeze_162: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_83, 0);  primals_83 = None
    unsqueeze_163: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, 2);  unsqueeze_162 = None
    unsqueeze_164: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_163, 3);  unsqueeze_163 = None
    unsqueeze_165: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_82, 0)
    unsqueeze_166: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_165, 2);  unsqueeze_165 = None
    unsqueeze_167: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, 3);  unsqueeze_166 = None
    mul_121: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_82, unsqueeze_167);  view_82 = unsqueeze_167 = None
    add_95: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_121, unsqueeze_164);  mul_121 = unsqueeze_164 = None
    squeeze_54: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_55, [2, 3]);  getitem_55 = None
    squeeze_55: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_27, [2, 3]);  rsqrt_27 = None
    alias_54: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_54);  squeeze_54 = None
    alias_55: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_55);  squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_29: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_95, primals_277, primals_278, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_122: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_29, 0.5)
    mul_123: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_29, 0.7071067811865476)
    erf_13: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_123);  mul_123 = None
    add_96: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_124: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_122, add_96);  mul_122 = add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_26: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_124);  mul_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_30: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_26, primals_279, primals_280, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_27: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_83: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_84, [384, 1, 1])
    mul_125: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_27, view_83);  clone_27 = view_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_97: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_93, mul_125);  add_93 = mul_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_84: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_97, [8, 1, 384, 196])
    var_mean_28 = torch.ops.aten.var_mean.correction(view_84, [2, 3], correction = 0, keepdim = True)
    getitem_56: "f32[8, 1, 1, 1]" = var_mean_28[0]
    getitem_57: "f32[8, 1, 1, 1]" = var_mean_28[1];  var_mean_28 = None
    add_98: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
    rsqrt_28: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    sub_42: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_84, getitem_57);  view_84 = None
    mul_126: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_28);  sub_42 = None
    view_85: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_126, [8, 384, 14, 14]);  mul_126 = None
    unsqueeze_168: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_86, 0);  primals_86 = None
    unsqueeze_169: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, 2);  unsqueeze_168 = None
    unsqueeze_170: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_169, 3);  unsqueeze_169 = None
    unsqueeze_171: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_85, 0)
    unsqueeze_172: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_171, 2);  unsqueeze_171 = None
    unsqueeze_173: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, 3);  unsqueeze_172 = None
    mul_127: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_85, unsqueeze_173);  view_85 = unsqueeze_173 = None
    add_99: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_127, unsqueeze_170);  mul_127 = unsqueeze_170 = None
    squeeze_56: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_57, [2, 3]);  getitem_57 = None
    squeeze_57: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_28, [2, 3]);  rsqrt_28 = None
    alias_56: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_56);  squeeze_56 = None
    alias_57: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_57);  squeeze_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_14: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d.default(add_99, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_43: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_14, add_99)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_86: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_87, [384, 1, 1])
    mul_128: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_43, view_86);  sub_43 = view_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_100: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_97, mul_128);  add_97 = mul_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_87: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_100, [8, 1, 384, 196])
    var_mean_29 = torch.ops.aten.var_mean.correction(view_87, [2, 3], correction = 0, keepdim = True)
    getitem_58: "f32[8, 1, 1, 1]" = var_mean_29[0]
    getitem_59: "f32[8, 1, 1, 1]" = var_mean_29[1];  var_mean_29 = None
    add_101: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
    rsqrt_29: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
    sub_44: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_87, getitem_59);  view_87 = None
    mul_129: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_29);  sub_44 = None
    view_88: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_129, [8, 384, 14, 14]);  mul_129 = None
    unsqueeze_174: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_89, 0);  primals_89 = None
    unsqueeze_175: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, 2);  unsqueeze_174 = None
    unsqueeze_176: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_175, 3);  unsqueeze_175 = None
    unsqueeze_177: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_88, 0)
    unsqueeze_178: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_177, 2);  unsqueeze_177 = None
    unsqueeze_179: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, 3);  unsqueeze_178 = None
    mul_130: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_88, unsqueeze_179);  view_88 = unsqueeze_179 = None
    add_102: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_130, unsqueeze_176);  mul_130 = unsqueeze_176 = None
    squeeze_58: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_59, [2, 3]);  getitem_59 = None
    squeeze_59: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_29, [2, 3]);  rsqrt_29 = None
    alias_58: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_58);  squeeze_58 = None
    alias_59: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_59);  squeeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_31: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_102, primals_281, primals_282, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_131: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_31, 0.5)
    mul_132: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_31, 0.7071067811865476)
    erf_14: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_132);  mul_132 = None
    add_103: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_133: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_131, add_103);  mul_131 = add_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_28: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_133);  mul_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_32: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_28, primals_283, primals_284, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_29: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_89: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_90, [384, 1, 1])
    mul_134: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_29, view_89);  clone_29 = view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_104: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_100, mul_134);  add_100 = mul_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_90: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_104, [8, 1, 384, 196])
    var_mean_30 = torch.ops.aten.var_mean.correction(view_90, [2, 3], correction = 0, keepdim = True)
    getitem_60: "f32[8, 1, 1, 1]" = var_mean_30[0]
    getitem_61: "f32[8, 1, 1, 1]" = var_mean_30[1];  var_mean_30 = None
    add_105: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
    rsqrt_30: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
    sub_45: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_90, getitem_61);  view_90 = None
    mul_135: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_30);  sub_45 = None
    view_91: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_135, [8, 384, 14, 14]);  mul_135 = None
    unsqueeze_180: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_92, 0);  primals_92 = None
    unsqueeze_181: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, 2);  unsqueeze_180 = None
    unsqueeze_182: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_181, 3);  unsqueeze_181 = None
    unsqueeze_183: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_91, 0)
    unsqueeze_184: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_183, 2);  unsqueeze_183 = None
    unsqueeze_185: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, 3);  unsqueeze_184 = None
    mul_136: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_91, unsqueeze_185);  view_91 = unsqueeze_185 = None
    add_106: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_136, unsqueeze_182);  mul_136 = unsqueeze_182 = None
    squeeze_60: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_61, [2, 3]);  getitem_61 = None
    squeeze_61: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_30, [2, 3]);  rsqrt_30 = None
    alias_60: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_60);  squeeze_60 = None
    alias_61: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_61);  squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_15: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d.default(add_106, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_46: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_15, add_106)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_92: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_93, [384, 1, 1])
    mul_137: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_46, view_92);  sub_46 = view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_107: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_104, mul_137);  add_104 = mul_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_93: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_107, [8, 1, 384, 196])
    var_mean_31 = torch.ops.aten.var_mean.correction(view_93, [2, 3], correction = 0, keepdim = True)
    getitem_62: "f32[8, 1, 1, 1]" = var_mean_31[0]
    getitem_63: "f32[8, 1, 1, 1]" = var_mean_31[1];  var_mean_31 = None
    add_108: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
    rsqrt_31: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    sub_47: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_93, getitem_63);  view_93 = None
    mul_138: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_31);  sub_47 = None
    view_94: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_138, [8, 384, 14, 14]);  mul_138 = None
    unsqueeze_186: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_95, 0);  primals_95 = None
    unsqueeze_187: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, 2);  unsqueeze_186 = None
    unsqueeze_188: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_187, 3);  unsqueeze_187 = None
    unsqueeze_189: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_94, 0)
    unsqueeze_190: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_189, 2);  unsqueeze_189 = None
    unsqueeze_191: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, 3);  unsqueeze_190 = None
    mul_139: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_94, unsqueeze_191);  view_94 = unsqueeze_191 = None
    add_109: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_139, unsqueeze_188);  mul_139 = unsqueeze_188 = None
    squeeze_62: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_63, [2, 3]);  getitem_63 = None
    squeeze_63: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_31, [2, 3]);  rsqrt_31 = None
    alias_62: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_62);  squeeze_62 = None
    alias_63: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_63);  squeeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_33: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_109, primals_285, primals_286, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_140: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_33, 0.5)
    mul_141: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_33, 0.7071067811865476)
    erf_15: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_141);  mul_141 = None
    add_110: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_142: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_140, add_110);  mul_140 = add_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_30: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_142);  mul_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_34: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_30, primals_287, primals_288, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_31: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_34)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_95: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_96, [384, 1, 1])
    mul_143: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_31, view_95);  clone_31 = view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_111: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_107, mul_143);  add_107 = mul_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_96: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_111, [8, 1, 384, 196])
    var_mean_32 = torch.ops.aten.var_mean.correction(view_96, [2, 3], correction = 0, keepdim = True)
    getitem_64: "f32[8, 1, 1, 1]" = var_mean_32[0]
    getitem_65: "f32[8, 1, 1, 1]" = var_mean_32[1];  var_mean_32 = None
    add_112: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
    rsqrt_32: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
    sub_48: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_96, getitem_65);  view_96 = None
    mul_144: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_32);  sub_48 = None
    view_97: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_144, [8, 384, 14, 14]);  mul_144 = None
    unsqueeze_192: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_98, 0);  primals_98 = None
    unsqueeze_193: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, 2);  unsqueeze_192 = None
    unsqueeze_194: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_193, 3);  unsqueeze_193 = None
    unsqueeze_195: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_97, 0)
    unsqueeze_196: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_195, 2);  unsqueeze_195 = None
    unsqueeze_197: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, 3);  unsqueeze_196 = None
    mul_145: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_97, unsqueeze_197);  view_97 = unsqueeze_197 = None
    add_113: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_145, unsqueeze_194);  mul_145 = unsqueeze_194 = None
    squeeze_64: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_65, [2, 3]);  getitem_65 = None
    squeeze_65: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_32, [2, 3]);  rsqrt_32 = None
    alias_64: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_64);  squeeze_64 = None
    alias_65: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_65);  squeeze_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_16: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d.default(add_113, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_49: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_16, add_113)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_98: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_99, [384, 1, 1])
    mul_146: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_49, view_98);  sub_49 = view_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_114: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_111, mul_146);  add_111 = mul_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_99: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_114, [8, 1, 384, 196])
    var_mean_33 = torch.ops.aten.var_mean.correction(view_99, [2, 3], correction = 0, keepdim = True)
    getitem_66: "f32[8, 1, 1, 1]" = var_mean_33[0]
    getitem_67: "f32[8, 1, 1, 1]" = var_mean_33[1];  var_mean_33 = None
    add_115: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
    rsqrt_33: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
    sub_50: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_99, getitem_67);  view_99 = None
    mul_147: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_33);  sub_50 = None
    view_100: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_147, [8, 384, 14, 14]);  mul_147 = None
    unsqueeze_198: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_101, 0);  primals_101 = None
    unsqueeze_199: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, 2);  unsqueeze_198 = None
    unsqueeze_200: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_199, 3);  unsqueeze_199 = None
    unsqueeze_201: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_100, 0)
    unsqueeze_202: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_201, 2);  unsqueeze_201 = None
    unsqueeze_203: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, 3);  unsqueeze_202 = None
    mul_148: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_100, unsqueeze_203);  view_100 = unsqueeze_203 = None
    add_116: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_148, unsqueeze_200);  mul_148 = unsqueeze_200 = None
    squeeze_66: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_67, [2, 3]);  getitem_67 = None
    squeeze_67: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_33, [2, 3]);  rsqrt_33 = None
    alias_66: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_66);  squeeze_66 = None
    alias_67: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_67);  squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_35: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_116, primals_289, primals_290, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_149: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_35, 0.5)
    mul_150: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_35, 0.7071067811865476)
    erf_16: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_150);  mul_150 = None
    add_117: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_151: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_149, add_117);  mul_149 = add_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_32: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_151);  mul_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_36: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_32, primals_291, primals_292, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_33: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_101: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_102, [384, 1, 1])
    mul_152: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_33, view_101);  clone_33 = view_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_118: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_114, mul_152);  add_114 = mul_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_102: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_118, [8, 1, 384, 196])
    var_mean_34 = torch.ops.aten.var_mean.correction(view_102, [2, 3], correction = 0, keepdim = True)
    getitem_68: "f32[8, 1, 1, 1]" = var_mean_34[0]
    getitem_69: "f32[8, 1, 1, 1]" = var_mean_34[1];  var_mean_34 = None
    add_119: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
    rsqrt_34: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
    sub_51: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_102, getitem_69);  view_102 = None
    mul_153: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_34);  sub_51 = None
    view_103: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_153, [8, 384, 14, 14]);  mul_153 = None
    unsqueeze_204: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_104, 0);  primals_104 = None
    unsqueeze_205: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, 2);  unsqueeze_204 = None
    unsqueeze_206: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 3);  unsqueeze_205 = None
    unsqueeze_207: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_103, 0)
    unsqueeze_208: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_207, 2);  unsqueeze_207 = None
    unsqueeze_209: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, 3);  unsqueeze_208 = None
    mul_154: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_103, unsqueeze_209);  view_103 = unsqueeze_209 = None
    add_120: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_154, unsqueeze_206);  mul_154 = unsqueeze_206 = None
    squeeze_68: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_69, [2, 3]);  getitem_69 = None
    squeeze_69: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_34, [2, 3]);  rsqrt_34 = None
    alias_68: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_68);  squeeze_68 = None
    alias_69: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_69);  squeeze_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_17: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d.default(add_120, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_52: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_17, add_120)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_104: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_105, [384, 1, 1])
    mul_155: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_52, view_104);  sub_52 = view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_121: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_118, mul_155);  add_118 = mul_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_105: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_121, [8, 1, 384, 196])
    var_mean_35 = torch.ops.aten.var_mean.correction(view_105, [2, 3], correction = 0, keepdim = True)
    getitem_70: "f32[8, 1, 1, 1]" = var_mean_35[0]
    getitem_71: "f32[8, 1, 1, 1]" = var_mean_35[1];  var_mean_35 = None
    add_122: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
    rsqrt_35: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    sub_53: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_105, getitem_71);  view_105 = None
    mul_156: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_35);  sub_53 = None
    view_106: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_156, [8, 384, 14, 14]);  mul_156 = None
    unsqueeze_210: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_107, 0);  primals_107 = None
    unsqueeze_211: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, 2);  unsqueeze_210 = None
    unsqueeze_212: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 3);  unsqueeze_211 = None
    unsqueeze_213: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_106, 0)
    unsqueeze_214: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 2);  unsqueeze_213 = None
    unsqueeze_215: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, 3);  unsqueeze_214 = None
    mul_157: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_106, unsqueeze_215);  view_106 = unsqueeze_215 = None
    add_123: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_157, unsqueeze_212);  mul_157 = unsqueeze_212 = None
    squeeze_70: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_71, [2, 3]);  getitem_71 = None
    squeeze_71: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_35, [2, 3]);  rsqrt_35 = None
    alias_70: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_70);  squeeze_70 = None
    alias_71: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_71);  squeeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_37: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_123, primals_293, primals_294, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_158: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_37, 0.5)
    mul_159: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_37, 0.7071067811865476)
    erf_17: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_159);  mul_159 = None
    add_124: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_160: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_158, add_124);  mul_158 = add_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_34: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_160);  mul_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_38: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_34, primals_295, primals_296, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_35: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_107: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_108, [384, 1, 1])
    mul_161: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_35, view_107);  clone_35 = view_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_125: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_121, mul_161);  add_121 = mul_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_108: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_125, [8, 1, 384, 196])
    var_mean_36 = torch.ops.aten.var_mean.correction(view_108, [2, 3], correction = 0, keepdim = True)
    getitem_72: "f32[8, 1, 1, 1]" = var_mean_36[0]
    getitem_73: "f32[8, 1, 1, 1]" = var_mean_36[1];  var_mean_36 = None
    add_126: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
    rsqrt_36: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
    sub_54: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_108, getitem_73);  view_108 = None
    mul_162: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_36);  sub_54 = None
    view_109: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_162, [8, 384, 14, 14]);  mul_162 = None
    unsqueeze_216: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_110, 0);  primals_110 = None
    unsqueeze_217: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, 2);  unsqueeze_216 = None
    unsqueeze_218: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 3);  unsqueeze_217 = None
    unsqueeze_219: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_109, 0)
    unsqueeze_220: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_219, 2);  unsqueeze_219 = None
    unsqueeze_221: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, 3);  unsqueeze_220 = None
    mul_163: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_109, unsqueeze_221);  view_109 = unsqueeze_221 = None
    add_127: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_163, unsqueeze_218);  mul_163 = unsqueeze_218 = None
    squeeze_72: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_73, [2, 3]);  getitem_73 = None
    squeeze_73: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_36, [2, 3]);  rsqrt_36 = None
    alias_72: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_72);  squeeze_72 = None
    alias_73: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_73);  squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_18: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d.default(add_127, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_55: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_18, add_127)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_110: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_111, [384, 1, 1])
    mul_164: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_55, view_110);  sub_55 = view_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_128: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_125, mul_164);  add_125 = mul_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_111: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_128, [8, 1, 384, 196])
    var_mean_37 = torch.ops.aten.var_mean.correction(view_111, [2, 3], correction = 0, keepdim = True)
    getitem_74: "f32[8, 1, 1, 1]" = var_mean_37[0]
    getitem_75: "f32[8, 1, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    add_129: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
    rsqrt_37: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
    sub_56: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_111, getitem_75);  view_111 = None
    mul_165: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_37);  sub_56 = None
    view_112: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_165, [8, 384, 14, 14]);  mul_165 = None
    unsqueeze_222: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_113, 0);  primals_113 = None
    unsqueeze_223: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, 2);  unsqueeze_222 = None
    unsqueeze_224: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 3);  unsqueeze_223 = None
    unsqueeze_225: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_112, 0)
    unsqueeze_226: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_225, 2);  unsqueeze_225 = None
    unsqueeze_227: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 3);  unsqueeze_226 = None
    mul_166: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_112, unsqueeze_227);  view_112 = unsqueeze_227 = None
    add_130: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_166, unsqueeze_224);  mul_166 = unsqueeze_224 = None
    squeeze_74: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_75, [2, 3]);  getitem_75 = None
    squeeze_75: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_37, [2, 3]);  rsqrt_37 = None
    alias_74: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_74);  squeeze_74 = None
    alias_75: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_75);  squeeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_39: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_130, primals_297, primals_298, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_167: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_39, 0.5)
    mul_168: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_39, 0.7071067811865476)
    erf_18: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_168);  mul_168 = None
    add_131: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_169: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_167, add_131);  mul_167 = add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_36: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_169);  mul_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_40: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_36, primals_299, primals_300, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_37: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_40)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_113: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_114, [384, 1, 1])
    mul_170: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_37, view_113);  clone_37 = view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_132: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_128, mul_170);  add_128 = mul_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_114: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_132, [8, 1, 384, 196])
    var_mean_38 = torch.ops.aten.var_mean.correction(view_114, [2, 3], correction = 0, keepdim = True)
    getitem_76: "f32[8, 1, 1, 1]" = var_mean_38[0]
    getitem_77: "f32[8, 1, 1, 1]" = var_mean_38[1];  var_mean_38 = None
    add_133: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
    rsqrt_38: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
    sub_57: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_114, getitem_77);  view_114 = None
    mul_171: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_38);  sub_57 = None
    view_115: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_171, [8, 384, 14, 14]);  mul_171 = None
    unsqueeze_228: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_116, 0);  primals_116 = None
    unsqueeze_229: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, 2);  unsqueeze_228 = None
    unsqueeze_230: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 3);  unsqueeze_229 = None
    unsqueeze_231: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_115, 0)
    unsqueeze_232: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_231, 2);  unsqueeze_231 = None
    unsqueeze_233: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 3);  unsqueeze_232 = None
    mul_172: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_115, unsqueeze_233);  view_115 = unsqueeze_233 = None
    add_134: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_172, unsqueeze_230);  mul_172 = unsqueeze_230 = None
    squeeze_76: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_77, [2, 3]);  getitem_77 = None
    squeeze_77: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_38, [2, 3]);  rsqrt_38 = None
    alias_76: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_76);  squeeze_76 = None
    alias_77: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_77);  squeeze_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_19: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d.default(add_134, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_58: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_19, add_134)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_116: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_117, [384, 1, 1])
    mul_173: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_58, view_116);  sub_58 = view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_135: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_132, mul_173);  add_132 = mul_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_117: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_135, [8, 1, 384, 196])
    var_mean_39 = torch.ops.aten.var_mean.correction(view_117, [2, 3], correction = 0, keepdim = True)
    getitem_78: "f32[8, 1, 1, 1]" = var_mean_39[0]
    getitem_79: "f32[8, 1, 1, 1]" = var_mean_39[1];  var_mean_39 = None
    add_136: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
    rsqrt_39: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
    sub_59: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_117, getitem_79);  view_117 = None
    mul_174: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_39);  sub_59 = None
    view_118: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_174, [8, 384, 14, 14]);  mul_174 = None
    unsqueeze_234: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_119, 0);  primals_119 = None
    unsqueeze_235: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, 2);  unsqueeze_234 = None
    unsqueeze_236: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 3);  unsqueeze_235 = None
    unsqueeze_237: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_118, 0)
    unsqueeze_238: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_237, 2);  unsqueeze_237 = None
    unsqueeze_239: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 3);  unsqueeze_238 = None
    mul_175: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_118, unsqueeze_239);  view_118 = unsqueeze_239 = None
    add_137: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_175, unsqueeze_236);  mul_175 = unsqueeze_236 = None
    squeeze_78: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_79, [2, 3]);  getitem_79 = None
    squeeze_79: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_39, [2, 3]);  rsqrt_39 = None
    alias_78: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_78);  squeeze_78 = None
    alias_79: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_79);  squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_41: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_137, primals_301, primals_302, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_176: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_41, 0.5)
    mul_177: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_41, 0.7071067811865476)
    erf_19: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_177);  mul_177 = None
    add_138: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_178: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_176, add_138);  mul_176 = add_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_38: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_178);  mul_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_42: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_38, primals_303, primals_304, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_39: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_42)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_119: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_120, [384, 1, 1])
    mul_179: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_39, view_119);  clone_39 = view_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_139: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_135, mul_179);  add_135 = mul_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_120: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_139, [8, 1, 384, 196])
    var_mean_40 = torch.ops.aten.var_mean.correction(view_120, [2, 3], correction = 0, keepdim = True)
    getitem_80: "f32[8, 1, 1, 1]" = var_mean_40[0]
    getitem_81: "f32[8, 1, 1, 1]" = var_mean_40[1];  var_mean_40 = None
    add_140: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
    rsqrt_40: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    sub_60: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_120, getitem_81);  view_120 = None
    mul_180: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_40);  sub_60 = None
    view_121: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_180, [8, 384, 14, 14]);  mul_180 = None
    unsqueeze_240: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_122, 0);  primals_122 = None
    unsqueeze_241: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, 2);  unsqueeze_240 = None
    unsqueeze_242: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 3);  unsqueeze_241 = None
    unsqueeze_243: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_121, 0)
    unsqueeze_244: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_243, 2);  unsqueeze_243 = None
    unsqueeze_245: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 3);  unsqueeze_244 = None
    mul_181: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_121, unsqueeze_245);  view_121 = unsqueeze_245 = None
    add_141: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_242);  mul_181 = unsqueeze_242 = None
    squeeze_80: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_81, [2, 3]);  getitem_81 = None
    squeeze_81: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_40, [2, 3]);  rsqrt_40 = None
    alias_80: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_80);  squeeze_80 = None
    alias_81: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_81);  squeeze_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_20: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d.default(add_141, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_61: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_20, add_141)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_122: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_123, [384, 1, 1])
    mul_182: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_61, view_122);  sub_61 = view_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_142: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_139, mul_182);  add_139 = mul_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_123: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_142, [8, 1, 384, 196])
    var_mean_41 = torch.ops.aten.var_mean.correction(view_123, [2, 3], correction = 0, keepdim = True)
    getitem_82: "f32[8, 1, 1, 1]" = var_mean_41[0]
    getitem_83: "f32[8, 1, 1, 1]" = var_mean_41[1];  var_mean_41 = None
    add_143: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
    rsqrt_41: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
    sub_62: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_123, getitem_83);  view_123 = None
    mul_183: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_41);  sub_62 = None
    view_124: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_183, [8, 384, 14, 14]);  mul_183 = None
    unsqueeze_246: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_125, 0);  primals_125 = None
    unsqueeze_247: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, 2);  unsqueeze_246 = None
    unsqueeze_248: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 3);  unsqueeze_247 = None
    unsqueeze_249: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_124, 0)
    unsqueeze_250: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_249, 2);  unsqueeze_249 = None
    unsqueeze_251: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 3);  unsqueeze_250 = None
    mul_184: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_124, unsqueeze_251);  view_124 = unsqueeze_251 = None
    add_144: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_184, unsqueeze_248);  mul_184 = unsqueeze_248 = None
    squeeze_82: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_83, [2, 3]);  getitem_83 = None
    squeeze_83: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_41, [2, 3]);  rsqrt_41 = None
    alias_82: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_82);  squeeze_82 = None
    alias_83: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_83);  squeeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_43: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_144, primals_305, primals_306, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_185: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_43, 0.5)
    mul_186: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_43, 0.7071067811865476)
    erf_20: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_186);  mul_186 = None
    add_145: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_187: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_185, add_145);  mul_185 = add_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_40: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_187);  mul_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_44: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_40, primals_307, primals_308, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_41: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_44)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_125: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_126, [384, 1, 1])
    mul_188: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_41, view_125);  clone_41 = view_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_146: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_142, mul_188);  add_142 = mul_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_126: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_146, [8, 1, 384, 196])
    var_mean_42 = torch.ops.aten.var_mean.correction(view_126, [2, 3], correction = 0, keepdim = True)
    getitem_84: "f32[8, 1, 1, 1]" = var_mean_42[0]
    getitem_85: "f32[8, 1, 1, 1]" = var_mean_42[1];  var_mean_42 = None
    add_147: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05);  getitem_84 = None
    rsqrt_42: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
    sub_63: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_126, getitem_85);  view_126 = None
    mul_189: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_42);  sub_63 = None
    view_127: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_189, [8, 384, 14, 14]);  mul_189 = None
    unsqueeze_252: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_128, 0);  primals_128 = None
    unsqueeze_253: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, 2);  unsqueeze_252 = None
    unsqueeze_254: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 3);  unsqueeze_253 = None
    unsqueeze_255: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_127, 0)
    unsqueeze_256: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 2);  unsqueeze_255 = None
    unsqueeze_257: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, 3);  unsqueeze_256 = None
    mul_190: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_127, unsqueeze_257);  view_127 = unsqueeze_257 = None
    add_148: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_190, unsqueeze_254);  mul_190 = unsqueeze_254 = None
    squeeze_84: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_85, [2, 3]);  getitem_85 = None
    squeeze_85: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_42, [2, 3]);  rsqrt_42 = None
    alias_84: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_84);  squeeze_84 = None
    alias_85: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_85);  squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_21: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d.default(add_148, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_64: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_21, add_148)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_128: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_129, [384, 1, 1])
    mul_191: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_64, view_128);  sub_64 = view_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_149: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_146, mul_191);  add_146 = mul_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_129: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_149, [8, 1, 384, 196])
    var_mean_43 = torch.ops.aten.var_mean.correction(view_129, [2, 3], correction = 0, keepdim = True)
    getitem_86: "f32[8, 1, 1, 1]" = var_mean_43[0]
    getitem_87: "f32[8, 1, 1, 1]" = var_mean_43[1];  var_mean_43 = None
    add_150: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
    rsqrt_43: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
    sub_65: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_129, getitem_87);  view_129 = None
    mul_192: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_43);  sub_65 = None
    view_130: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_192, [8, 384, 14, 14]);  mul_192 = None
    unsqueeze_258: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_131, 0);  primals_131 = None
    unsqueeze_259: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, 2);  unsqueeze_258 = None
    unsqueeze_260: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 3);  unsqueeze_259 = None
    unsqueeze_261: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_130, 0)
    unsqueeze_262: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 2);  unsqueeze_261 = None
    unsqueeze_263: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, 3);  unsqueeze_262 = None
    mul_193: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_130, unsqueeze_263);  view_130 = unsqueeze_263 = None
    add_151: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_193, unsqueeze_260);  mul_193 = unsqueeze_260 = None
    squeeze_86: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_87, [2, 3]);  getitem_87 = None
    squeeze_87: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_43, [2, 3]);  rsqrt_43 = None
    alias_86: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_86);  squeeze_86 = None
    alias_87: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_87);  squeeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_45: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_151, primals_309, primals_310, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_194: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_45, 0.5)
    mul_195: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_45, 0.7071067811865476)
    erf_21: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_195);  mul_195 = None
    add_152: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_196: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_194, add_152);  mul_194 = add_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_42: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_196);  mul_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_46: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_42, primals_311, primals_312, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_43: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_131: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_132, [384, 1, 1])
    mul_197: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_43, view_131);  clone_43 = view_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_153: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_149, mul_197);  add_149 = mul_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_132: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_153, [8, 1, 384, 196])
    var_mean_44 = torch.ops.aten.var_mean.correction(view_132, [2, 3], correction = 0, keepdim = True)
    getitem_88: "f32[8, 1, 1, 1]" = var_mean_44[0]
    getitem_89: "f32[8, 1, 1, 1]" = var_mean_44[1];  var_mean_44 = None
    add_154: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
    rsqrt_44: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
    sub_66: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_132, getitem_89);  view_132 = None
    mul_198: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_44);  sub_66 = None
    view_133: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_198, [8, 384, 14, 14]);  mul_198 = None
    unsqueeze_264: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_134, 0);  primals_134 = None
    unsqueeze_265: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, 2);  unsqueeze_264 = None
    unsqueeze_266: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 3);  unsqueeze_265 = None
    unsqueeze_267: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_133, 0)
    unsqueeze_268: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 2);  unsqueeze_267 = None
    unsqueeze_269: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, 3);  unsqueeze_268 = None
    mul_199: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_133, unsqueeze_269);  view_133 = unsqueeze_269 = None
    add_155: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_199, unsqueeze_266);  mul_199 = unsqueeze_266 = None
    squeeze_88: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_89, [2, 3]);  getitem_89 = None
    squeeze_89: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_44, [2, 3]);  rsqrt_44 = None
    alias_88: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_88);  squeeze_88 = None
    alias_89: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_89);  squeeze_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_22: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d.default(add_155, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_67: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_22, add_155)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_134: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_135, [384, 1, 1])
    mul_200: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_67, view_134);  sub_67 = view_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_156: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_153, mul_200);  add_153 = mul_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_135: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_156, [8, 1, 384, 196])
    var_mean_45 = torch.ops.aten.var_mean.correction(view_135, [2, 3], correction = 0, keepdim = True)
    getitem_90: "f32[8, 1, 1, 1]" = var_mean_45[0]
    getitem_91: "f32[8, 1, 1, 1]" = var_mean_45[1];  var_mean_45 = None
    add_157: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
    rsqrt_45: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
    sub_68: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_135, getitem_91);  view_135 = None
    mul_201: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_45);  sub_68 = None
    view_136: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_201, [8, 384, 14, 14]);  mul_201 = None
    unsqueeze_270: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_137, 0);  primals_137 = None
    unsqueeze_271: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 2);  unsqueeze_270 = None
    unsqueeze_272: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 3);  unsqueeze_271 = None
    unsqueeze_273: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_136, 0)
    unsqueeze_274: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 2);  unsqueeze_273 = None
    unsqueeze_275: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 3);  unsqueeze_274 = None
    mul_202: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_136, unsqueeze_275);  view_136 = unsqueeze_275 = None
    add_158: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_202, unsqueeze_272);  mul_202 = unsqueeze_272 = None
    squeeze_90: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_91, [2, 3]);  getitem_91 = None
    squeeze_91: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_45, [2, 3]);  rsqrt_45 = None
    alias_90: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_90);  squeeze_90 = None
    alias_91: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_91);  squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_47: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_158, primals_313, primals_314, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_203: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_47, 0.5)
    mul_204: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_47, 0.7071067811865476)
    erf_22: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_204);  mul_204 = None
    add_159: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_205: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_203, add_159);  mul_203 = add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_44: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_205);  mul_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_48: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_44, primals_315, primals_316, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_45: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_48)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_137: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_138, [384, 1, 1])
    mul_206: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_45, view_137);  clone_45 = view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_160: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_156, mul_206);  add_156 = mul_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_138: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_160, [8, 1, 384, 196])
    var_mean_46 = torch.ops.aten.var_mean.correction(view_138, [2, 3], correction = 0, keepdim = True)
    getitem_92: "f32[8, 1, 1, 1]" = var_mean_46[0]
    getitem_93: "f32[8, 1, 1, 1]" = var_mean_46[1];  var_mean_46 = None
    add_161: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
    rsqrt_46: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    sub_69: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_138, getitem_93);  view_138 = None
    mul_207: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_46);  sub_69 = None
    view_139: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_207, [8, 384, 14, 14]);  mul_207 = None
    unsqueeze_276: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_140, 0);  primals_140 = None
    unsqueeze_277: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 2);  unsqueeze_276 = None
    unsqueeze_278: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 3);  unsqueeze_277 = None
    unsqueeze_279: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_139, 0)
    unsqueeze_280: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 2);  unsqueeze_279 = None
    unsqueeze_281: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 3);  unsqueeze_280 = None
    mul_208: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_139, unsqueeze_281);  view_139 = unsqueeze_281 = None
    add_162: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_208, unsqueeze_278);  mul_208 = unsqueeze_278 = None
    squeeze_92: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_93, [2, 3]);  getitem_93 = None
    squeeze_93: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_46, [2, 3]);  rsqrt_46 = None
    alias_92: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_92);  squeeze_92 = None
    alias_93: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_93);  squeeze_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_23: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d.default(add_162, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_70: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_23, add_162)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_140: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_141, [384, 1, 1])
    mul_209: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_70, view_140);  sub_70 = view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_163: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_160, mul_209);  add_160 = mul_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_141: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_163, [8, 1, 384, 196])
    var_mean_47 = torch.ops.aten.var_mean.correction(view_141, [2, 3], correction = 0, keepdim = True)
    getitem_94: "f32[8, 1, 1, 1]" = var_mean_47[0]
    getitem_95: "f32[8, 1, 1, 1]" = var_mean_47[1];  var_mean_47 = None
    add_164: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
    rsqrt_47: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
    sub_71: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_141, getitem_95);  view_141 = None
    mul_210: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_47);  sub_71 = None
    view_142: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_210, [8, 384, 14, 14]);  mul_210 = None
    unsqueeze_282: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_143, 0);  primals_143 = None
    unsqueeze_283: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, 2);  unsqueeze_282 = None
    unsqueeze_284: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 3);  unsqueeze_283 = None
    unsqueeze_285: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_142, 0)
    unsqueeze_286: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 2);  unsqueeze_285 = None
    unsqueeze_287: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 3);  unsqueeze_286 = None
    mul_211: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_142, unsqueeze_287);  view_142 = unsqueeze_287 = None
    add_165: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_211, unsqueeze_284);  mul_211 = unsqueeze_284 = None
    squeeze_94: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_95, [2, 3]);  getitem_95 = None
    squeeze_95: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_47, [2, 3]);  rsqrt_47 = None
    alias_94: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_94);  squeeze_94 = None
    alias_95: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_95);  squeeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_49: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_165, primals_317, primals_318, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_212: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_49, 0.5)
    mul_213: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_49, 0.7071067811865476)
    erf_23: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_213);  mul_213 = None
    add_166: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_214: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_212, add_166);  mul_212 = add_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_46: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_214);  mul_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_50: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_46, primals_319, primals_320, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_47: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_50)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_143: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_144, [384, 1, 1])
    mul_215: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_47, view_143);  clone_47 = view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_167: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_163, mul_215);  add_163 = mul_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_144: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_167, [8, 1, 384, 196])
    var_mean_48 = torch.ops.aten.var_mean.correction(view_144, [2, 3], correction = 0, keepdim = True)
    getitem_96: "f32[8, 1, 1, 1]" = var_mean_48[0]
    getitem_97: "f32[8, 1, 1, 1]" = var_mean_48[1];  var_mean_48 = None
    add_168: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
    rsqrt_48: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
    sub_72: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_144, getitem_97);  view_144 = None
    mul_216: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_48);  sub_72 = None
    view_145: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_216, [8, 384, 14, 14]);  mul_216 = None
    unsqueeze_288: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_146, 0);  primals_146 = None
    unsqueeze_289: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 2);  unsqueeze_288 = None
    unsqueeze_290: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 3);  unsqueeze_289 = None
    unsqueeze_291: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_145, 0)
    unsqueeze_292: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 2);  unsqueeze_291 = None
    unsqueeze_293: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 3);  unsqueeze_292 = None
    mul_217: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_145, unsqueeze_293);  view_145 = unsqueeze_293 = None
    add_169: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_217, unsqueeze_290);  mul_217 = unsqueeze_290 = None
    squeeze_96: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_97, [2, 3]);  getitem_97 = None
    squeeze_97: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_48, [2, 3]);  rsqrt_48 = None
    alias_96: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_96);  squeeze_96 = None
    alias_97: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_97);  squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_24: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d.default(add_169, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_73: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_24, add_169)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_146: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_147, [384, 1, 1])
    mul_218: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_73, view_146);  sub_73 = view_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_170: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_167, mul_218);  add_167 = mul_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_147: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_170, [8, 1, 384, 196])
    var_mean_49 = torch.ops.aten.var_mean.correction(view_147, [2, 3], correction = 0, keepdim = True)
    getitem_98: "f32[8, 1, 1, 1]" = var_mean_49[0]
    getitem_99: "f32[8, 1, 1, 1]" = var_mean_49[1];  var_mean_49 = None
    add_171: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-05);  getitem_98 = None
    rsqrt_49: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
    sub_74: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_147, getitem_99);  view_147 = None
    mul_219: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_49);  sub_74 = None
    view_148: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_219, [8, 384, 14, 14]);  mul_219 = None
    unsqueeze_294: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_149, 0);  primals_149 = None
    unsqueeze_295: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, 2);  unsqueeze_294 = None
    unsqueeze_296: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 3);  unsqueeze_295 = None
    unsqueeze_297: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_148, 0)
    unsqueeze_298: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 2);  unsqueeze_297 = None
    unsqueeze_299: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 3);  unsqueeze_298 = None
    mul_220: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_148, unsqueeze_299);  view_148 = unsqueeze_299 = None
    add_172: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_220, unsqueeze_296);  mul_220 = unsqueeze_296 = None
    squeeze_98: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_99, [2, 3]);  getitem_99 = None
    squeeze_99: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_49, [2, 3]);  rsqrt_49 = None
    alias_98: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_98);  squeeze_98 = None
    alias_99: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_99);  squeeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_51: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_172, primals_321, primals_322, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_221: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_51, 0.5)
    mul_222: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_51, 0.7071067811865476)
    erf_24: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_222);  mul_222 = None
    add_173: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
    mul_223: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_221, add_173);  mul_221 = add_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_48: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_223);  mul_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_52: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_48, primals_323, primals_324, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_49: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_52)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_149: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_150, [384, 1, 1])
    mul_224: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_49, view_149);  clone_49 = view_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_174: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_170, mul_224);  add_170 = mul_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_150: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_174, [8, 1, 384, 196])
    var_mean_50 = torch.ops.aten.var_mean.correction(view_150, [2, 3], correction = 0, keepdim = True)
    getitem_100: "f32[8, 1, 1, 1]" = var_mean_50[0]
    getitem_101: "f32[8, 1, 1, 1]" = var_mean_50[1];  var_mean_50 = None
    add_175: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05);  getitem_100 = None
    rsqrt_50: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
    sub_75: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_150, getitem_101);  view_150 = None
    mul_225: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_50);  sub_75 = None
    view_151: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_225, [8, 384, 14, 14]);  mul_225 = None
    unsqueeze_300: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_152, 0);  primals_152 = None
    unsqueeze_301: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 2);  unsqueeze_300 = None
    unsqueeze_302: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 3);  unsqueeze_301 = None
    unsqueeze_303: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_151, 0)
    unsqueeze_304: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 2);  unsqueeze_303 = None
    unsqueeze_305: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, 3);  unsqueeze_304 = None
    mul_226: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_151, unsqueeze_305);  view_151 = unsqueeze_305 = None
    add_176: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_226, unsqueeze_302);  mul_226 = unsqueeze_302 = None
    squeeze_100: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_101, [2, 3]);  getitem_101 = None
    squeeze_101: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_50, [2, 3]);  rsqrt_50 = None
    alias_100: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_100);  squeeze_100 = None
    alias_101: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_101);  squeeze_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_25: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d.default(add_176, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_76: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_25, add_176)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_152: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_153, [384, 1, 1])
    mul_227: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_76, view_152);  sub_76 = view_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_177: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_174, mul_227);  add_174 = mul_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_153: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_177, [8, 1, 384, 196])
    var_mean_51 = torch.ops.aten.var_mean.correction(view_153, [2, 3], correction = 0, keepdim = True)
    getitem_102: "f32[8, 1, 1, 1]" = var_mean_51[0]
    getitem_103: "f32[8, 1, 1, 1]" = var_mean_51[1];  var_mean_51 = None
    add_178: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05);  getitem_102 = None
    rsqrt_51: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
    sub_77: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_153, getitem_103);  view_153 = None
    mul_228: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_51);  sub_77 = None
    view_154: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_228, [8, 384, 14, 14]);  mul_228 = None
    unsqueeze_306: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_155, 0);  primals_155 = None
    unsqueeze_307: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, 2);  unsqueeze_306 = None
    unsqueeze_308: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 3);  unsqueeze_307 = None
    unsqueeze_309: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_154, 0)
    unsqueeze_310: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 2);  unsqueeze_309 = None
    unsqueeze_311: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 3);  unsqueeze_310 = None
    mul_229: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_154, unsqueeze_311);  view_154 = unsqueeze_311 = None
    add_179: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_229, unsqueeze_308);  mul_229 = unsqueeze_308 = None
    squeeze_102: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_103, [2, 3]);  getitem_103 = None
    squeeze_103: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_51, [2, 3]);  rsqrt_51 = None
    alias_102: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_102);  squeeze_102 = None
    alias_103: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_103);  squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_53: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_179, primals_325, primals_326, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_230: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_53, 0.5)
    mul_231: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_53, 0.7071067811865476)
    erf_25: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_231);  mul_231 = None
    add_180: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
    mul_232: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_230, add_180);  mul_230 = add_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_50: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_232);  mul_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_54: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_50, primals_327, primals_328, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_51: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_54)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_155: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_156, [384, 1, 1])
    mul_233: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_51, view_155);  clone_51 = view_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_181: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_177, mul_233);  add_177 = mul_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_156: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_181, [8, 1, 384, 196])
    var_mean_52 = torch.ops.aten.var_mean.correction(view_156, [2, 3], correction = 0, keepdim = True)
    getitem_104: "f32[8, 1, 1, 1]" = var_mean_52[0]
    getitem_105: "f32[8, 1, 1, 1]" = var_mean_52[1];  var_mean_52 = None
    add_182: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
    rsqrt_52: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
    sub_78: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_156, getitem_105);  view_156 = None
    mul_234: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_52);  sub_78 = None
    view_157: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_234, [8, 384, 14, 14]);  mul_234 = None
    unsqueeze_312: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_158, 0);  primals_158 = None
    unsqueeze_313: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, 2);  unsqueeze_312 = None
    unsqueeze_314: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 3);  unsqueeze_313 = None
    unsqueeze_315: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_157, 0)
    unsqueeze_316: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 2);  unsqueeze_315 = None
    unsqueeze_317: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, 3);  unsqueeze_316 = None
    mul_235: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_157, unsqueeze_317);  view_157 = unsqueeze_317 = None
    add_183: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_235, unsqueeze_314);  mul_235 = unsqueeze_314 = None
    squeeze_104: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_105, [2, 3]);  getitem_105 = None
    squeeze_105: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_52, [2, 3]);  rsqrt_52 = None
    alias_104: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_104);  squeeze_104 = None
    alias_105: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_105);  squeeze_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_26: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d.default(add_183, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_79: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_26, add_183)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_158: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_159, [384, 1, 1])
    mul_236: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_79, view_158);  sub_79 = view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_184: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_181, mul_236);  add_181 = mul_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_159: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_184, [8, 1, 384, 196])
    var_mean_53 = torch.ops.aten.var_mean.correction(view_159, [2, 3], correction = 0, keepdim = True)
    getitem_106: "f32[8, 1, 1, 1]" = var_mean_53[0]
    getitem_107: "f32[8, 1, 1, 1]" = var_mean_53[1];  var_mean_53 = None
    add_185: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05);  getitem_106 = None
    rsqrt_53: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
    sub_80: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_159, getitem_107);  view_159 = None
    mul_237: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_53);  sub_80 = None
    view_160: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_237, [8, 384, 14, 14]);  mul_237 = None
    unsqueeze_318: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_161, 0);  primals_161 = None
    unsqueeze_319: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, 2);  unsqueeze_318 = None
    unsqueeze_320: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 3);  unsqueeze_319 = None
    unsqueeze_321: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_160, 0)
    unsqueeze_322: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 2);  unsqueeze_321 = None
    unsqueeze_323: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 3);  unsqueeze_322 = None
    mul_238: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_160, unsqueeze_323);  view_160 = unsqueeze_323 = None
    add_186: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_238, unsqueeze_320);  mul_238 = unsqueeze_320 = None
    squeeze_106: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_107, [2, 3]);  getitem_107 = None
    squeeze_107: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_53, [2, 3]);  rsqrt_53 = None
    alias_106: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_106);  squeeze_106 = None
    alias_107: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_107);  squeeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_55: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_186, primals_329, primals_330, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_239: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_55, 0.5)
    mul_240: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_55, 0.7071067811865476)
    erf_26: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_240);  mul_240 = None
    add_187: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
    mul_241: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_239, add_187);  mul_239 = add_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_52: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_241);  mul_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_56: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_52, primals_331, primals_332, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_53: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_56)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_161: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_162, [384, 1, 1])
    mul_242: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_53, view_161);  clone_53 = view_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_188: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_184, mul_242);  add_184 = mul_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_162: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_188, [8, 1, 384, 196])
    var_mean_54 = torch.ops.aten.var_mean.correction(view_162, [2, 3], correction = 0, keepdim = True)
    getitem_108: "f32[8, 1, 1, 1]" = var_mean_54[0]
    getitem_109: "f32[8, 1, 1, 1]" = var_mean_54[1];  var_mean_54 = None
    add_189: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
    rsqrt_54: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_189);  add_189 = None
    sub_81: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_162, getitem_109);  view_162 = None
    mul_243: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_54);  sub_81 = None
    view_163: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_243, [8, 384, 14, 14]);  mul_243 = None
    unsqueeze_324: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_164, 0);  primals_164 = None
    unsqueeze_325: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, 2);  unsqueeze_324 = None
    unsqueeze_326: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 3);  unsqueeze_325 = None
    unsqueeze_327: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_163, 0)
    unsqueeze_328: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 2);  unsqueeze_327 = None
    unsqueeze_329: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 3);  unsqueeze_328 = None
    mul_244: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_163, unsqueeze_329);  view_163 = unsqueeze_329 = None
    add_190: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_244, unsqueeze_326);  mul_244 = unsqueeze_326 = None
    squeeze_108: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_109, [2, 3]);  getitem_109 = None
    squeeze_109: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_54, [2, 3]);  rsqrt_54 = None
    alias_108: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_108);  squeeze_108 = None
    alias_109: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_109);  squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_27: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d.default(add_190, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_82: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_27, add_190)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_164: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_165, [384, 1, 1])
    mul_245: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_82, view_164);  sub_82 = view_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_191: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_188, mul_245);  add_188 = mul_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_165: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_191, [8, 1, 384, 196])
    var_mean_55 = torch.ops.aten.var_mean.correction(view_165, [2, 3], correction = 0, keepdim = True)
    getitem_110: "f32[8, 1, 1, 1]" = var_mean_55[0]
    getitem_111: "f32[8, 1, 1, 1]" = var_mean_55[1];  var_mean_55 = None
    add_192: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
    rsqrt_55: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
    sub_83: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_165, getitem_111);  view_165 = None
    mul_246: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_55);  sub_83 = None
    view_166: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_246, [8, 384, 14, 14]);  mul_246 = None
    unsqueeze_330: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_167, 0);  primals_167 = None
    unsqueeze_331: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, 2);  unsqueeze_330 = None
    unsqueeze_332: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 3);  unsqueeze_331 = None
    unsqueeze_333: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_166, 0)
    unsqueeze_334: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 2);  unsqueeze_333 = None
    unsqueeze_335: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 3);  unsqueeze_334 = None
    mul_247: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_166, unsqueeze_335);  view_166 = unsqueeze_335 = None
    add_193: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_247, unsqueeze_332);  mul_247 = unsqueeze_332 = None
    squeeze_110: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_111, [2, 3]);  getitem_111 = None
    squeeze_111: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_55, [2, 3]);  rsqrt_55 = None
    alias_110: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_110);  squeeze_110 = None
    alias_111: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_111);  squeeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_57: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_193, primals_333, primals_334, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_248: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_57, 0.5)
    mul_249: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_57, 0.7071067811865476)
    erf_27: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_249);  mul_249 = None
    add_194: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
    mul_250: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_248, add_194);  mul_248 = add_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_54: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_250);  mul_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_58: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_54, primals_335, primals_336, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_55: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_58)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_167: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_168, [384, 1, 1])
    mul_251: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_55, view_167);  clone_55 = view_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_195: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_191, mul_251);  add_191 = mul_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_168: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_195, [8, 1, 384, 196])
    var_mean_56 = torch.ops.aten.var_mean.correction(view_168, [2, 3], correction = 0, keepdim = True)
    getitem_112: "f32[8, 1, 1, 1]" = var_mean_56[0]
    getitem_113: "f32[8, 1, 1, 1]" = var_mean_56[1];  var_mean_56 = None
    add_196: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
    rsqrt_56: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_196);  add_196 = None
    sub_84: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_168, getitem_113);  view_168 = None
    mul_252: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_56);  sub_84 = None
    view_169: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_252, [8, 384, 14, 14]);  mul_252 = None
    unsqueeze_336: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_170, 0);  primals_170 = None
    unsqueeze_337: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, 2);  unsqueeze_336 = None
    unsqueeze_338: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 3);  unsqueeze_337 = None
    unsqueeze_339: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_169, 0)
    unsqueeze_340: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 2);  unsqueeze_339 = None
    unsqueeze_341: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, 3);  unsqueeze_340 = None
    mul_253: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_169, unsqueeze_341);  view_169 = unsqueeze_341 = None
    add_197: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_253, unsqueeze_338);  mul_253 = unsqueeze_338 = None
    squeeze_112: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_113, [2, 3]);  getitem_113 = None
    squeeze_113: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_56, [2, 3]);  rsqrt_56 = None
    alias_112: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_112);  squeeze_112 = None
    alias_113: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_113);  squeeze_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_28: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d.default(add_197, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_85: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_28, add_197)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_170: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_171, [384, 1, 1])
    mul_254: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_85, view_170);  sub_85 = view_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_198: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_195, mul_254);  add_195 = mul_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_171: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_198, [8, 1, 384, 196])
    var_mean_57 = torch.ops.aten.var_mean.correction(view_171, [2, 3], correction = 0, keepdim = True)
    getitem_114: "f32[8, 1, 1, 1]" = var_mean_57[0]
    getitem_115: "f32[8, 1, 1, 1]" = var_mean_57[1];  var_mean_57 = None
    add_199: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-05);  getitem_114 = None
    rsqrt_57: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
    sub_86: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_171, getitem_115);  view_171 = None
    mul_255: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_57);  sub_86 = None
    view_172: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_255, [8, 384, 14, 14]);  mul_255 = None
    unsqueeze_342: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_173, 0);  primals_173 = None
    unsqueeze_343: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 2);  unsqueeze_342 = None
    unsqueeze_344: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 3);  unsqueeze_343 = None
    unsqueeze_345: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_172, 0)
    unsqueeze_346: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_345, 2);  unsqueeze_345 = None
    unsqueeze_347: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 3);  unsqueeze_346 = None
    mul_256: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_172, unsqueeze_347);  view_172 = unsqueeze_347 = None
    add_200: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_256, unsqueeze_344);  mul_256 = unsqueeze_344 = None
    squeeze_114: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_115, [2, 3]);  getitem_115 = None
    squeeze_115: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_57, [2, 3]);  rsqrt_57 = None
    alias_114: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_114);  squeeze_114 = None
    alias_115: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_115);  squeeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_59: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_200, primals_337, primals_338, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_257: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_59, 0.5)
    mul_258: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_59, 0.7071067811865476)
    erf_28: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_258);  mul_258 = None
    add_201: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
    mul_259: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_257, add_201);  mul_257 = add_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_56: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_259);  mul_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_60: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_56, primals_339, primals_340, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_57: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_60)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_173: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_174, [384, 1, 1])
    mul_260: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_57, view_173);  clone_57 = view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_202: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_198, mul_260);  add_198 = mul_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_174: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_202, [8, 1, 384, 196])
    var_mean_58 = torch.ops.aten.var_mean.correction(view_174, [2, 3], correction = 0, keepdim = True)
    getitem_116: "f32[8, 1, 1, 1]" = var_mean_58[0]
    getitem_117: "f32[8, 1, 1, 1]" = var_mean_58[1];  var_mean_58 = None
    add_203: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-05);  getitem_116 = None
    rsqrt_58: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
    sub_87: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_174, getitem_117);  view_174 = None
    mul_261: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_58);  sub_87 = None
    view_175: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_261, [8, 384, 14, 14]);  mul_261 = None
    unsqueeze_348: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_176, 0);  primals_176 = None
    unsqueeze_349: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, 2);  unsqueeze_348 = None
    unsqueeze_350: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 3);  unsqueeze_349 = None
    unsqueeze_351: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_175, 0)
    unsqueeze_352: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 2);  unsqueeze_351 = None
    unsqueeze_353: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, 3);  unsqueeze_352 = None
    mul_262: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_175, unsqueeze_353);  view_175 = unsqueeze_353 = None
    add_204: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_262, unsqueeze_350);  mul_262 = unsqueeze_350 = None
    squeeze_116: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_117, [2, 3]);  getitem_117 = None
    squeeze_117: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_58, [2, 3]);  rsqrt_58 = None
    alias_116: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_116);  squeeze_116 = None
    alias_117: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_117);  squeeze_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_29: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d.default(add_204, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_88: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_29, add_204)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_176: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_177, [384, 1, 1])
    mul_263: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_88, view_176);  sub_88 = view_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_205: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_202, mul_263);  add_202 = mul_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_177: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_205, [8, 1, 384, 196])
    var_mean_59 = torch.ops.aten.var_mean.correction(view_177, [2, 3], correction = 0, keepdim = True)
    getitem_118: "f32[8, 1, 1, 1]" = var_mean_59[0]
    getitem_119: "f32[8, 1, 1, 1]" = var_mean_59[1];  var_mean_59 = None
    add_206: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05);  getitem_118 = None
    rsqrt_59: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_206);  add_206 = None
    sub_89: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_177, getitem_119);  view_177 = None
    mul_264: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_59);  sub_89 = None
    view_178: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_264, [8, 384, 14, 14]);  mul_264 = None
    unsqueeze_354: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_179, 0);  primals_179 = None
    unsqueeze_355: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, 2);  unsqueeze_354 = None
    unsqueeze_356: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 3);  unsqueeze_355 = None
    unsqueeze_357: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_178, 0)
    unsqueeze_358: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 2);  unsqueeze_357 = None
    unsqueeze_359: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 3);  unsqueeze_358 = None
    mul_265: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_178, unsqueeze_359);  view_178 = unsqueeze_359 = None
    add_207: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_265, unsqueeze_356);  mul_265 = unsqueeze_356 = None
    squeeze_118: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_119, [2, 3]);  getitem_119 = None
    squeeze_119: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_59, [2, 3]);  rsqrt_59 = None
    alias_118: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_118);  squeeze_118 = None
    alias_119: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_119);  squeeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_61: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_207, primals_341, primals_342, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_266: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_61, 0.5)
    mul_267: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_61, 0.7071067811865476)
    erf_29: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_267);  mul_267 = None
    add_208: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
    mul_268: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_266, add_208);  mul_266 = add_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_58: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_268);  mul_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_62: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_58, primals_343, primals_344, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_59: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_62)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_179: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_180, [384, 1, 1])
    mul_269: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_59, view_179);  clone_59 = view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_209: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_205, mul_269);  add_205 = mul_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:103, code: x = self.conv(x)
    convolution_63: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(add_209, primals_345, primals_346, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  primals_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_180: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(convolution_63, [8, 1, 768, 49])
    var_mean_60 = torch.ops.aten.var_mean.correction(view_180, [2, 3], correction = 0, keepdim = True)
    getitem_120: "f32[8, 1, 1, 1]" = var_mean_60[0]
    getitem_121: "f32[8, 1, 1, 1]" = var_mean_60[1];  var_mean_60 = None
    add_210: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-05);  getitem_120 = None
    rsqrt_60: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_210);  add_210 = None
    sub_90: "f32[8, 1, 768, 49]" = torch.ops.aten.sub.Tensor(view_180, getitem_121);  view_180 = None
    mul_270: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_60);  sub_90 = None
    view_181: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(mul_270, [8, 768, 7, 7]);  mul_270 = None
    unsqueeze_360: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_182, 0);  primals_182 = None
    unsqueeze_361: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, 2);  unsqueeze_360 = None
    unsqueeze_362: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 3);  unsqueeze_361 = None
    unsqueeze_363: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_181, 0)
    unsqueeze_364: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 2);  unsqueeze_363 = None
    unsqueeze_365: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, 3);  unsqueeze_364 = None
    mul_271: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(view_181, unsqueeze_365);  view_181 = unsqueeze_365 = None
    add_211: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_271, unsqueeze_362);  mul_271 = unsqueeze_362 = None
    squeeze_120: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_121, [2, 3]);  getitem_121 = None
    squeeze_121: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_60, [2, 3]);  rsqrt_60 = None
    alias_120: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_120);  squeeze_120 = None
    alias_121: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_121);  squeeze_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_30: "f32[8, 768, 7, 7]" = torch.ops.aten.avg_pool2d.default(add_211, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_91: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(avg_pool2d_30, add_211)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_182: "f32[768, 1, 1]" = torch.ops.aten.view.default(primals_183, [768, 1, 1])
    mul_272: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_91, view_182);  sub_91 = view_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_212: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(convolution_63, mul_272);  mul_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_183: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_212, [8, 1, 768, 49])
    var_mean_61 = torch.ops.aten.var_mean.correction(view_183, [2, 3], correction = 0, keepdim = True)
    getitem_122: "f32[8, 1, 1, 1]" = var_mean_61[0]
    getitem_123: "f32[8, 1, 1, 1]" = var_mean_61[1];  var_mean_61 = None
    add_213: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-05);  getitem_122 = None
    rsqrt_61: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
    sub_92: "f32[8, 1, 768, 49]" = torch.ops.aten.sub.Tensor(view_183, getitem_123);  view_183 = None
    mul_273: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_61);  sub_92 = None
    view_184: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(mul_273, [8, 768, 7, 7]);  mul_273 = None
    unsqueeze_366: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_185, 0);  primals_185 = None
    unsqueeze_367: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, 2);  unsqueeze_366 = None
    unsqueeze_368: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 3);  unsqueeze_367 = None
    unsqueeze_369: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_184, 0)
    unsqueeze_370: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_369, 2);  unsqueeze_369 = None
    unsqueeze_371: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 3);  unsqueeze_370 = None
    mul_274: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(view_184, unsqueeze_371);  view_184 = unsqueeze_371 = None
    add_214: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_274, unsqueeze_368);  mul_274 = unsqueeze_368 = None
    squeeze_122: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_123, [2, 3]);  getitem_123 = None
    squeeze_123: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_61, [2, 3]);  rsqrt_61 = None
    alias_122: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_122);  squeeze_122 = None
    alias_123: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_123);  squeeze_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_64: "f32[8, 3072, 7, 7]" = torch.ops.aten.convolution.default(add_214, primals_347, primals_348, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_275: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_64, 0.5)
    mul_276: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_64, 0.7071067811865476)
    erf_30: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_276);  mul_276 = None
    add_215: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
    mul_277: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_275, add_215);  mul_275 = add_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_60: "f32[8, 3072, 7, 7]" = torch.ops.aten.clone.default(mul_277);  mul_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_65: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(clone_60, primals_349, primals_350, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_61: "f32[8, 768, 7, 7]" = torch.ops.aten.clone.default(convolution_65)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_185: "f32[768, 1, 1]" = torch.ops.aten.view.default(primals_186, [768, 1, 1])
    mul_278: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(clone_61, view_185);  clone_61 = view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_216: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_212, mul_278);  add_212 = mul_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_186: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_216, [8, 1, 768, 49])
    var_mean_62 = torch.ops.aten.var_mean.correction(view_186, [2, 3], correction = 0, keepdim = True)
    getitem_124: "f32[8, 1, 1, 1]" = var_mean_62[0]
    getitem_125: "f32[8, 1, 1, 1]" = var_mean_62[1];  var_mean_62 = None
    add_217: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-05);  getitem_124 = None
    rsqrt_62: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_217);  add_217 = None
    sub_93: "f32[8, 1, 768, 49]" = torch.ops.aten.sub.Tensor(view_186, getitem_125);  view_186 = None
    mul_279: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(sub_93, rsqrt_62);  sub_93 = None
    view_187: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(mul_279, [8, 768, 7, 7]);  mul_279 = None
    unsqueeze_372: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_188, 0);  primals_188 = None
    unsqueeze_373: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, 2);  unsqueeze_372 = None
    unsqueeze_374: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 3);  unsqueeze_373 = None
    unsqueeze_375: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_187, 0)
    unsqueeze_376: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_375, 2);  unsqueeze_375 = None
    unsqueeze_377: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 3);  unsqueeze_376 = None
    mul_280: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(view_187, unsqueeze_377);  view_187 = unsqueeze_377 = None
    add_218: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_280, unsqueeze_374);  mul_280 = unsqueeze_374 = None
    squeeze_124: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_125, [2, 3]);  getitem_125 = None
    squeeze_125: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_62, [2, 3]);  rsqrt_62 = None
    alias_124: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_124);  squeeze_124 = None
    alias_125: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_125);  squeeze_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_31: "f32[8, 768, 7, 7]" = torch.ops.aten.avg_pool2d.default(add_218, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_94: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(avg_pool2d_31, add_218)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_188: "f32[768, 1, 1]" = torch.ops.aten.view.default(primals_189, [768, 1, 1])
    mul_281: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_94, view_188);  sub_94 = view_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_219: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_216, mul_281);  add_216 = mul_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_189: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_219, [8, 1, 768, 49])
    var_mean_63 = torch.ops.aten.var_mean.correction(view_189, [2, 3], correction = 0, keepdim = True)
    getitem_126: "f32[8, 1, 1, 1]" = var_mean_63[0]
    getitem_127: "f32[8, 1, 1, 1]" = var_mean_63[1];  var_mean_63 = None
    add_220: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-05);  getitem_126 = None
    rsqrt_63: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_220);  add_220 = None
    sub_95: "f32[8, 1, 768, 49]" = torch.ops.aten.sub.Tensor(view_189, getitem_127);  view_189 = None
    mul_282: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_63);  sub_95 = None
    view_190: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(mul_282, [8, 768, 7, 7]);  mul_282 = None
    unsqueeze_378: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_191, 0);  primals_191 = None
    unsqueeze_379: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, 2);  unsqueeze_378 = None
    unsqueeze_380: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 3);  unsqueeze_379 = None
    unsqueeze_381: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_190, 0)
    unsqueeze_382: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_381, 2);  unsqueeze_381 = None
    unsqueeze_383: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 3);  unsqueeze_382 = None
    mul_283: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(view_190, unsqueeze_383);  view_190 = unsqueeze_383 = None
    add_221: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_283, unsqueeze_380);  mul_283 = unsqueeze_380 = None
    squeeze_126: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_127, [2, 3]);  getitem_127 = None
    squeeze_127: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_63, [2, 3]);  rsqrt_63 = None
    alias_126: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_126);  squeeze_126 = None
    alias_127: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_127);  squeeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_66: "f32[8, 3072, 7, 7]" = torch.ops.aten.convolution.default(add_221, primals_351, primals_352, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_284: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_66, 0.5)
    mul_285: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_66, 0.7071067811865476)
    erf_31: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_285);  mul_285 = None
    add_222: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
    mul_286: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_284, add_222);  mul_284 = add_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_62: "f32[8, 3072, 7, 7]" = torch.ops.aten.clone.default(mul_286);  mul_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_67: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(clone_62, primals_353, primals_354, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_63: "f32[8, 768, 7, 7]" = torch.ops.aten.clone.default(convolution_67)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_191: "f32[768, 1, 1]" = torch.ops.aten.view.default(primals_192, [768, 1, 1])
    mul_287: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(clone_63, view_191);  clone_63 = view_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_223: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_219, mul_287);  add_219 = mul_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_192: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_223, [8, 1, 768, 49])
    var_mean_64 = torch.ops.aten.var_mean.correction(view_192, [2, 3], correction = 0, keepdim = True)
    getitem_128: "f32[8, 1, 1, 1]" = var_mean_64[0]
    getitem_129: "f32[8, 1, 1, 1]" = var_mean_64[1];  var_mean_64 = None
    add_224: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-05);  getitem_128 = None
    rsqrt_64: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_224);  add_224 = None
    sub_96: "f32[8, 1, 768, 49]" = torch.ops.aten.sub.Tensor(view_192, getitem_129);  view_192 = None
    mul_288: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(sub_96, rsqrt_64);  sub_96 = None
    view_193: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(mul_288, [8, 768, 7, 7]);  mul_288 = None
    unsqueeze_384: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_194, 0);  primals_194 = None
    unsqueeze_385: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, 2);  unsqueeze_384 = None
    unsqueeze_386: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 3);  unsqueeze_385 = None
    unsqueeze_387: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_193, 0)
    unsqueeze_388: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_387, 2);  unsqueeze_387 = None
    unsqueeze_389: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 3);  unsqueeze_388 = None
    mul_289: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(view_193, unsqueeze_389);  view_193 = unsqueeze_389 = None
    add_225: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_289, unsqueeze_386);  mul_289 = unsqueeze_386 = None
    squeeze_128: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_129, [2, 3]);  getitem_129 = None
    squeeze_129: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_64, [2, 3]);  rsqrt_64 = None
    alias_128: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_128);  squeeze_128 = None
    alias_129: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_129);  squeeze_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_32: "f32[8, 768, 7, 7]" = torch.ops.aten.avg_pool2d.default(add_225, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_97: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(avg_pool2d_32, add_225)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_194: "f32[768, 1, 1]" = torch.ops.aten.view.default(primals_195, [768, 1, 1])
    mul_290: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_97, view_194);  sub_97 = view_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_226: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_223, mul_290);  add_223 = mul_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_195: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_226, [8, 1, 768, 49])
    var_mean_65 = torch.ops.aten.var_mean.correction(view_195, [2, 3], correction = 0, keepdim = True)
    getitem_130: "f32[8, 1, 1, 1]" = var_mean_65[0]
    getitem_131: "f32[8, 1, 1, 1]" = var_mean_65[1];  var_mean_65 = None
    add_227: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
    rsqrt_65: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_227);  add_227 = None
    sub_98: "f32[8, 1, 768, 49]" = torch.ops.aten.sub.Tensor(view_195, getitem_131);  view_195 = None
    mul_291: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(sub_98, rsqrt_65);  sub_98 = None
    view_196: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(mul_291, [8, 768, 7, 7]);  mul_291 = None
    unsqueeze_390: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_197, 0);  primals_197 = None
    unsqueeze_391: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, 2);  unsqueeze_390 = None
    unsqueeze_392: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 3);  unsqueeze_391 = None
    unsqueeze_393: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_196, 0)
    unsqueeze_394: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 2);  unsqueeze_393 = None
    unsqueeze_395: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 3);  unsqueeze_394 = None
    mul_292: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(view_196, unsqueeze_395);  view_196 = unsqueeze_395 = None
    add_228: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_292, unsqueeze_392);  mul_292 = unsqueeze_392 = None
    squeeze_130: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_131, [2, 3]);  getitem_131 = None
    squeeze_131: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_65, [2, 3]);  rsqrt_65 = None
    alias_130: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_130);  squeeze_130 = None
    alias_131: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_131);  squeeze_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_68: "f32[8, 3072, 7, 7]" = torch.ops.aten.convolution.default(add_228, primals_355, primals_356, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_293: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_68, 0.5)
    mul_294: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_68, 0.7071067811865476)
    erf_32: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_294);  mul_294 = None
    add_229: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
    mul_295: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_293, add_229);  mul_293 = add_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_64: "f32[8, 3072, 7, 7]" = torch.ops.aten.clone.default(mul_295);  mul_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_69: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(clone_64, primals_357, primals_358, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_65: "f32[8, 768, 7, 7]" = torch.ops.aten.clone.default(convolution_69)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_197: "f32[768, 1, 1]" = torch.ops.aten.view.default(primals_198, [768, 1, 1])
    mul_296: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(clone_65, view_197);  clone_65 = view_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_230: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_226, mul_296);  add_226 = mul_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_198: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_230, [8, 1, 768, 49])
    var_mean_66 = torch.ops.aten.var_mean.correction(view_198, [2, 3], correction = 0, keepdim = True)
    getitem_132: "f32[8, 1, 1, 1]" = var_mean_66[0]
    getitem_133: "f32[8, 1, 1, 1]" = var_mean_66[1];  var_mean_66 = None
    add_231: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-05);  getitem_132 = None
    rsqrt_66: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_231);  add_231 = None
    sub_99: "f32[8, 1, 768, 49]" = torch.ops.aten.sub.Tensor(view_198, getitem_133);  view_198 = None
    mul_297: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(sub_99, rsqrt_66);  sub_99 = None
    view_199: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(mul_297, [8, 768, 7, 7]);  mul_297 = None
    unsqueeze_396: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_200, 0);  primals_200 = None
    unsqueeze_397: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, 2);  unsqueeze_396 = None
    unsqueeze_398: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 3);  unsqueeze_397 = None
    unsqueeze_399: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_199, 0)
    unsqueeze_400: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 2);  unsqueeze_399 = None
    unsqueeze_401: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, 3);  unsqueeze_400 = None
    mul_298: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(view_199, unsqueeze_401);  view_199 = unsqueeze_401 = None
    add_232: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_298, unsqueeze_398);  mul_298 = unsqueeze_398 = None
    squeeze_132: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_133, [2, 3]);  getitem_133 = None
    squeeze_133: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_66, [2, 3]);  rsqrt_66 = None
    alias_132: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_132);  squeeze_132 = None
    alias_133: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_133);  squeeze_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_33: "f32[8, 768, 7, 7]" = torch.ops.aten.avg_pool2d.default(add_232, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_100: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(avg_pool2d_33, add_232)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_200: "f32[768, 1, 1]" = torch.ops.aten.view.default(primals_201, [768, 1, 1])
    mul_299: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_100, view_200);  sub_100 = view_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_233: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_230, mul_299);  add_230 = mul_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_201: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_233, [8, 1, 768, 49])
    var_mean_67 = torch.ops.aten.var_mean.correction(view_201, [2, 3], correction = 0, keepdim = True)
    getitem_134: "f32[8, 1, 1, 1]" = var_mean_67[0]
    getitem_135: "f32[8, 1, 1, 1]" = var_mean_67[1];  var_mean_67 = None
    add_234: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_134, 1e-05);  getitem_134 = None
    rsqrt_67: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_234);  add_234 = None
    sub_101: "f32[8, 1, 768, 49]" = torch.ops.aten.sub.Tensor(view_201, getitem_135);  view_201 = None
    mul_300: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(sub_101, rsqrt_67);  sub_101 = None
    view_202: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(mul_300, [8, 768, 7, 7]);  mul_300 = None
    unsqueeze_402: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_203, 0);  primals_203 = None
    unsqueeze_403: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, 2);  unsqueeze_402 = None
    unsqueeze_404: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 3);  unsqueeze_403 = None
    unsqueeze_405: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_202, 0)
    unsqueeze_406: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 2);  unsqueeze_405 = None
    unsqueeze_407: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 3);  unsqueeze_406 = None
    mul_301: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(view_202, unsqueeze_407);  view_202 = unsqueeze_407 = None
    add_235: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_301, unsqueeze_404);  mul_301 = unsqueeze_404 = None
    squeeze_134: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_135, [2, 3]);  getitem_135 = None
    squeeze_135: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_67, [2, 3]);  rsqrt_67 = None
    alias_134: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_134);  squeeze_134 = None
    alias_135: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_135);  squeeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_70: "f32[8, 3072, 7, 7]" = torch.ops.aten.convolution.default(add_235, primals_359, primals_360, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_302: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_70, 0.5)
    mul_303: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_70, 0.7071067811865476)
    erf_33: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_303);  mul_303 = None
    add_236: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
    mul_304: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_302, add_236);  mul_302 = add_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_66: "f32[8, 3072, 7, 7]" = torch.ops.aten.clone.default(mul_304);  mul_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_71: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(clone_66, primals_361, primals_362, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_67: "f32[8, 768, 7, 7]" = torch.ops.aten.clone.default(convolution_71)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_203: "f32[768, 1, 1]" = torch.ops.aten.view.default(primals_204, [768, 1, 1])
    mul_305: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(clone_67, view_203);  clone_67 = view_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_237: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_233, mul_305);  add_233 = mul_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_204: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_237, [8, 1, 768, 49])
    var_mean_68 = torch.ops.aten.var_mean.correction(view_204, [2, 3], correction = 0, keepdim = True)
    getitem_136: "f32[8, 1, 1, 1]" = var_mean_68[0]
    getitem_137: "f32[8, 1, 1, 1]" = var_mean_68[1];  var_mean_68 = None
    add_238: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_136, 1e-05);  getitem_136 = None
    rsqrt_68: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_238);  add_238 = None
    sub_102: "f32[8, 1, 768, 49]" = torch.ops.aten.sub.Tensor(view_204, getitem_137);  view_204 = None
    mul_306: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(sub_102, rsqrt_68);  sub_102 = None
    view_205: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(mul_306, [8, 768, 7, 7]);  mul_306 = None
    unsqueeze_408: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_206, 0);  primals_206 = None
    unsqueeze_409: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, 2);  unsqueeze_408 = None
    unsqueeze_410: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 3);  unsqueeze_409 = None
    unsqueeze_411: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_205, 0)
    unsqueeze_412: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 2);  unsqueeze_411 = None
    unsqueeze_413: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, 3);  unsqueeze_412 = None
    mul_307: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(view_205, unsqueeze_413);  view_205 = unsqueeze_413 = None
    add_239: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_307, unsqueeze_410);  mul_307 = unsqueeze_410 = None
    squeeze_136: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_137, [2, 3]);  getitem_137 = None
    squeeze_137: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_68, [2, 3]);  rsqrt_68 = None
    alias_136: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_136);  squeeze_136 = None
    alias_137: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_137);  squeeze_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_34: "f32[8, 768, 7, 7]" = torch.ops.aten.avg_pool2d.default(add_239, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_103: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(avg_pool2d_34, add_239)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_206: "f32[768, 1, 1]" = torch.ops.aten.view.default(primals_207, [768, 1, 1])
    mul_308: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_103, view_206);  sub_103 = view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_240: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_237, mul_308);  add_237 = mul_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_207: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_240, [8, 1, 768, 49])
    var_mean_69 = torch.ops.aten.var_mean.correction(view_207, [2, 3], correction = 0, keepdim = True)
    getitem_138: "f32[8, 1, 1, 1]" = var_mean_69[0]
    getitem_139: "f32[8, 1, 1, 1]" = var_mean_69[1];  var_mean_69 = None
    add_241: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_138, 1e-05);  getitem_138 = None
    rsqrt_69: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_241);  add_241 = None
    sub_104: "f32[8, 1, 768, 49]" = torch.ops.aten.sub.Tensor(view_207, getitem_139);  view_207 = None
    mul_309: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(sub_104, rsqrt_69);  sub_104 = None
    view_208: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(mul_309, [8, 768, 7, 7]);  mul_309 = None
    unsqueeze_414: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_209, 0);  primals_209 = None
    unsqueeze_415: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, 2);  unsqueeze_414 = None
    unsqueeze_416: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 3);  unsqueeze_415 = None
    unsqueeze_417: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_208, 0)
    unsqueeze_418: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 2);  unsqueeze_417 = None
    unsqueeze_419: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 3);  unsqueeze_418 = None
    mul_310: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(view_208, unsqueeze_419);  view_208 = unsqueeze_419 = None
    add_242: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_310, unsqueeze_416);  mul_310 = unsqueeze_416 = None
    squeeze_138: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_139, [2, 3]);  getitem_139 = None
    squeeze_139: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_69, [2, 3]);  rsqrt_69 = None
    alias_138: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_138);  squeeze_138 = None
    alias_139: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_139);  squeeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_72: "f32[8, 3072, 7, 7]" = torch.ops.aten.convolution.default(add_242, primals_363, primals_364, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_311: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_72, 0.5)
    mul_312: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_72, 0.7071067811865476)
    erf_34: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_312);  mul_312 = None
    add_243: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
    mul_313: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_311, add_243);  mul_311 = add_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_68: "f32[8, 3072, 7, 7]" = torch.ops.aten.clone.default(mul_313);  mul_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_73: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(clone_68, primals_365, primals_366, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_69: "f32[8, 768, 7, 7]" = torch.ops.aten.clone.default(convolution_73)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_209: "f32[768, 1, 1]" = torch.ops.aten.view.default(primals_210, [768, 1, 1])
    mul_314: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(clone_69, view_209);  clone_69 = view_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_244: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_240, mul_314);  add_240 = mul_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_210: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_244, [8, 1, 768, 49])
    var_mean_70 = torch.ops.aten.var_mean.correction(view_210, [2, 3], correction = 0, keepdim = True)
    getitem_140: "f32[8, 1, 1, 1]" = var_mean_70[0]
    getitem_141: "f32[8, 1, 1, 1]" = var_mean_70[1];  var_mean_70 = None
    add_245: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_140, 1e-05);  getitem_140 = None
    rsqrt_70: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_245);  add_245 = None
    sub_105: "f32[8, 1, 768, 49]" = torch.ops.aten.sub.Tensor(view_210, getitem_141);  view_210 = None
    mul_315: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(sub_105, rsqrt_70);  sub_105 = None
    view_211: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(mul_315, [8, 768, 7, 7]);  mul_315 = None
    unsqueeze_420: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_212, 0);  primals_212 = None
    unsqueeze_421: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, 2);  unsqueeze_420 = None
    unsqueeze_422: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 3);  unsqueeze_421 = None
    unsqueeze_423: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_211, 0)
    unsqueeze_424: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 2);  unsqueeze_423 = None
    unsqueeze_425: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 3);  unsqueeze_424 = None
    mul_316: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(view_211, unsqueeze_425);  view_211 = unsqueeze_425 = None
    add_246: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_316, unsqueeze_422);  mul_316 = unsqueeze_422 = None
    squeeze_140: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_141, [2, 3]);  getitem_141 = None
    squeeze_141: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_70, [2, 3]);  rsqrt_70 = None
    alias_140: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_140);  squeeze_140 = None
    alias_141: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_141);  squeeze_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_35: "f32[8, 768, 7, 7]" = torch.ops.aten.avg_pool2d.default(add_246, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_106: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(avg_pool2d_35, add_246)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_212: "f32[768, 1, 1]" = torch.ops.aten.view.default(primals_213, [768, 1, 1])
    mul_317: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_106, view_212);  sub_106 = view_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_247: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_244, mul_317);  add_244 = mul_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_213: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_247, [8, 1, 768, 49])
    var_mean_71 = torch.ops.aten.var_mean.correction(view_213, [2, 3], correction = 0, keepdim = True)
    getitem_142: "f32[8, 1, 1, 1]" = var_mean_71[0]
    getitem_143: "f32[8, 1, 1, 1]" = var_mean_71[1];  var_mean_71 = None
    add_248: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_142, 1e-05);  getitem_142 = None
    rsqrt_71: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_248);  add_248 = None
    sub_107: "f32[8, 1, 768, 49]" = torch.ops.aten.sub.Tensor(view_213, getitem_143);  view_213 = None
    mul_318: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(sub_107, rsqrt_71);  sub_107 = None
    view_214: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(mul_318, [8, 768, 7, 7]);  mul_318 = None
    unsqueeze_426: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_215, 0);  primals_215 = None
    unsqueeze_427: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 2);  unsqueeze_426 = None
    unsqueeze_428: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 3);  unsqueeze_427 = None
    unsqueeze_429: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_214, 0)
    unsqueeze_430: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 2);  unsqueeze_429 = None
    unsqueeze_431: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 3);  unsqueeze_430 = None
    mul_319: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(view_214, unsqueeze_431);  view_214 = unsqueeze_431 = None
    add_249: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_319, unsqueeze_428);  mul_319 = unsqueeze_428 = None
    squeeze_142: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_143, [2, 3]);  getitem_143 = None
    squeeze_143: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_71, [2, 3]);  rsqrt_71 = None
    alias_142: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_142);  squeeze_142 = None
    alias_143: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_143);  squeeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_74: "f32[8, 3072, 7, 7]" = torch.ops.aten.convolution.default(add_249, primals_367, primals_368, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_320: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_74, 0.5)
    mul_321: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_74, 0.7071067811865476)
    erf_35: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_321);  mul_321 = None
    add_250: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
    mul_322: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_320, add_250);  mul_320 = add_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_70: "f32[8, 3072, 7, 7]" = torch.ops.aten.clone.default(mul_322);  mul_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_75: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(clone_70, primals_369, primals_370, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_71: "f32[8, 768, 7, 7]" = torch.ops.aten.clone.default(convolution_75)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_215: "f32[768, 1, 1]" = torch.ops.aten.view.default(primals_216, [768, 1, 1])
    mul_323: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(clone_71, view_215);  clone_71 = view_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_251: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_247, mul_323);  add_247 = mul_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean: "f32[8, 768, 1, 1]" = torch.ops.aten.mean.dim(add_251, [-1, -2], True);  add_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:68, code: x = x.permute(0, 2, 3, 1)
    permute: "f32[8, 1, 1, 768]" = torch.ops.aten.permute.default(mean, [0, 2, 3, 1]);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_72 = torch.ops.aten.var_mean.correction(permute, [3], correction = 0, keepdim = True)
    getitem_144: "f32[8, 1, 1, 1]" = var_mean_72[0]
    getitem_145: "f32[8, 1, 1, 1]" = var_mean_72[1];  var_mean_72 = None
    add_252: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-06);  getitem_144 = None
    rsqrt_72: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_252);  add_252 = None
    sub_108: "f32[8, 1, 1, 768]" = torch.ops.aten.sub.Tensor(permute, getitem_145);  permute = getitem_145 = None
    mul_324: "f32[8, 1, 1, 768]" = torch.ops.aten.mul.Tensor(sub_108, rsqrt_72);  sub_108 = None
    mul_325: "f32[8, 1, 1, 768]" = torch.ops.aten.mul.Tensor(mul_324, primals_217)
    add_253: "f32[8, 1, 1, 768]" = torch.ops.aten.add.Tensor(mul_325, primals_218);  mul_325 = primals_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:73, code: x = x.permute(0, 3, 1, 2)
    permute_1: "f32[8, 768, 1, 1]" = torch.ops.aten.permute.default(add_253, [0, 3, 1, 2]);  add_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:600, code: x = self.head.flatten(x)
    view_216: "f32[8, 768]" = torch.ops.aten.view.default(permute_1, [8, 768]);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:602, code: return x if pre_logits else self.head.fc(x)
    permute_2: "f32[768, 1000]" = torch.ops.aten.permute.default(primals_371, [1, 0]);  primals_371 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_372, view_216, permute_2);  primals_372 = None
    permute_3: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_72, 768);  rsqrt_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_144: "f32[8, 1]" = torch.ops.aten.alias.default(alias_142);  alias_142 = None
    alias_145: "f32[8, 1]" = torch.ops.aten.alias.default(alias_143);  alias_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_146: "f32[8, 1]" = torch.ops.aten.alias.default(alias_140);  alias_140 = None
    alias_147: "f32[8, 1]" = torch.ops.aten.alias.default(alias_141);  alias_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_148: "f32[8, 1]" = torch.ops.aten.alias.default(alias_138);  alias_138 = None
    alias_149: "f32[8, 1]" = torch.ops.aten.alias.default(alias_139);  alias_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_150: "f32[8, 1]" = torch.ops.aten.alias.default(alias_136);  alias_136 = None
    alias_151: "f32[8, 1]" = torch.ops.aten.alias.default(alias_137);  alias_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_152: "f32[8, 1]" = torch.ops.aten.alias.default(alias_134);  alias_134 = None
    alias_153: "f32[8, 1]" = torch.ops.aten.alias.default(alias_135);  alias_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_154: "f32[8, 1]" = torch.ops.aten.alias.default(alias_132);  alias_132 = None
    alias_155: "f32[8, 1]" = torch.ops.aten.alias.default(alias_133);  alias_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_156: "f32[8, 1]" = torch.ops.aten.alias.default(alias_130);  alias_130 = None
    alias_157: "f32[8, 1]" = torch.ops.aten.alias.default(alias_131);  alias_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_158: "f32[8, 1]" = torch.ops.aten.alias.default(alias_128);  alias_128 = None
    alias_159: "f32[8, 1]" = torch.ops.aten.alias.default(alias_129);  alias_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_160: "f32[8, 1]" = torch.ops.aten.alias.default(alias_126);  alias_126 = None
    alias_161: "f32[8, 1]" = torch.ops.aten.alias.default(alias_127);  alias_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_162: "f32[8, 1]" = torch.ops.aten.alias.default(alias_124);  alias_124 = None
    alias_163: "f32[8, 1]" = torch.ops.aten.alias.default(alias_125);  alias_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_164: "f32[8, 1]" = torch.ops.aten.alias.default(alias_122);  alias_122 = None
    alias_165: "f32[8, 1]" = torch.ops.aten.alias.default(alias_123);  alias_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_166: "f32[8, 1]" = torch.ops.aten.alias.default(alias_120);  alias_120 = None
    alias_167: "f32[8, 1]" = torch.ops.aten.alias.default(alias_121);  alias_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_168: "f32[8, 1]" = torch.ops.aten.alias.default(alias_118);  alias_118 = None
    alias_169: "f32[8, 1]" = torch.ops.aten.alias.default(alias_119);  alias_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_170: "f32[8, 1]" = torch.ops.aten.alias.default(alias_116);  alias_116 = None
    alias_171: "f32[8, 1]" = torch.ops.aten.alias.default(alias_117);  alias_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_172: "f32[8, 1]" = torch.ops.aten.alias.default(alias_114);  alias_114 = None
    alias_173: "f32[8, 1]" = torch.ops.aten.alias.default(alias_115);  alias_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_174: "f32[8, 1]" = torch.ops.aten.alias.default(alias_112);  alias_112 = None
    alias_175: "f32[8, 1]" = torch.ops.aten.alias.default(alias_113);  alias_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_176: "f32[8, 1]" = torch.ops.aten.alias.default(alias_110);  alias_110 = None
    alias_177: "f32[8, 1]" = torch.ops.aten.alias.default(alias_111);  alias_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_178: "f32[8, 1]" = torch.ops.aten.alias.default(alias_108);  alias_108 = None
    alias_179: "f32[8, 1]" = torch.ops.aten.alias.default(alias_109);  alias_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_180: "f32[8, 1]" = torch.ops.aten.alias.default(alias_106);  alias_106 = None
    alias_181: "f32[8, 1]" = torch.ops.aten.alias.default(alias_107);  alias_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_182: "f32[8, 1]" = torch.ops.aten.alias.default(alias_104);  alias_104 = None
    alias_183: "f32[8, 1]" = torch.ops.aten.alias.default(alias_105);  alias_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_184: "f32[8, 1]" = torch.ops.aten.alias.default(alias_102);  alias_102 = None
    alias_185: "f32[8, 1]" = torch.ops.aten.alias.default(alias_103);  alias_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_186: "f32[8, 1]" = torch.ops.aten.alias.default(alias_100);  alias_100 = None
    alias_187: "f32[8, 1]" = torch.ops.aten.alias.default(alias_101);  alias_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_188: "f32[8, 1]" = torch.ops.aten.alias.default(alias_98);  alias_98 = None
    alias_189: "f32[8, 1]" = torch.ops.aten.alias.default(alias_99);  alias_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_190: "f32[8, 1]" = torch.ops.aten.alias.default(alias_96);  alias_96 = None
    alias_191: "f32[8, 1]" = torch.ops.aten.alias.default(alias_97);  alias_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_192: "f32[8, 1]" = torch.ops.aten.alias.default(alias_94);  alias_94 = None
    alias_193: "f32[8, 1]" = torch.ops.aten.alias.default(alias_95);  alias_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_194: "f32[8, 1]" = torch.ops.aten.alias.default(alias_92);  alias_92 = None
    alias_195: "f32[8, 1]" = torch.ops.aten.alias.default(alias_93);  alias_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_196: "f32[8, 1]" = torch.ops.aten.alias.default(alias_90);  alias_90 = None
    alias_197: "f32[8, 1]" = torch.ops.aten.alias.default(alias_91);  alias_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_198: "f32[8, 1]" = torch.ops.aten.alias.default(alias_88);  alias_88 = None
    alias_199: "f32[8, 1]" = torch.ops.aten.alias.default(alias_89);  alias_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_200: "f32[8, 1]" = torch.ops.aten.alias.default(alias_86);  alias_86 = None
    alias_201: "f32[8, 1]" = torch.ops.aten.alias.default(alias_87);  alias_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_202: "f32[8, 1]" = torch.ops.aten.alias.default(alias_84);  alias_84 = None
    alias_203: "f32[8, 1]" = torch.ops.aten.alias.default(alias_85);  alias_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_204: "f32[8, 1]" = torch.ops.aten.alias.default(alias_82);  alias_82 = None
    alias_205: "f32[8, 1]" = torch.ops.aten.alias.default(alias_83);  alias_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_206: "f32[8, 1]" = torch.ops.aten.alias.default(alias_80);  alias_80 = None
    alias_207: "f32[8, 1]" = torch.ops.aten.alias.default(alias_81);  alias_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_208: "f32[8, 1]" = torch.ops.aten.alias.default(alias_78);  alias_78 = None
    alias_209: "f32[8, 1]" = torch.ops.aten.alias.default(alias_79);  alias_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_210: "f32[8, 1]" = torch.ops.aten.alias.default(alias_76);  alias_76 = None
    alias_211: "f32[8, 1]" = torch.ops.aten.alias.default(alias_77);  alias_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_212: "f32[8, 1]" = torch.ops.aten.alias.default(alias_74);  alias_74 = None
    alias_213: "f32[8, 1]" = torch.ops.aten.alias.default(alias_75);  alias_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_214: "f32[8, 1]" = torch.ops.aten.alias.default(alias_72);  alias_72 = None
    alias_215: "f32[8, 1]" = torch.ops.aten.alias.default(alias_73);  alias_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_216: "f32[8, 1]" = torch.ops.aten.alias.default(alias_70);  alias_70 = None
    alias_217: "f32[8, 1]" = torch.ops.aten.alias.default(alias_71);  alias_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_218: "f32[8, 1]" = torch.ops.aten.alias.default(alias_68);  alias_68 = None
    alias_219: "f32[8, 1]" = torch.ops.aten.alias.default(alias_69);  alias_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_220: "f32[8, 1]" = torch.ops.aten.alias.default(alias_66);  alias_66 = None
    alias_221: "f32[8, 1]" = torch.ops.aten.alias.default(alias_67);  alias_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_222: "f32[8, 1]" = torch.ops.aten.alias.default(alias_64);  alias_64 = None
    alias_223: "f32[8, 1]" = torch.ops.aten.alias.default(alias_65);  alias_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_224: "f32[8, 1]" = torch.ops.aten.alias.default(alias_62);  alias_62 = None
    alias_225: "f32[8, 1]" = torch.ops.aten.alias.default(alias_63);  alias_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_226: "f32[8, 1]" = torch.ops.aten.alias.default(alias_60);  alias_60 = None
    alias_227: "f32[8, 1]" = torch.ops.aten.alias.default(alias_61);  alias_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_228: "f32[8, 1]" = torch.ops.aten.alias.default(alias_58);  alias_58 = None
    alias_229: "f32[8, 1]" = torch.ops.aten.alias.default(alias_59);  alias_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_230: "f32[8, 1]" = torch.ops.aten.alias.default(alias_56);  alias_56 = None
    alias_231: "f32[8, 1]" = torch.ops.aten.alias.default(alias_57);  alias_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_232: "f32[8, 1]" = torch.ops.aten.alias.default(alias_54);  alias_54 = None
    alias_233: "f32[8, 1]" = torch.ops.aten.alias.default(alias_55);  alias_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_234: "f32[8, 1]" = torch.ops.aten.alias.default(alias_52);  alias_52 = None
    alias_235: "f32[8, 1]" = torch.ops.aten.alias.default(alias_53);  alias_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_236: "f32[8, 1]" = torch.ops.aten.alias.default(alias_50);  alias_50 = None
    alias_237: "f32[8, 1]" = torch.ops.aten.alias.default(alias_51);  alias_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_238: "f32[8, 1]" = torch.ops.aten.alias.default(alias_48);  alias_48 = None
    alias_239: "f32[8, 1]" = torch.ops.aten.alias.default(alias_49);  alias_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_240: "f32[8, 1]" = torch.ops.aten.alias.default(alias_46);  alias_46 = None
    alias_241: "f32[8, 1]" = torch.ops.aten.alias.default(alias_47);  alias_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_242: "f32[8, 1]" = torch.ops.aten.alias.default(alias_44);  alias_44 = None
    alias_243: "f32[8, 1]" = torch.ops.aten.alias.default(alias_45);  alias_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_244: "f32[8, 1]" = torch.ops.aten.alias.default(alias_42);  alias_42 = None
    alias_245: "f32[8, 1]" = torch.ops.aten.alias.default(alias_43);  alias_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_246: "f32[8, 1]" = torch.ops.aten.alias.default(alias_40);  alias_40 = None
    alias_247: "f32[8, 1]" = torch.ops.aten.alias.default(alias_41);  alias_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_248: "f32[8, 1]" = torch.ops.aten.alias.default(alias_38);  alias_38 = None
    alias_249: "f32[8, 1]" = torch.ops.aten.alias.default(alias_39);  alias_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_250: "f32[8, 1]" = torch.ops.aten.alias.default(alias_36);  alias_36 = None
    alias_251: "f32[8, 1]" = torch.ops.aten.alias.default(alias_37);  alias_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_252: "f32[8, 1]" = torch.ops.aten.alias.default(alias_34);  alias_34 = None
    alias_253: "f32[8, 1]" = torch.ops.aten.alias.default(alias_35);  alias_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_254: "f32[8, 1]" = torch.ops.aten.alias.default(alias_32);  alias_32 = None
    alias_255: "f32[8, 1]" = torch.ops.aten.alias.default(alias_33);  alias_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_256: "f32[8, 1]" = torch.ops.aten.alias.default(alias_30);  alias_30 = None
    alias_257: "f32[8, 1]" = torch.ops.aten.alias.default(alias_31);  alias_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_258: "f32[8, 1]" = torch.ops.aten.alias.default(alias_28);  alias_28 = None
    alias_259: "f32[8, 1]" = torch.ops.aten.alias.default(alias_29);  alias_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_260: "f32[8, 1]" = torch.ops.aten.alias.default(alias_26);  alias_26 = None
    alias_261: "f32[8, 1]" = torch.ops.aten.alias.default(alias_27);  alias_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_262: "f32[8, 1]" = torch.ops.aten.alias.default(alias_24);  alias_24 = None
    alias_263: "f32[8, 1]" = torch.ops.aten.alias.default(alias_25);  alias_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_264: "f32[8, 1]" = torch.ops.aten.alias.default(alias_22);  alias_22 = None
    alias_265: "f32[8, 1]" = torch.ops.aten.alias.default(alias_23);  alias_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_266: "f32[8, 1]" = torch.ops.aten.alias.default(alias_20);  alias_20 = None
    alias_267: "f32[8, 1]" = torch.ops.aten.alias.default(alias_21);  alias_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_268: "f32[8, 1]" = torch.ops.aten.alias.default(alias_18);  alias_18 = None
    alias_269: "f32[8, 1]" = torch.ops.aten.alias.default(alias_19);  alias_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_270: "f32[8, 1]" = torch.ops.aten.alias.default(alias_16);  alias_16 = None
    alias_271: "f32[8, 1]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_272: "f32[8, 1]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    alias_273: "f32[8, 1]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_274: "f32[8, 1]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    alias_275: "f32[8, 1]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_276: "f32[8, 1]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    alias_277: "f32[8, 1]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_278: "f32[8, 1]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    alias_279: "f32[8, 1]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_280: "f32[8, 1]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    alias_281: "f32[8, 1]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_282: "f32[8, 1]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    alias_283: "f32[8, 1]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_284: "f32[8, 1]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    alias_285: "f32[8, 1]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_286: "f32[8, 1]" = torch.ops.aten.alias.default(alias);  alias = None
    alias_287: "f32[8, 1]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    return [addmm, primals_1, primals_3, primals_4, primals_6, primals_7, primals_9, primals_10, primals_12, primals_13, primals_15, primals_16, primals_18, primals_19, primals_21, primals_22, primals_24, primals_25, primals_27, primals_28, primals_30, primals_31, primals_33, primals_34, primals_36, primals_37, primals_39, primals_40, primals_42, primals_43, primals_45, primals_46, primals_48, primals_49, primals_51, primals_52, primals_54, primals_55, primals_57, primals_58, primals_60, primals_61, primals_63, primals_64, primals_66, primals_67, primals_69, primals_70, primals_72, primals_73, primals_75, primals_76, primals_78, primals_79, primals_81, primals_82, primals_84, primals_85, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_99, primals_100, primals_102, primals_103, primals_105, primals_106, primals_108, primals_109, primals_111, primals_112, primals_114, primals_115, primals_117, primals_118, primals_120, primals_121, primals_123, primals_124, primals_126, primals_127, primals_129, primals_130, primals_132, primals_133, primals_135, primals_136, primals_138, primals_139, primals_141, primals_142, primals_144, primals_145, primals_147, primals_148, primals_150, primals_151, primals_153, primals_154, primals_156, primals_157, primals_159, primals_160, primals_162, primals_163, primals_165, primals_166, primals_168, primals_169, primals_171, primals_172, primals_174, primals_175, primals_177, primals_178, primals_180, primals_181, primals_183, primals_184, primals_186, primals_187, primals_189, primals_190, primals_192, primals_193, primals_195, primals_196, primals_198, primals_199, primals_201, primals_202, primals_204, primals_205, primals_207, primals_208, primals_210, primals_211, primals_213, primals_214, primals_216, primals_217, primals_219, primals_221, primals_223, primals_225, primals_227, primals_229, primals_231, primals_233, primals_235, primals_237, primals_239, primals_241, primals_243, primals_245, primals_247, primals_249, primals_251, primals_253, primals_255, primals_257, primals_259, primals_261, primals_263, primals_265, primals_267, primals_269, primals_271, primals_273, primals_275, primals_277, primals_279, primals_281, primals_283, primals_285, primals_287, primals_289, primals_291, primals_293, primals_295, primals_297, primals_299, primals_301, primals_303, primals_305, primals_307, primals_309, primals_311, primals_313, primals_315, primals_317, primals_319, primals_321, primals_323, primals_325, primals_327, primals_329, primals_331, primals_333, primals_335, primals_337, primals_339, primals_341, primals_343, primals_345, primals_347, primals_349, primals_351, primals_353, primals_355, primals_357, primals_359, primals_361, primals_363, primals_365, primals_367, primals_369, primals_373, convolution, add_1, avg_pool2d, add_4, convolution_1, clone, convolution_2, add_8, avg_pool2d_1, add_11, convolution_3, clone_2, convolution_4, add_15, avg_pool2d_2, add_18, convolution_5, clone_4, convolution_6, add_22, avg_pool2d_3, add_25, convolution_7, clone_6, convolution_8, add_29, avg_pool2d_4, add_32, convolution_9, clone_8, convolution_10, add_36, avg_pool2d_5, add_39, convolution_11, clone_10, convolution_12, add_41, convolution_13, add_43, avg_pool2d_6, add_46, convolution_14, clone_12, convolution_15, add_50, avg_pool2d_7, add_53, convolution_16, clone_14, convolution_17, add_57, avg_pool2d_8, add_60, convolution_18, clone_16, convolution_19, add_64, avg_pool2d_9, add_67, convolution_20, clone_18, convolution_21, add_71, avg_pool2d_10, add_74, convolution_22, clone_20, convolution_23, add_78, avg_pool2d_11, add_81, convolution_24, clone_22, convolution_25, add_83, convolution_26, add_85, avg_pool2d_12, add_88, convolution_27, clone_24, convolution_28, add_92, avg_pool2d_13, add_95, convolution_29, clone_26, convolution_30, add_99, avg_pool2d_14, add_102, convolution_31, clone_28, convolution_32, add_106, avg_pool2d_15, add_109, convolution_33, clone_30, convolution_34, add_113, avg_pool2d_16, add_116, convolution_35, clone_32, convolution_36, add_120, avg_pool2d_17, add_123, convolution_37, clone_34, convolution_38, add_127, avg_pool2d_18, add_130, convolution_39, clone_36, convolution_40, add_134, avg_pool2d_19, add_137, convolution_41, clone_38, convolution_42, add_141, avg_pool2d_20, add_144, convolution_43, clone_40, convolution_44, add_148, avg_pool2d_21, add_151, convolution_45, clone_42, convolution_46, add_155, avg_pool2d_22, add_158, convolution_47, clone_44, convolution_48, add_162, avg_pool2d_23, add_165, convolution_49, clone_46, convolution_50, add_169, avg_pool2d_24, add_172, convolution_51, clone_48, convolution_52, add_176, avg_pool2d_25, add_179, convolution_53, clone_50, convolution_54, add_183, avg_pool2d_26, add_186, convolution_55, clone_52, convolution_56, add_190, avg_pool2d_27, add_193, convolution_57, clone_54, convolution_58, add_197, avg_pool2d_28, add_200, convolution_59, clone_56, convolution_60, add_204, avg_pool2d_29, add_207, convolution_61, clone_58, convolution_62, add_209, convolution_63, add_211, avg_pool2d_30, add_214, convolution_64, clone_60, convolution_65, add_218, avg_pool2d_31, add_221, convolution_66, clone_62, convolution_67, add_225, avg_pool2d_32, add_228, convolution_68, clone_64, convolution_69, add_232, avg_pool2d_33, add_235, convolution_70, clone_66, convolution_71, add_239, avg_pool2d_34, add_242, convolution_72, clone_68, convolution_73, add_246, avg_pool2d_35, add_249, convolution_74, clone_70, convolution_75, mul_324, view_216, permute_3, div, alias_144, alias_145, alias_146, alias_147, alias_148, alias_149, alias_150, alias_151, alias_152, alias_153, alias_154, alias_155, alias_156, alias_157, alias_158, alias_159, alias_160, alias_161, alias_162, alias_163, alias_164, alias_165, alias_166, alias_167, alias_168, alias_169, alias_170, alias_171, alias_172, alias_173, alias_174, alias_175, alias_176, alias_177, alias_178, alias_179, alias_180, alias_181, alias_182, alias_183, alias_184, alias_185, alias_186, alias_187, alias_188, alias_189, alias_190, alias_191, alias_192, alias_193, alias_194, alias_195, alias_196, alias_197, alias_198, alias_199, alias_200, alias_201, alias_202, alias_203, alias_204, alias_205, alias_206, alias_207, alias_208, alias_209, alias_210, alias_211, alias_212, alias_213, alias_214, alias_215, alias_216, alias_217, alias_218, alias_219, alias_220, alias_221, alias_222, alias_223, alias_224, alias_225, alias_226, alias_227, alias_228, alias_229, alias_230, alias_231, alias_232, alias_233, alias_234, alias_235, alias_236, alias_237, alias_238, alias_239, alias_240, alias_241, alias_242, alias_243, alias_244, alias_245, alias_246, alias_247, alias_248, alias_249, alias_250, alias_251, alias_252, alias_253, alias_254, alias_255, alias_256, alias_257, alias_258, alias_259, alias_260, alias_261, alias_262, alias_263, alias_264, alias_265, alias_266, alias_267, alias_268, alias_269, alias_270, alias_271, alias_272, alias_273, alias_274, alias_275, alias_276, alias_277, alias_278, alias_279, alias_280, alias_281, alias_282, alias_283, alias_284, alias_285, alias_286, alias_287]
    