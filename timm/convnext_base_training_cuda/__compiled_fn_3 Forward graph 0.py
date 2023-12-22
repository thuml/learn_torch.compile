from __future__ import annotations



def forward(self, primals_1: "f32[128]", primals_2: "f32[128]", primals_3: "f32[128]", primals_4: "f32[128]", primals_5: "f32[128]", primals_6: "f32[128]", primals_7: "f32[128]", primals_8: "f32[128]", primals_9: "f32[128]", primals_10: "f32[128]", primals_11: "f32[128]", primals_12: "f32[128]", primals_13: "f32[128]", primals_14: "f32[256]", primals_15: "f32[256]", primals_16: "f32[256]", primals_17: "f32[256]", primals_18: "f32[256]", primals_19: "f32[256]", primals_20: "f32[256]", primals_21: "f32[256]", primals_22: "f32[256]", primals_23: "f32[256]", primals_24: "f32[256]", primals_25: "f32[512]", primals_26: "f32[512]", primals_27: "f32[512]", primals_28: "f32[512]", primals_29: "f32[512]", primals_30: "f32[512]", primals_31: "f32[512]", primals_32: "f32[512]", primals_33: "f32[512]", primals_34: "f32[512]", primals_35: "f32[512]", primals_36: "f32[512]", primals_37: "f32[512]", primals_38: "f32[512]", primals_39: "f32[512]", primals_40: "f32[512]", primals_41: "f32[512]", primals_42: "f32[512]", primals_43: "f32[512]", primals_44: "f32[512]", primals_45: "f32[512]", primals_46: "f32[512]", primals_47: "f32[512]", primals_48: "f32[512]", primals_49: "f32[512]", primals_50: "f32[512]", primals_51: "f32[512]", primals_52: "f32[512]", primals_53: "f32[512]", primals_54: "f32[512]", primals_55: "f32[512]", primals_56: "f32[512]", primals_57: "f32[512]", primals_58: "f32[512]", primals_59: "f32[512]", primals_60: "f32[512]", primals_61: "f32[512]", primals_62: "f32[512]", primals_63: "f32[512]", primals_64: "f32[512]", primals_65: "f32[512]", primals_66: "f32[512]", primals_67: "f32[512]", primals_68: "f32[512]", primals_69: "f32[512]", primals_70: "f32[512]", primals_71: "f32[512]", primals_72: "f32[512]", primals_73: "f32[512]", primals_74: "f32[512]", primals_75: "f32[512]", primals_76: "f32[512]", primals_77: "f32[512]", primals_78: "f32[512]", primals_79: "f32[512]", primals_80: "f32[512]", primals_81: "f32[512]", primals_82: "f32[512]", primals_83: "f32[512]", primals_84: "f32[512]", primals_85: "f32[512]", primals_86: "f32[512]", primals_87: "f32[512]", primals_88: "f32[512]", primals_89: "f32[512]", primals_90: "f32[512]", primals_91: "f32[512]", primals_92: "f32[512]", primals_93: "f32[512]", primals_94: "f32[512]", primals_95: "f32[512]", primals_96: "f32[512]", primals_97: "f32[512]", primals_98: "f32[512]", primals_99: "f32[512]", primals_100: "f32[512]", primals_101: "f32[512]", primals_102: "f32[512]", primals_103: "f32[512]", primals_104: "f32[512]", primals_105: "f32[512]", primals_106: "f32[512]", primals_107: "f32[512]", primals_108: "f32[1024]", primals_109: "f32[1024]", primals_110: "f32[1024]", primals_111: "f32[1024]", primals_112: "f32[1024]", primals_113: "f32[1024]", primals_114: "f32[1024]", primals_115: "f32[1024]", primals_116: "f32[1024]", primals_117: "f32[1024]", primals_118: "f32[1024]", primals_119: "f32[128, 3, 4, 4]", primals_120: "f32[128]", primals_121: "f32[128, 1, 7, 7]", primals_122: "f32[128]", primals_123: "f32[512, 128]", primals_124: "f32[512]", primals_125: "f32[128, 512]", primals_126: "f32[128]", primals_127: "f32[128, 1, 7, 7]", primals_128: "f32[128]", primals_129: "f32[512, 128]", primals_130: "f32[512]", primals_131: "f32[128, 512]", primals_132: "f32[128]", primals_133: "f32[128, 1, 7, 7]", primals_134: "f32[128]", primals_135: "f32[512, 128]", primals_136: "f32[512]", primals_137: "f32[128, 512]", primals_138: "f32[128]", primals_139: "f32[256, 128, 2, 2]", primals_140: "f32[256]", primals_141: "f32[256, 1, 7, 7]", primals_142: "f32[256]", primals_143: "f32[1024, 256]", primals_144: "f32[1024]", primals_145: "f32[256, 1024]", primals_146: "f32[256]", primals_147: "f32[256, 1, 7, 7]", primals_148: "f32[256]", primals_149: "f32[1024, 256]", primals_150: "f32[1024]", primals_151: "f32[256, 1024]", primals_152: "f32[256]", primals_153: "f32[256, 1, 7, 7]", primals_154: "f32[256]", primals_155: "f32[1024, 256]", primals_156: "f32[1024]", primals_157: "f32[256, 1024]", primals_158: "f32[256]", primals_159: "f32[512, 256, 2, 2]", primals_160: "f32[512]", primals_161: "f32[512, 1, 7, 7]", primals_162: "f32[512]", primals_163: "f32[2048, 512]", primals_164: "f32[2048]", primals_165: "f32[512, 2048]", primals_166: "f32[512]", primals_167: "f32[512, 1, 7, 7]", primals_168: "f32[512]", primals_169: "f32[2048, 512]", primals_170: "f32[2048]", primals_171: "f32[512, 2048]", primals_172: "f32[512]", primals_173: "f32[512, 1, 7, 7]", primals_174: "f32[512]", primals_175: "f32[2048, 512]", primals_176: "f32[2048]", primals_177: "f32[512, 2048]", primals_178: "f32[512]", primals_179: "f32[512, 1, 7, 7]", primals_180: "f32[512]", primals_181: "f32[2048, 512]", primals_182: "f32[2048]", primals_183: "f32[512, 2048]", primals_184: "f32[512]", primals_185: "f32[512, 1, 7, 7]", primals_186: "f32[512]", primals_187: "f32[2048, 512]", primals_188: "f32[2048]", primals_189: "f32[512, 2048]", primals_190: "f32[512]", primals_191: "f32[512, 1, 7, 7]", primals_192: "f32[512]", primals_193: "f32[2048, 512]", primals_194: "f32[2048]", primals_195: "f32[512, 2048]", primals_196: "f32[512]", primals_197: "f32[512, 1, 7, 7]", primals_198: "f32[512]", primals_199: "f32[2048, 512]", primals_200: "f32[2048]", primals_201: "f32[512, 2048]", primals_202: "f32[512]", primals_203: "f32[512, 1, 7, 7]", primals_204: "f32[512]", primals_205: "f32[2048, 512]", primals_206: "f32[2048]", primals_207: "f32[512, 2048]", primals_208: "f32[512]", primals_209: "f32[512, 1, 7, 7]", primals_210: "f32[512]", primals_211: "f32[2048, 512]", primals_212: "f32[2048]", primals_213: "f32[512, 2048]", primals_214: "f32[512]", primals_215: "f32[512, 1, 7, 7]", primals_216: "f32[512]", primals_217: "f32[2048, 512]", primals_218: "f32[2048]", primals_219: "f32[512, 2048]", primals_220: "f32[512]", primals_221: "f32[512, 1, 7, 7]", primals_222: "f32[512]", primals_223: "f32[2048, 512]", primals_224: "f32[2048]", primals_225: "f32[512, 2048]", primals_226: "f32[512]", primals_227: "f32[512, 1, 7, 7]", primals_228: "f32[512]", primals_229: "f32[2048, 512]", primals_230: "f32[2048]", primals_231: "f32[512, 2048]", primals_232: "f32[512]", primals_233: "f32[512, 1, 7, 7]", primals_234: "f32[512]", primals_235: "f32[2048, 512]", primals_236: "f32[2048]", primals_237: "f32[512, 2048]", primals_238: "f32[512]", primals_239: "f32[512, 1, 7, 7]", primals_240: "f32[512]", primals_241: "f32[2048, 512]", primals_242: "f32[2048]", primals_243: "f32[512, 2048]", primals_244: "f32[512]", primals_245: "f32[512, 1, 7, 7]", primals_246: "f32[512]", primals_247: "f32[2048, 512]", primals_248: "f32[2048]", primals_249: "f32[512, 2048]", primals_250: "f32[512]", primals_251: "f32[512, 1, 7, 7]", primals_252: "f32[512]", primals_253: "f32[2048, 512]", primals_254: "f32[2048]", primals_255: "f32[512, 2048]", primals_256: "f32[512]", primals_257: "f32[512, 1, 7, 7]", primals_258: "f32[512]", primals_259: "f32[2048, 512]", primals_260: "f32[2048]", primals_261: "f32[512, 2048]", primals_262: "f32[512]", primals_263: "f32[512, 1, 7, 7]", primals_264: "f32[512]", primals_265: "f32[2048, 512]", primals_266: "f32[2048]", primals_267: "f32[512, 2048]", primals_268: "f32[512]", primals_269: "f32[512, 1, 7, 7]", primals_270: "f32[512]", primals_271: "f32[2048, 512]", primals_272: "f32[2048]", primals_273: "f32[512, 2048]", primals_274: "f32[512]", primals_275: "f32[512, 1, 7, 7]", primals_276: "f32[512]", primals_277: "f32[2048, 512]", primals_278: "f32[2048]", primals_279: "f32[512, 2048]", primals_280: "f32[512]", primals_281: "f32[512, 1, 7, 7]", primals_282: "f32[512]", primals_283: "f32[2048, 512]", primals_284: "f32[2048]", primals_285: "f32[512, 2048]", primals_286: "f32[512]", primals_287: "f32[512, 1, 7, 7]", primals_288: "f32[512]", primals_289: "f32[2048, 512]", primals_290: "f32[2048]", primals_291: "f32[512, 2048]", primals_292: "f32[512]", primals_293: "f32[512, 1, 7, 7]", primals_294: "f32[512]", primals_295: "f32[2048, 512]", primals_296: "f32[2048]", primals_297: "f32[512, 2048]", primals_298: "f32[512]", primals_299: "f32[512, 1, 7, 7]", primals_300: "f32[512]", primals_301: "f32[2048, 512]", primals_302: "f32[2048]", primals_303: "f32[512, 2048]", primals_304: "f32[512]", primals_305: "f32[512, 1, 7, 7]", primals_306: "f32[512]", primals_307: "f32[2048, 512]", primals_308: "f32[2048]", primals_309: "f32[512, 2048]", primals_310: "f32[512]", primals_311: "f32[512, 1, 7, 7]", primals_312: "f32[512]", primals_313: "f32[2048, 512]", primals_314: "f32[2048]", primals_315: "f32[512, 2048]", primals_316: "f32[512]", primals_317: "f32[512, 1, 7, 7]", primals_318: "f32[512]", primals_319: "f32[2048, 512]", primals_320: "f32[2048]", primals_321: "f32[512, 2048]", primals_322: "f32[512]", primals_323: "f32[1024, 512, 2, 2]", primals_324: "f32[1024]", primals_325: "f32[1024, 1, 7, 7]", primals_326: "f32[1024]", primals_327: "f32[4096, 1024]", primals_328: "f32[4096]", primals_329: "f32[1024, 4096]", primals_330: "f32[1024]", primals_331: "f32[1024, 1, 7, 7]", primals_332: "f32[1024]", primals_333: "f32[4096, 1024]", primals_334: "f32[4096]", primals_335: "f32[1024, 4096]", primals_336: "f32[1024]", primals_337: "f32[1024, 1, 7, 7]", primals_338: "f32[1024]", primals_339: "f32[4096, 1024]", primals_340: "f32[4096]", primals_341: "f32[1024, 4096]", primals_342: "f32[1024]", primals_343: "f32[1000, 1024]", primals_344: "f32[1000]", primals_345: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:411, code: x = self.stem(x)
    convolution: "f32[8, 128, 56, 56]" = torch.ops.aten.convolution.default(primals_345, primals_119, primals_120, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  primals_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:68, code: x = x.permute(0, 2, 3, 1)
    permute: "f32[8, 56, 56, 128]" = torch.ops.aten.permute.default(convolution, [0, 2, 3, 1]);  convolution = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone: "f32[8, 56, 56, 128]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
    var_mean = torch.ops.aten.var_mean.correction(clone, [3], correction = 0, keepdim = True)
    getitem: "f32[8, 56, 56, 1]" = var_mean[0]
    getitem_1: "f32[8, 56, 56, 1]" = var_mean[1];  var_mean = None
    add: "f32[8, 56, 56, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
    rsqrt: "f32[8, 56, 56, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
    sub: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(clone, getitem_1);  clone = getitem_1 = None
    mul: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul, primals_1)
    add_1: "f32[8, 56, 56, 128]" = torch.ops.aten.add.Tensor(mul_1, primals_2);  mul_1 = primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:73, code: x = x.permute(0, 3, 1, 2)
    permute_1: "f32[8, 128, 56, 56]" = torch.ops.aten.permute.default(add_1, [0, 3, 1, 2]);  add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_1: "f32[8, 128, 56, 56]" = torch.ops.aten.convolution.default(permute_1, primals_121, primals_122, [1, 1], [3, 3], [1, 1], False, [0, 0], 128);  primals_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_2: "f32[8, 56, 56, 128]" = torch.ops.aten.permute.default(convolution_1, [0, 2, 3, 1]);  convolution_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_1: "f32[8, 56, 56, 128]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    var_mean_1 = torch.ops.aten.var_mean.correction(clone_1, [3], correction = 0, keepdim = True)
    getitem_2: "f32[8, 56, 56, 1]" = var_mean_1[0]
    getitem_3: "f32[8, 56, 56, 1]" = var_mean_1[1];  var_mean_1 = None
    add_2: "f32[8, 56, 56, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-06);  getitem_2 = None
    rsqrt_1: "f32[8, 56, 56, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
    sub_1: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(clone_1, getitem_3);  clone_1 = getitem_3 = None
    mul_2: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    mul_3: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_2, primals_3)
    add_3: "f32[8, 56, 56, 128]" = torch.ops.aten.add.Tensor(mul_3, primals_4);  mul_3 = primals_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view: "f32[25088, 128]" = torch.ops.aten.view.default(add_3, [25088, 128]);  add_3 = None
    permute_3: "f32[128, 512]" = torch.ops.aten.permute.default(primals_123, [1, 0]);  primals_123 = None
    addmm: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_124, view, permute_3);  primals_124 = None
    view_1: "f32[8, 56, 56, 512]" = torch.ops.aten.view.default(addmm, [8, 56, 56, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_4: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(view_1, 0.5)
    mul_5: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(view_1, 0.7071067811865476);  view_1 = None
    erf: "f32[8, 56, 56, 512]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
    add_4: "f32[8, 56, 56, 512]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_6: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(mul_4, add_4);  mul_4 = add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_2: "f32[8, 56, 56, 512]" = torch.ops.aten.clone.default(mul_6);  mul_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_2: "f32[25088, 512]" = torch.ops.aten.view.default(clone_2, [25088, 512]);  clone_2 = None
    permute_4: "f32[512, 128]" = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
    addmm_1: "f32[25088, 128]" = torch.ops.aten.addmm.default(primals_126, view_2, permute_4);  primals_126 = None
    view_3: "f32[8, 56, 56, 128]" = torch.ops.aten.view.default(addmm_1, [8, 56, 56, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_3: "f32[8, 56, 56, 128]" = torch.ops.aten.clone.default(view_3);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_5: "f32[8, 128, 56, 56]" = torch.ops.aten.permute.default(clone_3, [0, 3, 1, 2]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_4: "f32[1, 128, 1, 1]" = torch.ops.aten.view.default(primals_5, [1, -1, 1, 1])
    mul_7: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(permute_5, view_4);  permute_5 = view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_5: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_7, permute_1);  mul_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_2: "f32[8, 128, 56, 56]" = torch.ops.aten.convolution.default(add_5, primals_127, primals_128, [1, 1], [3, 3], [1, 1], False, [0, 0], 128);  primals_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_6: "f32[8, 56, 56, 128]" = torch.ops.aten.permute.default(convolution_2, [0, 2, 3, 1]);  convolution_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_4: "f32[8, 56, 56, 128]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
    var_mean_2 = torch.ops.aten.var_mean.correction(clone_4, [3], correction = 0, keepdim = True)
    getitem_4: "f32[8, 56, 56, 1]" = var_mean_2[0]
    getitem_5: "f32[8, 56, 56, 1]" = var_mean_2[1];  var_mean_2 = None
    add_6: "f32[8, 56, 56, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-06);  getitem_4 = None
    rsqrt_2: "f32[8, 56, 56, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_2: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(clone_4, getitem_5);  clone_4 = getitem_5 = None
    mul_8: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    mul_9: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_8, primals_6)
    add_7: "f32[8, 56, 56, 128]" = torch.ops.aten.add.Tensor(mul_9, primals_7);  mul_9 = primals_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_5: "f32[25088, 128]" = torch.ops.aten.view.default(add_7, [25088, 128]);  add_7 = None
    permute_7: "f32[128, 512]" = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
    addmm_2: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_130, view_5, permute_7);  primals_130 = None
    view_6: "f32[8, 56, 56, 512]" = torch.ops.aten.view.default(addmm_2, [8, 56, 56, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_10: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(view_6, 0.5)
    mul_11: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(view_6, 0.7071067811865476);  view_6 = None
    erf_1: "f32[8, 56, 56, 512]" = torch.ops.aten.erf.default(mul_11);  mul_11 = None
    add_8: "f32[8, 56, 56, 512]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_12: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(mul_10, add_8);  mul_10 = add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_5: "f32[8, 56, 56, 512]" = torch.ops.aten.clone.default(mul_12);  mul_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_7: "f32[25088, 512]" = torch.ops.aten.view.default(clone_5, [25088, 512]);  clone_5 = None
    permute_8: "f32[512, 128]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
    addmm_3: "f32[25088, 128]" = torch.ops.aten.addmm.default(primals_132, view_7, permute_8);  primals_132 = None
    view_8: "f32[8, 56, 56, 128]" = torch.ops.aten.view.default(addmm_3, [8, 56, 56, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_6: "f32[8, 56, 56, 128]" = torch.ops.aten.clone.default(view_8);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_9: "f32[8, 128, 56, 56]" = torch.ops.aten.permute.default(clone_6, [0, 3, 1, 2]);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_9: "f32[1, 128, 1, 1]" = torch.ops.aten.view.default(primals_8, [1, -1, 1, 1])
    mul_13: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(permute_9, view_9);  permute_9 = view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_9: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_13, add_5);  mul_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_3: "f32[8, 128, 56, 56]" = torch.ops.aten.convolution.default(add_9, primals_133, primals_134, [1, 1], [3, 3], [1, 1], False, [0, 0], 128);  primals_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_10: "f32[8, 56, 56, 128]" = torch.ops.aten.permute.default(convolution_3, [0, 2, 3, 1]);  convolution_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_7: "f32[8, 56, 56, 128]" = torch.ops.aten.clone.default(permute_10, memory_format = torch.contiguous_format);  permute_10 = None
    var_mean_3 = torch.ops.aten.var_mean.correction(clone_7, [3], correction = 0, keepdim = True)
    getitem_6: "f32[8, 56, 56, 1]" = var_mean_3[0]
    getitem_7: "f32[8, 56, 56, 1]" = var_mean_3[1];  var_mean_3 = None
    add_10: "f32[8, 56, 56, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-06);  getitem_6 = None
    rsqrt_3: "f32[8, 56, 56, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
    sub_3: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(clone_7, getitem_7);  clone_7 = getitem_7 = None
    mul_14: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    mul_15: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_14, primals_9)
    add_11: "f32[8, 56, 56, 128]" = torch.ops.aten.add.Tensor(mul_15, primals_10);  mul_15 = primals_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_10: "f32[25088, 128]" = torch.ops.aten.view.default(add_11, [25088, 128]);  add_11 = None
    permute_11: "f32[128, 512]" = torch.ops.aten.permute.default(primals_135, [1, 0]);  primals_135 = None
    addmm_4: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_136, view_10, permute_11);  primals_136 = None
    view_11: "f32[8, 56, 56, 512]" = torch.ops.aten.view.default(addmm_4, [8, 56, 56, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_16: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(view_11, 0.5)
    mul_17: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(view_11, 0.7071067811865476);  view_11 = None
    erf_2: "f32[8, 56, 56, 512]" = torch.ops.aten.erf.default(mul_17);  mul_17 = None
    add_12: "f32[8, 56, 56, 512]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_18: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(mul_16, add_12);  mul_16 = add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_8: "f32[8, 56, 56, 512]" = torch.ops.aten.clone.default(mul_18);  mul_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_12: "f32[25088, 512]" = torch.ops.aten.view.default(clone_8, [25088, 512]);  clone_8 = None
    permute_12: "f32[512, 128]" = torch.ops.aten.permute.default(primals_137, [1, 0]);  primals_137 = None
    addmm_5: "f32[25088, 128]" = torch.ops.aten.addmm.default(primals_138, view_12, permute_12);  primals_138 = None
    view_13: "f32[8, 56, 56, 128]" = torch.ops.aten.view.default(addmm_5, [8, 56, 56, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_9: "f32[8, 56, 56, 128]" = torch.ops.aten.clone.default(view_13);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_13: "f32[8, 128, 56, 56]" = torch.ops.aten.permute.default(clone_9, [0, 3, 1, 2]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_14: "f32[1, 128, 1, 1]" = torch.ops.aten.view.default(primals_11, [1, -1, 1, 1])
    mul_19: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(permute_13, view_14);  permute_13 = view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_13: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_19, add_9);  mul_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:68, code: x = x.permute(0, 2, 3, 1)
    permute_14: "f32[8, 56, 56, 128]" = torch.ops.aten.permute.default(add_13, [0, 2, 3, 1]);  add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_4 = torch.ops.aten.var_mean.correction(permute_14, [3], correction = 0, keepdim = True)
    getitem_8: "f32[8, 56, 56, 1]" = var_mean_4[0]
    getitem_9: "f32[8, 56, 56, 1]" = var_mean_4[1];  var_mean_4 = None
    add_14: "f32[8, 56, 56, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-06);  getitem_8 = None
    rsqrt_4: "f32[8, 56, 56, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    sub_4: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(permute_14, getitem_9);  permute_14 = getitem_9 = None
    mul_20: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    mul_21: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_20, primals_12)
    add_15: "f32[8, 56, 56, 128]" = torch.ops.aten.add.Tensor(mul_21, primals_13);  mul_21 = primals_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:73, code: x = x.permute(0, 3, 1, 2)
    permute_15: "f32[8, 128, 56, 56]" = torch.ops.aten.permute.default(add_15, [0, 3, 1, 2]);  add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:229, code: x = self.downsample(x)
    convolution_4: "f32[8, 256, 28, 28]" = torch.ops.aten.convolution.default(permute_15, primals_139, primals_140, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_5: "f32[8, 256, 28, 28]" = torch.ops.aten.convolution.default(convolution_4, primals_141, primals_142, [1, 1], [3, 3], [1, 1], False, [0, 0], 256);  primals_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_16: "f32[8, 28, 28, 256]" = torch.ops.aten.permute.default(convolution_5, [0, 2, 3, 1]);  convolution_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_10: "f32[8, 28, 28, 256]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
    var_mean_5 = torch.ops.aten.var_mean.correction(clone_10, [3], correction = 0, keepdim = True)
    getitem_10: "f32[8, 28, 28, 1]" = var_mean_5[0]
    getitem_11: "f32[8, 28, 28, 1]" = var_mean_5[1];  var_mean_5 = None
    add_16: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-06);  getitem_10 = None
    rsqrt_5: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_5: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(clone_10, getitem_11);  clone_10 = getitem_11 = None
    mul_22: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    mul_23: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_22, primals_14)
    add_17: "f32[8, 28, 28, 256]" = torch.ops.aten.add.Tensor(mul_23, primals_15);  mul_23 = primals_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_15: "f32[6272, 256]" = torch.ops.aten.view.default(add_17, [6272, 256]);  add_17 = None
    permute_17: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_143, [1, 0]);  primals_143 = None
    addmm_6: "f32[6272, 1024]" = torch.ops.aten.addmm.default(primals_144, view_15, permute_17);  primals_144 = None
    view_16: "f32[8, 28, 28, 1024]" = torch.ops.aten.view.default(addmm_6, [8, 28, 28, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_24: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(view_16, 0.5)
    mul_25: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(view_16, 0.7071067811865476);  view_16 = None
    erf_3: "f32[8, 28, 28, 1024]" = torch.ops.aten.erf.default(mul_25);  mul_25 = None
    add_18: "f32[8, 28, 28, 1024]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_26: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(mul_24, add_18);  mul_24 = add_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_11: "f32[8, 28, 28, 1024]" = torch.ops.aten.clone.default(mul_26);  mul_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_17: "f32[6272, 1024]" = torch.ops.aten.view.default(clone_11, [6272, 1024]);  clone_11 = None
    permute_18: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_145, [1, 0]);  primals_145 = None
    addmm_7: "f32[6272, 256]" = torch.ops.aten.addmm.default(primals_146, view_17, permute_18);  primals_146 = None
    view_18: "f32[8, 28, 28, 256]" = torch.ops.aten.view.default(addmm_7, [8, 28, 28, 256])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_12: "f32[8, 28, 28, 256]" = torch.ops.aten.clone.default(view_18);  view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_19: "f32[8, 256, 28, 28]" = torch.ops.aten.permute.default(clone_12, [0, 3, 1, 2]);  clone_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_19: "f32[1, 256, 1, 1]" = torch.ops.aten.view.default(primals_16, [1, -1, 1, 1])
    mul_27: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(permute_19, view_19);  permute_19 = view_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_19: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(mul_27, convolution_4);  mul_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_6: "f32[8, 256, 28, 28]" = torch.ops.aten.convolution.default(add_19, primals_147, primals_148, [1, 1], [3, 3], [1, 1], False, [0, 0], 256);  primals_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_20: "f32[8, 28, 28, 256]" = torch.ops.aten.permute.default(convolution_6, [0, 2, 3, 1]);  convolution_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_13: "f32[8, 28, 28, 256]" = torch.ops.aten.clone.default(permute_20, memory_format = torch.contiguous_format);  permute_20 = None
    var_mean_6 = torch.ops.aten.var_mean.correction(clone_13, [3], correction = 0, keepdim = True)
    getitem_12: "f32[8, 28, 28, 1]" = var_mean_6[0]
    getitem_13: "f32[8, 28, 28, 1]" = var_mean_6[1];  var_mean_6 = None
    add_20: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-06);  getitem_12 = None
    rsqrt_6: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
    sub_6: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(clone_13, getitem_13);  clone_13 = getitem_13 = None
    mul_28: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    mul_29: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_28, primals_17)
    add_21: "f32[8, 28, 28, 256]" = torch.ops.aten.add.Tensor(mul_29, primals_18);  mul_29 = primals_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_20: "f32[6272, 256]" = torch.ops.aten.view.default(add_21, [6272, 256]);  add_21 = None
    permute_21: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_149, [1, 0]);  primals_149 = None
    addmm_8: "f32[6272, 1024]" = torch.ops.aten.addmm.default(primals_150, view_20, permute_21);  primals_150 = None
    view_21: "f32[8, 28, 28, 1024]" = torch.ops.aten.view.default(addmm_8, [8, 28, 28, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_30: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(view_21, 0.5)
    mul_31: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(view_21, 0.7071067811865476);  view_21 = None
    erf_4: "f32[8, 28, 28, 1024]" = torch.ops.aten.erf.default(mul_31);  mul_31 = None
    add_22: "f32[8, 28, 28, 1024]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_32: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(mul_30, add_22);  mul_30 = add_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_14: "f32[8, 28, 28, 1024]" = torch.ops.aten.clone.default(mul_32);  mul_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_22: "f32[6272, 1024]" = torch.ops.aten.view.default(clone_14, [6272, 1024]);  clone_14 = None
    permute_22: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_151, [1, 0]);  primals_151 = None
    addmm_9: "f32[6272, 256]" = torch.ops.aten.addmm.default(primals_152, view_22, permute_22);  primals_152 = None
    view_23: "f32[8, 28, 28, 256]" = torch.ops.aten.view.default(addmm_9, [8, 28, 28, 256])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_15: "f32[8, 28, 28, 256]" = torch.ops.aten.clone.default(view_23);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_23: "f32[8, 256, 28, 28]" = torch.ops.aten.permute.default(clone_15, [0, 3, 1, 2]);  clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_24: "f32[1, 256, 1, 1]" = torch.ops.aten.view.default(primals_19, [1, -1, 1, 1])
    mul_33: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(permute_23, view_24);  permute_23 = view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_23: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(mul_33, add_19);  mul_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_7: "f32[8, 256, 28, 28]" = torch.ops.aten.convolution.default(add_23, primals_153, primals_154, [1, 1], [3, 3], [1, 1], False, [0, 0], 256);  primals_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_24: "f32[8, 28, 28, 256]" = torch.ops.aten.permute.default(convolution_7, [0, 2, 3, 1]);  convolution_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_16: "f32[8, 28, 28, 256]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
    var_mean_7 = torch.ops.aten.var_mean.correction(clone_16, [3], correction = 0, keepdim = True)
    getitem_14: "f32[8, 28, 28, 1]" = var_mean_7[0]
    getitem_15: "f32[8, 28, 28, 1]" = var_mean_7[1];  var_mean_7 = None
    add_24: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
    rsqrt_7: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
    sub_7: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(clone_16, getitem_15);  clone_16 = getitem_15 = None
    mul_34: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    mul_35: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_34, primals_20)
    add_25: "f32[8, 28, 28, 256]" = torch.ops.aten.add.Tensor(mul_35, primals_21);  mul_35 = primals_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_25: "f32[6272, 256]" = torch.ops.aten.view.default(add_25, [6272, 256]);  add_25 = None
    permute_25: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_155, [1, 0]);  primals_155 = None
    addmm_10: "f32[6272, 1024]" = torch.ops.aten.addmm.default(primals_156, view_25, permute_25);  primals_156 = None
    view_26: "f32[8, 28, 28, 1024]" = torch.ops.aten.view.default(addmm_10, [8, 28, 28, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_36: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(view_26, 0.5)
    mul_37: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(view_26, 0.7071067811865476);  view_26 = None
    erf_5: "f32[8, 28, 28, 1024]" = torch.ops.aten.erf.default(mul_37);  mul_37 = None
    add_26: "f32[8, 28, 28, 1024]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_38: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(mul_36, add_26);  mul_36 = add_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_17: "f32[8, 28, 28, 1024]" = torch.ops.aten.clone.default(mul_38);  mul_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_27: "f32[6272, 1024]" = torch.ops.aten.view.default(clone_17, [6272, 1024]);  clone_17 = None
    permute_26: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_157, [1, 0]);  primals_157 = None
    addmm_11: "f32[6272, 256]" = torch.ops.aten.addmm.default(primals_158, view_27, permute_26);  primals_158 = None
    view_28: "f32[8, 28, 28, 256]" = torch.ops.aten.view.default(addmm_11, [8, 28, 28, 256])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_18: "f32[8, 28, 28, 256]" = torch.ops.aten.clone.default(view_28);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_27: "f32[8, 256, 28, 28]" = torch.ops.aten.permute.default(clone_18, [0, 3, 1, 2]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_29: "f32[1, 256, 1, 1]" = torch.ops.aten.view.default(primals_22, [1, -1, 1, 1])
    mul_39: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(permute_27, view_29);  permute_27 = view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_27: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(mul_39, add_23);  mul_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:68, code: x = x.permute(0, 2, 3, 1)
    permute_28: "f32[8, 28, 28, 256]" = torch.ops.aten.permute.default(add_27, [0, 2, 3, 1]);  add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_8 = torch.ops.aten.var_mean.correction(permute_28, [3], correction = 0, keepdim = True)
    getitem_16: "f32[8, 28, 28, 1]" = var_mean_8[0]
    getitem_17: "f32[8, 28, 28, 1]" = var_mean_8[1];  var_mean_8 = None
    add_28: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-06);  getitem_16 = None
    rsqrt_8: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_8: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(permute_28, getitem_17);  permute_28 = getitem_17 = None
    mul_40: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    mul_41: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_40, primals_23)
    add_29: "f32[8, 28, 28, 256]" = torch.ops.aten.add.Tensor(mul_41, primals_24);  mul_41 = primals_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:73, code: x = x.permute(0, 3, 1, 2)
    permute_29: "f32[8, 256, 28, 28]" = torch.ops.aten.permute.default(add_29, [0, 3, 1, 2]);  add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:229, code: x = self.downsample(x)
    convolution_8: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(permute_29, primals_159, primals_160, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_9: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(convolution_8, primals_161, primals_162, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_30: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_9, [0, 2, 3, 1]);  convolution_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_19: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
    var_mean_9 = torch.ops.aten.var_mean.correction(clone_19, [3], correction = 0, keepdim = True)
    getitem_18: "f32[8, 14, 14, 1]" = var_mean_9[0]
    getitem_19: "f32[8, 14, 14, 1]" = var_mean_9[1];  var_mean_9 = None
    add_30: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-06);  getitem_18 = None
    rsqrt_9: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
    sub_9: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_19, getitem_19);  clone_19 = getitem_19 = None
    mul_42: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    mul_43: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_42, primals_25)
    add_31: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_43, primals_26);  mul_43 = primals_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_30: "f32[1568, 512]" = torch.ops.aten.view.default(add_31, [1568, 512]);  add_31 = None
    permute_31: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_163, [1, 0]);  primals_163 = None
    addmm_12: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_164, view_30, permute_31);  primals_164 = None
    view_31: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_12, [8, 14, 14, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_44: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_31, 0.5)
    mul_45: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_31, 0.7071067811865476);  view_31 = None
    erf_6: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_45);  mul_45 = None
    add_32: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_46: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_44, add_32);  mul_44 = add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_20: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_46);  mul_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_32: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_20, [1568, 2048]);  clone_20 = None
    permute_32: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_165, [1, 0]);  primals_165 = None
    addmm_13: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_166, view_32, permute_32);  primals_166 = None
    view_33: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_13, [8, 14, 14, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_21: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_33);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_33: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_21, [0, 3, 1, 2]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_34: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_27, [1, -1, 1, 1])
    mul_47: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_33, view_34);  permute_33 = view_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_33: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_47, convolution_8);  mul_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_10: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_33, primals_167, primals_168, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_34: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_10, [0, 2, 3, 1]);  convolution_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_22: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
    var_mean_10 = torch.ops.aten.var_mean.correction(clone_22, [3], correction = 0, keepdim = True)
    getitem_20: "f32[8, 14, 14, 1]" = var_mean_10[0]
    getitem_21: "f32[8, 14, 14, 1]" = var_mean_10[1];  var_mean_10 = None
    add_34: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-06);  getitem_20 = None
    rsqrt_10: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
    sub_10: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_22, getitem_21);  clone_22 = getitem_21 = None
    mul_48: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    mul_49: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_48, primals_28)
    add_35: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_49, primals_29);  mul_49 = primals_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_35: "f32[1568, 512]" = torch.ops.aten.view.default(add_35, [1568, 512]);  add_35 = None
    permute_35: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_169, [1, 0]);  primals_169 = None
    addmm_14: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_170, view_35, permute_35);  primals_170 = None
    view_36: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_14, [8, 14, 14, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_50: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_36, 0.5)
    mul_51: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_36, 0.7071067811865476);  view_36 = None
    erf_7: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
    add_36: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_52: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_50, add_36);  mul_50 = add_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_23: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_52);  mul_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_37: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_23, [1568, 2048]);  clone_23 = None
    permute_36: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_171, [1, 0]);  primals_171 = None
    addmm_15: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_172, view_37, permute_36);  primals_172 = None
    view_38: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_15, [8, 14, 14, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_24: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_38);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_37: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_24, [0, 3, 1, 2]);  clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_39: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_30, [1, -1, 1, 1])
    mul_53: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_37, view_39);  permute_37 = view_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_37: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_53, add_33);  mul_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_11: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_37, primals_173, primals_174, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_38: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_11, [0, 2, 3, 1]);  convolution_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_25: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    var_mean_11 = torch.ops.aten.var_mean.correction(clone_25, [3], correction = 0, keepdim = True)
    getitem_22: "f32[8, 14, 14, 1]" = var_mean_11[0]
    getitem_23: "f32[8, 14, 14, 1]" = var_mean_11[1];  var_mean_11 = None
    add_38: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-06);  getitem_22 = None
    rsqrt_11: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_11: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_25, getitem_23);  clone_25 = getitem_23 = None
    mul_54: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    mul_55: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_54, primals_31)
    add_39: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_55, primals_32);  mul_55 = primals_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_40: "f32[1568, 512]" = torch.ops.aten.view.default(add_39, [1568, 512]);  add_39 = None
    permute_39: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_175, [1, 0]);  primals_175 = None
    addmm_16: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_176, view_40, permute_39);  primals_176 = None
    view_41: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_16, [8, 14, 14, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_56: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_41, 0.5)
    mul_57: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476);  view_41 = None
    erf_8: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_57);  mul_57 = None
    add_40: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_58: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_56, add_40);  mul_56 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_26: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_58);  mul_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_42: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_26, [1568, 2048]);  clone_26 = None
    permute_40: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_177, [1, 0]);  primals_177 = None
    addmm_17: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_178, view_42, permute_40);  primals_178 = None
    view_43: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_17, [8, 14, 14, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_27: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_43);  view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_41: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_27, [0, 3, 1, 2]);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_44: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_33, [1, -1, 1, 1])
    mul_59: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_41, view_44);  permute_41 = view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_41: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_59, add_37);  mul_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_12: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_41, primals_179, primals_180, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_42: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_12, [0, 2, 3, 1]);  convolution_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_28: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_42, memory_format = torch.contiguous_format);  permute_42 = None
    var_mean_12 = torch.ops.aten.var_mean.correction(clone_28, [3], correction = 0, keepdim = True)
    getitem_24: "f32[8, 14, 14, 1]" = var_mean_12[0]
    getitem_25: "f32[8, 14, 14, 1]" = var_mean_12[1];  var_mean_12 = None
    add_42: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-06);  getitem_24 = None
    rsqrt_12: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_12: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_28, getitem_25);  clone_28 = getitem_25 = None
    mul_60: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    mul_61: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_60, primals_34)
    add_43: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_61, primals_35);  mul_61 = primals_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_45: "f32[1568, 512]" = torch.ops.aten.view.default(add_43, [1568, 512]);  add_43 = None
    permute_43: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_181, [1, 0]);  primals_181 = None
    addmm_18: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_182, view_45, permute_43);  primals_182 = None
    view_46: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_18, [8, 14, 14, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_62: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_46, 0.5)
    mul_63: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_46, 0.7071067811865476);  view_46 = None
    erf_9: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
    add_44: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_64: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_62, add_44);  mul_62 = add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_29: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_64);  mul_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_47: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_29, [1568, 2048]);  clone_29 = None
    permute_44: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_183, [1, 0]);  primals_183 = None
    addmm_19: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_184, view_47, permute_44);  primals_184 = None
    view_48: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_19, [8, 14, 14, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_30: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_48);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_45: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_30, [0, 3, 1, 2]);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_49: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_36, [1, -1, 1, 1])
    mul_65: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_45, view_49);  permute_45 = view_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_45: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_65, add_41);  mul_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_13: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_45, primals_185, primals_186, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_46: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_13, [0, 2, 3, 1]);  convolution_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_31: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
    var_mean_13 = torch.ops.aten.var_mean.correction(clone_31, [3], correction = 0, keepdim = True)
    getitem_26: "f32[8, 14, 14, 1]" = var_mean_13[0]
    getitem_27: "f32[8, 14, 14, 1]" = var_mean_13[1];  var_mean_13 = None
    add_46: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-06);  getitem_26 = None
    rsqrt_13: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    sub_13: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_31, getitem_27);  clone_31 = getitem_27 = None
    mul_66: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    mul_67: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_66, primals_37)
    add_47: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_67, primals_38);  mul_67 = primals_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_50: "f32[1568, 512]" = torch.ops.aten.view.default(add_47, [1568, 512]);  add_47 = None
    permute_47: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_187, [1, 0]);  primals_187 = None
    addmm_20: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_188, view_50, permute_47);  primals_188 = None
    view_51: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_20, [8, 14, 14, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_68: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_51, 0.5)
    mul_69: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_51, 0.7071067811865476);  view_51 = None
    erf_10: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
    add_48: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_70: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_68, add_48);  mul_68 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_32: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_70);  mul_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_52: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_32, [1568, 2048]);  clone_32 = None
    permute_48: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_189, [1, 0]);  primals_189 = None
    addmm_21: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_190, view_52, permute_48);  primals_190 = None
    view_53: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_21, [8, 14, 14, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_33: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_53);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_49: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_33, [0, 3, 1, 2]);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_54: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_39, [1, -1, 1, 1])
    mul_71: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_49, view_54);  permute_49 = view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_49: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_71, add_45);  mul_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_14: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_49, primals_191, primals_192, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_50: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_14, [0, 2, 3, 1]);  convolution_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_34: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_50, memory_format = torch.contiguous_format);  permute_50 = None
    var_mean_14 = torch.ops.aten.var_mean.correction(clone_34, [3], correction = 0, keepdim = True)
    getitem_28: "f32[8, 14, 14, 1]" = var_mean_14[0]
    getitem_29: "f32[8, 14, 14, 1]" = var_mean_14[1];  var_mean_14 = None
    add_50: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-06);  getitem_28 = None
    rsqrt_14: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    sub_14: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_34, getitem_29);  clone_34 = getitem_29 = None
    mul_72: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    mul_73: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_72, primals_40)
    add_51: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_73, primals_41);  mul_73 = primals_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_55: "f32[1568, 512]" = torch.ops.aten.view.default(add_51, [1568, 512]);  add_51 = None
    permute_51: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_193, [1, 0]);  primals_193 = None
    addmm_22: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_194, view_55, permute_51);  primals_194 = None
    view_56: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_22, [8, 14, 14, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_74: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_56, 0.5)
    mul_75: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_56, 0.7071067811865476);  view_56 = None
    erf_11: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
    add_52: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_76: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_74, add_52);  mul_74 = add_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_35: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_76);  mul_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_57: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_35, [1568, 2048]);  clone_35 = None
    permute_52: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_195, [1, 0]);  primals_195 = None
    addmm_23: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_196, view_57, permute_52);  primals_196 = None
    view_58: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_23, [8, 14, 14, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_36: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_58);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_53: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_36, [0, 3, 1, 2]);  clone_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_59: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_42, [1, -1, 1, 1])
    mul_77: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_53, view_59);  permute_53 = view_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_53: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_77, add_49);  mul_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_15: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_53, primals_197, primals_198, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_54: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_15, [0, 2, 3, 1]);  convolution_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_37: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_54, memory_format = torch.contiguous_format);  permute_54 = None
    var_mean_15 = torch.ops.aten.var_mean.correction(clone_37, [3], correction = 0, keepdim = True)
    getitem_30: "f32[8, 14, 14, 1]" = var_mean_15[0]
    getitem_31: "f32[8, 14, 14, 1]" = var_mean_15[1];  var_mean_15 = None
    add_54: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-06);  getitem_30 = None
    rsqrt_15: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
    sub_15: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_37, getitem_31);  clone_37 = getitem_31 = None
    mul_78: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    mul_79: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_78, primals_43)
    add_55: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_79, primals_44);  mul_79 = primals_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_60: "f32[1568, 512]" = torch.ops.aten.view.default(add_55, [1568, 512]);  add_55 = None
    permute_55: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_199, [1, 0]);  primals_199 = None
    addmm_24: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_200, view_60, permute_55);  primals_200 = None
    view_61: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_24, [8, 14, 14, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_80: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_61, 0.5)
    mul_81: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_61, 0.7071067811865476);  view_61 = None
    erf_12: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_81);  mul_81 = None
    add_56: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_82: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_80, add_56);  mul_80 = add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_38: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_82);  mul_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_62: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_38, [1568, 2048]);  clone_38 = None
    permute_56: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_201, [1, 0]);  primals_201 = None
    addmm_25: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_202, view_62, permute_56);  primals_202 = None
    view_63: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_25, [8, 14, 14, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_39: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_63);  view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_57: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_39, [0, 3, 1, 2]);  clone_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_64: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_45, [1, -1, 1, 1])
    mul_83: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_57, view_64);  permute_57 = view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_57: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_83, add_53);  mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_16: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_57, primals_203, primals_204, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_58: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_16, [0, 2, 3, 1]);  convolution_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_40: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_58, memory_format = torch.contiguous_format);  permute_58 = None
    var_mean_16 = torch.ops.aten.var_mean.correction(clone_40, [3], correction = 0, keepdim = True)
    getitem_32: "f32[8, 14, 14, 1]" = var_mean_16[0]
    getitem_33: "f32[8, 14, 14, 1]" = var_mean_16[1];  var_mean_16 = None
    add_58: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
    rsqrt_16: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    sub_16: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_40, getitem_33);  clone_40 = getitem_33 = None
    mul_84: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    mul_85: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_84, primals_46)
    add_59: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_85, primals_47);  mul_85 = primals_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_65: "f32[1568, 512]" = torch.ops.aten.view.default(add_59, [1568, 512]);  add_59 = None
    permute_59: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_205, [1, 0]);  primals_205 = None
    addmm_26: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_206, view_65, permute_59);  primals_206 = None
    view_66: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_26, [8, 14, 14, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_86: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_66, 0.5)
    mul_87: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_66, 0.7071067811865476);  view_66 = None
    erf_13: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
    add_60: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_88: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_86, add_60);  mul_86 = add_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_41: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_88);  mul_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_67: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_41, [1568, 2048]);  clone_41 = None
    permute_60: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_207, [1, 0]);  primals_207 = None
    addmm_27: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_208, view_67, permute_60);  primals_208 = None
    view_68: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_27, [8, 14, 14, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_42: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_68);  view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_61: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_42, [0, 3, 1, 2]);  clone_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_69: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_48, [1, -1, 1, 1])
    mul_89: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_61, view_69);  permute_61 = view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_61: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_89, add_57);  mul_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_17: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_61, primals_209, primals_210, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_62: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_17, [0, 2, 3, 1]);  convolution_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_43: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    var_mean_17 = torch.ops.aten.var_mean.correction(clone_43, [3], correction = 0, keepdim = True)
    getitem_34: "f32[8, 14, 14, 1]" = var_mean_17[0]
    getitem_35: "f32[8, 14, 14, 1]" = var_mean_17[1];  var_mean_17 = None
    add_62: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-06);  getitem_34 = None
    rsqrt_17: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    sub_17: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_43, getitem_35);  clone_43 = getitem_35 = None
    mul_90: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    mul_91: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_90, primals_49)
    add_63: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_91, primals_50);  mul_91 = primals_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_70: "f32[1568, 512]" = torch.ops.aten.view.default(add_63, [1568, 512]);  add_63 = None
    permute_63: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_211, [1, 0]);  primals_211 = None
    addmm_28: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_212, view_70, permute_63);  primals_212 = None
    view_71: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_28, [8, 14, 14, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_92: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_71, 0.5)
    mul_93: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_71, 0.7071067811865476);  view_71 = None
    erf_14: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_93);  mul_93 = None
    add_64: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_94: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_92, add_64);  mul_92 = add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_44: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_94);  mul_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_72: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_44, [1568, 2048]);  clone_44 = None
    permute_64: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_213, [1, 0]);  primals_213 = None
    addmm_29: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_214, view_72, permute_64);  primals_214 = None
    view_73: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_29, [8, 14, 14, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_45: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_73);  view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_65: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_45, [0, 3, 1, 2]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_74: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_51, [1, -1, 1, 1])
    mul_95: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_65, view_74);  permute_65 = view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_65: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_95, add_61);  mul_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_18: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_65, primals_215, primals_216, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_66: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_18, [0, 2, 3, 1]);  convolution_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_46: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_66, memory_format = torch.contiguous_format);  permute_66 = None
    var_mean_18 = torch.ops.aten.var_mean.correction(clone_46, [3], correction = 0, keepdim = True)
    getitem_36: "f32[8, 14, 14, 1]" = var_mean_18[0]
    getitem_37: "f32[8, 14, 14, 1]" = var_mean_18[1];  var_mean_18 = None
    add_66: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-06);  getitem_36 = None
    rsqrt_18: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    sub_18: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_46, getitem_37);  clone_46 = getitem_37 = None
    mul_96: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    mul_97: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_96, primals_52)
    add_67: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_97, primals_53);  mul_97 = primals_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_75: "f32[1568, 512]" = torch.ops.aten.view.default(add_67, [1568, 512]);  add_67 = None
    permute_67: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_217, [1, 0]);  primals_217 = None
    addmm_30: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_218, view_75, permute_67);  primals_218 = None
    view_76: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_30, [8, 14, 14, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_98: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_76, 0.5)
    mul_99: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_76, 0.7071067811865476);  view_76 = None
    erf_15: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_99);  mul_99 = None
    add_68: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_100: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_98, add_68);  mul_98 = add_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_47: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_100);  mul_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_77: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_47, [1568, 2048]);  clone_47 = None
    permute_68: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_219, [1, 0]);  primals_219 = None
    addmm_31: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_220, view_77, permute_68);  primals_220 = None
    view_78: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_31, [8, 14, 14, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_48: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_78);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_69: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_48, [0, 3, 1, 2]);  clone_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_79: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_54, [1, -1, 1, 1])
    mul_101: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_69, view_79);  permute_69 = view_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_69: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_101, add_65);  mul_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_19: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_69, primals_221, primals_222, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_70: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_19, [0, 2, 3, 1]);  convolution_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_49: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
    var_mean_19 = torch.ops.aten.var_mean.correction(clone_49, [3], correction = 0, keepdim = True)
    getitem_38: "f32[8, 14, 14, 1]" = var_mean_19[0]
    getitem_39: "f32[8, 14, 14, 1]" = var_mean_19[1];  var_mean_19 = None
    add_70: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-06);  getitem_38 = None
    rsqrt_19: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    sub_19: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_49, getitem_39);  clone_49 = getitem_39 = None
    mul_102: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    mul_103: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_102, primals_55)
    add_71: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_103, primals_56);  mul_103 = primals_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_80: "f32[1568, 512]" = torch.ops.aten.view.default(add_71, [1568, 512]);  add_71 = None
    permute_71: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_223, [1, 0]);  primals_223 = None
    addmm_32: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_224, view_80, permute_71);  primals_224 = None
    view_81: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_32, [8, 14, 14, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_104: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_81, 0.5)
    mul_105: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_81, 0.7071067811865476);  view_81 = None
    erf_16: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_105);  mul_105 = None
    add_72: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_106: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_104, add_72);  mul_104 = add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_50: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_106);  mul_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_82: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_50, [1568, 2048]);  clone_50 = None
    permute_72: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_225, [1, 0]);  primals_225 = None
    addmm_33: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_226, view_82, permute_72);  primals_226 = None
    view_83: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_33, [8, 14, 14, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_51: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_83);  view_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_73: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_51, [0, 3, 1, 2]);  clone_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_84: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_57, [1, -1, 1, 1])
    mul_107: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_73, view_84);  permute_73 = view_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_73: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_107, add_69);  mul_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_20: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_73, primals_227, primals_228, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_74: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_20, [0, 2, 3, 1]);  convolution_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_52: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format);  permute_74 = None
    var_mean_20 = torch.ops.aten.var_mean.correction(clone_52, [3], correction = 0, keepdim = True)
    getitem_40: "f32[8, 14, 14, 1]" = var_mean_20[0]
    getitem_41: "f32[8, 14, 14, 1]" = var_mean_20[1];  var_mean_20 = None
    add_74: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-06);  getitem_40 = None
    rsqrt_20: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_20: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_52, getitem_41);  clone_52 = getitem_41 = None
    mul_108: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    mul_109: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_108, primals_58)
    add_75: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_109, primals_59);  mul_109 = primals_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_85: "f32[1568, 512]" = torch.ops.aten.view.default(add_75, [1568, 512]);  add_75 = None
    permute_75: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_229, [1, 0]);  primals_229 = None
    addmm_34: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_230, view_85, permute_75);  primals_230 = None
    view_86: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_34, [8, 14, 14, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_110: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_86, 0.5)
    mul_111: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_86, 0.7071067811865476);  view_86 = None
    erf_17: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_111);  mul_111 = None
    add_76: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_112: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_110, add_76);  mul_110 = add_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_53: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_112);  mul_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_87: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_53, [1568, 2048]);  clone_53 = None
    permute_76: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_231, [1, 0]);  primals_231 = None
    addmm_35: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_232, view_87, permute_76);  primals_232 = None
    view_88: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_35, [8, 14, 14, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_54: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_88);  view_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_77: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_54, [0, 3, 1, 2]);  clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_89: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_60, [1, -1, 1, 1])
    mul_113: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_77, view_89);  permute_77 = view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_77: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_113, add_73);  mul_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_21: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_77, primals_233, primals_234, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_78: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_21, [0, 2, 3, 1]);  convolution_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_55: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_78, memory_format = torch.contiguous_format);  permute_78 = None
    var_mean_21 = torch.ops.aten.var_mean.correction(clone_55, [3], correction = 0, keepdim = True)
    getitem_42: "f32[8, 14, 14, 1]" = var_mean_21[0]
    getitem_43: "f32[8, 14, 14, 1]" = var_mean_21[1];  var_mean_21 = None
    add_78: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-06);  getitem_42 = None
    rsqrt_21: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    sub_21: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_55, getitem_43);  clone_55 = getitem_43 = None
    mul_114: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    mul_115: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_114, primals_61)
    add_79: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_115, primals_62);  mul_115 = primals_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_90: "f32[1568, 512]" = torch.ops.aten.view.default(add_79, [1568, 512]);  add_79 = None
    permute_79: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_235, [1, 0]);  primals_235 = None
    addmm_36: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_236, view_90, permute_79);  primals_236 = None
    view_91: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_36, [8, 14, 14, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_116: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_91, 0.5)
    mul_117: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_91, 0.7071067811865476);  view_91 = None
    erf_18: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_117);  mul_117 = None
    add_80: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_118: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_116, add_80);  mul_116 = add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_56: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_118);  mul_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_92: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_56, [1568, 2048]);  clone_56 = None
    permute_80: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_237, [1, 0]);  primals_237 = None
    addmm_37: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_238, view_92, permute_80);  primals_238 = None
    view_93: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_37, [8, 14, 14, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_57: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_93);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_81: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_57, [0, 3, 1, 2]);  clone_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_94: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_63, [1, -1, 1, 1])
    mul_119: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_81, view_94);  permute_81 = view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_81: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_119, add_77);  mul_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_22: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_81, primals_239, primals_240, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_82: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_22, [0, 2, 3, 1]);  convolution_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_58: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
    var_mean_22 = torch.ops.aten.var_mean.correction(clone_58, [3], correction = 0, keepdim = True)
    getitem_44: "f32[8, 14, 14, 1]" = var_mean_22[0]
    getitem_45: "f32[8, 14, 14, 1]" = var_mean_22[1];  var_mean_22 = None
    add_82: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-06);  getitem_44 = None
    rsqrt_22: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
    sub_22: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_58, getitem_45);  clone_58 = getitem_45 = None
    mul_120: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    mul_121: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_120, primals_64)
    add_83: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_121, primals_65);  mul_121 = primals_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_95: "f32[1568, 512]" = torch.ops.aten.view.default(add_83, [1568, 512]);  add_83 = None
    permute_83: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_241, [1, 0]);  primals_241 = None
    addmm_38: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_242, view_95, permute_83);  primals_242 = None
    view_96: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_38, [8, 14, 14, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_122: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_96, 0.5)
    mul_123: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_96, 0.7071067811865476);  view_96 = None
    erf_19: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_123);  mul_123 = None
    add_84: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_124: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_122, add_84);  mul_122 = add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_59: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_124);  mul_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_97: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_59, [1568, 2048]);  clone_59 = None
    permute_84: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_243, [1, 0]);  primals_243 = None
    addmm_39: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_244, view_97, permute_84);  primals_244 = None
    view_98: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_39, [8, 14, 14, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_60: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_98);  view_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_85: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_60, [0, 3, 1, 2]);  clone_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_99: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_66, [1, -1, 1, 1])
    mul_125: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_85, view_99);  permute_85 = view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_85: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_125, add_81);  mul_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_23: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_85, primals_245, primals_246, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_86: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_23, [0, 2, 3, 1]);  convolution_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_61: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_86, memory_format = torch.contiguous_format);  permute_86 = None
    var_mean_23 = torch.ops.aten.var_mean.correction(clone_61, [3], correction = 0, keepdim = True)
    getitem_46: "f32[8, 14, 14, 1]" = var_mean_23[0]
    getitem_47: "f32[8, 14, 14, 1]" = var_mean_23[1];  var_mean_23 = None
    add_86: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-06);  getitem_46 = None
    rsqrt_23: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
    sub_23: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_61, getitem_47);  clone_61 = getitem_47 = None
    mul_126: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    mul_127: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_126, primals_67)
    add_87: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_127, primals_68);  mul_127 = primals_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_100: "f32[1568, 512]" = torch.ops.aten.view.default(add_87, [1568, 512]);  add_87 = None
    permute_87: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_247, [1, 0]);  primals_247 = None
    addmm_40: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_248, view_100, permute_87);  primals_248 = None
    view_101: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_40, [8, 14, 14, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_128: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_101, 0.5)
    mul_129: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_101, 0.7071067811865476);  view_101 = None
    erf_20: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_129);  mul_129 = None
    add_88: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_130: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_128, add_88);  mul_128 = add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_62: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_130);  mul_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_102: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_62, [1568, 2048]);  clone_62 = None
    permute_88: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_249, [1, 0]);  primals_249 = None
    addmm_41: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_250, view_102, permute_88);  primals_250 = None
    view_103: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_41, [8, 14, 14, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_63: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_103);  view_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_89: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_63, [0, 3, 1, 2]);  clone_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_104: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_69, [1, -1, 1, 1])
    mul_131: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_89, view_104);  permute_89 = view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_89: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_131, add_85);  mul_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_24: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_89, primals_251, primals_252, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_90: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_24, [0, 2, 3, 1]);  convolution_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_64: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
    var_mean_24 = torch.ops.aten.var_mean.correction(clone_64, [3], correction = 0, keepdim = True)
    getitem_48: "f32[8, 14, 14, 1]" = var_mean_24[0]
    getitem_49: "f32[8, 14, 14, 1]" = var_mean_24[1];  var_mean_24 = None
    add_90: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-06);  getitem_48 = None
    rsqrt_24: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
    sub_24: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_64, getitem_49);  clone_64 = getitem_49 = None
    mul_132: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    mul_133: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_132, primals_70)
    add_91: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_133, primals_71);  mul_133 = primals_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_105: "f32[1568, 512]" = torch.ops.aten.view.default(add_91, [1568, 512]);  add_91 = None
    permute_91: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_253, [1, 0]);  primals_253 = None
    addmm_42: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_254, view_105, permute_91);  primals_254 = None
    view_106: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_42, [8, 14, 14, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_134: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_106, 0.5)
    mul_135: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_106, 0.7071067811865476);  view_106 = None
    erf_21: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_135);  mul_135 = None
    add_92: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_136: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_134, add_92);  mul_134 = add_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_65: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_136);  mul_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_107: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_65, [1568, 2048]);  clone_65 = None
    permute_92: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_255, [1, 0]);  primals_255 = None
    addmm_43: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_256, view_107, permute_92);  primals_256 = None
    view_108: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_43, [8, 14, 14, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_66: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_108);  view_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_93: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_66, [0, 3, 1, 2]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_109: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_72, [1, -1, 1, 1])
    mul_137: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_93, view_109);  permute_93 = view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_93: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_137, add_89);  mul_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_25: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_93, primals_257, primals_258, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_94: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_25, [0, 2, 3, 1]);  convolution_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_67: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_94, memory_format = torch.contiguous_format);  permute_94 = None
    var_mean_25 = torch.ops.aten.var_mean.correction(clone_67, [3], correction = 0, keepdim = True)
    getitem_50: "f32[8, 14, 14, 1]" = var_mean_25[0]
    getitem_51: "f32[8, 14, 14, 1]" = var_mean_25[1];  var_mean_25 = None
    add_94: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-06);  getitem_50 = None
    rsqrt_25: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
    sub_25: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_67, getitem_51);  clone_67 = getitem_51 = None
    mul_138: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    mul_139: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_138, primals_73)
    add_95: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_139, primals_74);  mul_139 = primals_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_110: "f32[1568, 512]" = torch.ops.aten.view.default(add_95, [1568, 512]);  add_95 = None
    permute_95: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_259, [1, 0]);  primals_259 = None
    addmm_44: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_260, view_110, permute_95);  primals_260 = None
    view_111: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_44, [8, 14, 14, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_140: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_111, 0.5)
    mul_141: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_111, 0.7071067811865476);  view_111 = None
    erf_22: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_141);  mul_141 = None
    add_96: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_142: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_140, add_96);  mul_140 = add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_68: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_142);  mul_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_112: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_68, [1568, 2048]);  clone_68 = None
    permute_96: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_261, [1, 0]);  primals_261 = None
    addmm_45: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_262, view_112, permute_96);  primals_262 = None
    view_113: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_45, [8, 14, 14, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_69: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_113);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_97: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_69, [0, 3, 1, 2]);  clone_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_114: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_75, [1, -1, 1, 1])
    mul_143: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_97, view_114);  permute_97 = view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_97: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_143, add_93);  mul_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_26: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_97, primals_263, primals_264, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_98: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_26, [0, 2, 3, 1]);  convolution_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_70: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_98, memory_format = torch.contiguous_format);  permute_98 = None
    var_mean_26 = torch.ops.aten.var_mean.correction(clone_70, [3], correction = 0, keepdim = True)
    getitem_52: "f32[8, 14, 14, 1]" = var_mean_26[0]
    getitem_53: "f32[8, 14, 14, 1]" = var_mean_26[1];  var_mean_26 = None
    add_98: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-06);  getitem_52 = None
    rsqrt_26: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    sub_26: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_70, getitem_53);  clone_70 = getitem_53 = None
    mul_144: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    mul_145: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_144, primals_76)
    add_99: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_145, primals_77);  mul_145 = primals_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_115: "f32[1568, 512]" = torch.ops.aten.view.default(add_99, [1568, 512]);  add_99 = None
    permute_99: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_265, [1, 0]);  primals_265 = None
    addmm_46: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_266, view_115, permute_99);  primals_266 = None
    view_116: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_46, [8, 14, 14, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_146: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_116, 0.5)
    mul_147: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_116, 0.7071067811865476);  view_116 = None
    erf_23: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_147);  mul_147 = None
    add_100: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_148: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_146, add_100);  mul_146 = add_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_71: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_148);  mul_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_117: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_71, [1568, 2048]);  clone_71 = None
    permute_100: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_267, [1, 0]);  primals_267 = None
    addmm_47: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_268, view_117, permute_100);  primals_268 = None
    view_118: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_47, [8, 14, 14, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_72: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_118);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_101: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_72, [0, 3, 1, 2]);  clone_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_119: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_78, [1, -1, 1, 1])
    mul_149: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_101, view_119);  permute_101 = view_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_101: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_149, add_97);  mul_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_27: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_101, primals_269, primals_270, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_102: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_27, [0, 2, 3, 1]);  convolution_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_73: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None
    var_mean_27 = torch.ops.aten.var_mean.correction(clone_73, [3], correction = 0, keepdim = True)
    getitem_54: "f32[8, 14, 14, 1]" = var_mean_27[0]
    getitem_55: "f32[8, 14, 14, 1]" = var_mean_27[1];  var_mean_27 = None
    add_102: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-06);  getitem_54 = None
    rsqrt_27: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
    sub_27: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_73, getitem_55);  clone_73 = getitem_55 = None
    mul_150: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
    mul_151: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_150, primals_79)
    add_103: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_151, primals_80);  mul_151 = primals_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_120: "f32[1568, 512]" = torch.ops.aten.view.default(add_103, [1568, 512]);  add_103 = None
    permute_103: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_271, [1, 0]);  primals_271 = None
    addmm_48: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_272, view_120, permute_103);  primals_272 = None
    view_121: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_48, [8, 14, 14, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_152: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_121, 0.5)
    mul_153: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_121, 0.7071067811865476);  view_121 = None
    erf_24: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_153);  mul_153 = None
    add_104: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
    mul_154: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_152, add_104);  mul_152 = add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_74: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_154);  mul_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_122: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_74, [1568, 2048]);  clone_74 = None
    permute_104: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_273, [1, 0]);  primals_273 = None
    addmm_49: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_274, view_122, permute_104);  primals_274 = None
    view_123: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_49, [8, 14, 14, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_75: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_123);  view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_105: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_75, [0, 3, 1, 2]);  clone_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_124: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_81, [1, -1, 1, 1])
    mul_155: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_105, view_124);  permute_105 = view_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_105: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_155, add_101);  mul_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_28: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_105, primals_275, primals_276, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_106: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_28, [0, 2, 3, 1]);  convolution_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_76: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
    var_mean_28 = torch.ops.aten.var_mean.correction(clone_76, [3], correction = 0, keepdim = True)
    getitem_56: "f32[8, 14, 14, 1]" = var_mean_28[0]
    getitem_57: "f32[8, 14, 14, 1]" = var_mean_28[1];  var_mean_28 = None
    add_106: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-06);  getitem_56 = None
    rsqrt_28: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
    sub_28: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_76, getitem_57);  clone_76 = getitem_57 = None
    mul_156: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
    mul_157: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_156, primals_82)
    add_107: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_157, primals_83);  mul_157 = primals_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_125: "f32[1568, 512]" = torch.ops.aten.view.default(add_107, [1568, 512]);  add_107 = None
    permute_107: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_277, [1, 0]);  primals_277 = None
    addmm_50: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_278, view_125, permute_107);  primals_278 = None
    view_126: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_50, [8, 14, 14, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_158: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_126, 0.5)
    mul_159: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_126, 0.7071067811865476);  view_126 = None
    erf_25: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_159);  mul_159 = None
    add_108: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
    mul_160: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_158, add_108);  mul_158 = add_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_77: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_160);  mul_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_127: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_77, [1568, 2048]);  clone_77 = None
    permute_108: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_279, [1, 0]);  primals_279 = None
    addmm_51: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_280, view_127, permute_108);  primals_280 = None
    view_128: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_51, [8, 14, 14, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_78: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_128);  view_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_109: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_78, [0, 3, 1, 2]);  clone_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_129: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_84, [1, -1, 1, 1])
    mul_161: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_109, view_129);  permute_109 = view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_109: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_161, add_105);  mul_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_29: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_109, primals_281, primals_282, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_110: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_29, [0, 2, 3, 1]);  convolution_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_79: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None
    var_mean_29 = torch.ops.aten.var_mean.correction(clone_79, [3], correction = 0, keepdim = True)
    getitem_58: "f32[8, 14, 14, 1]" = var_mean_29[0]
    getitem_59: "f32[8, 14, 14, 1]" = var_mean_29[1];  var_mean_29 = None
    add_110: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-06);  getitem_58 = None
    rsqrt_29: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
    sub_29: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_79, getitem_59);  clone_79 = getitem_59 = None
    mul_162: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
    mul_163: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_162, primals_85)
    add_111: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_163, primals_86);  mul_163 = primals_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_130: "f32[1568, 512]" = torch.ops.aten.view.default(add_111, [1568, 512]);  add_111 = None
    permute_111: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_283, [1, 0]);  primals_283 = None
    addmm_52: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_284, view_130, permute_111);  primals_284 = None
    view_131: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_52, [8, 14, 14, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_164: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_131, 0.5)
    mul_165: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_131, 0.7071067811865476);  view_131 = None
    erf_26: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_165);  mul_165 = None
    add_112: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
    mul_166: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_164, add_112);  mul_164 = add_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_80: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_166);  mul_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_132: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_80, [1568, 2048]);  clone_80 = None
    permute_112: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_285, [1, 0]);  primals_285 = None
    addmm_53: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_286, view_132, permute_112);  primals_286 = None
    view_133: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_53, [8, 14, 14, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_81: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_133);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_113: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_81, [0, 3, 1, 2]);  clone_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_134: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_87, [1, -1, 1, 1])
    mul_167: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_113, view_134);  permute_113 = view_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_113: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_167, add_109);  mul_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_30: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_113, primals_287, primals_288, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_114: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_30, [0, 2, 3, 1]);  convolution_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_82: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
    var_mean_30 = torch.ops.aten.var_mean.correction(clone_82, [3], correction = 0, keepdim = True)
    getitem_60: "f32[8, 14, 14, 1]" = var_mean_30[0]
    getitem_61: "f32[8, 14, 14, 1]" = var_mean_30[1];  var_mean_30 = None
    add_114: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-06);  getitem_60 = None
    rsqrt_30: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
    sub_30: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_82, getitem_61);  clone_82 = getitem_61 = None
    mul_168: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
    mul_169: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_168, primals_88)
    add_115: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_169, primals_89);  mul_169 = primals_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_135: "f32[1568, 512]" = torch.ops.aten.view.default(add_115, [1568, 512]);  add_115 = None
    permute_115: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_289, [1, 0]);  primals_289 = None
    addmm_54: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_290, view_135, permute_115);  primals_290 = None
    view_136: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_54, [8, 14, 14, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_170: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_136, 0.5)
    mul_171: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_136, 0.7071067811865476);  view_136 = None
    erf_27: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_171);  mul_171 = None
    add_116: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
    mul_172: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_170, add_116);  mul_170 = add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_83: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_172);  mul_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_137: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_83, [1568, 2048]);  clone_83 = None
    permute_116: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_291, [1, 0]);  primals_291 = None
    addmm_55: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_292, view_137, permute_116);  primals_292 = None
    view_138: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_55, [8, 14, 14, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_84: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_138);  view_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_117: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_84, [0, 3, 1, 2]);  clone_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_139: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_90, [1, -1, 1, 1])
    mul_173: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_117, view_139);  permute_117 = view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_117: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_173, add_113);  mul_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_31: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_117, primals_293, primals_294, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_118: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_31, [0, 2, 3, 1]);  convolution_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_85: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_118, memory_format = torch.contiguous_format);  permute_118 = None
    var_mean_31 = torch.ops.aten.var_mean.correction(clone_85, [3], correction = 0, keepdim = True)
    getitem_62: "f32[8, 14, 14, 1]" = var_mean_31[0]
    getitem_63: "f32[8, 14, 14, 1]" = var_mean_31[1];  var_mean_31 = None
    add_118: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-06);  getitem_62 = None
    rsqrt_31: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
    sub_31: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_85, getitem_63);  clone_85 = getitem_63 = None
    mul_174: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
    mul_175: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_174, primals_91)
    add_119: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_175, primals_92);  mul_175 = primals_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_140: "f32[1568, 512]" = torch.ops.aten.view.default(add_119, [1568, 512]);  add_119 = None
    permute_119: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_295, [1, 0]);  primals_295 = None
    addmm_56: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_296, view_140, permute_119);  primals_296 = None
    view_141: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_56, [8, 14, 14, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_176: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_141, 0.5)
    mul_177: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_141, 0.7071067811865476);  view_141 = None
    erf_28: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_177);  mul_177 = None
    add_120: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
    mul_178: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_176, add_120);  mul_176 = add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_86: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_178);  mul_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_142: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_86, [1568, 2048]);  clone_86 = None
    permute_120: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_297, [1, 0]);  primals_297 = None
    addmm_57: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_298, view_142, permute_120);  primals_298 = None
    view_143: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_57, [8, 14, 14, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_87: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_143);  view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_121: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_87, [0, 3, 1, 2]);  clone_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_144: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_93, [1, -1, 1, 1])
    mul_179: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_121, view_144);  permute_121 = view_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_121: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_179, add_117);  mul_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_32: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_121, primals_299, primals_300, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_122: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_32, [0, 2, 3, 1]);  convolution_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_88: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
    var_mean_32 = torch.ops.aten.var_mean.correction(clone_88, [3], correction = 0, keepdim = True)
    getitem_64: "f32[8, 14, 14, 1]" = var_mean_32[0]
    getitem_65: "f32[8, 14, 14, 1]" = var_mean_32[1];  var_mean_32 = None
    add_122: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-06);  getitem_64 = None
    rsqrt_32: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    sub_32: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_88, getitem_65);  clone_88 = getitem_65 = None
    mul_180: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
    mul_181: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_180, primals_94)
    add_123: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_181, primals_95);  mul_181 = primals_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_145: "f32[1568, 512]" = torch.ops.aten.view.default(add_123, [1568, 512]);  add_123 = None
    permute_123: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_301, [1, 0]);  primals_301 = None
    addmm_58: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_302, view_145, permute_123);  primals_302 = None
    view_146: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_58, [8, 14, 14, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_182: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_146, 0.5)
    mul_183: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_146, 0.7071067811865476);  view_146 = None
    erf_29: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_183);  mul_183 = None
    add_124: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
    mul_184: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_182, add_124);  mul_182 = add_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_89: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_184);  mul_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_147: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_89, [1568, 2048]);  clone_89 = None
    permute_124: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_303, [1, 0]);  primals_303 = None
    addmm_59: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_304, view_147, permute_124);  primals_304 = None
    view_148: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_59, [8, 14, 14, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_90: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_148);  view_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_125: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_90, [0, 3, 1, 2]);  clone_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_149: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_96, [1, -1, 1, 1])
    mul_185: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_125, view_149);  permute_125 = view_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_125: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_185, add_121);  mul_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_33: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_125, primals_305, primals_306, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_126: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_33, [0, 2, 3, 1]);  convolution_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_91: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
    var_mean_33 = torch.ops.aten.var_mean.correction(clone_91, [3], correction = 0, keepdim = True)
    getitem_66: "f32[8, 14, 14, 1]" = var_mean_33[0]
    getitem_67: "f32[8, 14, 14, 1]" = var_mean_33[1];  var_mean_33 = None
    add_126: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-06);  getitem_66 = None
    rsqrt_33: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
    sub_33: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_91, getitem_67);  clone_91 = getitem_67 = None
    mul_186: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
    mul_187: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_186, primals_97)
    add_127: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_187, primals_98);  mul_187 = primals_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_150: "f32[1568, 512]" = torch.ops.aten.view.default(add_127, [1568, 512]);  add_127 = None
    permute_127: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_307, [1, 0]);  primals_307 = None
    addmm_60: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_308, view_150, permute_127);  primals_308 = None
    view_151: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_60, [8, 14, 14, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_188: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_151, 0.5)
    mul_189: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476);  view_151 = None
    erf_30: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_189);  mul_189 = None
    add_128: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
    mul_190: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_188, add_128);  mul_188 = add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_92: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_190);  mul_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_152: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_92, [1568, 2048]);  clone_92 = None
    permute_128: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_309, [1, 0]);  primals_309 = None
    addmm_61: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_310, view_152, permute_128);  primals_310 = None
    view_153: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_61, [8, 14, 14, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_93: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_153);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_129: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_93, [0, 3, 1, 2]);  clone_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_154: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_99, [1, -1, 1, 1])
    mul_191: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_129, view_154);  permute_129 = view_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_129: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_191, add_125);  mul_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_34: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_129, primals_311, primals_312, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_130: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_34, [0, 2, 3, 1]);  convolution_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_94: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_130, memory_format = torch.contiguous_format);  permute_130 = None
    var_mean_34 = torch.ops.aten.var_mean.correction(clone_94, [3], correction = 0, keepdim = True)
    getitem_68: "f32[8, 14, 14, 1]" = var_mean_34[0]
    getitem_69: "f32[8, 14, 14, 1]" = var_mean_34[1];  var_mean_34 = None
    add_130: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-06);  getitem_68 = None
    rsqrt_34: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
    sub_34: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_94, getitem_69);  clone_94 = getitem_69 = None
    mul_192: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
    mul_193: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_192, primals_100)
    add_131: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_193, primals_101);  mul_193 = primals_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_155: "f32[1568, 512]" = torch.ops.aten.view.default(add_131, [1568, 512]);  add_131 = None
    permute_131: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_313, [1, 0]);  primals_313 = None
    addmm_62: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_314, view_155, permute_131);  primals_314 = None
    view_156: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_62, [8, 14, 14, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_194: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_156, 0.5)
    mul_195: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_156, 0.7071067811865476);  view_156 = None
    erf_31: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_195);  mul_195 = None
    add_132: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
    mul_196: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_194, add_132);  mul_194 = add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_95: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_196);  mul_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_157: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_95, [1568, 2048]);  clone_95 = None
    permute_132: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_315, [1, 0]);  primals_315 = None
    addmm_63: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_316, view_157, permute_132);  primals_316 = None
    view_158: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_63, [8, 14, 14, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_96: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_158);  view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_133: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_96, [0, 3, 1, 2]);  clone_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_159: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_102, [1, -1, 1, 1])
    mul_197: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_133, view_159);  permute_133 = view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_133: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_197, add_129);  mul_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_35: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_133, primals_317, primals_318, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_134: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_35, [0, 2, 3, 1]);  convolution_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_97: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_134, memory_format = torch.contiguous_format);  permute_134 = None
    var_mean_35 = torch.ops.aten.var_mean.correction(clone_97, [3], correction = 0, keepdim = True)
    getitem_70: "f32[8, 14, 14, 1]" = var_mean_35[0]
    getitem_71: "f32[8, 14, 14, 1]" = var_mean_35[1];  var_mean_35 = None
    add_134: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-06);  getitem_70 = None
    rsqrt_35: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
    sub_35: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_97, getitem_71);  clone_97 = getitem_71 = None
    mul_198: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
    mul_199: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_198, primals_103)
    add_135: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_199, primals_104);  mul_199 = primals_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_160: "f32[1568, 512]" = torch.ops.aten.view.default(add_135, [1568, 512]);  add_135 = None
    permute_135: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_319, [1, 0]);  primals_319 = None
    addmm_64: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_320, view_160, permute_135);  primals_320 = None
    view_161: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_64, [8, 14, 14, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_200: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_161, 0.5)
    mul_201: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_161, 0.7071067811865476);  view_161 = None
    erf_32: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_201);  mul_201 = None
    add_136: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
    mul_202: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_200, add_136);  mul_200 = add_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_98: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_202);  mul_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_162: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_98, [1568, 2048]);  clone_98 = None
    permute_136: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_321, [1, 0]);  primals_321 = None
    addmm_65: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_322, view_162, permute_136);  primals_322 = None
    view_163: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_65, [8, 14, 14, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_99: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_163);  view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_137: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_99, [0, 3, 1, 2]);  clone_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_164: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_105, [1, -1, 1, 1])
    mul_203: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_137, view_164);  permute_137 = view_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_137: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_203, add_133);  mul_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:68, code: x = x.permute(0, 2, 3, 1)
    permute_138: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(add_137, [0, 2, 3, 1]);  add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_36 = torch.ops.aten.var_mean.correction(permute_138, [3], correction = 0, keepdim = True)
    getitem_72: "f32[8, 14, 14, 1]" = var_mean_36[0]
    getitem_73: "f32[8, 14, 14, 1]" = var_mean_36[1];  var_mean_36 = None
    add_138: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-06);  getitem_72 = None
    rsqrt_36: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
    sub_36: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(permute_138, getitem_73);  permute_138 = getitem_73 = None
    mul_204: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
    mul_205: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_204, primals_106)
    add_139: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_205, primals_107);  mul_205 = primals_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:73, code: x = x.permute(0, 3, 1, 2)
    permute_139: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(add_139, [0, 3, 1, 2]);  add_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:229, code: x = self.downsample(x)
    convolution_36: "f32[8, 1024, 7, 7]" = torch.ops.aten.convolution.default(permute_139, primals_323, primals_324, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_37: "f32[8, 1024, 7, 7]" = torch.ops.aten.convolution.default(convolution_36, primals_325, primals_326, [1, 1], [3, 3], [1, 1], False, [0, 0], 1024);  primals_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_140: "f32[8, 7, 7, 1024]" = torch.ops.aten.permute.default(convolution_37, [0, 2, 3, 1]);  convolution_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_100: "f32[8, 7, 7, 1024]" = torch.ops.aten.clone.default(permute_140, memory_format = torch.contiguous_format);  permute_140 = None
    var_mean_37 = torch.ops.aten.var_mean.correction(clone_100, [3], correction = 0, keepdim = True)
    getitem_74: "f32[8, 7, 7, 1]" = var_mean_37[0]
    getitem_75: "f32[8, 7, 7, 1]" = var_mean_37[1];  var_mean_37 = None
    add_140: "f32[8, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-06);  getitem_74 = None
    rsqrt_37: "f32[8, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    sub_37: "f32[8, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(clone_100, getitem_75);  clone_100 = getitem_75 = None
    mul_206: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
    mul_207: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_206, primals_108)
    add_141: "f32[8, 7, 7, 1024]" = torch.ops.aten.add.Tensor(mul_207, primals_109);  mul_207 = primals_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_165: "f32[392, 1024]" = torch.ops.aten.view.default(add_141, [392, 1024]);  add_141 = None
    permute_141: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_327, [1, 0]);  primals_327 = None
    addmm_66: "f32[392, 4096]" = torch.ops.aten.addmm.default(primals_328, view_165, permute_141);  primals_328 = None
    view_166: "f32[8, 7, 7, 4096]" = torch.ops.aten.view.default(addmm_66, [8, 7, 7, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_208: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(view_166, 0.5)
    mul_209: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(view_166, 0.7071067811865476);  view_166 = None
    erf_33: "f32[8, 7, 7, 4096]" = torch.ops.aten.erf.default(mul_209);  mul_209 = None
    add_142: "f32[8, 7, 7, 4096]" = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
    mul_210: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(mul_208, add_142);  mul_208 = add_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_101: "f32[8, 7, 7, 4096]" = torch.ops.aten.clone.default(mul_210);  mul_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_167: "f32[392, 4096]" = torch.ops.aten.view.default(clone_101, [392, 4096]);  clone_101 = None
    permute_142: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_329, [1, 0]);  primals_329 = None
    addmm_67: "f32[392, 1024]" = torch.ops.aten.addmm.default(primals_330, view_167, permute_142);  primals_330 = None
    view_168: "f32[8, 7, 7, 1024]" = torch.ops.aten.view.default(addmm_67, [8, 7, 7, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_102: "f32[8, 7, 7, 1024]" = torch.ops.aten.clone.default(view_168);  view_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_143: "f32[8, 1024, 7, 7]" = torch.ops.aten.permute.default(clone_102, [0, 3, 1, 2]);  clone_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_169: "f32[1, 1024, 1, 1]" = torch.ops.aten.view.default(primals_110, [1, -1, 1, 1])
    mul_211: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(permute_143, view_169);  permute_143 = view_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_143: "f32[8, 1024, 7, 7]" = torch.ops.aten.add.Tensor(mul_211, convolution_36);  mul_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_38: "f32[8, 1024, 7, 7]" = torch.ops.aten.convolution.default(add_143, primals_331, primals_332, [1, 1], [3, 3], [1, 1], False, [0, 0], 1024);  primals_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_144: "f32[8, 7, 7, 1024]" = torch.ops.aten.permute.default(convolution_38, [0, 2, 3, 1]);  convolution_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_103: "f32[8, 7, 7, 1024]" = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None
    var_mean_38 = torch.ops.aten.var_mean.correction(clone_103, [3], correction = 0, keepdim = True)
    getitem_76: "f32[8, 7, 7, 1]" = var_mean_38[0]
    getitem_77: "f32[8, 7, 7, 1]" = var_mean_38[1];  var_mean_38 = None
    add_144: "f32[8, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-06);  getitem_76 = None
    rsqrt_38: "f32[8, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    sub_38: "f32[8, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(clone_103, getitem_77);  clone_103 = getitem_77 = None
    mul_212: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
    mul_213: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_212, primals_111)
    add_145: "f32[8, 7, 7, 1024]" = torch.ops.aten.add.Tensor(mul_213, primals_112);  mul_213 = primals_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_170: "f32[392, 1024]" = torch.ops.aten.view.default(add_145, [392, 1024]);  add_145 = None
    permute_145: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_333, [1, 0]);  primals_333 = None
    addmm_68: "f32[392, 4096]" = torch.ops.aten.addmm.default(primals_334, view_170, permute_145);  primals_334 = None
    view_171: "f32[8, 7, 7, 4096]" = torch.ops.aten.view.default(addmm_68, [8, 7, 7, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_214: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(view_171, 0.5)
    mul_215: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(view_171, 0.7071067811865476);  view_171 = None
    erf_34: "f32[8, 7, 7, 4096]" = torch.ops.aten.erf.default(mul_215);  mul_215 = None
    add_146: "f32[8, 7, 7, 4096]" = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
    mul_216: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(mul_214, add_146);  mul_214 = add_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_104: "f32[8, 7, 7, 4096]" = torch.ops.aten.clone.default(mul_216);  mul_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_172: "f32[392, 4096]" = torch.ops.aten.view.default(clone_104, [392, 4096]);  clone_104 = None
    permute_146: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_335, [1, 0]);  primals_335 = None
    addmm_69: "f32[392, 1024]" = torch.ops.aten.addmm.default(primals_336, view_172, permute_146);  primals_336 = None
    view_173: "f32[8, 7, 7, 1024]" = torch.ops.aten.view.default(addmm_69, [8, 7, 7, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_105: "f32[8, 7, 7, 1024]" = torch.ops.aten.clone.default(view_173);  view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_147: "f32[8, 1024, 7, 7]" = torch.ops.aten.permute.default(clone_105, [0, 3, 1, 2]);  clone_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_174: "f32[1, 1024, 1, 1]" = torch.ops.aten.view.default(primals_113, [1, -1, 1, 1])
    mul_217: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(permute_147, view_174);  permute_147 = view_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_147: "f32[8, 1024, 7, 7]" = torch.ops.aten.add.Tensor(mul_217, add_143);  mul_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_39: "f32[8, 1024, 7, 7]" = torch.ops.aten.convolution.default(add_147, primals_337, primals_338, [1, 1], [3, 3], [1, 1], False, [0, 0], 1024);  primals_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_148: "f32[8, 7, 7, 1024]" = torch.ops.aten.permute.default(convolution_39, [0, 2, 3, 1]);  convolution_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_106: "f32[8, 7, 7, 1024]" = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None
    var_mean_39 = torch.ops.aten.var_mean.correction(clone_106, [3], correction = 0, keepdim = True)
    getitem_78: "f32[8, 7, 7, 1]" = var_mean_39[0]
    getitem_79: "f32[8, 7, 7, 1]" = var_mean_39[1];  var_mean_39 = None
    add_148: "f32[8, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-06);  getitem_78 = None
    rsqrt_39: "f32[8, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
    sub_39: "f32[8, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(clone_106, getitem_79);  clone_106 = getitem_79 = None
    mul_218: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
    mul_219: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_218, primals_114)
    add_149: "f32[8, 7, 7, 1024]" = torch.ops.aten.add.Tensor(mul_219, primals_115);  mul_219 = primals_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_175: "f32[392, 1024]" = torch.ops.aten.view.default(add_149, [392, 1024]);  add_149 = None
    permute_149: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_339, [1, 0]);  primals_339 = None
    addmm_70: "f32[392, 4096]" = torch.ops.aten.addmm.default(primals_340, view_175, permute_149);  primals_340 = None
    view_176: "f32[8, 7, 7, 4096]" = torch.ops.aten.view.default(addmm_70, [8, 7, 7, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_220: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(view_176, 0.5)
    mul_221: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(view_176, 0.7071067811865476);  view_176 = None
    erf_35: "f32[8, 7, 7, 4096]" = torch.ops.aten.erf.default(mul_221);  mul_221 = None
    add_150: "f32[8, 7, 7, 4096]" = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
    mul_222: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(mul_220, add_150);  mul_220 = add_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_107: "f32[8, 7, 7, 4096]" = torch.ops.aten.clone.default(mul_222);  mul_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_177: "f32[392, 4096]" = torch.ops.aten.view.default(clone_107, [392, 4096]);  clone_107 = None
    permute_150: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_341, [1, 0]);  primals_341 = None
    addmm_71: "f32[392, 1024]" = torch.ops.aten.addmm.default(primals_342, view_177, permute_150);  primals_342 = None
    view_178: "f32[8, 7, 7, 1024]" = torch.ops.aten.view.default(addmm_71, [8, 7, 7, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_108: "f32[8, 7, 7, 1024]" = torch.ops.aten.clone.default(view_178);  view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_151: "f32[8, 1024, 7, 7]" = torch.ops.aten.permute.default(clone_108, [0, 3, 1, 2]);  clone_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_179: "f32[1, 1024, 1, 1]" = torch.ops.aten.view.default(primals_116, [1, -1, 1, 1])
    mul_223: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(permute_151, view_179);  permute_151 = view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_151: "f32[8, 1024, 7, 7]" = torch.ops.aten.add.Tensor(mul_223, add_147);  mul_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean: "f32[8, 1024, 1, 1]" = torch.ops.aten.mean.dim(add_151, [-1, -2], True);  add_151 = None
    as_strided: "f32[8, 1024, 1, 1]" = torch.ops.aten.as_strided.default(mean, [8, 1024, 1, 1], [1024, 1, 1024, 1024]);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:68, code: x = x.permute(0, 2, 3, 1)
    permute_152: "f32[8, 1, 1, 1024]" = torch.ops.aten.permute.default(as_strided, [0, 2, 3, 1]);  as_strided = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_40 = torch.ops.aten.var_mean.correction(permute_152, [3], correction = 0, keepdim = True)
    getitem_80: "f32[8, 1, 1, 1]" = var_mean_40[0]
    getitem_81: "f32[8, 1, 1, 1]" = var_mean_40[1];  var_mean_40 = None
    add_152: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-06);  getitem_80 = None
    rsqrt_40: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
    sub_40: "f32[8, 1, 1, 1024]" = torch.ops.aten.sub.Tensor(permute_152, getitem_81);  permute_152 = getitem_81 = None
    mul_224: "f32[8, 1, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
    mul_225: "f32[8, 1, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_224, primals_117)
    add_153: "f32[8, 1, 1, 1024]" = torch.ops.aten.add.Tensor(mul_225, primals_118);  mul_225 = primals_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:73, code: x = x.permute(0, 3, 1, 2)
    permute_153: "f32[8, 1024, 1, 1]" = torch.ops.aten.permute.default(add_153, [0, 3, 1, 2]);  add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:202, code: x = self.flatten(x)
    view_180: "f32[8, 1024]" = torch.ops.aten.view.default(permute_153, [8, 1024]);  permute_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:204, code: x = self.drop(x)
    clone_109: "f32[8, 1024]" = torch.ops.aten.clone.default(view_180);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:207, code: x = self.fc(x)
    permute_154: "f32[1024, 1000]" = torch.ops.aten.permute.default(primals_343, [1, 0]);  primals_343 = None
    addmm_72: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_344, clone_109, permute_154);  primals_344 = None
    permute_155: "f32[1000, 1024]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 1024);  rsqrt_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_162: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_150, [1, 0]);  permute_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_166: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_149, [1, 0]);  permute_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_2: "f32[8, 7, 7, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 1024);  rsqrt_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_172: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_146, [1, 0]);  permute_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_176: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_3: "f32[8, 7, 7, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 1024);  rsqrt_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_182: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_186: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_4: "f32[8, 7, 7, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 1024);  rsqrt_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_5: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 512);  rsqrt_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_194: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_198: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_135, [1, 0]);  permute_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_6: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 512);  rsqrt_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_204: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_208: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_7: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 512);  rsqrt_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_214: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_218: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_127, [1, 0]);  permute_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_8: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 512);  rsqrt_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_224: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_228: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_9: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 512);  rsqrt_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_234: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_238: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_10: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 512);  rsqrt_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_244: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_248: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_115, [1, 0]);  permute_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_11: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 512);  rsqrt_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_254: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_258: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_12: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 512);  rsqrt_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_264: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_268: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_13: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 512);  rsqrt_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_274: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_104, [1, 0]);  permute_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_278: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_14: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 512);  rsqrt_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_284: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_288: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_15: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 512);  rsqrt_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_294: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_298: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_95, [1, 0]);  permute_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_16: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 512);  rsqrt_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_304: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_308: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_91, [1, 0]);  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_17: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 512);  rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_314: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_318: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_18: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 512);  rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_324: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_84, [1, 0]);  permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_328: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_19: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 512);  rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_334: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_338: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_20: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 512);  rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_344: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_348: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_21: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 512);  rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_354: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_358: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_22: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 512);  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_364: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_368: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_23: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 512);  rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_374: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_378: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_24: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 512);  rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_384: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_388: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_25: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 512);  rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_394: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_398: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_26: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 512);  rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_404: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_408: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_27: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 512);  rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_414: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_418: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_28: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 512);  rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_424: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_428: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_29: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 512);  rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_434: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_438: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_30: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 512);  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_444: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_448: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_31: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 512);  rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_454: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_458: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_32: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 512);  rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_33: "f32[8, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 256);  rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_466: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_470: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_34: "f32[8, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 256);  rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_476: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_480: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_35: "f32[8, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 256);  rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_486: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_490: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_36: "f32[8, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 256);  rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_37: "f32[8, 56, 56, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 128);  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_498: "f32[128, 512]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_502: "f32[512, 128]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_38: "f32[8, 56, 56, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 128);  rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_508: "f32[128, 512]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_512: "f32[512, 128]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_39: "f32[8, 56, 56, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 128);  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_518: "f32[128, 512]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_522: "f32[512, 128]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_40: "f32[8, 56, 56, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 128);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_41: "f32[8, 56, 56, 1]" = torch.ops.aten.div.Tensor(rsqrt, 128);  rsqrt = None
    return [addmm_72, primals_1, primals_3, primals_5, primals_6, primals_8, primals_9, primals_11, primals_12, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_27, primals_28, primals_30, primals_31, primals_33, primals_34, primals_36, primals_37, primals_39, primals_40, primals_42, primals_43, primals_45, primals_46, primals_48, primals_49, primals_51, primals_52, primals_54, primals_55, primals_57, primals_58, primals_60, primals_61, primals_63, primals_64, primals_66, primals_67, primals_69, primals_70, primals_72, primals_73, primals_75, primals_76, primals_78, primals_79, primals_81, primals_82, primals_84, primals_85, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_99, primals_100, primals_102, primals_103, primals_105, primals_106, primals_108, primals_110, primals_111, primals_113, primals_114, primals_116, primals_117, primals_119, primals_121, primals_127, primals_133, primals_139, primals_141, primals_147, primals_153, primals_159, primals_161, primals_167, primals_173, primals_179, primals_185, primals_191, primals_197, primals_203, primals_209, primals_215, primals_221, primals_227, primals_233, primals_239, primals_245, primals_251, primals_257, primals_263, primals_269, primals_275, primals_281, primals_287, primals_293, primals_299, primals_305, primals_311, primals_317, primals_323, primals_325, primals_331, primals_337, primals_345, mul, permute_1, mul_2, view, addmm, view_2, addmm_1, add_5, mul_8, view_5, addmm_2, view_7, addmm_3, add_9, mul_14, view_10, addmm_4, view_12, addmm_5, mul_20, permute_15, convolution_4, mul_22, view_15, addmm_6, view_17, addmm_7, add_19, mul_28, view_20, addmm_8, view_22, addmm_9, add_23, mul_34, view_25, addmm_10, view_27, addmm_11, mul_40, permute_29, convolution_8, mul_42, view_30, addmm_12, view_32, addmm_13, add_33, mul_48, view_35, addmm_14, view_37, addmm_15, add_37, mul_54, view_40, addmm_16, view_42, addmm_17, add_41, mul_60, view_45, addmm_18, view_47, addmm_19, add_45, mul_66, view_50, addmm_20, view_52, addmm_21, add_49, mul_72, view_55, addmm_22, view_57, addmm_23, add_53, mul_78, view_60, addmm_24, view_62, addmm_25, add_57, mul_84, view_65, addmm_26, view_67, addmm_27, add_61, mul_90, view_70, addmm_28, view_72, addmm_29, add_65, mul_96, view_75, addmm_30, view_77, addmm_31, add_69, mul_102, view_80, addmm_32, view_82, addmm_33, add_73, mul_108, view_85, addmm_34, view_87, addmm_35, add_77, mul_114, view_90, addmm_36, view_92, addmm_37, add_81, mul_120, view_95, addmm_38, view_97, addmm_39, add_85, mul_126, view_100, addmm_40, view_102, addmm_41, add_89, mul_132, view_105, addmm_42, view_107, addmm_43, add_93, mul_138, view_110, addmm_44, view_112, addmm_45, add_97, mul_144, view_115, addmm_46, view_117, addmm_47, add_101, mul_150, view_120, addmm_48, view_122, addmm_49, add_105, mul_156, view_125, addmm_50, view_127, addmm_51, add_109, mul_162, view_130, addmm_52, view_132, addmm_53, add_113, mul_168, view_135, addmm_54, view_137, addmm_55, add_117, mul_174, view_140, addmm_56, view_142, addmm_57, add_121, mul_180, view_145, addmm_58, view_147, addmm_59, add_125, mul_186, view_150, addmm_60, view_152, addmm_61, add_129, mul_192, view_155, addmm_62, view_157, addmm_63, add_133, mul_198, view_160, addmm_64, view_162, addmm_65, mul_204, permute_139, convolution_36, mul_206, view_165, addmm_66, view_167, addmm_67, add_143, mul_212, view_170, addmm_68, view_172, addmm_69, add_147, mul_218, view_175, addmm_70, view_177, addmm_71, mul_224, clone_109, permute_155, div, permute_162, permute_166, div_2, permute_172, permute_176, div_3, permute_182, permute_186, div_4, div_5, permute_194, permute_198, div_6, permute_204, permute_208, div_7, permute_214, permute_218, div_8, permute_224, permute_228, div_9, permute_234, permute_238, div_10, permute_244, permute_248, div_11, permute_254, permute_258, div_12, permute_264, permute_268, div_13, permute_274, permute_278, div_14, permute_284, permute_288, div_15, permute_294, permute_298, div_16, permute_304, permute_308, div_17, permute_314, permute_318, div_18, permute_324, permute_328, div_19, permute_334, permute_338, div_20, permute_344, permute_348, div_21, permute_354, permute_358, div_22, permute_364, permute_368, div_23, permute_374, permute_378, div_24, permute_384, permute_388, div_25, permute_394, permute_398, div_26, permute_404, permute_408, div_27, permute_414, permute_418, div_28, permute_424, permute_428, div_29, permute_434, permute_438, div_30, permute_444, permute_448, div_31, permute_454, permute_458, div_32, div_33, permute_466, permute_470, div_34, permute_476, permute_480, div_35, permute_486, permute_490, div_36, div_37, permute_498, permute_502, div_38, permute_508, permute_512, div_39, permute_518, permute_522, div_40, div_41]
    