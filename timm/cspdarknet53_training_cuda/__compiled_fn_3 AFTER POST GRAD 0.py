from __future__ import annotations



def forward(self, primals_1: "f32[32]", primals_2: "f32[32]", primals_3: "f32[64]", primals_4: "f32[64]", primals_5: "f32[128]", primals_6: "f32[128]", primals_7: "f32[32]", primals_8: "f32[32]", primals_9: "f32[64]", primals_10: "f32[64]", primals_11: "f32[64]", primals_12: "f32[64]", primals_13: "f32[64]", primals_14: "f32[64]", primals_15: "f32[128]", primals_16: "f32[128]", primals_17: "f32[128]", primals_18: "f32[128]", primals_19: "f32[64]", primals_20: "f32[64]", primals_21: "f32[64]", primals_22: "f32[64]", primals_23: "f32[64]", primals_24: "f32[64]", primals_25: "f32[64]", primals_26: "f32[64]", primals_27: "f32[64]", primals_28: "f32[64]", primals_29: "f32[128]", primals_30: "f32[128]", primals_31: "f32[256]", primals_32: "f32[256]", primals_33: "f32[256]", primals_34: "f32[256]", primals_35: "f32[128]", primals_36: "f32[128]", primals_37: "f32[128]", primals_38: "f32[128]", primals_39: "f32[128]", primals_40: "f32[128]", primals_41: "f32[128]", primals_42: "f32[128]", primals_43: "f32[128]", primals_44: "f32[128]", primals_45: "f32[128]", primals_46: "f32[128]", primals_47: "f32[128]", primals_48: "f32[128]", primals_49: "f32[128]", primals_50: "f32[128]", primals_51: "f32[128]", primals_52: "f32[128]", primals_53: "f32[128]", primals_54: "f32[128]", primals_55: "f32[128]", primals_56: "f32[128]", primals_57: "f32[128]", primals_58: "f32[128]", primals_59: "f32[128]", primals_60: "f32[128]", primals_61: "f32[128]", primals_62: "f32[128]", primals_63: "f32[128]", primals_64: "f32[128]", primals_65: "f32[128]", primals_66: "f32[128]", primals_67: "f32[128]", primals_68: "f32[128]", primals_69: "f32[256]", primals_70: "f32[256]", primals_71: "f32[512]", primals_72: "f32[512]", primals_73: "f32[512]", primals_74: "f32[512]", primals_75: "f32[256]", primals_76: "f32[256]", primals_77: "f32[256]", primals_78: "f32[256]", primals_79: "f32[256]", primals_80: "f32[256]", primals_81: "f32[256]", primals_82: "f32[256]", primals_83: "f32[256]", primals_84: "f32[256]", primals_85: "f32[256]", primals_86: "f32[256]", primals_87: "f32[256]", primals_88: "f32[256]", primals_89: "f32[256]", primals_90: "f32[256]", primals_91: "f32[256]", primals_92: "f32[256]", primals_93: "f32[256]", primals_94: "f32[256]", primals_95: "f32[256]", primals_96: "f32[256]", primals_97: "f32[256]", primals_98: "f32[256]", primals_99: "f32[256]", primals_100: "f32[256]", primals_101: "f32[256]", primals_102: "f32[256]", primals_103: "f32[256]", primals_104: "f32[256]", primals_105: "f32[256]", primals_106: "f32[256]", primals_107: "f32[256]", primals_108: "f32[256]", primals_109: "f32[512]", primals_110: "f32[512]", primals_111: "f32[1024]", primals_112: "f32[1024]", primals_113: "f32[1024]", primals_114: "f32[1024]", primals_115: "f32[512]", primals_116: "f32[512]", primals_117: "f32[512]", primals_118: "f32[512]", primals_119: "f32[512]", primals_120: "f32[512]", primals_121: "f32[512]", primals_122: "f32[512]", primals_123: "f32[512]", primals_124: "f32[512]", primals_125: "f32[512]", primals_126: "f32[512]", primals_127: "f32[512]", primals_128: "f32[512]", primals_129: "f32[512]", primals_130: "f32[512]", primals_131: "f32[512]", primals_132: "f32[512]", primals_133: "f32[1024]", primals_134: "f32[1024]", primals_135: "f32[32, 3, 3, 3]", primals_136: "f32[64, 32, 3, 3]", primals_137: "f32[128, 64, 1, 1]", primals_138: "f32[32, 64, 1, 1]", primals_139: "f32[64, 32, 3, 3]", primals_140: "f32[64, 64, 1, 1]", primals_141: "f32[64, 128, 1, 1]", primals_142: "f32[128, 64, 3, 3]", primals_143: "f32[128, 128, 1, 1]", primals_144: "f32[64, 64, 1, 1]", primals_145: "f32[64, 64, 3, 3]", primals_146: "f32[64, 64, 1, 1]", primals_147: "f32[64, 64, 3, 3]", primals_148: "f32[64, 64, 1, 1]", primals_149: "f32[128, 128, 1, 1]", primals_150: "f32[256, 128, 3, 3]", primals_151: "f32[256, 256, 1, 1]", primals_152: "f32[128, 128, 1, 1]", primals_153: "f32[128, 128, 3, 3]", primals_154: "f32[128, 128, 1, 1]", primals_155: "f32[128, 128, 3, 3]", primals_156: "f32[128, 128, 1, 1]", primals_157: "f32[128, 128, 3, 3]", primals_158: "f32[128, 128, 1, 1]", primals_159: "f32[128, 128, 3, 3]", primals_160: "f32[128, 128, 1, 1]", primals_161: "f32[128, 128, 3, 3]", primals_162: "f32[128, 128, 1, 1]", primals_163: "f32[128, 128, 3, 3]", primals_164: "f32[128, 128, 1, 1]", primals_165: "f32[128, 128, 3, 3]", primals_166: "f32[128, 128, 1, 1]", primals_167: "f32[128, 128, 3, 3]", primals_168: "f32[128, 128, 1, 1]", primals_169: "f32[256, 256, 1, 1]", primals_170: "f32[512, 256, 3, 3]", primals_171: "f32[512, 512, 1, 1]", primals_172: "f32[256, 256, 1, 1]", primals_173: "f32[256, 256, 3, 3]", primals_174: "f32[256, 256, 1, 1]", primals_175: "f32[256, 256, 3, 3]", primals_176: "f32[256, 256, 1, 1]", primals_177: "f32[256, 256, 3, 3]", primals_178: "f32[256, 256, 1, 1]", primals_179: "f32[256, 256, 3, 3]", primals_180: "f32[256, 256, 1, 1]", primals_181: "f32[256, 256, 3, 3]", primals_182: "f32[256, 256, 1, 1]", primals_183: "f32[256, 256, 3, 3]", primals_184: "f32[256, 256, 1, 1]", primals_185: "f32[256, 256, 3, 3]", primals_186: "f32[256, 256, 1, 1]", primals_187: "f32[256, 256, 3, 3]", primals_188: "f32[256, 256, 1, 1]", primals_189: "f32[512, 512, 1, 1]", primals_190: "f32[1024, 512, 3, 3]", primals_191: "f32[1024, 1024, 1, 1]", primals_192: "f32[512, 512, 1, 1]", primals_193: "f32[512, 512, 3, 3]", primals_194: "f32[512, 512, 1, 1]", primals_195: "f32[512, 512, 3, 3]", primals_196: "f32[512, 512, 1, 1]", primals_197: "f32[512, 512, 3, 3]", primals_198: "f32[512, 512, 1, 1]", primals_199: "f32[512, 512, 3, 3]", primals_200: "f32[512, 512, 1, 1]", primals_201: "f32[1024, 1024, 1, 1]", primals_202: "f32[1000, 1024]", primals_203: "f32[1000]", primals_204: "i64[]", primals_205: "f32[32]", primals_206: "f32[32]", primals_207: "i64[]", primals_208: "f32[64]", primals_209: "f32[64]", primals_210: "i64[]", primals_211: "f32[128]", primals_212: "f32[128]", primals_213: "i64[]", primals_214: "f32[32]", primals_215: "f32[32]", primals_216: "i64[]", primals_217: "f32[64]", primals_218: "f32[64]", primals_219: "i64[]", primals_220: "f32[64]", primals_221: "f32[64]", primals_222: "i64[]", primals_223: "f32[64]", primals_224: "f32[64]", primals_225: "i64[]", primals_226: "f32[128]", primals_227: "f32[128]", primals_228: "i64[]", primals_229: "f32[128]", primals_230: "f32[128]", primals_231: "i64[]", primals_232: "f32[64]", primals_233: "f32[64]", primals_234: "i64[]", primals_235: "f32[64]", primals_236: "f32[64]", primals_237: "i64[]", primals_238: "f32[64]", primals_239: "f32[64]", primals_240: "i64[]", primals_241: "f32[64]", primals_242: "f32[64]", primals_243: "i64[]", primals_244: "f32[64]", primals_245: "f32[64]", primals_246: "i64[]", primals_247: "f32[128]", primals_248: "f32[128]", primals_249: "i64[]", primals_250: "f32[256]", primals_251: "f32[256]", primals_252: "i64[]", primals_253: "f32[256]", primals_254: "f32[256]", primals_255: "i64[]", primals_256: "f32[128]", primals_257: "f32[128]", primals_258: "i64[]", primals_259: "f32[128]", primals_260: "f32[128]", primals_261: "i64[]", primals_262: "f32[128]", primals_263: "f32[128]", primals_264: "i64[]", primals_265: "f32[128]", primals_266: "f32[128]", primals_267: "i64[]", primals_268: "f32[128]", primals_269: "f32[128]", primals_270: "i64[]", primals_271: "f32[128]", primals_272: "f32[128]", primals_273: "i64[]", primals_274: "f32[128]", primals_275: "f32[128]", primals_276: "i64[]", primals_277: "f32[128]", primals_278: "f32[128]", primals_279: "i64[]", primals_280: "f32[128]", primals_281: "f32[128]", primals_282: "i64[]", primals_283: "f32[128]", primals_284: "f32[128]", primals_285: "i64[]", primals_286: "f32[128]", primals_287: "f32[128]", primals_288: "i64[]", primals_289: "f32[128]", primals_290: "f32[128]", primals_291: "i64[]", primals_292: "f32[128]", primals_293: "f32[128]", primals_294: "i64[]", primals_295: "f32[128]", primals_296: "f32[128]", primals_297: "i64[]", primals_298: "f32[128]", primals_299: "f32[128]", primals_300: "i64[]", primals_301: "f32[128]", primals_302: "f32[128]", primals_303: "i64[]", primals_304: "f32[128]", primals_305: "f32[128]", primals_306: "i64[]", primals_307: "f32[256]", primals_308: "f32[256]", primals_309: "i64[]", primals_310: "f32[512]", primals_311: "f32[512]", primals_312: "i64[]", primals_313: "f32[512]", primals_314: "f32[512]", primals_315: "i64[]", primals_316: "f32[256]", primals_317: "f32[256]", primals_318: "i64[]", primals_319: "f32[256]", primals_320: "f32[256]", primals_321: "i64[]", primals_322: "f32[256]", primals_323: "f32[256]", primals_324: "i64[]", primals_325: "f32[256]", primals_326: "f32[256]", primals_327: "i64[]", primals_328: "f32[256]", primals_329: "f32[256]", primals_330: "i64[]", primals_331: "f32[256]", primals_332: "f32[256]", primals_333: "i64[]", primals_334: "f32[256]", primals_335: "f32[256]", primals_336: "i64[]", primals_337: "f32[256]", primals_338: "f32[256]", primals_339: "i64[]", primals_340: "f32[256]", primals_341: "f32[256]", primals_342: "i64[]", primals_343: "f32[256]", primals_344: "f32[256]", primals_345: "i64[]", primals_346: "f32[256]", primals_347: "f32[256]", primals_348: "i64[]", primals_349: "f32[256]", primals_350: "f32[256]", primals_351: "i64[]", primals_352: "f32[256]", primals_353: "f32[256]", primals_354: "i64[]", primals_355: "f32[256]", primals_356: "f32[256]", primals_357: "i64[]", primals_358: "f32[256]", primals_359: "f32[256]", primals_360: "i64[]", primals_361: "f32[256]", primals_362: "f32[256]", primals_363: "i64[]", primals_364: "f32[256]", primals_365: "f32[256]", primals_366: "i64[]", primals_367: "f32[512]", primals_368: "f32[512]", primals_369: "i64[]", primals_370: "f32[1024]", primals_371: "f32[1024]", primals_372: "i64[]", primals_373: "f32[1024]", primals_374: "f32[1024]", primals_375: "i64[]", primals_376: "f32[512]", primals_377: "f32[512]", primals_378: "i64[]", primals_379: "f32[512]", primals_380: "f32[512]", primals_381: "i64[]", primals_382: "f32[512]", primals_383: "f32[512]", primals_384: "i64[]", primals_385: "f32[512]", primals_386: "f32[512]", primals_387: "i64[]", primals_388: "f32[512]", primals_389: "f32[512]", primals_390: "i64[]", primals_391: "f32[512]", primals_392: "f32[512]", primals_393: "i64[]", primals_394: "f32[512]", primals_395: "f32[512]", primals_396: "i64[]", primals_397: "f32[512]", primals_398: "f32[512]", primals_399: "i64[]", primals_400: "f32[512]", primals_401: "f32[512]", primals_402: "i64[]", primals_403: "f32[1024]", primals_404: "f32[1024]", primals_405: "f32[8, 3, 256, 256]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution: "f32[8, 32, 256, 256]" = torch.ops.aten.convolution.default(primals_405, primals_135, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_204, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
    getitem: "f32[1, 32, 1, 1]" = var_mean[0]
    getitem_1: "f32[1, 32, 1, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
    rsqrt: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 32, 256, 256]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
    mul: "f32[8, 32, 256, 256]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    squeeze: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
    squeeze_1: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
    mul_1: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
    mul_2: "f32[32]" = torch.ops.aten.mul.Tensor(primals_205, 0.9)
    add_2: "f32[32]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[32]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.0000019073522708);  squeeze_2 = None
    mul_4: "f32[32]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[32]" = torch.ops.aten.mul.Tensor(primals_206, 0.9)
    add_3: "f32[32]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    unsqueeze: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_1, -1)
    unsqueeze_1: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    mul_6: "f32[8, 32, 256, 256]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
    unsqueeze_2: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1);  primals_2 = None
    unsqueeze_3: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    add_4: "f32[8, 32, 256, 256]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt: "b8[8, 32, 256, 256]" = torch.ops.aten.gt.Scalar(add_4, 0)
    mul_7: "f32[8, 32, 256, 256]" = torch.ops.aten.mul.Tensor(add_4, 0.01)
    where: "f32[8, 32, 256, 256]" = torch.ops.aten.where.self(gt, add_4, mul_7);  gt = add_4 = mul_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:126, code: x = self.conv(x)
    convolution_1: "f32[8, 64, 128, 128]" = torch.ops.aten.convolution.default(where, primals_136, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_207, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_2: "f32[1, 64, 1, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 64, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
    rsqrt_1: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_1: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
    mul_8: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_3: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
    squeeze_4: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
    mul_9: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
    mul_10: "f32[64]" = torch.ops.aten.mul.Tensor(primals_208, 0.9)
    add_7: "f32[64]" = torch.ops.aten.add.Tensor(mul_9, mul_10);  mul_9 = mul_10 = None
    squeeze_5: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
    mul_11: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000076294527394);  squeeze_5 = None
    mul_12: "f32[64]" = torch.ops.aten.mul.Tensor(mul_11, 0.1);  mul_11 = None
    mul_13: "f32[64]" = torch.ops.aten.mul.Tensor(primals_209, 0.9)
    add_8: "f32[64]" = torch.ops.aten.add.Tensor(mul_12, mul_13);  mul_12 = mul_13 = None
    unsqueeze_4: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1)
    unsqueeze_5: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_14: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_8, unsqueeze_5);  mul_8 = unsqueeze_5 = None
    unsqueeze_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_4, -1);  primals_4 = None
    unsqueeze_7: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_9: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_7);  mul_14 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_1: "b8[8, 64, 128, 128]" = torch.ops.aten.gt.Scalar(add_9, 0)
    mul_15: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(add_9, 0.01)
    where_1: "f32[8, 64, 128, 128]" = torch.ops.aten.where.self(gt_1, add_9, mul_15);  gt_1 = add_9 = mul_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_2: "f32[8, 128, 128, 128]" = torch.ops.aten.convolution.default(where_1, primals_137, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_210, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 128, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 128, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
    rsqrt_2: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_2: "f32[8, 128, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
    mul_16: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_7: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_17: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_18: "f32[128]" = torch.ops.aten.mul.Tensor(primals_211, 0.9)
    add_12: "f32[128]" = torch.ops.aten.add.Tensor(mul_17, mul_18);  mul_17 = mul_18 = None
    squeeze_8: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_19: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0000076294527394);  squeeze_8 = None
    mul_20: "f32[128]" = torch.ops.aten.mul.Tensor(mul_19, 0.1);  mul_19 = None
    mul_21: "f32[128]" = torch.ops.aten.mul.Tensor(primals_212, 0.9)
    add_13: "f32[128]" = torch.ops.aten.add.Tensor(mul_20, mul_21);  mul_20 = mul_21 = None
    unsqueeze_8: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_9: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_22: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_9);  mul_16 = unsqueeze_9 = None
    unsqueeze_10: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_11: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_14: "f32[8, 128, 128, 128]" = torch.ops.aten.add.Tensor(mul_22, unsqueeze_11);  mul_22 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_2: "b8[8, 128, 128, 128]" = torch.ops.aten.gt.Scalar(add_14, 0)
    mul_23: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(add_14, 0.01)
    where_2: "f32[8, 128, 128, 128]" = torch.ops.aten.where.self(gt_2, add_14, mul_23);  gt_2 = add_14 = mul_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(where_2, [64, 64], 1)
    getitem_9: "f32[8, 64, 128, 128]" = split_with_sizes_1[1]
    convolution_3: "f32[8, 32, 128, 128]" = torch.ops.aten.convolution.default(getitem_9, primals_138, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_15: "i64[]" = torch.ops.aten.add.Tensor(primals_213, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
    getitem_10: "f32[1, 32, 1, 1]" = var_mean_3[0]
    getitem_11: "f32[1, 32, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_16: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
    rsqrt_3: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_3: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_11)
    mul_24: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_9: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
    squeeze_10: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
    mul_25: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
    mul_26: "f32[32]" = torch.ops.aten.mul.Tensor(primals_214, 0.9)
    add_17: "f32[32]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
    squeeze_11: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
    mul_27: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.0000076294527394);  squeeze_11 = None
    mul_28: "f32[32]" = torch.ops.aten.mul.Tensor(mul_27, 0.1);  mul_27 = None
    mul_29: "f32[32]" = torch.ops.aten.mul.Tensor(primals_215, 0.9)
    add_18: "f32[32]" = torch.ops.aten.add.Tensor(mul_28, mul_29);  mul_28 = mul_29 = None
    unsqueeze_12: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1)
    unsqueeze_13: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_30: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(mul_24, unsqueeze_13);  mul_24 = unsqueeze_13 = None
    unsqueeze_14: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1);  primals_8 = None
    unsqueeze_15: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_19: "f32[8, 32, 128, 128]" = torch.ops.aten.add.Tensor(mul_30, unsqueeze_15);  mul_30 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_3: "b8[8, 32, 128, 128]" = torch.ops.aten.gt.Scalar(add_19, 0)
    mul_31: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(add_19, 0.01)
    where_3: "f32[8, 32, 128, 128]" = torch.ops.aten.where.self(gt_3, add_19, mul_31);  gt_3 = add_19 = mul_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_4: "f32[8, 64, 128, 128]" = torch.ops.aten.convolution.default(where_3, primals_139, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_20: "i64[]" = torch.ops.aten.add.Tensor(primals_216, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_4 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
    getitem_12: "f32[1, 64, 1, 1]" = var_mean_4[0]
    getitem_13: "f32[1, 64, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_21: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
    rsqrt_4: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_4: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_13)
    mul_32: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_12: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
    squeeze_13: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
    mul_33: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
    mul_34: "f32[64]" = torch.ops.aten.mul.Tensor(primals_217, 0.9)
    add_22: "f32[64]" = torch.ops.aten.add.Tensor(mul_33, mul_34);  mul_33 = mul_34 = None
    squeeze_14: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
    mul_35: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0000076294527394);  squeeze_14 = None
    mul_36: "f32[64]" = torch.ops.aten.mul.Tensor(mul_35, 0.1);  mul_35 = None
    mul_37: "f32[64]" = torch.ops.aten.mul.Tensor(primals_218, 0.9)
    add_23: "f32[64]" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
    unsqueeze_16: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1)
    unsqueeze_17: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    mul_38: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_32, unsqueeze_17);  mul_32 = unsqueeze_17 = None
    unsqueeze_18: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_10, -1);  primals_10 = None
    unsqueeze_19: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    add_24: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_38, unsqueeze_19);  mul_38 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_4: "b8[8, 64, 128, 128]" = torch.ops.aten.gt.Scalar(add_24, 0)
    mul_39: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(add_24, 0.01)
    where_4: "f32[8, 64, 128, 128]" = torch.ops.aten.where.self(gt_4, add_24, mul_39);  gt_4 = add_24 = mul_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_25: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(where_4, getitem_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_5: "f32[8, 64, 128, 128]" = torch.ops.aten.convolution.default(add_25, primals_140, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_26: "i64[]" = torch.ops.aten.add.Tensor(primals_219, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_5 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
    getitem_14: "f32[1, 64, 1, 1]" = var_mean_5[0]
    getitem_15: "f32[1, 64, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_27: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
    rsqrt_5: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
    sub_5: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_15)
    mul_40: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    squeeze_15: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
    squeeze_16: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
    mul_41: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
    mul_42: "f32[64]" = torch.ops.aten.mul.Tensor(primals_220, 0.9)
    add_28: "f32[64]" = torch.ops.aten.add.Tensor(mul_41, mul_42);  mul_41 = mul_42 = None
    squeeze_17: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
    mul_43: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0000076294527394);  squeeze_17 = None
    mul_44: "f32[64]" = torch.ops.aten.mul.Tensor(mul_43, 0.1);  mul_43 = None
    mul_45: "f32[64]" = torch.ops.aten.mul.Tensor(primals_221, 0.9)
    add_29: "f32[64]" = torch.ops.aten.add.Tensor(mul_44, mul_45);  mul_44 = mul_45 = None
    unsqueeze_20: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_21: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_46: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_40, unsqueeze_21);  mul_40 = unsqueeze_21 = None
    unsqueeze_22: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_23: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_30: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_46, unsqueeze_23);  mul_46 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_5: "b8[8, 64, 128, 128]" = torch.ops.aten.gt.Scalar(add_30, 0)
    mul_47: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(add_30, 0.01)
    where_5: "f32[8, 64, 128, 128]" = torch.ops.aten.where.self(gt_5, add_30, mul_47);  gt_5 = add_30 = mul_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:339, code: out = self.conv_transition(torch.cat([xs, xb], dim=1))
    getitem_16: "f32[8, 64, 128, 128]" = split_with_sizes_1[0];  split_with_sizes_1 = None
    cat: "f32[8, 128, 128, 128]" = torch.ops.aten.cat.default([getitem_16, where_5], 1);  getitem_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_6: "f32[8, 64, 128, 128]" = torch.ops.aten.convolution.default(cat, primals_141, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_31: "i64[]" = torch.ops.aten.add.Tensor(primals_222, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_6 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
    getitem_18: "f32[1, 64, 1, 1]" = var_mean_6[0]
    getitem_19: "f32[1, 64, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_32: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
    rsqrt_6: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_6: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_19)
    mul_48: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    squeeze_18: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
    squeeze_19: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
    mul_49: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
    mul_50: "f32[64]" = torch.ops.aten.mul.Tensor(primals_223, 0.9)
    add_33: "f32[64]" = torch.ops.aten.add.Tensor(mul_49, mul_50);  mul_49 = mul_50 = None
    squeeze_20: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
    mul_51: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0000076294527394);  squeeze_20 = None
    mul_52: "f32[64]" = torch.ops.aten.mul.Tensor(mul_51, 0.1);  mul_51 = None
    mul_53: "f32[64]" = torch.ops.aten.mul.Tensor(primals_224, 0.9)
    add_34: "f32[64]" = torch.ops.aten.add.Tensor(mul_52, mul_53);  mul_52 = mul_53 = None
    unsqueeze_24: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1)
    unsqueeze_25: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    mul_54: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_48, unsqueeze_25);  mul_48 = unsqueeze_25 = None
    unsqueeze_26: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1);  primals_14 = None
    unsqueeze_27: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    add_35: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_54, unsqueeze_27);  mul_54 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_6: "b8[8, 64, 128, 128]" = torch.ops.aten.gt.Scalar(add_35, 0)
    mul_55: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(add_35, 0.01)
    where_6: "f32[8, 64, 128, 128]" = torch.ops.aten.where.self(gt_6, add_35, mul_55);  gt_6 = add_35 = mul_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:126, code: x = self.conv(x)
    convolution_7: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(where_6, primals_142, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_36: "i64[]" = torch.ops.aten.add.Tensor(primals_225, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_7 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
    getitem_20: "f32[1, 128, 1, 1]" = var_mean_7[0]
    getitem_21: "f32[1, 128, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_37: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
    rsqrt_7: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    sub_7: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_21)
    mul_56: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    squeeze_21: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
    squeeze_22: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
    mul_57: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
    mul_58: "f32[128]" = torch.ops.aten.mul.Tensor(primals_226, 0.9)
    add_38: "f32[128]" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
    squeeze_23: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
    mul_59: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.000030518509476);  squeeze_23 = None
    mul_60: "f32[128]" = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
    mul_61: "f32[128]" = torch.ops.aten.mul.Tensor(primals_227, 0.9)
    add_39: "f32[128]" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
    unsqueeze_28: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1)
    unsqueeze_29: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_62: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_29);  mul_56 = unsqueeze_29 = None
    unsqueeze_30: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_16, -1);  primals_16 = None
    unsqueeze_31: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_40: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_31);  mul_62 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_7: "b8[8, 128, 64, 64]" = torch.ops.aten.gt.Scalar(add_40, 0)
    mul_63: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(add_40, 0.01)
    where_7: "f32[8, 128, 64, 64]" = torch.ops.aten.where.self(gt_7, add_40, mul_63);  gt_7 = add_40 = mul_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_8: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(where_7, primals_143, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_41: "i64[]" = torch.ops.aten.add.Tensor(primals_228, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_8 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
    getitem_22: "f32[1, 128, 1, 1]" = var_mean_8[0]
    getitem_23: "f32[1, 128, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_42: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
    rsqrt_8: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_8: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_23)
    mul_64: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    squeeze_24: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
    squeeze_25: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
    mul_65: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
    mul_66: "f32[128]" = torch.ops.aten.mul.Tensor(primals_229, 0.9)
    add_43: "f32[128]" = torch.ops.aten.add.Tensor(mul_65, mul_66);  mul_65 = mul_66 = None
    squeeze_26: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
    mul_67: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.000030518509476);  squeeze_26 = None
    mul_68: "f32[128]" = torch.ops.aten.mul.Tensor(mul_67, 0.1);  mul_67 = None
    mul_69: "f32[128]" = torch.ops.aten.mul.Tensor(primals_230, 0.9)
    add_44: "f32[128]" = torch.ops.aten.add.Tensor(mul_68, mul_69);  mul_68 = mul_69 = None
    unsqueeze_32: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_33: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    mul_70: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_64, unsqueeze_33);  mul_64 = unsqueeze_33 = None
    unsqueeze_34: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_35: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    add_45: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(mul_70, unsqueeze_35);  mul_70 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_8: "b8[8, 128, 64, 64]" = torch.ops.aten.gt.Scalar(add_45, 0)
    mul_71: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(add_45, 0.01)
    where_8: "f32[8, 128, 64, 64]" = torch.ops.aten.where.self(gt_8, add_45, mul_71);  gt_8 = add_45 = mul_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    split_with_sizes_4 = torch.ops.aten.split_with_sizes.default(where_8, [64, 64], 1)
    getitem_27: "f32[8, 64, 64, 64]" = split_with_sizes_4[1]
    convolution_9: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(getitem_27, primals_144, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_46: "i64[]" = torch.ops.aten.add.Tensor(primals_231, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_9 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
    getitem_28: "f32[1, 64, 1, 1]" = var_mean_9[0]
    getitem_29: "f32[1, 64, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_47: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
    rsqrt_9: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    sub_9: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_9, getitem_29)
    mul_72: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    squeeze_27: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
    squeeze_28: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
    mul_73: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
    mul_74: "f32[64]" = torch.ops.aten.mul.Tensor(primals_232, 0.9)
    add_48: "f32[64]" = torch.ops.aten.add.Tensor(mul_73, mul_74);  mul_73 = mul_74 = None
    squeeze_29: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
    mul_75: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.000030518509476);  squeeze_29 = None
    mul_76: "f32[64]" = torch.ops.aten.mul.Tensor(mul_75, 0.1);  mul_75 = None
    mul_77: "f32[64]" = torch.ops.aten.mul.Tensor(primals_233, 0.9)
    add_49: "f32[64]" = torch.ops.aten.add.Tensor(mul_76, mul_77);  mul_76 = mul_77 = None
    unsqueeze_36: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1)
    unsqueeze_37: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_78: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_72, unsqueeze_37);  mul_72 = unsqueeze_37 = None
    unsqueeze_38: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1);  primals_20 = None
    unsqueeze_39: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_50: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_78, unsqueeze_39);  mul_78 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_9: "b8[8, 64, 64, 64]" = torch.ops.aten.gt.Scalar(add_50, 0)
    mul_79: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_50, 0.01)
    where_9: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(gt_9, add_50, mul_79);  gt_9 = add_50 = mul_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_10: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(where_9, primals_145, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_51: "i64[]" = torch.ops.aten.add.Tensor(primals_234, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_10 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
    getitem_30: "f32[1, 64, 1, 1]" = var_mean_10[0]
    getitem_31: "f32[1, 64, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_52: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
    rsqrt_10: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_10: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_31)
    mul_80: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    squeeze_30: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
    squeeze_31: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
    mul_81: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
    mul_82: "f32[64]" = torch.ops.aten.mul.Tensor(primals_235, 0.9)
    add_53: "f32[64]" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
    squeeze_32: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
    mul_83: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.000030518509476);  squeeze_32 = None
    mul_84: "f32[64]" = torch.ops.aten.mul.Tensor(mul_83, 0.1);  mul_83 = None
    mul_85: "f32[64]" = torch.ops.aten.mul.Tensor(primals_236, 0.9)
    add_54: "f32[64]" = torch.ops.aten.add.Tensor(mul_84, mul_85);  mul_84 = mul_85 = None
    unsqueeze_40: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1)
    unsqueeze_41: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    mul_86: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_80, unsqueeze_41);  mul_80 = unsqueeze_41 = None
    unsqueeze_42: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_22, -1);  primals_22 = None
    unsqueeze_43: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    add_55: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_86, unsqueeze_43);  mul_86 = unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_10: "b8[8, 64, 64, 64]" = torch.ops.aten.gt.Scalar(add_55, 0)
    mul_87: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_55, 0.01)
    where_10: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(gt_10, add_55, mul_87);  gt_10 = add_55 = mul_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_56: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(where_10, getitem_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_11: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(add_56, primals_146, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_57: "i64[]" = torch.ops.aten.add.Tensor(primals_237, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_11 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
    getitem_32: "f32[1, 64, 1, 1]" = var_mean_11[0]
    getitem_33: "f32[1, 64, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_58: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
    rsqrt_11: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    sub_11: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_33)
    mul_88: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    squeeze_33: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
    squeeze_34: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
    mul_89: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
    mul_90: "f32[64]" = torch.ops.aten.mul.Tensor(primals_238, 0.9)
    add_59: "f32[64]" = torch.ops.aten.add.Tensor(mul_89, mul_90);  mul_89 = mul_90 = None
    squeeze_35: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
    mul_91: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.000030518509476);  squeeze_35 = None
    mul_92: "f32[64]" = torch.ops.aten.mul.Tensor(mul_91, 0.1);  mul_91 = None
    mul_93: "f32[64]" = torch.ops.aten.mul.Tensor(primals_239, 0.9)
    add_60: "f32[64]" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
    unsqueeze_44: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_45: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_94: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_88, unsqueeze_45);  mul_88 = unsqueeze_45 = None
    unsqueeze_46: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_47: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_61: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_94, unsqueeze_47);  mul_94 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_11: "b8[8, 64, 64, 64]" = torch.ops.aten.gt.Scalar(add_61, 0)
    mul_95: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_61, 0.01)
    where_11: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(gt_11, add_61, mul_95);  gt_11 = add_61 = mul_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_12: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(where_11, primals_147, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_62: "i64[]" = torch.ops.aten.add.Tensor(primals_240, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_12 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
    getitem_34: "f32[1, 64, 1, 1]" = var_mean_12[0]
    getitem_35: "f32[1, 64, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_63: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
    rsqrt_12: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    sub_12: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_12, getitem_35)
    mul_96: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    squeeze_36: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
    squeeze_37: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
    mul_97: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
    mul_98: "f32[64]" = torch.ops.aten.mul.Tensor(primals_241, 0.9)
    add_64: "f32[64]" = torch.ops.aten.add.Tensor(mul_97, mul_98);  mul_97 = mul_98 = None
    squeeze_38: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
    mul_99: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.000030518509476);  squeeze_38 = None
    mul_100: "f32[64]" = torch.ops.aten.mul.Tensor(mul_99, 0.1);  mul_99 = None
    mul_101: "f32[64]" = torch.ops.aten.mul.Tensor(primals_242, 0.9)
    add_65: "f32[64]" = torch.ops.aten.add.Tensor(mul_100, mul_101);  mul_100 = mul_101 = None
    unsqueeze_48: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_25, -1)
    unsqueeze_49: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    mul_102: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_96, unsqueeze_49);  mul_96 = unsqueeze_49 = None
    unsqueeze_50: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1);  primals_26 = None
    unsqueeze_51: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    add_66: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_102, unsqueeze_51);  mul_102 = unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_12: "b8[8, 64, 64, 64]" = torch.ops.aten.gt.Scalar(add_66, 0)
    mul_103: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_66, 0.01)
    where_12: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(gt_12, add_66, mul_103);  gt_12 = add_66 = mul_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_67: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(where_12, add_56)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_13: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(add_67, primals_148, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_68: "i64[]" = torch.ops.aten.add.Tensor(primals_243, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_13 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
    getitem_36: "f32[1, 64, 1, 1]" = var_mean_13[0]
    getitem_37: "f32[1, 64, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_69: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
    rsqrt_13: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
    sub_13: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_37)
    mul_104: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    squeeze_39: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
    squeeze_40: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
    mul_105: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
    mul_106: "f32[64]" = torch.ops.aten.mul.Tensor(primals_244, 0.9)
    add_70: "f32[64]" = torch.ops.aten.add.Tensor(mul_105, mul_106);  mul_105 = mul_106 = None
    squeeze_41: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
    mul_107: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.000030518509476);  squeeze_41 = None
    mul_108: "f32[64]" = torch.ops.aten.mul.Tensor(mul_107, 0.1);  mul_107 = None
    mul_109: "f32[64]" = torch.ops.aten.mul.Tensor(primals_245, 0.9)
    add_71: "f32[64]" = torch.ops.aten.add.Tensor(mul_108, mul_109);  mul_108 = mul_109 = None
    unsqueeze_52: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1)
    unsqueeze_53: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_110: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_104, unsqueeze_53);  mul_104 = unsqueeze_53 = None
    unsqueeze_54: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_28, -1);  primals_28 = None
    unsqueeze_55: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_72: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_110, unsqueeze_55);  mul_110 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_13: "b8[8, 64, 64, 64]" = torch.ops.aten.gt.Scalar(add_72, 0)
    mul_111: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_72, 0.01)
    where_13: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(gt_13, add_72, mul_111);  gt_13 = add_72 = mul_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:339, code: out = self.conv_transition(torch.cat([xs, xb], dim=1))
    getitem_38: "f32[8, 64, 64, 64]" = split_with_sizes_4[0];  split_with_sizes_4 = None
    cat_1: "f32[8, 128, 64, 64]" = torch.ops.aten.cat.default([getitem_38, where_13], 1);  getitem_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_14: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(cat_1, primals_149, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_73: "i64[]" = torch.ops.aten.add.Tensor(primals_246, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_14 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
    getitem_40: "f32[1, 128, 1, 1]" = var_mean_14[0]
    getitem_41: "f32[1, 128, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_74: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05)
    rsqrt_14: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_14: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_41)
    mul_112: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    squeeze_42: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
    squeeze_43: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
    mul_113: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
    mul_114: "f32[128]" = torch.ops.aten.mul.Tensor(primals_247, 0.9)
    add_75: "f32[128]" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
    squeeze_44: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
    mul_115: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.000030518509476);  squeeze_44 = None
    mul_116: "f32[128]" = torch.ops.aten.mul.Tensor(mul_115, 0.1);  mul_115 = None
    mul_117: "f32[128]" = torch.ops.aten.mul.Tensor(primals_248, 0.9)
    add_76: "f32[128]" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
    unsqueeze_56: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_57: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    mul_118: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_57);  mul_112 = unsqueeze_57 = None
    unsqueeze_58: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_59: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    add_77: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_59);  mul_118 = unsqueeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_14: "b8[8, 128, 64, 64]" = torch.ops.aten.gt.Scalar(add_77, 0)
    mul_119: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(add_77, 0.01)
    where_14: "f32[8, 128, 64, 64]" = torch.ops.aten.where.self(gt_14, add_77, mul_119);  gt_14 = add_77 = mul_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:126, code: x = self.conv(x)
    convolution_15: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(where_14, primals_150, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_78: "i64[]" = torch.ops.aten.add.Tensor(primals_249, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_15 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
    getitem_42: "f32[1, 256, 1, 1]" = var_mean_15[0]
    getitem_43: "f32[1, 256, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_79: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05)
    rsqrt_15: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
    sub_15: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_43)
    mul_120: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    squeeze_45: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
    squeeze_46: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
    mul_121: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
    mul_122: "f32[256]" = torch.ops.aten.mul.Tensor(primals_250, 0.9)
    add_80: "f32[256]" = torch.ops.aten.add.Tensor(mul_121, mul_122);  mul_121 = mul_122 = None
    squeeze_47: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
    mul_123: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0001220852154804);  squeeze_47 = None
    mul_124: "f32[256]" = torch.ops.aten.mul.Tensor(mul_123, 0.1);  mul_123 = None
    mul_125: "f32[256]" = torch.ops.aten.mul.Tensor(primals_251, 0.9)
    add_81: "f32[256]" = torch.ops.aten.add.Tensor(mul_124, mul_125);  mul_124 = mul_125 = None
    unsqueeze_60: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1)
    unsqueeze_61: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_126: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_120, unsqueeze_61);  mul_120 = unsqueeze_61 = None
    unsqueeze_62: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1);  primals_32 = None
    unsqueeze_63: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_82: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(mul_126, unsqueeze_63);  mul_126 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_15: "b8[8, 256, 32, 32]" = torch.ops.aten.gt.Scalar(add_82, 0)
    mul_127: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(add_82, 0.01)
    where_15: "f32[8, 256, 32, 32]" = torch.ops.aten.where.self(gt_15, add_82, mul_127);  gt_15 = add_82 = mul_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_16: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(where_15, primals_151, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_83: "i64[]" = torch.ops.aten.add.Tensor(primals_252, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_16 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
    getitem_44: "f32[1, 256, 1, 1]" = var_mean_16[0]
    getitem_45: "f32[1, 256, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_84: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05)
    rsqrt_16: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_16: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_45)
    mul_128: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    squeeze_48: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
    squeeze_49: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
    mul_129: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
    mul_130: "f32[256]" = torch.ops.aten.mul.Tensor(primals_253, 0.9)
    add_85: "f32[256]" = torch.ops.aten.add.Tensor(mul_129, mul_130);  mul_129 = mul_130 = None
    squeeze_50: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
    mul_131: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0001220852154804);  squeeze_50 = None
    mul_132: "f32[256]" = torch.ops.aten.mul.Tensor(mul_131, 0.1);  mul_131 = None
    mul_133: "f32[256]" = torch.ops.aten.mul.Tensor(primals_254, 0.9)
    add_86: "f32[256]" = torch.ops.aten.add.Tensor(mul_132, mul_133);  mul_132 = mul_133 = None
    unsqueeze_64: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1)
    unsqueeze_65: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    mul_134: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_128, unsqueeze_65);  mul_128 = unsqueeze_65 = None
    unsqueeze_66: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_34, -1);  primals_34 = None
    unsqueeze_67: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    add_87: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(mul_134, unsqueeze_67);  mul_134 = unsqueeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_16: "b8[8, 256, 32, 32]" = torch.ops.aten.gt.Scalar(add_87, 0)
    mul_135: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(add_87, 0.01)
    where_16: "f32[8, 256, 32, 32]" = torch.ops.aten.where.self(gt_16, add_87, mul_135);  gt_16 = add_87 = mul_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    split_with_sizes_7 = torch.ops.aten.split_with_sizes.default(where_16, [128, 128], 1)
    getitem_49: "f32[8, 128, 32, 32]" = split_with_sizes_7[1]
    convolution_17: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(getitem_49, primals_152, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_88: "i64[]" = torch.ops.aten.add.Tensor(primals_255, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_17 = torch.ops.aten.var_mean.correction(convolution_17, [0, 2, 3], correction = 0, keepdim = True)
    getitem_50: "f32[1, 128, 1, 1]" = var_mean_17[0]
    getitem_51: "f32[1, 128, 1, 1]" = var_mean_17[1];  var_mean_17 = None
    add_89: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05)
    rsqrt_17: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
    sub_17: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_17, getitem_51)
    mul_136: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    squeeze_51: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
    squeeze_52: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
    mul_137: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
    mul_138: "f32[128]" = torch.ops.aten.mul.Tensor(primals_256, 0.9)
    add_90: "f32[128]" = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
    squeeze_53: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
    mul_139: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0001220852154804);  squeeze_53 = None
    mul_140: "f32[128]" = torch.ops.aten.mul.Tensor(mul_139, 0.1);  mul_139 = None
    mul_141: "f32[128]" = torch.ops.aten.mul.Tensor(primals_257, 0.9)
    add_91: "f32[128]" = torch.ops.aten.add.Tensor(mul_140, mul_141);  mul_140 = mul_141 = None
    unsqueeze_68: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_69: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_142: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_69);  mul_136 = unsqueeze_69 = None
    unsqueeze_70: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_71: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_92: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_142, unsqueeze_71);  mul_142 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_17: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_92, 0)
    mul_143: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_92, 0.01)
    where_17: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_17, add_92, mul_143);  gt_17 = add_92 = mul_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_18: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(where_17, primals_153, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_93: "i64[]" = torch.ops.aten.add.Tensor(primals_258, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_18 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
    getitem_52: "f32[1, 128, 1, 1]" = var_mean_18[0]
    getitem_53: "f32[1, 128, 1, 1]" = var_mean_18[1];  var_mean_18 = None
    add_94: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05)
    rsqrt_18: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
    sub_18: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_53)
    mul_144: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    squeeze_54: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
    squeeze_55: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
    mul_145: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
    mul_146: "f32[128]" = torch.ops.aten.mul.Tensor(primals_259, 0.9)
    add_95: "f32[128]" = torch.ops.aten.add.Tensor(mul_145, mul_146);  mul_145 = mul_146 = None
    squeeze_56: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
    mul_147: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0001220852154804);  squeeze_56 = None
    mul_148: "f32[128]" = torch.ops.aten.mul.Tensor(mul_147, 0.1);  mul_147 = None
    mul_149: "f32[128]" = torch.ops.aten.mul.Tensor(primals_260, 0.9)
    add_96: "f32[128]" = torch.ops.aten.add.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None
    unsqueeze_72: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1)
    unsqueeze_73: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    mul_150: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_144, unsqueeze_73);  mul_144 = unsqueeze_73 = None
    unsqueeze_74: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1);  primals_38 = None
    unsqueeze_75: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    add_97: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_150, unsqueeze_75);  mul_150 = unsqueeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_18: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_97, 0)
    mul_151: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_97, 0.01)
    where_18: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_18, add_97, mul_151);  gt_18 = add_97 = mul_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_98: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(where_18, getitem_49)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_19: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(add_98, primals_154, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_99: "i64[]" = torch.ops.aten.add.Tensor(primals_261, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_19 = torch.ops.aten.var_mean.correction(convolution_19, [0, 2, 3], correction = 0, keepdim = True)
    getitem_54: "f32[1, 128, 1, 1]" = var_mean_19[0]
    getitem_55: "f32[1, 128, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_100: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05)
    rsqrt_19: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
    sub_19: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_55)
    mul_152: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    squeeze_57: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
    squeeze_58: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
    mul_153: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
    mul_154: "f32[128]" = torch.ops.aten.mul.Tensor(primals_262, 0.9)
    add_101: "f32[128]" = torch.ops.aten.add.Tensor(mul_153, mul_154);  mul_153 = mul_154 = None
    squeeze_59: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
    mul_155: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0001220852154804);  squeeze_59 = None
    mul_156: "f32[128]" = torch.ops.aten.mul.Tensor(mul_155, 0.1);  mul_155 = None
    mul_157: "f32[128]" = torch.ops.aten.mul.Tensor(primals_263, 0.9)
    add_102: "f32[128]" = torch.ops.aten.add.Tensor(mul_156, mul_157);  mul_156 = mul_157 = None
    unsqueeze_76: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1)
    unsqueeze_77: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_158: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_152, unsqueeze_77);  mul_152 = unsqueeze_77 = None
    unsqueeze_78: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_40, -1);  primals_40 = None
    unsqueeze_79: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_103: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_158, unsqueeze_79);  mul_158 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_19: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_103, 0)
    mul_159: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_103, 0.01)
    where_19: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_19, add_103, mul_159);  gt_19 = add_103 = mul_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_20: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(where_19, primals_155, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_104: "i64[]" = torch.ops.aten.add.Tensor(primals_264, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_20 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
    getitem_56: "f32[1, 128, 1, 1]" = var_mean_20[0]
    getitem_57: "f32[1, 128, 1, 1]" = var_mean_20[1];  var_mean_20 = None
    add_105: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05)
    rsqrt_20: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
    sub_20: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_57)
    mul_160: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    squeeze_60: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
    squeeze_61: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
    mul_161: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
    mul_162: "f32[128]" = torch.ops.aten.mul.Tensor(primals_265, 0.9)
    add_106: "f32[128]" = torch.ops.aten.add.Tensor(mul_161, mul_162);  mul_161 = mul_162 = None
    squeeze_62: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
    mul_163: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0001220852154804);  squeeze_62 = None
    mul_164: "f32[128]" = torch.ops.aten.mul.Tensor(mul_163, 0.1);  mul_163 = None
    mul_165: "f32[128]" = torch.ops.aten.mul.Tensor(primals_266, 0.9)
    add_107: "f32[128]" = torch.ops.aten.add.Tensor(mul_164, mul_165);  mul_164 = mul_165 = None
    unsqueeze_80: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_81: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    mul_166: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_160, unsqueeze_81);  mul_160 = unsqueeze_81 = None
    unsqueeze_82: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_83: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    add_108: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_166, unsqueeze_83);  mul_166 = unsqueeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_20: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_108, 0)
    mul_167: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_108, 0.01)
    where_20: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_20, add_108, mul_167);  gt_20 = add_108 = mul_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_109: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(where_20, add_98)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_21: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(add_109, primals_156, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_110: "i64[]" = torch.ops.aten.add.Tensor(primals_267, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_21 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
    getitem_58: "f32[1, 128, 1, 1]" = var_mean_21[0]
    getitem_59: "f32[1, 128, 1, 1]" = var_mean_21[1];  var_mean_21 = None
    add_111: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05)
    rsqrt_21: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
    sub_21: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_21, getitem_59)
    mul_168: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    squeeze_63: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
    squeeze_64: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
    mul_169: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
    mul_170: "f32[128]" = torch.ops.aten.mul.Tensor(primals_268, 0.9)
    add_112: "f32[128]" = torch.ops.aten.add.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None
    squeeze_65: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
    mul_171: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0001220852154804);  squeeze_65 = None
    mul_172: "f32[128]" = torch.ops.aten.mul.Tensor(mul_171, 0.1);  mul_171 = None
    mul_173: "f32[128]" = torch.ops.aten.mul.Tensor(primals_269, 0.9)
    add_113: "f32[128]" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
    unsqueeze_84: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1)
    unsqueeze_85: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_174: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_168, unsqueeze_85);  mul_168 = unsqueeze_85 = None
    unsqueeze_86: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1);  primals_44 = None
    unsqueeze_87: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_114: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_174, unsqueeze_87);  mul_174 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_21: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_114, 0)
    mul_175: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_114, 0.01)
    where_21: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_21, add_114, mul_175);  gt_21 = add_114 = mul_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_22: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(where_21, primals_157, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_115: "i64[]" = torch.ops.aten.add.Tensor(primals_270, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_22 = torch.ops.aten.var_mean.correction(convolution_22, [0, 2, 3], correction = 0, keepdim = True)
    getitem_60: "f32[1, 128, 1, 1]" = var_mean_22[0]
    getitem_61: "f32[1, 128, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_116: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05)
    rsqrt_22: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
    sub_22: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_22, getitem_61)
    mul_176: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    squeeze_66: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
    squeeze_67: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
    mul_177: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
    mul_178: "f32[128]" = torch.ops.aten.mul.Tensor(primals_271, 0.9)
    add_117: "f32[128]" = torch.ops.aten.add.Tensor(mul_177, mul_178);  mul_177 = mul_178 = None
    squeeze_68: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
    mul_179: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0001220852154804);  squeeze_68 = None
    mul_180: "f32[128]" = torch.ops.aten.mul.Tensor(mul_179, 0.1);  mul_179 = None
    mul_181: "f32[128]" = torch.ops.aten.mul.Tensor(primals_272, 0.9)
    add_118: "f32[128]" = torch.ops.aten.add.Tensor(mul_180, mul_181);  mul_180 = mul_181 = None
    unsqueeze_88: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1)
    unsqueeze_89: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    mul_182: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_176, unsqueeze_89);  mul_176 = unsqueeze_89 = None
    unsqueeze_90: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_46, -1);  primals_46 = None
    unsqueeze_91: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    add_119: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_182, unsqueeze_91);  mul_182 = unsqueeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_22: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_119, 0)
    mul_183: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_119, 0.01)
    where_22: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_22, add_119, mul_183);  gt_22 = add_119 = mul_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_120: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(where_22, add_109)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_23: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(add_120, primals_158, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_121: "i64[]" = torch.ops.aten.add.Tensor(primals_273, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_23 = torch.ops.aten.var_mean.correction(convolution_23, [0, 2, 3], correction = 0, keepdim = True)
    getitem_62: "f32[1, 128, 1, 1]" = var_mean_23[0]
    getitem_63: "f32[1, 128, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_122: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
    rsqrt_23: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    sub_23: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_23, getitem_63)
    mul_184: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    squeeze_69: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
    squeeze_70: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
    mul_185: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
    mul_186: "f32[128]" = torch.ops.aten.mul.Tensor(primals_274, 0.9)
    add_123: "f32[128]" = torch.ops.aten.add.Tensor(mul_185, mul_186);  mul_185 = mul_186 = None
    squeeze_71: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
    mul_187: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0001220852154804);  squeeze_71 = None
    mul_188: "f32[128]" = torch.ops.aten.mul.Tensor(mul_187, 0.1);  mul_187 = None
    mul_189: "f32[128]" = torch.ops.aten.mul.Tensor(primals_275, 0.9)
    add_124: "f32[128]" = torch.ops.aten.add.Tensor(mul_188, mul_189);  mul_188 = mul_189 = None
    unsqueeze_92: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_93: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_190: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_184, unsqueeze_93);  mul_184 = unsqueeze_93 = None
    unsqueeze_94: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_95: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_125: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_190, unsqueeze_95);  mul_190 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_23: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_125, 0)
    mul_191: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_125, 0.01)
    where_23: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_23, add_125, mul_191);  gt_23 = add_125 = mul_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_24: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(where_23, primals_159, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_126: "i64[]" = torch.ops.aten.add.Tensor(primals_276, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_24 = torch.ops.aten.var_mean.correction(convolution_24, [0, 2, 3], correction = 0, keepdim = True)
    getitem_64: "f32[1, 128, 1, 1]" = var_mean_24[0]
    getitem_65: "f32[1, 128, 1, 1]" = var_mean_24[1];  var_mean_24 = None
    add_127: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
    rsqrt_24: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
    sub_24: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_24, getitem_65)
    mul_192: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    squeeze_72: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
    squeeze_73: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
    mul_193: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
    mul_194: "f32[128]" = torch.ops.aten.mul.Tensor(primals_277, 0.9)
    add_128: "f32[128]" = torch.ops.aten.add.Tensor(mul_193, mul_194);  mul_193 = mul_194 = None
    squeeze_74: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
    mul_195: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0001220852154804);  squeeze_74 = None
    mul_196: "f32[128]" = torch.ops.aten.mul.Tensor(mul_195, 0.1);  mul_195 = None
    mul_197: "f32[128]" = torch.ops.aten.mul.Tensor(primals_278, 0.9)
    add_129: "f32[128]" = torch.ops.aten.add.Tensor(mul_196, mul_197);  mul_196 = mul_197 = None
    unsqueeze_96: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1)
    unsqueeze_97: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    mul_198: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_192, unsqueeze_97);  mul_192 = unsqueeze_97 = None
    unsqueeze_98: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1);  primals_50 = None
    unsqueeze_99: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    add_130: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_198, unsqueeze_99);  mul_198 = unsqueeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_24: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_130, 0)
    mul_199: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_130, 0.01)
    where_24: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_24, add_130, mul_199);  gt_24 = add_130 = mul_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_131: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(where_24, add_120)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_25: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(add_131, primals_160, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_132: "i64[]" = torch.ops.aten.add.Tensor(primals_279, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_25 = torch.ops.aten.var_mean.correction(convolution_25, [0, 2, 3], correction = 0, keepdim = True)
    getitem_66: "f32[1, 128, 1, 1]" = var_mean_25[0]
    getitem_67: "f32[1, 128, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_133: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05)
    rsqrt_25: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
    sub_25: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_67)
    mul_200: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    squeeze_75: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
    squeeze_76: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
    mul_201: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
    mul_202: "f32[128]" = torch.ops.aten.mul.Tensor(primals_280, 0.9)
    add_134: "f32[128]" = torch.ops.aten.add.Tensor(mul_201, mul_202);  mul_201 = mul_202 = None
    squeeze_77: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
    mul_203: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0001220852154804);  squeeze_77 = None
    mul_204: "f32[128]" = torch.ops.aten.mul.Tensor(mul_203, 0.1);  mul_203 = None
    mul_205: "f32[128]" = torch.ops.aten.mul.Tensor(primals_281, 0.9)
    add_135: "f32[128]" = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None
    unsqueeze_100: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1)
    unsqueeze_101: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_206: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_200, unsqueeze_101);  mul_200 = unsqueeze_101 = None
    unsqueeze_102: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_52, -1);  primals_52 = None
    unsqueeze_103: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_136: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_206, unsqueeze_103);  mul_206 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_25: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_136, 0)
    mul_207: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_136, 0.01)
    where_25: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_25, add_136, mul_207);  gt_25 = add_136 = mul_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_26: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(where_25, primals_161, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_137: "i64[]" = torch.ops.aten.add.Tensor(primals_282, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_26 = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
    getitem_68: "f32[1, 128, 1, 1]" = var_mean_26[0]
    getitem_69: "f32[1, 128, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_138: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05)
    rsqrt_26: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
    sub_26: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_69)
    mul_208: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    squeeze_78: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
    squeeze_79: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
    mul_209: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
    mul_210: "f32[128]" = torch.ops.aten.mul.Tensor(primals_283, 0.9)
    add_139: "f32[128]" = torch.ops.aten.add.Tensor(mul_209, mul_210);  mul_209 = mul_210 = None
    squeeze_80: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
    mul_211: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0001220852154804);  squeeze_80 = None
    mul_212: "f32[128]" = torch.ops.aten.mul.Tensor(mul_211, 0.1);  mul_211 = None
    mul_213: "f32[128]" = torch.ops.aten.mul.Tensor(primals_284, 0.9)
    add_140: "f32[128]" = torch.ops.aten.add.Tensor(mul_212, mul_213);  mul_212 = mul_213 = None
    unsqueeze_104: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
    unsqueeze_105: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    mul_214: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_208, unsqueeze_105);  mul_208 = unsqueeze_105 = None
    unsqueeze_106: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
    unsqueeze_107: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    add_141: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_214, unsqueeze_107);  mul_214 = unsqueeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_26: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_141, 0)
    mul_215: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_141, 0.01)
    where_26: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_26, add_141, mul_215);  gt_26 = add_141 = mul_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_142: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(where_26, add_131)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_27: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(add_142, primals_162, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_143: "i64[]" = torch.ops.aten.add.Tensor(primals_285, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_27 = torch.ops.aten.var_mean.correction(convolution_27, [0, 2, 3], correction = 0, keepdim = True)
    getitem_70: "f32[1, 128, 1, 1]" = var_mean_27[0]
    getitem_71: "f32[1, 128, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_144: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05)
    rsqrt_27: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    sub_27: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_27, getitem_71)
    mul_216: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
    squeeze_81: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
    squeeze_82: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
    mul_217: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
    mul_218: "f32[128]" = torch.ops.aten.mul.Tensor(primals_286, 0.9)
    add_145: "f32[128]" = torch.ops.aten.add.Tensor(mul_217, mul_218);  mul_217 = mul_218 = None
    squeeze_83: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
    mul_219: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0001220852154804);  squeeze_83 = None
    mul_220: "f32[128]" = torch.ops.aten.mul.Tensor(mul_219, 0.1);  mul_219 = None
    mul_221: "f32[128]" = torch.ops.aten.mul.Tensor(primals_287, 0.9)
    add_146: "f32[128]" = torch.ops.aten.add.Tensor(mul_220, mul_221);  mul_220 = mul_221 = None
    unsqueeze_108: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1)
    unsqueeze_109: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_222: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_216, unsqueeze_109);  mul_216 = unsqueeze_109 = None
    unsqueeze_110: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1);  primals_56 = None
    unsqueeze_111: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_147: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_222, unsqueeze_111);  mul_222 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_27: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_147, 0)
    mul_223: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_147, 0.01)
    where_27: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_27, add_147, mul_223);  gt_27 = add_147 = mul_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_28: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(where_27, primals_163, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_148: "i64[]" = torch.ops.aten.add.Tensor(primals_288, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_28 = torch.ops.aten.var_mean.correction(convolution_28, [0, 2, 3], correction = 0, keepdim = True)
    getitem_72: "f32[1, 128, 1, 1]" = var_mean_28[0]
    getitem_73: "f32[1, 128, 1, 1]" = var_mean_28[1];  var_mean_28 = None
    add_149: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05)
    rsqrt_28: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
    sub_28: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_28, getitem_73)
    mul_224: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
    squeeze_84: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
    squeeze_85: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
    mul_225: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
    mul_226: "f32[128]" = torch.ops.aten.mul.Tensor(primals_289, 0.9)
    add_150: "f32[128]" = torch.ops.aten.add.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None
    squeeze_86: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
    mul_227: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0001220852154804);  squeeze_86 = None
    mul_228: "f32[128]" = torch.ops.aten.mul.Tensor(mul_227, 0.1);  mul_227 = None
    mul_229: "f32[128]" = torch.ops.aten.mul.Tensor(primals_290, 0.9)
    add_151: "f32[128]" = torch.ops.aten.add.Tensor(mul_228, mul_229);  mul_228 = mul_229 = None
    unsqueeze_112: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1)
    unsqueeze_113: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    mul_230: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_224, unsqueeze_113);  mul_224 = unsqueeze_113 = None
    unsqueeze_114: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_58, -1);  primals_58 = None
    unsqueeze_115: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    add_152: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_115);  mul_230 = unsqueeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_28: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_152, 0)
    mul_231: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_152, 0.01)
    where_28: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_28, add_152, mul_231);  gt_28 = add_152 = mul_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_153: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(where_28, add_142)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_29: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(add_153, primals_164, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_154: "i64[]" = torch.ops.aten.add.Tensor(primals_291, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_29 = torch.ops.aten.var_mean.correction(convolution_29, [0, 2, 3], correction = 0, keepdim = True)
    getitem_74: "f32[1, 128, 1, 1]" = var_mean_29[0]
    getitem_75: "f32[1, 128, 1, 1]" = var_mean_29[1];  var_mean_29 = None
    add_155: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05)
    rsqrt_29: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
    sub_29: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_29, getitem_75)
    mul_232: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
    squeeze_87: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
    squeeze_88: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
    mul_233: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
    mul_234: "f32[128]" = torch.ops.aten.mul.Tensor(primals_292, 0.9)
    add_156: "f32[128]" = torch.ops.aten.add.Tensor(mul_233, mul_234);  mul_233 = mul_234 = None
    squeeze_89: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
    mul_235: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0001220852154804);  squeeze_89 = None
    mul_236: "f32[128]" = torch.ops.aten.mul.Tensor(mul_235, 0.1);  mul_235 = None
    mul_237: "f32[128]" = torch.ops.aten.mul.Tensor(primals_293, 0.9)
    add_157: "f32[128]" = torch.ops.aten.add.Tensor(mul_236, mul_237);  mul_236 = mul_237 = None
    unsqueeze_116: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
    unsqueeze_117: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_238: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_232, unsqueeze_117);  mul_232 = unsqueeze_117 = None
    unsqueeze_118: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
    unsqueeze_119: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_158: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_238, unsqueeze_119);  mul_238 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_29: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_158, 0)
    mul_239: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_158, 0.01)
    where_29: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_29, add_158, mul_239);  gt_29 = add_158 = mul_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_30: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(where_29, primals_165, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_159: "i64[]" = torch.ops.aten.add.Tensor(primals_294, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_30 = torch.ops.aten.var_mean.correction(convolution_30, [0, 2, 3], correction = 0, keepdim = True)
    getitem_76: "f32[1, 128, 1, 1]" = var_mean_30[0]
    getitem_77: "f32[1, 128, 1, 1]" = var_mean_30[1];  var_mean_30 = None
    add_160: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05)
    rsqrt_30: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_160);  add_160 = None
    sub_30: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_30, getitem_77)
    mul_240: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
    squeeze_90: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
    squeeze_91: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
    mul_241: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
    mul_242: "f32[128]" = torch.ops.aten.mul.Tensor(primals_295, 0.9)
    add_161: "f32[128]" = torch.ops.aten.add.Tensor(mul_241, mul_242);  mul_241 = mul_242 = None
    squeeze_92: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
    mul_243: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0001220852154804);  squeeze_92 = None
    mul_244: "f32[128]" = torch.ops.aten.mul.Tensor(mul_243, 0.1);  mul_243 = None
    mul_245: "f32[128]" = torch.ops.aten.mul.Tensor(primals_296, 0.9)
    add_162: "f32[128]" = torch.ops.aten.add.Tensor(mul_244, mul_245);  mul_244 = mul_245 = None
    unsqueeze_120: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_61, -1)
    unsqueeze_121: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    mul_246: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_240, unsqueeze_121);  mul_240 = unsqueeze_121 = None
    unsqueeze_122: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1);  primals_62 = None
    unsqueeze_123: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    add_163: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_246, unsqueeze_123);  mul_246 = unsqueeze_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_30: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_163, 0)
    mul_247: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_163, 0.01)
    where_30: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_30, add_163, mul_247);  gt_30 = add_163 = mul_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_164: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(where_30, add_153)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_31: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(add_164, primals_166, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_165: "i64[]" = torch.ops.aten.add.Tensor(primals_297, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_31 = torch.ops.aten.var_mean.correction(convolution_31, [0, 2, 3], correction = 0, keepdim = True)
    getitem_78: "f32[1, 128, 1, 1]" = var_mean_31[0]
    getitem_79: "f32[1, 128, 1, 1]" = var_mean_31[1];  var_mean_31 = None
    add_166: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05)
    rsqrt_31: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
    sub_31: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_31, getitem_79)
    mul_248: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
    squeeze_93: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
    squeeze_94: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
    mul_249: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
    mul_250: "f32[128]" = torch.ops.aten.mul.Tensor(primals_298, 0.9)
    add_167: "f32[128]" = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
    squeeze_95: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
    mul_251: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0001220852154804);  squeeze_95 = None
    mul_252: "f32[128]" = torch.ops.aten.mul.Tensor(mul_251, 0.1);  mul_251 = None
    mul_253: "f32[128]" = torch.ops.aten.mul.Tensor(primals_299, 0.9)
    add_168: "f32[128]" = torch.ops.aten.add.Tensor(mul_252, mul_253);  mul_252 = mul_253 = None
    unsqueeze_124: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1)
    unsqueeze_125: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_254: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_248, unsqueeze_125);  mul_248 = unsqueeze_125 = None
    unsqueeze_126: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_64, -1);  primals_64 = None
    unsqueeze_127: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_169: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_254, unsqueeze_127);  mul_254 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_31: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_169, 0)
    mul_255: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_169, 0.01)
    where_31: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_31, add_169, mul_255);  gt_31 = add_169 = mul_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_32: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(where_31, primals_167, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_170: "i64[]" = torch.ops.aten.add.Tensor(primals_300, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_32 = torch.ops.aten.var_mean.correction(convolution_32, [0, 2, 3], correction = 0, keepdim = True)
    getitem_80: "f32[1, 128, 1, 1]" = var_mean_32[0]
    getitem_81: "f32[1, 128, 1, 1]" = var_mean_32[1];  var_mean_32 = None
    add_171: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05)
    rsqrt_32: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
    sub_32: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_32, getitem_81)
    mul_256: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
    squeeze_96: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
    squeeze_97: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
    mul_257: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
    mul_258: "f32[128]" = torch.ops.aten.mul.Tensor(primals_301, 0.9)
    add_172: "f32[128]" = torch.ops.aten.add.Tensor(mul_257, mul_258);  mul_257 = mul_258 = None
    squeeze_98: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
    mul_259: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0001220852154804);  squeeze_98 = None
    mul_260: "f32[128]" = torch.ops.aten.mul.Tensor(mul_259, 0.1);  mul_259 = None
    mul_261: "f32[128]" = torch.ops.aten.mul.Tensor(primals_302, 0.9)
    add_173: "f32[128]" = torch.ops.aten.add.Tensor(mul_260, mul_261);  mul_260 = mul_261 = None
    unsqueeze_128: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1)
    unsqueeze_129: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    mul_262: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_256, unsqueeze_129);  mul_256 = unsqueeze_129 = None
    unsqueeze_130: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
    unsqueeze_131: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    add_174: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_262, unsqueeze_131);  mul_262 = unsqueeze_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_32: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_174, 0)
    mul_263: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_174, 0.01)
    where_32: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_32, add_174, mul_263);  gt_32 = add_174 = mul_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_175: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(where_32, add_164)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_33: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(add_175, primals_168, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_176: "i64[]" = torch.ops.aten.add.Tensor(primals_303, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_33 = torch.ops.aten.var_mean.correction(convolution_33, [0, 2, 3], correction = 0, keepdim = True)
    getitem_82: "f32[1, 128, 1, 1]" = var_mean_33[0]
    getitem_83: "f32[1, 128, 1, 1]" = var_mean_33[1];  var_mean_33 = None
    add_177: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05)
    rsqrt_33: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_177);  add_177 = None
    sub_33: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_33, getitem_83)
    mul_264: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
    squeeze_99: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
    squeeze_100: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
    mul_265: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
    mul_266: "f32[128]" = torch.ops.aten.mul.Tensor(primals_304, 0.9)
    add_178: "f32[128]" = torch.ops.aten.add.Tensor(mul_265, mul_266);  mul_265 = mul_266 = None
    squeeze_101: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
    mul_267: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0001220852154804);  squeeze_101 = None
    mul_268: "f32[128]" = torch.ops.aten.mul.Tensor(mul_267, 0.1);  mul_267 = None
    mul_269: "f32[128]" = torch.ops.aten.mul.Tensor(primals_305, 0.9)
    add_179: "f32[128]" = torch.ops.aten.add.Tensor(mul_268, mul_269);  mul_268 = mul_269 = None
    unsqueeze_132: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_67, -1)
    unsqueeze_133: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_270: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_264, unsqueeze_133);  mul_264 = unsqueeze_133 = None
    unsqueeze_134: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1);  primals_68 = None
    unsqueeze_135: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_180: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_270, unsqueeze_135);  mul_270 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_33: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_180, 0)
    mul_271: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_180, 0.01)
    where_33: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_33, add_180, mul_271);  gt_33 = add_180 = mul_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:339, code: out = self.conv_transition(torch.cat([xs, xb], dim=1))
    getitem_84: "f32[8, 128, 32, 32]" = split_with_sizes_7[0];  split_with_sizes_7 = None
    cat_2: "f32[8, 256, 32, 32]" = torch.ops.aten.cat.default([getitem_84, where_33], 1);  getitem_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_34: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(cat_2, primals_169, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_181: "i64[]" = torch.ops.aten.add.Tensor(primals_306, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_34 = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
    getitem_86: "f32[1, 256, 1, 1]" = var_mean_34[0]
    getitem_87: "f32[1, 256, 1, 1]" = var_mean_34[1];  var_mean_34 = None
    add_182: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05)
    rsqrt_34: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
    sub_34: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_34, getitem_87)
    mul_272: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
    squeeze_102: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
    squeeze_103: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
    mul_273: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
    mul_274: "f32[256]" = torch.ops.aten.mul.Tensor(primals_307, 0.9)
    add_183: "f32[256]" = torch.ops.aten.add.Tensor(mul_273, mul_274);  mul_273 = mul_274 = None
    squeeze_104: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
    mul_275: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0001220852154804);  squeeze_104 = None
    mul_276: "f32[256]" = torch.ops.aten.mul.Tensor(mul_275, 0.1);  mul_275 = None
    mul_277: "f32[256]" = torch.ops.aten.mul.Tensor(primals_308, 0.9)
    add_184: "f32[256]" = torch.ops.aten.add.Tensor(mul_276, mul_277);  mul_276 = mul_277 = None
    unsqueeze_136: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1)
    unsqueeze_137: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    mul_278: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_272, unsqueeze_137);  mul_272 = unsqueeze_137 = None
    unsqueeze_138: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_70, -1);  primals_70 = None
    unsqueeze_139: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    add_185: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(mul_278, unsqueeze_139);  mul_278 = unsqueeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_34: "b8[8, 256, 32, 32]" = torch.ops.aten.gt.Scalar(add_185, 0)
    mul_279: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(add_185, 0.01)
    where_34: "f32[8, 256, 32, 32]" = torch.ops.aten.where.self(gt_34, add_185, mul_279);  gt_34 = add_185 = mul_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:126, code: x = self.conv(x)
    convolution_35: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(where_34, primals_170, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_186: "i64[]" = torch.ops.aten.add.Tensor(primals_309, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_35 = torch.ops.aten.var_mean.correction(convolution_35, [0, 2, 3], correction = 0, keepdim = True)
    getitem_88: "f32[1, 512, 1, 1]" = var_mean_35[0]
    getitem_89: "f32[1, 512, 1, 1]" = var_mean_35[1];  var_mean_35 = None
    add_187: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05)
    rsqrt_35: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_187);  add_187 = None
    sub_35: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_35, getitem_89)
    mul_280: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
    squeeze_105: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
    squeeze_106: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
    mul_281: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
    mul_282: "f32[512]" = torch.ops.aten.mul.Tensor(primals_310, 0.9)
    add_188: "f32[512]" = torch.ops.aten.add.Tensor(mul_281, mul_282);  mul_281 = mul_282 = None
    squeeze_107: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
    mul_283: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_107, 1.0004885197850513);  squeeze_107 = None
    mul_284: "f32[512]" = torch.ops.aten.mul.Tensor(mul_283, 0.1);  mul_283 = None
    mul_285: "f32[512]" = torch.ops.aten.mul.Tensor(primals_311, 0.9)
    add_189: "f32[512]" = torch.ops.aten.add.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None
    unsqueeze_140: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
    unsqueeze_141: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_286: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_141);  mul_280 = unsqueeze_141 = None
    unsqueeze_142: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
    unsqueeze_143: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_190: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_286, unsqueeze_143);  mul_286 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_35: "b8[8, 512, 16, 16]" = torch.ops.aten.gt.Scalar(add_190, 0)
    mul_287: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(add_190, 0.01)
    where_35: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(gt_35, add_190, mul_287);  gt_35 = add_190 = mul_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_36: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(where_35, primals_171, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_191: "i64[]" = torch.ops.aten.add.Tensor(primals_312, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_36 = torch.ops.aten.var_mean.correction(convolution_36, [0, 2, 3], correction = 0, keepdim = True)
    getitem_90: "f32[1, 512, 1, 1]" = var_mean_36[0]
    getitem_91: "f32[1, 512, 1, 1]" = var_mean_36[1];  var_mean_36 = None
    add_192: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05)
    rsqrt_36: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
    sub_36: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_36, getitem_91)
    mul_288: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
    squeeze_108: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
    squeeze_109: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
    mul_289: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
    mul_290: "f32[512]" = torch.ops.aten.mul.Tensor(primals_313, 0.9)
    add_193: "f32[512]" = torch.ops.aten.add.Tensor(mul_289, mul_290);  mul_289 = mul_290 = None
    squeeze_110: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
    mul_291: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0004885197850513);  squeeze_110 = None
    mul_292: "f32[512]" = torch.ops.aten.mul.Tensor(mul_291, 0.1);  mul_291 = None
    mul_293: "f32[512]" = torch.ops.aten.mul.Tensor(primals_314, 0.9)
    add_194: "f32[512]" = torch.ops.aten.add.Tensor(mul_292, mul_293);  mul_292 = mul_293 = None
    unsqueeze_144: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_73, -1)
    unsqueeze_145: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    mul_294: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_288, unsqueeze_145);  mul_288 = unsqueeze_145 = None
    unsqueeze_146: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1);  primals_74 = None
    unsqueeze_147: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    add_195: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_294, unsqueeze_147);  mul_294 = unsqueeze_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_36: "b8[8, 512, 16, 16]" = torch.ops.aten.gt.Scalar(add_195, 0)
    mul_295: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(add_195, 0.01)
    where_36: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(gt_36, add_195, mul_295);  gt_36 = add_195 = mul_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    split_with_sizes_10 = torch.ops.aten.split_with_sizes.default(where_36, [256, 256], 1)
    getitem_95: "f32[8, 256, 16, 16]" = split_with_sizes_10[1]
    convolution_37: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(getitem_95, primals_172, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_196: "i64[]" = torch.ops.aten.add.Tensor(primals_315, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_37 = torch.ops.aten.var_mean.correction(convolution_37, [0, 2, 3], correction = 0, keepdim = True)
    getitem_96: "f32[1, 256, 1, 1]" = var_mean_37[0]
    getitem_97: "f32[1, 256, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    add_197: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05)
    rsqrt_37: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
    sub_37: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_37, getitem_97)
    mul_296: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
    squeeze_111: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
    squeeze_112: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
    mul_297: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
    mul_298: "f32[256]" = torch.ops.aten.mul.Tensor(primals_316, 0.9)
    add_198: "f32[256]" = torch.ops.aten.add.Tensor(mul_297, mul_298);  mul_297 = mul_298 = None
    squeeze_113: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
    mul_299: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0004885197850513);  squeeze_113 = None
    mul_300: "f32[256]" = torch.ops.aten.mul.Tensor(mul_299, 0.1);  mul_299 = None
    mul_301: "f32[256]" = torch.ops.aten.mul.Tensor(primals_317, 0.9)
    add_199: "f32[256]" = torch.ops.aten.add.Tensor(mul_300, mul_301);  mul_300 = mul_301 = None
    unsqueeze_148: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1)
    unsqueeze_149: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_302: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_296, unsqueeze_149);  mul_296 = unsqueeze_149 = None
    unsqueeze_150: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_76, -1);  primals_76 = None
    unsqueeze_151: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_200: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_302, unsqueeze_151);  mul_302 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_37: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_200, 0)
    mul_303: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_200, 0.01)
    where_37: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_37, add_200, mul_303);  gt_37 = add_200 = mul_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_38: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(where_37, primals_173, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_201: "i64[]" = torch.ops.aten.add.Tensor(primals_318, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_38 = torch.ops.aten.var_mean.correction(convolution_38, [0, 2, 3], correction = 0, keepdim = True)
    getitem_98: "f32[1, 256, 1, 1]" = var_mean_38[0]
    getitem_99: "f32[1, 256, 1, 1]" = var_mean_38[1];  var_mean_38 = None
    add_202: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-05)
    rsqrt_38: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_202);  add_202 = None
    sub_38: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_38, getitem_99)
    mul_304: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
    squeeze_114: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
    squeeze_115: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
    mul_305: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
    mul_306: "f32[256]" = torch.ops.aten.mul.Tensor(primals_319, 0.9)
    add_203: "f32[256]" = torch.ops.aten.add.Tensor(mul_305, mul_306);  mul_305 = mul_306 = None
    squeeze_116: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_98, [0, 2, 3]);  getitem_98 = None
    mul_307: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_116, 1.0004885197850513);  squeeze_116 = None
    mul_308: "f32[256]" = torch.ops.aten.mul.Tensor(mul_307, 0.1);  mul_307 = None
    mul_309: "f32[256]" = torch.ops.aten.mul.Tensor(primals_320, 0.9)
    add_204: "f32[256]" = torch.ops.aten.add.Tensor(mul_308, mul_309);  mul_308 = mul_309 = None
    unsqueeze_152: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1)
    unsqueeze_153: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    mul_310: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_304, unsqueeze_153);  mul_304 = unsqueeze_153 = None
    unsqueeze_154: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
    unsqueeze_155: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    add_205: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_310, unsqueeze_155);  mul_310 = unsqueeze_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_38: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_205, 0)
    mul_311: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_205, 0.01)
    where_38: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_38, add_205, mul_311);  gt_38 = add_205 = mul_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_206: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(where_38, getitem_95)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_39: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(add_206, primals_174, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_207: "i64[]" = torch.ops.aten.add.Tensor(primals_321, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_39 = torch.ops.aten.var_mean.correction(convolution_39, [0, 2, 3], correction = 0, keepdim = True)
    getitem_100: "f32[1, 256, 1, 1]" = var_mean_39[0]
    getitem_101: "f32[1, 256, 1, 1]" = var_mean_39[1];  var_mean_39 = None
    add_208: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05)
    rsqrt_39: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_208);  add_208 = None
    sub_39: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_39, getitem_101)
    mul_312: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
    squeeze_117: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2, 3]);  getitem_101 = None
    squeeze_118: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
    mul_313: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1)
    mul_314: "f32[256]" = torch.ops.aten.mul.Tensor(primals_322, 0.9)
    add_209: "f32[256]" = torch.ops.aten.add.Tensor(mul_313, mul_314);  mul_313 = mul_314 = None
    squeeze_119: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_100, [0, 2, 3]);  getitem_100 = None
    mul_315: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_119, 1.0004885197850513);  squeeze_119 = None
    mul_316: "f32[256]" = torch.ops.aten.mul.Tensor(mul_315, 0.1);  mul_315 = None
    mul_317: "f32[256]" = torch.ops.aten.mul.Tensor(primals_323, 0.9)
    add_210: "f32[256]" = torch.ops.aten.add.Tensor(mul_316, mul_317);  mul_316 = mul_317 = None
    unsqueeze_156: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_79, -1)
    unsqueeze_157: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_318: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_312, unsqueeze_157);  mul_312 = unsqueeze_157 = None
    unsqueeze_158: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1);  primals_80 = None
    unsqueeze_159: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_211: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_318, unsqueeze_159);  mul_318 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_39: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_211, 0)
    mul_319: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_211, 0.01)
    where_39: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_39, add_211, mul_319);  gt_39 = add_211 = mul_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_40: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(where_39, primals_175, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_212: "i64[]" = torch.ops.aten.add.Tensor(primals_324, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_40 = torch.ops.aten.var_mean.correction(convolution_40, [0, 2, 3], correction = 0, keepdim = True)
    getitem_102: "f32[1, 256, 1, 1]" = var_mean_40[0]
    getitem_103: "f32[1, 256, 1, 1]" = var_mean_40[1];  var_mean_40 = None
    add_213: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05)
    rsqrt_40: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
    sub_40: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_40, getitem_103)
    mul_320: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
    squeeze_120: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
    squeeze_121: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
    mul_321: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1)
    mul_322: "f32[256]" = torch.ops.aten.mul.Tensor(primals_325, 0.9)
    add_214: "f32[256]" = torch.ops.aten.add.Tensor(mul_321, mul_322);  mul_321 = mul_322 = None
    squeeze_122: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_102, [0, 2, 3]);  getitem_102 = None
    mul_323: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_122, 1.0004885197850513);  squeeze_122 = None
    mul_324: "f32[256]" = torch.ops.aten.mul.Tensor(mul_323, 0.1);  mul_323 = None
    mul_325: "f32[256]" = torch.ops.aten.mul.Tensor(primals_326, 0.9)
    add_215: "f32[256]" = torch.ops.aten.add.Tensor(mul_324, mul_325);  mul_324 = mul_325 = None
    unsqueeze_160: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1)
    unsqueeze_161: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    mul_326: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_320, unsqueeze_161);  mul_320 = unsqueeze_161 = None
    unsqueeze_162: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_82, -1);  primals_82 = None
    unsqueeze_163: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    add_216: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_326, unsqueeze_163);  mul_326 = unsqueeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_40: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_216, 0)
    mul_327: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_216, 0.01)
    where_40: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_40, add_216, mul_327);  gt_40 = add_216 = mul_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_217: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(where_40, add_206)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_41: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(add_217, primals_176, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_218: "i64[]" = torch.ops.aten.add.Tensor(primals_327, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_41 = torch.ops.aten.var_mean.correction(convolution_41, [0, 2, 3], correction = 0, keepdim = True)
    getitem_104: "f32[1, 256, 1, 1]" = var_mean_41[0]
    getitem_105: "f32[1, 256, 1, 1]" = var_mean_41[1];  var_mean_41 = None
    add_219: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05)
    rsqrt_41: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_219);  add_219 = None
    sub_41: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_41, getitem_105)
    mul_328: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
    squeeze_123: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_105, [0, 2, 3]);  getitem_105 = None
    squeeze_124: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
    mul_329: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
    mul_330: "f32[256]" = torch.ops.aten.mul.Tensor(primals_328, 0.9)
    add_220: "f32[256]" = torch.ops.aten.add.Tensor(mul_329, mul_330);  mul_329 = mul_330 = None
    squeeze_125: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_104, [0, 2, 3]);  getitem_104 = None
    mul_331: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_125, 1.0004885197850513);  squeeze_125 = None
    mul_332: "f32[256]" = torch.ops.aten.mul.Tensor(mul_331, 0.1);  mul_331 = None
    mul_333: "f32[256]" = torch.ops.aten.mul.Tensor(primals_329, 0.9)
    add_221: "f32[256]" = torch.ops.aten.add.Tensor(mul_332, mul_333);  mul_332 = mul_333 = None
    unsqueeze_164: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1)
    unsqueeze_165: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_334: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_328, unsqueeze_165);  mul_328 = unsqueeze_165 = None
    unsqueeze_166: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1);  primals_84 = None
    unsqueeze_167: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_222: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_334, unsqueeze_167);  mul_334 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_41: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_222, 0)
    mul_335: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_222, 0.01)
    where_41: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_41, add_222, mul_335);  gt_41 = add_222 = mul_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_42: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(where_41, primals_177, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_223: "i64[]" = torch.ops.aten.add.Tensor(primals_330, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_42 = torch.ops.aten.var_mean.correction(convolution_42, [0, 2, 3], correction = 0, keepdim = True)
    getitem_106: "f32[1, 256, 1, 1]" = var_mean_42[0]
    getitem_107: "f32[1, 256, 1, 1]" = var_mean_42[1];  var_mean_42 = None
    add_224: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05)
    rsqrt_42: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_224);  add_224 = None
    sub_42: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_42, getitem_107)
    mul_336: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
    squeeze_126: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
    squeeze_127: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
    mul_337: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_126, 0.1)
    mul_338: "f32[256]" = torch.ops.aten.mul.Tensor(primals_331, 0.9)
    add_225: "f32[256]" = torch.ops.aten.add.Tensor(mul_337, mul_338);  mul_337 = mul_338 = None
    squeeze_128: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_106, [0, 2, 3]);  getitem_106 = None
    mul_339: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_128, 1.0004885197850513);  squeeze_128 = None
    mul_340: "f32[256]" = torch.ops.aten.mul.Tensor(mul_339, 0.1);  mul_339 = None
    mul_341: "f32[256]" = torch.ops.aten.mul.Tensor(primals_332, 0.9)
    add_226: "f32[256]" = torch.ops.aten.add.Tensor(mul_340, mul_341);  mul_340 = mul_341 = None
    unsqueeze_168: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_85, -1)
    unsqueeze_169: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    mul_342: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_336, unsqueeze_169);  mul_336 = unsqueeze_169 = None
    unsqueeze_170: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_86, -1);  primals_86 = None
    unsqueeze_171: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    add_227: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_342, unsqueeze_171);  mul_342 = unsqueeze_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_42: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_227, 0)
    mul_343: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_227, 0.01)
    where_42: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_42, add_227, mul_343);  gt_42 = add_227 = mul_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_228: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(where_42, add_217)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_43: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(add_228, primals_178, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_229: "i64[]" = torch.ops.aten.add.Tensor(primals_333, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_43 = torch.ops.aten.var_mean.correction(convolution_43, [0, 2, 3], correction = 0, keepdim = True)
    getitem_108: "f32[1, 256, 1, 1]" = var_mean_43[0]
    getitem_109: "f32[1, 256, 1, 1]" = var_mean_43[1];  var_mean_43 = None
    add_230: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05)
    rsqrt_43: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_230);  add_230 = None
    sub_43: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_43, getitem_109)
    mul_344: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
    squeeze_129: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_109, [0, 2, 3]);  getitem_109 = None
    squeeze_130: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
    mul_345: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_129, 0.1)
    mul_346: "f32[256]" = torch.ops.aten.mul.Tensor(primals_334, 0.9)
    add_231: "f32[256]" = torch.ops.aten.add.Tensor(mul_345, mul_346);  mul_345 = mul_346 = None
    squeeze_131: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_108, [0, 2, 3]);  getitem_108 = None
    mul_347: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_131, 1.0004885197850513);  squeeze_131 = None
    mul_348: "f32[256]" = torch.ops.aten.mul.Tensor(mul_347, 0.1);  mul_347 = None
    mul_349: "f32[256]" = torch.ops.aten.mul.Tensor(primals_335, 0.9)
    add_232: "f32[256]" = torch.ops.aten.add.Tensor(mul_348, mul_349);  mul_348 = mul_349 = None
    unsqueeze_172: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_87, -1)
    unsqueeze_173: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_350: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_344, unsqueeze_173);  mul_344 = unsqueeze_173 = None
    unsqueeze_174: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_88, -1);  primals_88 = None
    unsqueeze_175: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_233: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_350, unsqueeze_175);  mul_350 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_43: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_233, 0)
    mul_351: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_233, 0.01)
    where_43: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_43, add_233, mul_351);  gt_43 = add_233 = mul_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_44: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(where_43, primals_179, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_234: "i64[]" = torch.ops.aten.add.Tensor(primals_336, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_44 = torch.ops.aten.var_mean.correction(convolution_44, [0, 2, 3], correction = 0, keepdim = True)
    getitem_110: "f32[1, 256, 1, 1]" = var_mean_44[0]
    getitem_111: "f32[1, 256, 1, 1]" = var_mean_44[1];  var_mean_44 = None
    add_235: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05)
    rsqrt_44: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_235);  add_235 = None
    sub_44: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_44, getitem_111)
    mul_352: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
    squeeze_132: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_111, [0, 2, 3]);  getitem_111 = None
    squeeze_133: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
    mul_353: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_132, 0.1)
    mul_354: "f32[256]" = torch.ops.aten.mul.Tensor(primals_337, 0.9)
    add_236: "f32[256]" = torch.ops.aten.add.Tensor(mul_353, mul_354);  mul_353 = mul_354 = None
    squeeze_134: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_110, [0, 2, 3]);  getitem_110 = None
    mul_355: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_134, 1.0004885197850513);  squeeze_134 = None
    mul_356: "f32[256]" = torch.ops.aten.mul.Tensor(mul_355, 0.1);  mul_355 = None
    mul_357: "f32[256]" = torch.ops.aten.mul.Tensor(primals_338, 0.9)
    add_237: "f32[256]" = torch.ops.aten.add.Tensor(mul_356, mul_357);  mul_356 = mul_357 = None
    unsqueeze_176: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_89, -1)
    unsqueeze_177: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    mul_358: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_352, unsqueeze_177);  mul_352 = unsqueeze_177 = None
    unsqueeze_178: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1);  primals_90 = None
    unsqueeze_179: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    add_238: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_358, unsqueeze_179);  mul_358 = unsqueeze_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_44: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_238, 0)
    mul_359: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_238, 0.01)
    where_44: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_44, add_238, mul_359);  gt_44 = add_238 = mul_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_239: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(where_44, add_228)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_45: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(add_239, primals_180, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_240: "i64[]" = torch.ops.aten.add.Tensor(primals_339, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_45 = torch.ops.aten.var_mean.correction(convolution_45, [0, 2, 3], correction = 0, keepdim = True)
    getitem_112: "f32[1, 256, 1, 1]" = var_mean_45[0]
    getitem_113: "f32[1, 256, 1, 1]" = var_mean_45[1];  var_mean_45 = None
    add_241: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05)
    rsqrt_45: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_241);  add_241 = None
    sub_45: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_45, getitem_113)
    mul_360: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
    squeeze_135: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2, 3]);  getitem_113 = None
    squeeze_136: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
    mul_361: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_135, 0.1)
    mul_362: "f32[256]" = torch.ops.aten.mul.Tensor(primals_340, 0.9)
    add_242: "f32[256]" = torch.ops.aten.add.Tensor(mul_361, mul_362);  mul_361 = mul_362 = None
    squeeze_137: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_112, [0, 2, 3]);  getitem_112 = None
    mul_363: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_137, 1.0004885197850513);  squeeze_137 = None
    mul_364: "f32[256]" = torch.ops.aten.mul.Tensor(mul_363, 0.1);  mul_363 = None
    mul_365: "f32[256]" = torch.ops.aten.mul.Tensor(primals_341, 0.9)
    add_243: "f32[256]" = torch.ops.aten.add.Tensor(mul_364, mul_365);  mul_364 = mul_365 = None
    unsqueeze_180: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_91, -1)
    unsqueeze_181: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_366: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_360, unsqueeze_181);  mul_360 = unsqueeze_181 = None
    unsqueeze_182: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1);  primals_92 = None
    unsqueeze_183: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_244: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_366, unsqueeze_183);  mul_366 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_45: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_244, 0)
    mul_367: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_244, 0.01)
    where_45: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_45, add_244, mul_367);  gt_45 = add_244 = mul_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_46: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(where_45, primals_181, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_245: "i64[]" = torch.ops.aten.add.Tensor(primals_342, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_46 = torch.ops.aten.var_mean.correction(convolution_46, [0, 2, 3], correction = 0, keepdim = True)
    getitem_114: "f32[1, 256, 1, 1]" = var_mean_46[0]
    getitem_115: "f32[1, 256, 1, 1]" = var_mean_46[1];  var_mean_46 = None
    add_246: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-05)
    rsqrt_46: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_246);  add_246 = None
    sub_46: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_46, getitem_115)
    mul_368: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
    squeeze_138: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_115, [0, 2, 3]);  getitem_115 = None
    squeeze_139: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
    mul_369: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_138, 0.1)
    mul_370: "f32[256]" = torch.ops.aten.mul.Tensor(primals_343, 0.9)
    add_247: "f32[256]" = torch.ops.aten.add.Tensor(mul_369, mul_370);  mul_369 = mul_370 = None
    squeeze_140: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_114, [0, 2, 3]);  getitem_114 = None
    mul_371: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_140, 1.0004885197850513);  squeeze_140 = None
    mul_372: "f32[256]" = torch.ops.aten.mul.Tensor(mul_371, 0.1);  mul_371 = None
    mul_373: "f32[256]" = torch.ops.aten.mul.Tensor(primals_344, 0.9)
    add_248: "f32[256]" = torch.ops.aten.add.Tensor(mul_372, mul_373);  mul_372 = mul_373 = None
    unsqueeze_184: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_93, -1)
    unsqueeze_185: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    mul_374: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_368, unsqueeze_185);  mul_368 = unsqueeze_185 = None
    unsqueeze_186: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_94, -1);  primals_94 = None
    unsqueeze_187: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    add_249: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_374, unsqueeze_187);  mul_374 = unsqueeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_46: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_249, 0)
    mul_375: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_249, 0.01)
    where_46: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_46, add_249, mul_375);  gt_46 = add_249 = mul_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_250: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(where_46, add_239)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_47: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(add_250, primals_182, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_251: "i64[]" = torch.ops.aten.add.Tensor(primals_345, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_47 = torch.ops.aten.var_mean.correction(convolution_47, [0, 2, 3], correction = 0, keepdim = True)
    getitem_116: "f32[1, 256, 1, 1]" = var_mean_47[0]
    getitem_117: "f32[1, 256, 1, 1]" = var_mean_47[1];  var_mean_47 = None
    add_252: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-05)
    rsqrt_47: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_252);  add_252 = None
    sub_47: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_47, getitem_117)
    mul_376: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
    squeeze_141: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_117, [0, 2, 3]);  getitem_117 = None
    squeeze_142: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
    mul_377: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_141, 0.1)
    mul_378: "f32[256]" = torch.ops.aten.mul.Tensor(primals_346, 0.9)
    add_253: "f32[256]" = torch.ops.aten.add.Tensor(mul_377, mul_378);  mul_377 = mul_378 = None
    squeeze_143: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_116, [0, 2, 3]);  getitem_116 = None
    mul_379: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_143, 1.0004885197850513);  squeeze_143 = None
    mul_380: "f32[256]" = torch.ops.aten.mul.Tensor(mul_379, 0.1);  mul_379 = None
    mul_381: "f32[256]" = torch.ops.aten.mul.Tensor(primals_347, 0.9)
    add_254: "f32[256]" = torch.ops.aten.add.Tensor(mul_380, mul_381);  mul_380 = mul_381 = None
    unsqueeze_188: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1)
    unsqueeze_189: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_382: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_376, unsqueeze_189);  mul_376 = unsqueeze_189 = None
    unsqueeze_190: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1);  primals_96 = None
    unsqueeze_191: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_255: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_382, unsqueeze_191);  mul_382 = unsqueeze_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_47: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_255, 0)
    mul_383: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_255, 0.01)
    where_47: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_47, add_255, mul_383);  gt_47 = add_255 = mul_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_48: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(where_47, primals_183, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_256: "i64[]" = torch.ops.aten.add.Tensor(primals_348, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_48 = torch.ops.aten.var_mean.correction(convolution_48, [0, 2, 3], correction = 0, keepdim = True)
    getitem_118: "f32[1, 256, 1, 1]" = var_mean_48[0]
    getitem_119: "f32[1, 256, 1, 1]" = var_mean_48[1];  var_mean_48 = None
    add_257: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05)
    rsqrt_48: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_257);  add_257 = None
    sub_48: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_48, getitem_119)
    mul_384: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
    squeeze_144: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_119, [0, 2, 3]);  getitem_119 = None
    squeeze_145: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
    mul_385: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_144, 0.1)
    mul_386: "f32[256]" = torch.ops.aten.mul.Tensor(primals_349, 0.9)
    add_258: "f32[256]" = torch.ops.aten.add.Tensor(mul_385, mul_386);  mul_385 = mul_386 = None
    squeeze_146: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_118, [0, 2, 3]);  getitem_118 = None
    mul_387: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_146, 1.0004885197850513);  squeeze_146 = None
    mul_388: "f32[256]" = torch.ops.aten.mul.Tensor(mul_387, 0.1);  mul_387 = None
    mul_389: "f32[256]" = torch.ops.aten.mul.Tensor(primals_350, 0.9)
    add_259: "f32[256]" = torch.ops.aten.add.Tensor(mul_388, mul_389);  mul_388 = mul_389 = None
    unsqueeze_192: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_97, -1)
    unsqueeze_193: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    mul_390: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_384, unsqueeze_193);  mul_384 = unsqueeze_193 = None
    unsqueeze_194: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1);  primals_98 = None
    unsqueeze_195: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    add_260: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_390, unsqueeze_195);  mul_390 = unsqueeze_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_48: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_260, 0)
    mul_391: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_260, 0.01)
    where_48: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_48, add_260, mul_391);  gt_48 = add_260 = mul_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_261: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(where_48, add_250)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_49: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(add_261, primals_184, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_262: "i64[]" = torch.ops.aten.add.Tensor(primals_351, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_49 = torch.ops.aten.var_mean.correction(convolution_49, [0, 2, 3], correction = 0, keepdim = True)
    getitem_120: "f32[1, 256, 1, 1]" = var_mean_49[0]
    getitem_121: "f32[1, 256, 1, 1]" = var_mean_49[1];  var_mean_49 = None
    add_263: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-05)
    rsqrt_49: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_263);  add_263 = None
    sub_49: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_49, getitem_121)
    mul_392: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
    squeeze_147: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_121, [0, 2, 3]);  getitem_121 = None
    squeeze_148: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
    mul_393: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_147, 0.1)
    mul_394: "f32[256]" = torch.ops.aten.mul.Tensor(primals_352, 0.9)
    add_264: "f32[256]" = torch.ops.aten.add.Tensor(mul_393, mul_394);  mul_393 = mul_394 = None
    squeeze_149: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_120, [0, 2, 3]);  getitem_120 = None
    mul_395: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_149, 1.0004885197850513);  squeeze_149 = None
    mul_396: "f32[256]" = torch.ops.aten.mul.Tensor(mul_395, 0.1);  mul_395 = None
    mul_397: "f32[256]" = torch.ops.aten.mul.Tensor(primals_353, 0.9)
    add_265: "f32[256]" = torch.ops.aten.add.Tensor(mul_396, mul_397);  mul_396 = mul_397 = None
    unsqueeze_196: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_99, -1)
    unsqueeze_197: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_398: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_392, unsqueeze_197);  mul_392 = unsqueeze_197 = None
    unsqueeze_198: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_100, -1);  primals_100 = None
    unsqueeze_199: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_266: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_398, unsqueeze_199);  mul_398 = unsqueeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_49: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_266, 0)
    mul_399: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_266, 0.01)
    where_49: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_49, add_266, mul_399);  gt_49 = add_266 = mul_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_50: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(where_49, primals_185, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_267: "i64[]" = torch.ops.aten.add.Tensor(primals_354, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_50 = torch.ops.aten.var_mean.correction(convolution_50, [0, 2, 3], correction = 0, keepdim = True)
    getitem_122: "f32[1, 256, 1, 1]" = var_mean_50[0]
    getitem_123: "f32[1, 256, 1, 1]" = var_mean_50[1];  var_mean_50 = None
    add_268: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-05)
    rsqrt_50: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_268);  add_268 = None
    sub_50: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_50, getitem_123)
    mul_400: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
    squeeze_150: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_123, [0, 2, 3]);  getitem_123 = None
    squeeze_151: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
    mul_401: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_150, 0.1)
    mul_402: "f32[256]" = torch.ops.aten.mul.Tensor(primals_355, 0.9)
    add_269: "f32[256]" = torch.ops.aten.add.Tensor(mul_401, mul_402);  mul_401 = mul_402 = None
    squeeze_152: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_122, [0, 2, 3]);  getitem_122 = None
    mul_403: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_152, 1.0004885197850513);  squeeze_152 = None
    mul_404: "f32[256]" = torch.ops.aten.mul.Tensor(mul_403, 0.1);  mul_403 = None
    mul_405: "f32[256]" = torch.ops.aten.mul.Tensor(primals_356, 0.9)
    add_270: "f32[256]" = torch.ops.aten.add.Tensor(mul_404, mul_405);  mul_404 = mul_405 = None
    unsqueeze_200: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_101, -1)
    unsqueeze_201: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    mul_406: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_400, unsqueeze_201);  mul_400 = unsqueeze_201 = None
    unsqueeze_202: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_102, -1);  primals_102 = None
    unsqueeze_203: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    add_271: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_406, unsqueeze_203);  mul_406 = unsqueeze_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_50: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_271, 0)
    mul_407: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_271, 0.01)
    where_50: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_50, add_271, mul_407);  gt_50 = add_271 = mul_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_272: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(where_50, add_261)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_51: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(add_272, primals_186, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_273: "i64[]" = torch.ops.aten.add.Tensor(primals_357, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_51 = torch.ops.aten.var_mean.correction(convolution_51, [0, 2, 3], correction = 0, keepdim = True)
    getitem_124: "f32[1, 256, 1, 1]" = var_mean_51[0]
    getitem_125: "f32[1, 256, 1, 1]" = var_mean_51[1];  var_mean_51 = None
    add_274: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-05)
    rsqrt_51: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_274);  add_274 = None
    sub_51: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_51, getitem_125)
    mul_408: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
    squeeze_153: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_125, [0, 2, 3]);  getitem_125 = None
    squeeze_154: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
    mul_409: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_153, 0.1)
    mul_410: "f32[256]" = torch.ops.aten.mul.Tensor(primals_358, 0.9)
    add_275: "f32[256]" = torch.ops.aten.add.Tensor(mul_409, mul_410);  mul_409 = mul_410 = None
    squeeze_155: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_124, [0, 2, 3]);  getitem_124 = None
    mul_411: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_155, 1.0004885197850513);  squeeze_155 = None
    mul_412: "f32[256]" = torch.ops.aten.mul.Tensor(mul_411, 0.1);  mul_411 = None
    mul_413: "f32[256]" = torch.ops.aten.mul.Tensor(primals_359, 0.9)
    add_276: "f32[256]" = torch.ops.aten.add.Tensor(mul_412, mul_413);  mul_412 = mul_413 = None
    unsqueeze_204: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_103, -1)
    unsqueeze_205: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_414: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_408, unsqueeze_205);  mul_408 = unsqueeze_205 = None
    unsqueeze_206: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_104, -1);  primals_104 = None
    unsqueeze_207: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_277: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_414, unsqueeze_207);  mul_414 = unsqueeze_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_51: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_277, 0)
    mul_415: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_277, 0.01)
    where_51: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_51, add_277, mul_415);  gt_51 = add_277 = mul_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_52: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(where_51, primals_187, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_278: "i64[]" = torch.ops.aten.add.Tensor(primals_360, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_52 = torch.ops.aten.var_mean.correction(convolution_52, [0, 2, 3], correction = 0, keepdim = True)
    getitem_126: "f32[1, 256, 1, 1]" = var_mean_52[0]
    getitem_127: "f32[1, 256, 1, 1]" = var_mean_52[1];  var_mean_52 = None
    add_279: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-05)
    rsqrt_52: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_279);  add_279 = None
    sub_52: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_52, getitem_127)
    mul_416: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
    squeeze_156: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_127, [0, 2, 3]);  getitem_127 = None
    squeeze_157: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
    mul_417: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_156, 0.1)
    mul_418: "f32[256]" = torch.ops.aten.mul.Tensor(primals_361, 0.9)
    add_280: "f32[256]" = torch.ops.aten.add.Tensor(mul_417, mul_418);  mul_417 = mul_418 = None
    squeeze_158: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_126, [0, 2, 3]);  getitem_126 = None
    mul_419: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_158, 1.0004885197850513);  squeeze_158 = None
    mul_420: "f32[256]" = torch.ops.aten.mul.Tensor(mul_419, 0.1);  mul_419 = None
    mul_421: "f32[256]" = torch.ops.aten.mul.Tensor(primals_362, 0.9)
    add_281: "f32[256]" = torch.ops.aten.add.Tensor(mul_420, mul_421);  mul_420 = mul_421 = None
    unsqueeze_208: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_105, -1)
    unsqueeze_209: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    mul_422: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_416, unsqueeze_209);  mul_416 = unsqueeze_209 = None
    unsqueeze_210: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_106, -1);  primals_106 = None
    unsqueeze_211: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    add_282: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_422, unsqueeze_211);  mul_422 = unsqueeze_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_52: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_282, 0)
    mul_423: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_282, 0.01)
    where_52: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_52, add_282, mul_423);  gt_52 = add_282 = mul_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_283: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(where_52, add_272)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_53: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(add_283, primals_188, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_284: "i64[]" = torch.ops.aten.add.Tensor(primals_363, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_53 = torch.ops.aten.var_mean.correction(convolution_53, [0, 2, 3], correction = 0, keepdim = True)
    getitem_128: "f32[1, 256, 1, 1]" = var_mean_53[0]
    getitem_129: "f32[1, 256, 1, 1]" = var_mean_53[1];  var_mean_53 = None
    add_285: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-05)
    rsqrt_53: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_285);  add_285 = None
    sub_53: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_53, getitem_129)
    mul_424: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = None
    squeeze_159: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_129, [0, 2, 3]);  getitem_129 = None
    squeeze_160: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_53, [0, 2, 3]);  rsqrt_53 = None
    mul_425: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_159, 0.1)
    mul_426: "f32[256]" = torch.ops.aten.mul.Tensor(primals_364, 0.9)
    add_286: "f32[256]" = torch.ops.aten.add.Tensor(mul_425, mul_426);  mul_425 = mul_426 = None
    squeeze_161: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_128, [0, 2, 3]);  getitem_128 = None
    mul_427: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_161, 1.0004885197850513);  squeeze_161 = None
    mul_428: "f32[256]" = torch.ops.aten.mul.Tensor(mul_427, 0.1);  mul_427 = None
    mul_429: "f32[256]" = torch.ops.aten.mul.Tensor(primals_365, 0.9)
    add_287: "f32[256]" = torch.ops.aten.add.Tensor(mul_428, mul_429);  mul_428 = mul_429 = None
    unsqueeze_212: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_107, -1)
    unsqueeze_213: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_430: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_424, unsqueeze_213);  mul_424 = unsqueeze_213 = None
    unsqueeze_214: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_108, -1);  primals_108 = None
    unsqueeze_215: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_288: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_430, unsqueeze_215);  mul_430 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_53: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_288, 0)
    mul_431: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_288, 0.01)
    where_53: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_53, add_288, mul_431);  gt_53 = add_288 = mul_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:339, code: out = self.conv_transition(torch.cat([xs, xb], dim=1))
    getitem_130: "f32[8, 256, 16, 16]" = split_with_sizes_10[0];  split_with_sizes_10 = None
    cat_3: "f32[8, 512, 16, 16]" = torch.ops.aten.cat.default([getitem_130, where_53], 1);  getitem_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_54: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(cat_3, primals_189, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_289: "i64[]" = torch.ops.aten.add.Tensor(primals_366, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_54 = torch.ops.aten.var_mean.correction(convolution_54, [0, 2, 3], correction = 0, keepdim = True)
    getitem_132: "f32[1, 512, 1, 1]" = var_mean_54[0]
    getitem_133: "f32[1, 512, 1, 1]" = var_mean_54[1];  var_mean_54 = None
    add_290: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-05)
    rsqrt_54: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_290);  add_290 = None
    sub_54: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_54, getitem_133)
    mul_432: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = None
    squeeze_162: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_133, [0, 2, 3]);  getitem_133 = None
    squeeze_163: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2, 3]);  rsqrt_54 = None
    mul_433: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_162, 0.1)
    mul_434: "f32[512]" = torch.ops.aten.mul.Tensor(primals_367, 0.9)
    add_291: "f32[512]" = torch.ops.aten.add.Tensor(mul_433, mul_434);  mul_433 = mul_434 = None
    squeeze_164: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_132, [0, 2, 3]);  getitem_132 = None
    mul_435: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_164, 1.0004885197850513);  squeeze_164 = None
    mul_436: "f32[512]" = torch.ops.aten.mul.Tensor(mul_435, 0.1);  mul_435 = None
    mul_437: "f32[512]" = torch.ops.aten.mul.Tensor(primals_368, 0.9)
    add_292: "f32[512]" = torch.ops.aten.add.Tensor(mul_436, mul_437);  mul_436 = mul_437 = None
    unsqueeze_216: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_109, -1)
    unsqueeze_217: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    mul_438: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_432, unsqueeze_217);  mul_432 = unsqueeze_217 = None
    unsqueeze_218: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_110, -1);  primals_110 = None
    unsqueeze_219: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    add_293: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_438, unsqueeze_219);  mul_438 = unsqueeze_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_54: "b8[8, 512, 16, 16]" = torch.ops.aten.gt.Scalar(add_293, 0)
    mul_439: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(add_293, 0.01)
    where_54: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(gt_54, add_293, mul_439);  gt_54 = add_293 = mul_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:126, code: x = self.conv(x)
    convolution_55: "f32[8, 1024, 8, 8]" = torch.ops.aten.convolution.default(where_54, primals_190, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_294: "i64[]" = torch.ops.aten.add.Tensor(primals_369, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_55 = torch.ops.aten.var_mean.correction(convolution_55, [0, 2, 3], correction = 0, keepdim = True)
    getitem_134: "f32[1, 1024, 1, 1]" = var_mean_55[0]
    getitem_135: "f32[1, 1024, 1, 1]" = var_mean_55[1];  var_mean_55 = None
    add_295: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_134, 1e-05)
    rsqrt_55: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_295);  add_295 = None
    sub_55: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_55, getitem_135)
    mul_440: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = None
    squeeze_165: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_135, [0, 2, 3]);  getitem_135 = None
    squeeze_166: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2, 3]);  rsqrt_55 = None
    mul_441: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_165, 0.1)
    mul_442: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_370, 0.9)
    add_296: "f32[1024]" = torch.ops.aten.add.Tensor(mul_441, mul_442);  mul_441 = mul_442 = None
    squeeze_167: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_134, [0, 2, 3]);  getitem_134 = None
    mul_443: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_167, 1.0019569471624266);  squeeze_167 = None
    mul_444: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_443, 0.1);  mul_443 = None
    mul_445: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_371, 0.9)
    add_297: "f32[1024]" = torch.ops.aten.add.Tensor(mul_444, mul_445);  mul_444 = mul_445 = None
    unsqueeze_220: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_111, -1)
    unsqueeze_221: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_446: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(mul_440, unsqueeze_221);  mul_440 = unsqueeze_221 = None
    unsqueeze_222: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_112, -1);  primals_112 = None
    unsqueeze_223: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_298: "f32[8, 1024, 8, 8]" = torch.ops.aten.add.Tensor(mul_446, unsqueeze_223);  mul_446 = unsqueeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_55: "b8[8, 1024, 8, 8]" = torch.ops.aten.gt.Scalar(add_298, 0)
    mul_447: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(add_298, 0.01)
    where_55: "f32[8, 1024, 8, 8]" = torch.ops.aten.where.self(gt_55, add_298, mul_447);  gt_55 = add_298 = mul_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_56: "f32[8, 1024, 8, 8]" = torch.ops.aten.convolution.default(where_55, primals_191, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_299: "i64[]" = torch.ops.aten.add.Tensor(primals_372, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_56 = torch.ops.aten.var_mean.correction(convolution_56, [0, 2, 3], correction = 0, keepdim = True)
    getitem_136: "f32[1, 1024, 1, 1]" = var_mean_56[0]
    getitem_137: "f32[1, 1024, 1, 1]" = var_mean_56[1];  var_mean_56 = None
    add_300: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_136, 1e-05)
    rsqrt_56: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_300);  add_300 = None
    sub_56: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_56, getitem_137)
    mul_448: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = None
    squeeze_168: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_137, [0, 2, 3]);  getitem_137 = None
    squeeze_169: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_56, [0, 2, 3]);  rsqrt_56 = None
    mul_449: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_168, 0.1)
    mul_450: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_373, 0.9)
    add_301: "f32[1024]" = torch.ops.aten.add.Tensor(mul_449, mul_450);  mul_449 = mul_450 = None
    squeeze_170: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_136, [0, 2, 3]);  getitem_136 = None
    mul_451: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_170, 1.0019569471624266);  squeeze_170 = None
    mul_452: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_451, 0.1);  mul_451 = None
    mul_453: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_374, 0.9)
    add_302: "f32[1024]" = torch.ops.aten.add.Tensor(mul_452, mul_453);  mul_452 = mul_453 = None
    unsqueeze_224: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_113, -1)
    unsqueeze_225: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    mul_454: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(mul_448, unsqueeze_225);  mul_448 = unsqueeze_225 = None
    unsqueeze_226: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_114, -1);  primals_114 = None
    unsqueeze_227: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    add_303: "f32[8, 1024, 8, 8]" = torch.ops.aten.add.Tensor(mul_454, unsqueeze_227);  mul_454 = unsqueeze_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_56: "b8[8, 1024, 8, 8]" = torch.ops.aten.gt.Scalar(add_303, 0)
    mul_455: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(add_303, 0.01)
    where_56: "f32[8, 1024, 8, 8]" = torch.ops.aten.where.self(gt_56, add_303, mul_455);  gt_56 = add_303 = mul_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    split_with_sizes_13 = torch.ops.aten.split_with_sizes.default(where_56, [512, 512], 1)
    getitem_141: "f32[8, 512, 8, 8]" = split_with_sizes_13[1]
    convolution_57: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(getitem_141, primals_192, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_304: "i64[]" = torch.ops.aten.add.Tensor(primals_375, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_57 = torch.ops.aten.var_mean.correction(convolution_57, [0, 2, 3], correction = 0, keepdim = True)
    getitem_142: "f32[1, 512, 1, 1]" = var_mean_57[0]
    getitem_143: "f32[1, 512, 1, 1]" = var_mean_57[1];  var_mean_57 = None
    add_305: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_142, 1e-05)
    rsqrt_57: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_305);  add_305 = None
    sub_57: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_57, getitem_143)
    mul_456: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = None
    squeeze_171: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_143, [0, 2, 3]);  getitem_143 = None
    squeeze_172: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_57, [0, 2, 3]);  rsqrt_57 = None
    mul_457: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_171, 0.1)
    mul_458: "f32[512]" = torch.ops.aten.mul.Tensor(primals_376, 0.9)
    add_306: "f32[512]" = torch.ops.aten.add.Tensor(mul_457, mul_458);  mul_457 = mul_458 = None
    squeeze_173: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_142, [0, 2, 3]);  getitem_142 = None
    mul_459: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_173, 1.0019569471624266);  squeeze_173 = None
    mul_460: "f32[512]" = torch.ops.aten.mul.Tensor(mul_459, 0.1);  mul_459 = None
    mul_461: "f32[512]" = torch.ops.aten.mul.Tensor(primals_377, 0.9)
    add_307: "f32[512]" = torch.ops.aten.add.Tensor(mul_460, mul_461);  mul_460 = mul_461 = None
    unsqueeze_228: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_115, -1)
    unsqueeze_229: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_462: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_456, unsqueeze_229);  mul_456 = unsqueeze_229 = None
    unsqueeze_230: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_116, -1);  primals_116 = None
    unsqueeze_231: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_308: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_462, unsqueeze_231);  mul_462 = unsqueeze_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_57: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(add_308, 0)
    mul_463: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_308, 0.01)
    where_57: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_57, add_308, mul_463);  gt_57 = add_308 = mul_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_58: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(where_57, primals_193, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_309: "i64[]" = torch.ops.aten.add.Tensor(primals_378, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_58 = torch.ops.aten.var_mean.correction(convolution_58, [0, 2, 3], correction = 0, keepdim = True)
    getitem_144: "f32[1, 512, 1, 1]" = var_mean_58[0]
    getitem_145: "f32[1, 512, 1, 1]" = var_mean_58[1];  var_mean_58 = None
    add_310: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-05)
    rsqrt_58: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_310);  add_310 = None
    sub_58: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_58, getitem_145)
    mul_464: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_58);  sub_58 = None
    squeeze_174: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_145, [0, 2, 3]);  getitem_145 = None
    squeeze_175: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_58, [0, 2, 3]);  rsqrt_58 = None
    mul_465: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_174, 0.1)
    mul_466: "f32[512]" = torch.ops.aten.mul.Tensor(primals_379, 0.9)
    add_311: "f32[512]" = torch.ops.aten.add.Tensor(mul_465, mul_466);  mul_465 = mul_466 = None
    squeeze_176: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_144, [0, 2, 3]);  getitem_144 = None
    mul_467: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_176, 1.0019569471624266);  squeeze_176 = None
    mul_468: "f32[512]" = torch.ops.aten.mul.Tensor(mul_467, 0.1);  mul_467 = None
    mul_469: "f32[512]" = torch.ops.aten.mul.Tensor(primals_380, 0.9)
    add_312: "f32[512]" = torch.ops.aten.add.Tensor(mul_468, mul_469);  mul_468 = mul_469 = None
    unsqueeze_232: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_117, -1)
    unsqueeze_233: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    mul_470: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_464, unsqueeze_233);  mul_464 = unsqueeze_233 = None
    unsqueeze_234: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_118, -1);  primals_118 = None
    unsqueeze_235: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    add_313: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_470, unsqueeze_235);  mul_470 = unsqueeze_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_58: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(add_313, 0)
    mul_471: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_313, 0.01)
    where_58: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_58, add_313, mul_471);  gt_58 = add_313 = mul_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_314: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(where_58, getitem_141)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_59: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(add_314, primals_194, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_315: "i64[]" = torch.ops.aten.add.Tensor(primals_381, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_59 = torch.ops.aten.var_mean.correction(convolution_59, [0, 2, 3], correction = 0, keepdim = True)
    getitem_146: "f32[1, 512, 1, 1]" = var_mean_59[0]
    getitem_147: "f32[1, 512, 1, 1]" = var_mean_59[1];  var_mean_59 = None
    add_316: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_146, 1e-05)
    rsqrt_59: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_316);  add_316 = None
    sub_59: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_59, getitem_147)
    mul_472: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_59);  sub_59 = None
    squeeze_177: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_147, [0, 2, 3]);  getitem_147 = None
    squeeze_178: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_59, [0, 2, 3]);  rsqrt_59 = None
    mul_473: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_177, 0.1)
    mul_474: "f32[512]" = torch.ops.aten.mul.Tensor(primals_382, 0.9)
    add_317: "f32[512]" = torch.ops.aten.add.Tensor(mul_473, mul_474);  mul_473 = mul_474 = None
    squeeze_179: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_146, [0, 2, 3]);  getitem_146 = None
    mul_475: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_179, 1.0019569471624266);  squeeze_179 = None
    mul_476: "f32[512]" = torch.ops.aten.mul.Tensor(mul_475, 0.1);  mul_475 = None
    mul_477: "f32[512]" = torch.ops.aten.mul.Tensor(primals_383, 0.9)
    add_318: "f32[512]" = torch.ops.aten.add.Tensor(mul_476, mul_477);  mul_476 = mul_477 = None
    unsqueeze_236: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_119, -1)
    unsqueeze_237: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_478: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_472, unsqueeze_237);  mul_472 = unsqueeze_237 = None
    unsqueeze_238: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_120, -1);  primals_120 = None
    unsqueeze_239: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_319: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_478, unsqueeze_239);  mul_478 = unsqueeze_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_59: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(add_319, 0)
    mul_479: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_319, 0.01)
    where_59: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_59, add_319, mul_479);  gt_59 = add_319 = mul_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_60: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(where_59, primals_195, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_320: "i64[]" = torch.ops.aten.add.Tensor(primals_384, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_60 = torch.ops.aten.var_mean.correction(convolution_60, [0, 2, 3], correction = 0, keepdim = True)
    getitem_148: "f32[1, 512, 1, 1]" = var_mean_60[0]
    getitem_149: "f32[1, 512, 1, 1]" = var_mean_60[1];  var_mean_60 = None
    add_321: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_148, 1e-05)
    rsqrt_60: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_321);  add_321 = None
    sub_60: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_60, getitem_149)
    mul_480: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_60);  sub_60 = None
    squeeze_180: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_149, [0, 2, 3]);  getitem_149 = None
    squeeze_181: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_60, [0, 2, 3]);  rsqrt_60 = None
    mul_481: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_180, 0.1)
    mul_482: "f32[512]" = torch.ops.aten.mul.Tensor(primals_385, 0.9)
    add_322: "f32[512]" = torch.ops.aten.add.Tensor(mul_481, mul_482);  mul_481 = mul_482 = None
    squeeze_182: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_148, [0, 2, 3]);  getitem_148 = None
    mul_483: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_182, 1.0019569471624266);  squeeze_182 = None
    mul_484: "f32[512]" = torch.ops.aten.mul.Tensor(mul_483, 0.1);  mul_483 = None
    mul_485: "f32[512]" = torch.ops.aten.mul.Tensor(primals_386, 0.9)
    add_323: "f32[512]" = torch.ops.aten.add.Tensor(mul_484, mul_485);  mul_484 = mul_485 = None
    unsqueeze_240: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_121, -1)
    unsqueeze_241: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    mul_486: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_480, unsqueeze_241);  mul_480 = unsqueeze_241 = None
    unsqueeze_242: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_122, -1);  primals_122 = None
    unsqueeze_243: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    add_324: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_486, unsqueeze_243);  mul_486 = unsqueeze_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_60: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(add_324, 0)
    mul_487: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_324, 0.01)
    where_60: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_60, add_324, mul_487);  gt_60 = add_324 = mul_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_325: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(where_60, add_314)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_61: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(add_325, primals_196, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_326: "i64[]" = torch.ops.aten.add.Tensor(primals_387, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_61 = torch.ops.aten.var_mean.correction(convolution_61, [0, 2, 3], correction = 0, keepdim = True)
    getitem_150: "f32[1, 512, 1, 1]" = var_mean_61[0]
    getitem_151: "f32[1, 512, 1, 1]" = var_mean_61[1];  var_mean_61 = None
    add_327: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_150, 1e-05)
    rsqrt_61: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_327);  add_327 = None
    sub_61: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_61, getitem_151)
    mul_488: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_61);  sub_61 = None
    squeeze_183: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_151, [0, 2, 3]);  getitem_151 = None
    squeeze_184: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_61, [0, 2, 3]);  rsqrt_61 = None
    mul_489: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_183, 0.1)
    mul_490: "f32[512]" = torch.ops.aten.mul.Tensor(primals_388, 0.9)
    add_328: "f32[512]" = torch.ops.aten.add.Tensor(mul_489, mul_490);  mul_489 = mul_490 = None
    squeeze_185: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_150, [0, 2, 3]);  getitem_150 = None
    mul_491: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_185, 1.0019569471624266);  squeeze_185 = None
    mul_492: "f32[512]" = torch.ops.aten.mul.Tensor(mul_491, 0.1);  mul_491 = None
    mul_493: "f32[512]" = torch.ops.aten.mul.Tensor(primals_389, 0.9)
    add_329: "f32[512]" = torch.ops.aten.add.Tensor(mul_492, mul_493);  mul_492 = mul_493 = None
    unsqueeze_244: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_123, -1)
    unsqueeze_245: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_494: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_488, unsqueeze_245);  mul_488 = unsqueeze_245 = None
    unsqueeze_246: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_124, -1);  primals_124 = None
    unsqueeze_247: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_330: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_494, unsqueeze_247);  mul_494 = unsqueeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_61: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(add_330, 0)
    mul_495: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_330, 0.01)
    where_61: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_61, add_330, mul_495);  gt_61 = add_330 = mul_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_62: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(where_61, primals_197, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_331: "i64[]" = torch.ops.aten.add.Tensor(primals_390, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_62 = torch.ops.aten.var_mean.correction(convolution_62, [0, 2, 3], correction = 0, keepdim = True)
    getitem_152: "f32[1, 512, 1, 1]" = var_mean_62[0]
    getitem_153: "f32[1, 512, 1, 1]" = var_mean_62[1];  var_mean_62 = None
    add_332: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_152, 1e-05)
    rsqrt_62: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_332);  add_332 = None
    sub_62: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_62, getitem_153)
    mul_496: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_62);  sub_62 = None
    squeeze_186: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_153, [0, 2, 3]);  getitem_153 = None
    squeeze_187: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_62, [0, 2, 3]);  rsqrt_62 = None
    mul_497: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_186, 0.1)
    mul_498: "f32[512]" = torch.ops.aten.mul.Tensor(primals_391, 0.9)
    add_333: "f32[512]" = torch.ops.aten.add.Tensor(mul_497, mul_498);  mul_497 = mul_498 = None
    squeeze_188: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_152, [0, 2, 3]);  getitem_152 = None
    mul_499: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_188, 1.0019569471624266);  squeeze_188 = None
    mul_500: "f32[512]" = torch.ops.aten.mul.Tensor(mul_499, 0.1);  mul_499 = None
    mul_501: "f32[512]" = torch.ops.aten.mul.Tensor(primals_392, 0.9)
    add_334: "f32[512]" = torch.ops.aten.add.Tensor(mul_500, mul_501);  mul_500 = mul_501 = None
    unsqueeze_248: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_125, -1)
    unsqueeze_249: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
    mul_502: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_496, unsqueeze_249);  mul_496 = unsqueeze_249 = None
    unsqueeze_250: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_126, -1);  primals_126 = None
    unsqueeze_251: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
    add_335: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_502, unsqueeze_251);  mul_502 = unsqueeze_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_62: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(add_335, 0)
    mul_503: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_335, 0.01)
    where_62: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_62, add_335, mul_503);  gt_62 = add_335 = mul_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_336: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(where_62, add_325)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_63: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(add_336, primals_198, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_337: "i64[]" = torch.ops.aten.add.Tensor(primals_393, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_63 = torch.ops.aten.var_mean.correction(convolution_63, [0, 2, 3], correction = 0, keepdim = True)
    getitem_154: "f32[1, 512, 1, 1]" = var_mean_63[0]
    getitem_155: "f32[1, 512, 1, 1]" = var_mean_63[1];  var_mean_63 = None
    add_338: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_154, 1e-05)
    rsqrt_63: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_338);  add_338 = None
    sub_63: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_63, getitem_155)
    mul_504: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_63);  sub_63 = None
    squeeze_189: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_155, [0, 2, 3]);  getitem_155 = None
    squeeze_190: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_63, [0, 2, 3]);  rsqrt_63 = None
    mul_505: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_189, 0.1)
    mul_506: "f32[512]" = torch.ops.aten.mul.Tensor(primals_394, 0.9)
    add_339: "f32[512]" = torch.ops.aten.add.Tensor(mul_505, mul_506);  mul_505 = mul_506 = None
    squeeze_191: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_154, [0, 2, 3]);  getitem_154 = None
    mul_507: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_191, 1.0019569471624266);  squeeze_191 = None
    mul_508: "f32[512]" = torch.ops.aten.mul.Tensor(mul_507, 0.1);  mul_507 = None
    mul_509: "f32[512]" = torch.ops.aten.mul.Tensor(primals_395, 0.9)
    add_340: "f32[512]" = torch.ops.aten.add.Tensor(mul_508, mul_509);  mul_508 = mul_509 = None
    unsqueeze_252: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_127, -1)
    unsqueeze_253: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
    mul_510: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_504, unsqueeze_253);  mul_504 = unsqueeze_253 = None
    unsqueeze_254: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_128, -1);  primals_128 = None
    unsqueeze_255: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
    add_341: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_510, unsqueeze_255);  mul_510 = unsqueeze_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_63: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(add_341, 0)
    mul_511: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_341, 0.01)
    where_63: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_63, add_341, mul_511);  gt_63 = add_341 = mul_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_64: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(where_63, primals_199, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_342: "i64[]" = torch.ops.aten.add.Tensor(primals_396, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_64 = torch.ops.aten.var_mean.correction(convolution_64, [0, 2, 3], correction = 0, keepdim = True)
    getitem_156: "f32[1, 512, 1, 1]" = var_mean_64[0]
    getitem_157: "f32[1, 512, 1, 1]" = var_mean_64[1];  var_mean_64 = None
    add_343: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_156, 1e-05)
    rsqrt_64: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_343);  add_343 = None
    sub_64: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_64, getitem_157)
    mul_512: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_64);  sub_64 = None
    squeeze_192: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_157, [0, 2, 3]);  getitem_157 = None
    squeeze_193: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_64, [0, 2, 3]);  rsqrt_64 = None
    mul_513: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_192, 0.1)
    mul_514: "f32[512]" = torch.ops.aten.mul.Tensor(primals_397, 0.9)
    add_344: "f32[512]" = torch.ops.aten.add.Tensor(mul_513, mul_514);  mul_513 = mul_514 = None
    squeeze_194: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_156, [0, 2, 3]);  getitem_156 = None
    mul_515: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_194, 1.0019569471624266);  squeeze_194 = None
    mul_516: "f32[512]" = torch.ops.aten.mul.Tensor(mul_515, 0.1);  mul_515 = None
    mul_517: "f32[512]" = torch.ops.aten.mul.Tensor(primals_398, 0.9)
    add_345: "f32[512]" = torch.ops.aten.add.Tensor(mul_516, mul_517);  mul_516 = mul_517 = None
    unsqueeze_256: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_129, -1)
    unsqueeze_257: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
    mul_518: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_512, unsqueeze_257);  mul_512 = unsqueeze_257 = None
    unsqueeze_258: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_130, -1);  primals_130 = None
    unsqueeze_259: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
    add_346: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_518, unsqueeze_259);  mul_518 = unsqueeze_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_64: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(add_346, 0)
    mul_519: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_346, 0.01)
    where_64: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_64, add_346, mul_519);  gt_64 = add_346 = mul_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_347: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(where_64, add_336)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_65: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(add_347, primals_200, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_348: "i64[]" = torch.ops.aten.add.Tensor(primals_399, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_65 = torch.ops.aten.var_mean.correction(convolution_65, [0, 2, 3], correction = 0, keepdim = True)
    getitem_158: "f32[1, 512, 1, 1]" = var_mean_65[0]
    getitem_159: "f32[1, 512, 1, 1]" = var_mean_65[1];  var_mean_65 = None
    add_349: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-05)
    rsqrt_65: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_349);  add_349 = None
    sub_65: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_65, getitem_159)
    mul_520: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_65);  sub_65 = None
    squeeze_195: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_159, [0, 2, 3]);  getitem_159 = None
    squeeze_196: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_65, [0, 2, 3]);  rsqrt_65 = None
    mul_521: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_195, 0.1)
    mul_522: "f32[512]" = torch.ops.aten.mul.Tensor(primals_400, 0.9)
    add_350: "f32[512]" = torch.ops.aten.add.Tensor(mul_521, mul_522);  mul_521 = mul_522 = None
    squeeze_197: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_158, [0, 2, 3]);  getitem_158 = None
    mul_523: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_197, 1.0019569471624266);  squeeze_197 = None
    mul_524: "f32[512]" = torch.ops.aten.mul.Tensor(mul_523, 0.1);  mul_523 = None
    mul_525: "f32[512]" = torch.ops.aten.mul.Tensor(primals_401, 0.9)
    add_351: "f32[512]" = torch.ops.aten.add.Tensor(mul_524, mul_525);  mul_524 = mul_525 = None
    unsqueeze_260: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_131, -1)
    unsqueeze_261: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
    mul_526: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_520, unsqueeze_261);  mul_520 = unsqueeze_261 = None
    unsqueeze_262: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_132, -1);  primals_132 = None
    unsqueeze_263: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
    add_352: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_526, unsqueeze_263);  mul_526 = unsqueeze_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_65: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(add_352, 0)
    mul_527: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_352, 0.01)
    where_65: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_65, add_352, mul_527);  gt_65 = add_352 = mul_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:339, code: out = self.conv_transition(torch.cat([xs, xb], dim=1))
    getitem_160: "f32[8, 512, 8, 8]" = split_with_sizes_13[0];  split_with_sizes_13 = None
    cat_4: "f32[8, 1024, 8, 8]" = torch.ops.aten.cat.default([getitem_160, where_65], 1);  getitem_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_66: "f32[8, 1024, 8, 8]" = torch.ops.aten.convolution.default(cat_4, primals_201, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_353: "i64[]" = torch.ops.aten.add.Tensor(primals_402, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_66 = torch.ops.aten.var_mean.correction(convolution_66, [0, 2, 3], correction = 0, keepdim = True)
    getitem_162: "f32[1, 1024, 1, 1]" = var_mean_66[0]
    getitem_163: "f32[1, 1024, 1, 1]" = var_mean_66[1];  var_mean_66 = None
    add_354: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_162, 1e-05)
    rsqrt_66: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_354);  add_354 = None
    sub_66: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_66, getitem_163)
    mul_528: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_66);  sub_66 = None
    squeeze_198: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_163, [0, 2, 3]);  getitem_163 = None
    squeeze_199: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_66, [0, 2, 3]);  rsqrt_66 = None
    mul_529: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_198, 0.1)
    mul_530: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_403, 0.9)
    add_355: "f32[1024]" = torch.ops.aten.add.Tensor(mul_529, mul_530);  mul_529 = mul_530 = None
    squeeze_200: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_162, [0, 2, 3]);  getitem_162 = None
    mul_531: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_200, 1.0019569471624266);  squeeze_200 = None
    mul_532: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_531, 0.1);  mul_531 = None
    mul_533: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_404, 0.9)
    add_356: "f32[1024]" = torch.ops.aten.add.Tensor(mul_532, mul_533);  mul_532 = mul_533 = None
    unsqueeze_264: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_133, -1)
    unsqueeze_265: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
    mul_534: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(mul_528, unsqueeze_265);  mul_528 = unsqueeze_265 = None
    unsqueeze_266: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_134, -1);  primals_134 = None
    unsqueeze_267: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
    add_357: "f32[8, 1024, 8, 8]" = torch.ops.aten.add.Tensor(mul_534, unsqueeze_267);  mul_534 = unsqueeze_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_66: "b8[8, 1024, 8, 8]" = torch.ops.aten.gt.Scalar(add_357, 0)
    mul_535: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(add_357, 0.01)
    where_66: "f32[8, 1024, 8, 8]" = torch.ops.aten.where.self(gt_66, add_357, mul_535);  gt_66 = add_357 = mul_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean: "f32[8, 1024, 1, 1]" = torch.ops.aten.mean.dim(where_66, [-1, -2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view: "f32[8, 1024]" = torch.ops.aten.reshape.default(mean, [8, 1024]);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    permute: "f32[1024, 1000]" = torch.ops.aten.permute.default(primals_202, [1, 0]);  primals_202 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_203, view, permute);  primals_203 = None
    permute_1: "f32[1000, 1024]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_67: "b8[8, 1024, 8, 8]" = torch.ops.aten.gt.Scalar(where_66, 0);  where_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_268: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_198, 0);  squeeze_198 = None
    unsqueeze_269: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, 2);  unsqueeze_268 = None
    unsqueeze_270: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 3);  unsqueeze_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_68: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(where_65, 0);  where_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_280: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_195, 0);  squeeze_195 = None
    unsqueeze_281: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 2);  unsqueeze_280 = None
    unsqueeze_282: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 3);  unsqueeze_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_69: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(where_64, 0);  where_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_292: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_192, 0);  squeeze_192 = None
    unsqueeze_293: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 2);  unsqueeze_292 = None
    unsqueeze_294: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 3);  unsqueeze_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_304: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_189, 0);  squeeze_189 = None
    unsqueeze_305: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, 2);  unsqueeze_304 = None
    unsqueeze_306: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 3);  unsqueeze_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_71: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(where_62, 0);  where_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_316: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_186, 0);  squeeze_186 = None
    unsqueeze_317: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, 2);  unsqueeze_316 = None
    unsqueeze_318: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 3);  unsqueeze_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_328: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_183, 0);  squeeze_183 = None
    unsqueeze_329: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 2);  unsqueeze_328 = None
    unsqueeze_330: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 3);  unsqueeze_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_73: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(where_60, 0);  where_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_340: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_180, 0);  squeeze_180 = None
    unsqueeze_341: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, 2);  unsqueeze_340 = None
    unsqueeze_342: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 3);  unsqueeze_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_352: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_177, 0);  squeeze_177 = None
    unsqueeze_353: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, 2);  unsqueeze_352 = None
    unsqueeze_354: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 3);  unsqueeze_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_75: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(where_58, 0);  where_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_364: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_174, 0);  squeeze_174 = None
    unsqueeze_365: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, 2);  unsqueeze_364 = None
    unsqueeze_366: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 3);  unsqueeze_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_376: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_171, 0);  squeeze_171 = None
    unsqueeze_377: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 2);  unsqueeze_376 = None
    unsqueeze_378: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 3);  unsqueeze_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_98: "f32[8, 1024, 8, 8]" = torch.ops.aten.alias.default(where_56);  where_56 = None
    alias_99: "f32[8, 1024, 8, 8]" = torch.ops.aten.alias.default(alias_98);  alias_98 = None
    gt_77: "b8[8, 1024, 8, 8]" = torch.ops.aten.gt.Scalar(alias_99, 0);  alias_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_388: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_168, 0);  squeeze_168 = None
    unsqueeze_389: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 2);  unsqueeze_388 = None
    unsqueeze_390: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 3);  unsqueeze_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_400: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_165, 0);  squeeze_165 = None
    unsqueeze_401: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, 2);  unsqueeze_400 = None
    unsqueeze_402: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 3);  unsqueeze_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_412: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_162, 0);  squeeze_162 = None
    unsqueeze_413: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, 2);  unsqueeze_412 = None
    unsqueeze_414: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 3);  unsqueeze_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_80: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(where_53, 0);  where_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_424: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_159, 0);  squeeze_159 = None
    unsqueeze_425: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 2);  unsqueeze_424 = None
    unsqueeze_426: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 3);  unsqueeze_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_81: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(where_52, 0);  where_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_436: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_156, 0);  squeeze_156 = None
    unsqueeze_437: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 2);  unsqueeze_436 = None
    unsqueeze_438: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 3);  unsqueeze_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_448: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_153, 0);  squeeze_153 = None
    unsqueeze_449: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, 2);  unsqueeze_448 = None
    unsqueeze_450: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 3);  unsqueeze_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_83: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(where_50, 0);  where_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_460: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_150, 0);  squeeze_150 = None
    unsqueeze_461: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, 2);  unsqueeze_460 = None
    unsqueeze_462: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 3);  unsqueeze_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_472: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_147, 0);  squeeze_147 = None
    unsqueeze_473: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, 2);  unsqueeze_472 = None
    unsqueeze_474: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 3);  unsqueeze_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_85: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(where_48, 0);  where_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_484: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
    unsqueeze_485: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 2);  unsqueeze_484 = None
    unsqueeze_486: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 3);  unsqueeze_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_496: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_141, 0);  squeeze_141 = None
    unsqueeze_497: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, 2);  unsqueeze_496 = None
    unsqueeze_498: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 3);  unsqueeze_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_87: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(where_46, 0);  where_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_508: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_138, 0);  squeeze_138 = None
    unsqueeze_509: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, 2);  unsqueeze_508 = None
    unsqueeze_510: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 3);  unsqueeze_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_520: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
    unsqueeze_521: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, 2);  unsqueeze_520 = None
    unsqueeze_522: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 3);  unsqueeze_521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_89: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(where_44, 0);  where_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_532: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_132, 0);  squeeze_132 = None
    unsqueeze_533: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, 2);  unsqueeze_532 = None
    unsqueeze_534: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 3);  unsqueeze_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_544: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
    unsqueeze_545: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, 2);  unsqueeze_544 = None
    unsqueeze_546: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 3);  unsqueeze_545 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_91: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(where_42, 0);  where_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_556: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
    unsqueeze_557: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, 2);  unsqueeze_556 = None
    unsqueeze_558: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 3);  unsqueeze_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_568: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
    unsqueeze_569: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, 2);  unsqueeze_568 = None
    unsqueeze_570: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 3);  unsqueeze_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_93: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(where_40, 0);  where_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_580: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
    unsqueeze_581: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, 2);  unsqueeze_580 = None
    unsqueeze_582: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 3);  unsqueeze_581 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_592: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
    unsqueeze_593: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 2);  unsqueeze_592 = None
    unsqueeze_594: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 3);  unsqueeze_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_95: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(where_38, 0);  where_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_604: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
    unsqueeze_605: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, 2);  unsqueeze_604 = None
    unsqueeze_606: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 3);  unsqueeze_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_616: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
    unsqueeze_617: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, 2);  unsqueeze_616 = None
    unsqueeze_618: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 3);  unsqueeze_617 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_158: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(where_36);  where_36 = None
    alias_159: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(alias_158);  alias_158 = None
    gt_97: "b8[8, 512, 16, 16]" = torch.ops.aten.gt.Scalar(alias_159, 0);  alias_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_628: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
    unsqueeze_629: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, 2);  unsqueeze_628 = None
    unsqueeze_630: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 3);  unsqueeze_629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_640: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
    unsqueeze_641: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, 2);  unsqueeze_640 = None
    unsqueeze_642: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 3);  unsqueeze_641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_652: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
    unsqueeze_653: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, 2);  unsqueeze_652 = None
    unsqueeze_654: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 3);  unsqueeze_653 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_100: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(where_33, 0);  where_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_664: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
    unsqueeze_665: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, 2);  unsqueeze_664 = None
    unsqueeze_666: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 3);  unsqueeze_665 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_101: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(where_32, 0);  where_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_676: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
    unsqueeze_677: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, 2);  unsqueeze_676 = None
    unsqueeze_678: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 3);  unsqueeze_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_688: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
    unsqueeze_689: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, 2);  unsqueeze_688 = None
    unsqueeze_690: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 3);  unsqueeze_689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_103: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(where_30, 0);  where_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_700: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    unsqueeze_701: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, 2);  unsqueeze_700 = None
    unsqueeze_702: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 3);  unsqueeze_701 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_712: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
    unsqueeze_713: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, 2);  unsqueeze_712 = None
    unsqueeze_714: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 3);  unsqueeze_713 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_105: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(where_28, 0);  where_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_724: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    unsqueeze_725: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, 2);  unsqueeze_724 = None
    unsqueeze_726: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 3);  unsqueeze_725 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_736: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
    unsqueeze_737: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, 2);  unsqueeze_736 = None
    unsqueeze_738: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 3);  unsqueeze_737 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_107: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(where_26, 0);  where_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_748: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_749: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, 2);  unsqueeze_748 = None
    unsqueeze_750: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 3);  unsqueeze_749 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_760: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_761: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, 2);  unsqueeze_760 = None
    unsqueeze_762: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 3);  unsqueeze_761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_109: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(where_24, 0);  where_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_772: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_773: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, 2);  unsqueeze_772 = None
    unsqueeze_774: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 3);  unsqueeze_773 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_784: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_785: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_784, 2);  unsqueeze_784 = None
    unsqueeze_786: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 3);  unsqueeze_785 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_111: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(where_22, 0);  where_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_796: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_797: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_796, 2);  unsqueeze_796 = None
    unsqueeze_798: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_797, 3);  unsqueeze_797 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_808: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_809: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_808, 2);  unsqueeze_808 = None
    unsqueeze_810: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 3);  unsqueeze_809 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_113: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(where_20, 0);  where_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_820: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_821: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_820, 2);  unsqueeze_820 = None
    unsqueeze_822: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 3);  unsqueeze_821 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_832: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_833: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_832, 2);  unsqueeze_832 = None
    unsqueeze_834: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_833, 3);  unsqueeze_833 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_115: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(where_18, 0);  where_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_844: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_845: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_844, 2);  unsqueeze_844 = None
    unsqueeze_846: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_845, 3);  unsqueeze_845 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_856: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_857: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_856, 2);  unsqueeze_856 = None
    unsqueeze_858: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_857, 3);  unsqueeze_857 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_218: "f32[8, 256, 32, 32]" = torch.ops.aten.alias.default(where_16);  where_16 = None
    alias_219: "f32[8, 256, 32, 32]" = torch.ops.aten.alias.default(alias_218);  alias_218 = None
    gt_117: "b8[8, 256, 32, 32]" = torch.ops.aten.gt.Scalar(alias_219, 0);  alias_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_868: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_869: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_868, 2);  unsqueeze_868 = None
    unsqueeze_870: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_869, 3);  unsqueeze_869 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_880: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_881: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_880, 2);  unsqueeze_880 = None
    unsqueeze_882: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_881, 3);  unsqueeze_881 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_892: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_893: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_892, 2);  unsqueeze_892 = None
    unsqueeze_894: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_893, 3);  unsqueeze_893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_120: "b8[8, 64, 64, 64]" = torch.ops.aten.gt.Scalar(where_13, 0);  where_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_904: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_905: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_904, 2);  unsqueeze_904 = None
    unsqueeze_906: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_905, 3);  unsqueeze_905 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_121: "b8[8, 64, 64, 64]" = torch.ops.aten.gt.Scalar(where_12, 0);  where_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_916: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_917: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_916, 2);  unsqueeze_916 = None
    unsqueeze_918: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_917, 3);  unsqueeze_917 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_928: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_929: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_928, 2);  unsqueeze_928 = None
    unsqueeze_930: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_929, 3);  unsqueeze_929 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_123: "b8[8, 64, 64, 64]" = torch.ops.aten.gt.Scalar(where_10, 0);  where_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_940: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_941: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_940, 2);  unsqueeze_940 = None
    unsqueeze_942: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_941, 3);  unsqueeze_941 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_952: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_953: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_952, 2);  unsqueeze_952 = None
    unsqueeze_954: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_953, 3);  unsqueeze_953 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_242: "f32[8, 128, 64, 64]" = torch.ops.aten.alias.default(where_8);  where_8 = None
    alias_243: "f32[8, 128, 64, 64]" = torch.ops.aten.alias.default(alias_242);  alias_242 = None
    gt_125: "b8[8, 128, 64, 64]" = torch.ops.aten.gt.Scalar(alias_243, 0);  alias_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_964: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_965: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_964, 2);  unsqueeze_964 = None
    unsqueeze_966: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_965, 3);  unsqueeze_965 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_976: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_977: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_976, 2);  unsqueeze_976 = None
    unsqueeze_978: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_977, 3);  unsqueeze_977 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_988: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_989: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_988, 2);  unsqueeze_988 = None
    unsqueeze_990: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_989, 3);  unsqueeze_989 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_128: "b8[8, 64, 128, 128]" = torch.ops.aten.gt.Scalar(where_5, 0);  where_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1000: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_1001: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1000, 2);  unsqueeze_1000 = None
    unsqueeze_1002: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1001, 3);  unsqueeze_1001 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_129: "b8[8, 64, 128, 128]" = torch.ops.aten.gt.Scalar(where_4, 0);  where_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1012: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_1013: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1012, 2);  unsqueeze_1012 = None
    unsqueeze_1014: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1013, 3);  unsqueeze_1013 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1024: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_1025: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1024, 2);  unsqueeze_1024 = None
    unsqueeze_1026: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1025, 3);  unsqueeze_1025 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_260: "f32[8, 128, 128, 128]" = torch.ops.aten.alias.default(where_2);  where_2 = None
    alias_261: "f32[8, 128, 128, 128]" = torch.ops.aten.alias.default(alias_260);  alias_260 = None
    gt_131: "b8[8, 128, 128, 128]" = torch.ops.aten.gt.Scalar(alias_261, 0);  alias_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1036: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_1037: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1036, 2);  unsqueeze_1036 = None
    unsqueeze_1038: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1037, 3);  unsqueeze_1037 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1048: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_1049: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1048, 2);  unsqueeze_1048 = None
    unsqueeze_1050: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1049, 3);  unsqueeze_1049 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1060: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_1061: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1060, 2);  unsqueeze_1060 = None
    unsqueeze_1062: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1061, 3);  unsqueeze_1061 = None
    
    # No stacktrace found for following nodes
    copy_: "i64[]" = torch.ops.aten.copy_.default(primals_204, add);  primals_204 = add = None
    copy__1: "f32[32]" = torch.ops.aten.copy_.default(primals_205, add_2);  primals_205 = add_2 = None
    copy__2: "f32[32]" = torch.ops.aten.copy_.default(primals_206, add_3);  primals_206 = add_3 = None
    copy__3: "i64[]" = torch.ops.aten.copy_.default(primals_207, add_5);  primals_207 = add_5 = None
    copy__4: "f32[64]" = torch.ops.aten.copy_.default(primals_208, add_7);  primals_208 = add_7 = None
    copy__5: "f32[64]" = torch.ops.aten.copy_.default(primals_209, add_8);  primals_209 = add_8 = None
    copy__6: "i64[]" = torch.ops.aten.copy_.default(primals_210, add_10);  primals_210 = add_10 = None
    copy__7: "f32[128]" = torch.ops.aten.copy_.default(primals_211, add_12);  primals_211 = add_12 = None
    copy__8: "f32[128]" = torch.ops.aten.copy_.default(primals_212, add_13);  primals_212 = add_13 = None
    copy__9: "i64[]" = torch.ops.aten.copy_.default(primals_213, add_15);  primals_213 = add_15 = None
    copy__10: "f32[32]" = torch.ops.aten.copy_.default(primals_214, add_17);  primals_214 = add_17 = None
    copy__11: "f32[32]" = torch.ops.aten.copy_.default(primals_215, add_18);  primals_215 = add_18 = None
    copy__12: "i64[]" = torch.ops.aten.copy_.default(primals_216, add_20);  primals_216 = add_20 = None
    copy__13: "f32[64]" = torch.ops.aten.copy_.default(primals_217, add_22);  primals_217 = add_22 = None
    copy__14: "f32[64]" = torch.ops.aten.copy_.default(primals_218, add_23);  primals_218 = add_23 = None
    copy__15: "i64[]" = torch.ops.aten.copy_.default(primals_219, add_26);  primals_219 = add_26 = None
    copy__16: "f32[64]" = torch.ops.aten.copy_.default(primals_220, add_28);  primals_220 = add_28 = None
    copy__17: "f32[64]" = torch.ops.aten.copy_.default(primals_221, add_29);  primals_221 = add_29 = None
    copy__18: "i64[]" = torch.ops.aten.copy_.default(primals_222, add_31);  primals_222 = add_31 = None
    copy__19: "f32[64]" = torch.ops.aten.copy_.default(primals_223, add_33);  primals_223 = add_33 = None
    copy__20: "f32[64]" = torch.ops.aten.copy_.default(primals_224, add_34);  primals_224 = add_34 = None
    copy__21: "i64[]" = torch.ops.aten.copy_.default(primals_225, add_36);  primals_225 = add_36 = None
    copy__22: "f32[128]" = torch.ops.aten.copy_.default(primals_226, add_38);  primals_226 = add_38 = None
    copy__23: "f32[128]" = torch.ops.aten.copy_.default(primals_227, add_39);  primals_227 = add_39 = None
    copy__24: "i64[]" = torch.ops.aten.copy_.default(primals_228, add_41);  primals_228 = add_41 = None
    copy__25: "f32[128]" = torch.ops.aten.copy_.default(primals_229, add_43);  primals_229 = add_43 = None
    copy__26: "f32[128]" = torch.ops.aten.copy_.default(primals_230, add_44);  primals_230 = add_44 = None
    copy__27: "i64[]" = torch.ops.aten.copy_.default(primals_231, add_46);  primals_231 = add_46 = None
    copy__28: "f32[64]" = torch.ops.aten.copy_.default(primals_232, add_48);  primals_232 = add_48 = None
    copy__29: "f32[64]" = torch.ops.aten.copy_.default(primals_233, add_49);  primals_233 = add_49 = None
    copy__30: "i64[]" = torch.ops.aten.copy_.default(primals_234, add_51);  primals_234 = add_51 = None
    copy__31: "f32[64]" = torch.ops.aten.copy_.default(primals_235, add_53);  primals_235 = add_53 = None
    copy__32: "f32[64]" = torch.ops.aten.copy_.default(primals_236, add_54);  primals_236 = add_54 = None
    copy__33: "i64[]" = torch.ops.aten.copy_.default(primals_237, add_57);  primals_237 = add_57 = None
    copy__34: "f32[64]" = torch.ops.aten.copy_.default(primals_238, add_59);  primals_238 = add_59 = None
    copy__35: "f32[64]" = torch.ops.aten.copy_.default(primals_239, add_60);  primals_239 = add_60 = None
    copy__36: "i64[]" = torch.ops.aten.copy_.default(primals_240, add_62);  primals_240 = add_62 = None
    copy__37: "f32[64]" = torch.ops.aten.copy_.default(primals_241, add_64);  primals_241 = add_64 = None
    copy__38: "f32[64]" = torch.ops.aten.copy_.default(primals_242, add_65);  primals_242 = add_65 = None
    copy__39: "i64[]" = torch.ops.aten.copy_.default(primals_243, add_68);  primals_243 = add_68 = None
    copy__40: "f32[64]" = torch.ops.aten.copy_.default(primals_244, add_70);  primals_244 = add_70 = None
    copy__41: "f32[64]" = torch.ops.aten.copy_.default(primals_245, add_71);  primals_245 = add_71 = None
    copy__42: "i64[]" = torch.ops.aten.copy_.default(primals_246, add_73);  primals_246 = add_73 = None
    copy__43: "f32[128]" = torch.ops.aten.copy_.default(primals_247, add_75);  primals_247 = add_75 = None
    copy__44: "f32[128]" = torch.ops.aten.copy_.default(primals_248, add_76);  primals_248 = add_76 = None
    copy__45: "i64[]" = torch.ops.aten.copy_.default(primals_249, add_78);  primals_249 = add_78 = None
    copy__46: "f32[256]" = torch.ops.aten.copy_.default(primals_250, add_80);  primals_250 = add_80 = None
    copy__47: "f32[256]" = torch.ops.aten.copy_.default(primals_251, add_81);  primals_251 = add_81 = None
    copy__48: "i64[]" = torch.ops.aten.copy_.default(primals_252, add_83);  primals_252 = add_83 = None
    copy__49: "f32[256]" = torch.ops.aten.copy_.default(primals_253, add_85);  primals_253 = add_85 = None
    copy__50: "f32[256]" = torch.ops.aten.copy_.default(primals_254, add_86);  primals_254 = add_86 = None
    copy__51: "i64[]" = torch.ops.aten.copy_.default(primals_255, add_88);  primals_255 = add_88 = None
    copy__52: "f32[128]" = torch.ops.aten.copy_.default(primals_256, add_90);  primals_256 = add_90 = None
    copy__53: "f32[128]" = torch.ops.aten.copy_.default(primals_257, add_91);  primals_257 = add_91 = None
    copy__54: "i64[]" = torch.ops.aten.copy_.default(primals_258, add_93);  primals_258 = add_93 = None
    copy__55: "f32[128]" = torch.ops.aten.copy_.default(primals_259, add_95);  primals_259 = add_95 = None
    copy__56: "f32[128]" = torch.ops.aten.copy_.default(primals_260, add_96);  primals_260 = add_96 = None
    copy__57: "i64[]" = torch.ops.aten.copy_.default(primals_261, add_99);  primals_261 = add_99 = None
    copy__58: "f32[128]" = torch.ops.aten.copy_.default(primals_262, add_101);  primals_262 = add_101 = None
    copy__59: "f32[128]" = torch.ops.aten.copy_.default(primals_263, add_102);  primals_263 = add_102 = None
    copy__60: "i64[]" = torch.ops.aten.copy_.default(primals_264, add_104);  primals_264 = add_104 = None
    copy__61: "f32[128]" = torch.ops.aten.copy_.default(primals_265, add_106);  primals_265 = add_106 = None
    copy__62: "f32[128]" = torch.ops.aten.copy_.default(primals_266, add_107);  primals_266 = add_107 = None
    copy__63: "i64[]" = torch.ops.aten.copy_.default(primals_267, add_110);  primals_267 = add_110 = None
    copy__64: "f32[128]" = torch.ops.aten.copy_.default(primals_268, add_112);  primals_268 = add_112 = None
    copy__65: "f32[128]" = torch.ops.aten.copy_.default(primals_269, add_113);  primals_269 = add_113 = None
    copy__66: "i64[]" = torch.ops.aten.copy_.default(primals_270, add_115);  primals_270 = add_115 = None
    copy__67: "f32[128]" = torch.ops.aten.copy_.default(primals_271, add_117);  primals_271 = add_117 = None
    copy__68: "f32[128]" = torch.ops.aten.copy_.default(primals_272, add_118);  primals_272 = add_118 = None
    copy__69: "i64[]" = torch.ops.aten.copy_.default(primals_273, add_121);  primals_273 = add_121 = None
    copy__70: "f32[128]" = torch.ops.aten.copy_.default(primals_274, add_123);  primals_274 = add_123 = None
    copy__71: "f32[128]" = torch.ops.aten.copy_.default(primals_275, add_124);  primals_275 = add_124 = None
    copy__72: "i64[]" = torch.ops.aten.copy_.default(primals_276, add_126);  primals_276 = add_126 = None
    copy__73: "f32[128]" = torch.ops.aten.copy_.default(primals_277, add_128);  primals_277 = add_128 = None
    copy__74: "f32[128]" = torch.ops.aten.copy_.default(primals_278, add_129);  primals_278 = add_129 = None
    copy__75: "i64[]" = torch.ops.aten.copy_.default(primals_279, add_132);  primals_279 = add_132 = None
    copy__76: "f32[128]" = torch.ops.aten.copy_.default(primals_280, add_134);  primals_280 = add_134 = None
    copy__77: "f32[128]" = torch.ops.aten.copy_.default(primals_281, add_135);  primals_281 = add_135 = None
    copy__78: "i64[]" = torch.ops.aten.copy_.default(primals_282, add_137);  primals_282 = add_137 = None
    copy__79: "f32[128]" = torch.ops.aten.copy_.default(primals_283, add_139);  primals_283 = add_139 = None
    copy__80: "f32[128]" = torch.ops.aten.copy_.default(primals_284, add_140);  primals_284 = add_140 = None
    copy__81: "i64[]" = torch.ops.aten.copy_.default(primals_285, add_143);  primals_285 = add_143 = None
    copy__82: "f32[128]" = torch.ops.aten.copy_.default(primals_286, add_145);  primals_286 = add_145 = None
    copy__83: "f32[128]" = torch.ops.aten.copy_.default(primals_287, add_146);  primals_287 = add_146 = None
    copy__84: "i64[]" = torch.ops.aten.copy_.default(primals_288, add_148);  primals_288 = add_148 = None
    copy__85: "f32[128]" = torch.ops.aten.copy_.default(primals_289, add_150);  primals_289 = add_150 = None
    copy__86: "f32[128]" = torch.ops.aten.copy_.default(primals_290, add_151);  primals_290 = add_151 = None
    copy__87: "i64[]" = torch.ops.aten.copy_.default(primals_291, add_154);  primals_291 = add_154 = None
    copy__88: "f32[128]" = torch.ops.aten.copy_.default(primals_292, add_156);  primals_292 = add_156 = None
    copy__89: "f32[128]" = torch.ops.aten.copy_.default(primals_293, add_157);  primals_293 = add_157 = None
    copy__90: "i64[]" = torch.ops.aten.copy_.default(primals_294, add_159);  primals_294 = add_159 = None
    copy__91: "f32[128]" = torch.ops.aten.copy_.default(primals_295, add_161);  primals_295 = add_161 = None
    copy__92: "f32[128]" = torch.ops.aten.copy_.default(primals_296, add_162);  primals_296 = add_162 = None
    copy__93: "i64[]" = torch.ops.aten.copy_.default(primals_297, add_165);  primals_297 = add_165 = None
    copy__94: "f32[128]" = torch.ops.aten.copy_.default(primals_298, add_167);  primals_298 = add_167 = None
    copy__95: "f32[128]" = torch.ops.aten.copy_.default(primals_299, add_168);  primals_299 = add_168 = None
    copy__96: "i64[]" = torch.ops.aten.copy_.default(primals_300, add_170);  primals_300 = add_170 = None
    copy__97: "f32[128]" = torch.ops.aten.copy_.default(primals_301, add_172);  primals_301 = add_172 = None
    copy__98: "f32[128]" = torch.ops.aten.copy_.default(primals_302, add_173);  primals_302 = add_173 = None
    copy__99: "i64[]" = torch.ops.aten.copy_.default(primals_303, add_176);  primals_303 = add_176 = None
    copy__100: "f32[128]" = torch.ops.aten.copy_.default(primals_304, add_178);  primals_304 = add_178 = None
    copy__101: "f32[128]" = torch.ops.aten.copy_.default(primals_305, add_179);  primals_305 = add_179 = None
    copy__102: "i64[]" = torch.ops.aten.copy_.default(primals_306, add_181);  primals_306 = add_181 = None
    copy__103: "f32[256]" = torch.ops.aten.copy_.default(primals_307, add_183);  primals_307 = add_183 = None
    copy__104: "f32[256]" = torch.ops.aten.copy_.default(primals_308, add_184);  primals_308 = add_184 = None
    copy__105: "i64[]" = torch.ops.aten.copy_.default(primals_309, add_186);  primals_309 = add_186 = None
    copy__106: "f32[512]" = torch.ops.aten.copy_.default(primals_310, add_188);  primals_310 = add_188 = None
    copy__107: "f32[512]" = torch.ops.aten.copy_.default(primals_311, add_189);  primals_311 = add_189 = None
    copy__108: "i64[]" = torch.ops.aten.copy_.default(primals_312, add_191);  primals_312 = add_191 = None
    copy__109: "f32[512]" = torch.ops.aten.copy_.default(primals_313, add_193);  primals_313 = add_193 = None
    copy__110: "f32[512]" = torch.ops.aten.copy_.default(primals_314, add_194);  primals_314 = add_194 = None
    copy__111: "i64[]" = torch.ops.aten.copy_.default(primals_315, add_196);  primals_315 = add_196 = None
    copy__112: "f32[256]" = torch.ops.aten.copy_.default(primals_316, add_198);  primals_316 = add_198 = None
    copy__113: "f32[256]" = torch.ops.aten.copy_.default(primals_317, add_199);  primals_317 = add_199 = None
    copy__114: "i64[]" = torch.ops.aten.copy_.default(primals_318, add_201);  primals_318 = add_201 = None
    copy__115: "f32[256]" = torch.ops.aten.copy_.default(primals_319, add_203);  primals_319 = add_203 = None
    copy__116: "f32[256]" = torch.ops.aten.copy_.default(primals_320, add_204);  primals_320 = add_204 = None
    copy__117: "i64[]" = torch.ops.aten.copy_.default(primals_321, add_207);  primals_321 = add_207 = None
    copy__118: "f32[256]" = torch.ops.aten.copy_.default(primals_322, add_209);  primals_322 = add_209 = None
    copy__119: "f32[256]" = torch.ops.aten.copy_.default(primals_323, add_210);  primals_323 = add_210 = None
    copy__120: "i64[]" = torch.ops.aten.copy_.default(primals_324, add_212);  primals_324 = add_212 = None
    copy__121: "f32[256]" = torch.ops.aten.copy_.default(primals_325, add_214);  primals_325 = add_214 = None
    copy__122: "f32[256]" = torch.ops.aten.copy_.default(primals_326, add_215);  primals_326 = add_215 = None
    copy__123: "i64[]" = torch.ops.aten.copy_.default(primals_327, add_218);  primals_327 = add_218 = None
    copy__124: "f32[256]" = torch.ops.aten.copy_.default(primals_328, add_220);  primals_328 = add_220 = None
    copy__125: "f32[256]" = torch.ops.aten.copy_.default(primals_329, add_221);  primals_329 = add_221 = None
    copy__126: "i64[]" = torch.ops.aten.copy_.default(primals_330, add_223);  primals_330 = add_223 = None
    copy__127: "f32[256]" = torch.ops.aten.copy_.default(primals_331, add_225);  primals_331 = add_225 = None
    copy__128: "f32[256]" = torch.ops.aten.copy_.default(primals_332, add_226);  primals_332 = add_226 = None
    copy__129: "i64[]" = torch.ops.aten.copy_.default(primals_333, add_229);  primals_333 = add_229 = None
    copy__130: "f32[256]" = torch.ops.aten.copy_.default(primals_334, add_231);  primals_334 = add_231 = None
    copy__131: "f32[256]" = torch.ops.aten.copy_.default(primals_335, add_232);  primals_335 = add_232 = None
    copy__132: "i64[]" = torch.ops.aten.copy_.default(primals_336, add_234);  primals_336 = add_234 = None
    copy__133: "f32[256]" = torch.ops.aten.copy_.default(primals_337, add_236);  primals_337 = add_236 = None
    copy__134: "f32[256]" = torch.ops.aten.copy_.default(primals_338, add_237);  primals_338 = add_237 = None
    copy__135: "i64[]" = torch.ops.aten.copy_.default(primals_339, add_240);  primals_339 = add_240 = None
    copy__136: "f32[256]" = torch.ops.aten.copy_.default(primals_340, add_242);  primals_340 = add_242 = None
    copy__137: "f32[256]" = torch.ops.aten.copy_.default(primals_341, add_243);  primals_341 = add_243 = None
    copy__138: "i64[]" = torch.ops.aten.copy_.default(primals_342, add_245);  primals_342 = add_245 = None
    copy__139: "f32[256]" = torch.ops.aten.copy_.default(primals_343, add_247);  primals_343 = add_247 = None
    copy__140: "f32[256]" = torch.ops.aten.copy_.default(primals_344, add_248);  primals_344 = add_248 = None
    copy__141: "i64[]" = torch.ops.aten.copy_.default(primals_345, add_251);  primals_345 = add_251 = None
    copy__142: "f32[256]" = torch.ops.aten.copy_.default(primals_346, add_253);  primals_346 = add_253 = None
    copy__143: "f32[256]" = torch.ops.aten.copy_.default(primals_347, add_254);  primals_347 = add_254 = None
    copy__144: "i64[]" = torch.ops.aten.copy_.default(primals_348, add_256);  primals_348 = add_256 = None
    copy__145: "f32[256]" = torch.ops.aten.copy_.default(primals_349, add_258);  primals_349 = add_258 = None
    copy__146: "f32[256]" = torch.ops.aten.copy_.default(primals_350, add_259);  primals_350 = add_259 = None
    copy__147: "i64[]" = torch.ops.aten.copy_.default(primals_351, add_262);  primals_351 = add_262 = None
    copy__148: "f32[256]" = torch.ops.aten.copy_.default(primals_352, add_264);  primals_352 = add_264 = None
    copy__149: "f32[256]" = torch.ops.aten.copy_.default(primals_353, add_265);  primals_353 = add_265 = None
    copy__150: "i64[]" = torch.ops.aten.copy_.default(primals_354, add_267);  primals_354 = add_267 = None
    copy__151: "f32[256]" = torch.ops.aten.copy_.default(primals_355, add_269);  primals_355 = add_269 = None
    copy__152: "f32[256]" = torch.ops.aten.copy_.default(primals_356, add_270);  primals_356 = add_270 = None
    copy__153: "i64[]" = torch.ops.aten.copy_.default(primals_357, add_273);  primals_357 = add_273 = None
    copy__154: "f32[256]" = torch.ops.aten.copy_.default(primals_358, add_275);  primals_358 = add_275 = None
    copy__155: "f32[256]" = torch.ops.aten.copy_.default(primals_359, add_276);  primals_359 = add_276 = None
    copy__156: "i64[]" = torch.ops.aten.copy_.default(primals_360, add_278);  primals_360 = add_278 = None
    copy__157: "f32[256]" = torch.ops.aten.copy_.default(primals_361, add_280);  primals_361 = add_280 = None
    copy__158: "f32[256]" = torch.ops.aten.copy_.default(primals_362, add_281);  primals_362 = add_281 = None
    copy__159: "i64[]" = torch.ops.aten.copy_.default(primals_363, add_284);  primals_363 = add_284 = None
    copy__160: "f32[256]" = torch.ops.aten.copy_.default(primals_364, add_286);  primals_364 = add_286 = None
    copy__161: "f32[256]" = torch.ops.aten.copy_.default(primals_365, add_287);  primals_365 = add_287 = None
    copy__162: "i64[]" = torch.ops.aten.copy_.default(primals_366, add_289);  primals_366 = add_289 = None
    copy__163: "f32[512]" = torch.ops.aten.copy_.default(primals_367, add_291);  primals_367 = add_291 = None
    copy__164: "f32[512]" = torch.ops.aten.copy_.default(primals_368, add_292);  primals_368 = add_292 = None
    copy__165: "i64[]" = torch.ops.aten.copy_.default(primals_369, add_294);  primals_369 = add_294 = None
    copy__166: "f32[1024]" = torch.ops.aten.copy_.default(primals_370, add_296);  primals_370 = add_296 = None
    copy__167: "f32[1024]" = torch.ops.aten.copy_.default(primals_371, add_297);  primals_371 = add_297 = None
    copy__168: "i64[]" = torch.ops.aten.copy_.default(primals_372, add_299);  primals_372 = add_299 = None
    copy__169: "f32[1024]" = torch.ops.aten.copy_.default(primals_373, add_301);  primals_373 = add_301 = None
    copy__170: "f32[1024]" = torch.ops.aten.copy_.default(primals_374, add_302);  primals_374 = add_302 = None
    copy__171: "i64[]" = torch.ops.aten.copy_.default(primals_375, add_304);  primals_375 = add_304 = None
    copy__172: "f32[512]" = torch.ops.aten.copy_.default(primals_376, add_306);  primals_376 = add_306 = None
    copy__173: "f32[512]" = torch.ops.aten.copy_.default(primals_377, add_307);  primals_377 = add_307 = None
    copy__174: "i64[]" = torch.ops.aten.copy_.default(primals_378, add_309);  primals_378 = add_309 = None
    copy__175: "f32[512]" = torch.ops.aten.copy_.default(primals_379, add_311);  primals_379 = add_311 = None
    copy__176: "f32[512]" = torch.ops.aten.copy_.default(primals_380, add_312);  primals_380 = add_312 = None
    copy__177: "i64[]" = torch.ops.aten.copy_.default(primals_381, add_315);  primals_381 = add_315 = None
    copy__178: "f32[512]" = torch.ops.aten.copy_.default(primals_382, add_317);  primals_382 = add_317 = None
    copy__179: "f32[512]" = torch.ops.aten.copy_.default(primals_383, add_318);  primals_383 = add_318 = None
    copy__180: "i64[]" = torch.ops.aten.copy_.default(primals_384, add_320);  primals_384 = add_320 = None
    copy__181: "f32[512]" = torch.ops.aten.copy_.default(primals_385, add_322);  primals_385 = add_322 = None
    copy__182: "f32[512]" = torch.ops.aten.copy_.default(primals_386, add_323);  primals_386 = add_323 = None
    copy__183: "i64[]" = torch.ops.aten.copy_.default(primals_387, add_326);  primals_387 = add_326 = None
    copy__184: "f32[512]" = torch.ops.aten.copy_.default(primals_388, add_328);  primals_388 = add_328 = None
    copy__185: "f32[512]" = torch.ops.aten.copy_.default(primals_389, add_329);  primals_389 = add_329 = None
    copy__186: "i64[]" = torch.ops.aten.copy_.default(primals_390, add_331);  primals_390 = add_331 = None
    copy__187: "f32[512]" = torch.ops.aten.copy_.default(primals_391, add_333);  primals_391 = add_333 = None
    copy__188: "f32[512]" = torch.ops.aten.copy_.default(primals_392, add_334);  primals_392 = add_334 = None
    copy__189: "i64[]" = torch.ops.aten.copy_.default(primals_393, add_337);  primals_393 = add_337 = None
    copy__190: "f32[512]" = torch.ops.aten.copy_.default(primals_394, add_339);  primals_394 = add_339 = None
    copy__191: "f32[512]" = torch.ops.aten.copy_.default(primals_395, add_340);  primals_395 = add_340 = None
    copy__192: "i64[]" = torch.ops.aten.copy_.default(primals_396, add_342);  primals_396 = add_342 = None
    copy__193: "f32[512]" = torch.ops.aten.copy_.default(primals_397, add_344);  primals_397 = add_344 = None
    copy__194: "f32[512]" = torch.ops.aten.copy_.default(primals_398, add_345);  primals_398 = add_345 = None
    copy__195: "i64[]" = torch.ops.aten.copy_.default(primals_399, add_348);  primals_399 = add_348 = None
    copy__196: "f32[512]" = torch.ops.aten.copy_.default(primals_400, add_350);  primals_400 = add_350 = None
    copy__197: "f32[512]" = torch.ops.aten.copy_.default(primals_401, add_351);  primals_401 = add_351 = None
    copy__198: "i64[]" = torch.ops.aten.copy_.default(primals_402, add_353);  primals_402 = add_353 = None
    copy__199: "f32[1024]" = torch.ops.aten.copy_.default(primals_403, add_355);  primals_403 = add_355 = None
    copy__200: "f32[1024]" = torch.ops.aten.copy_.default(primals_404, add_356);  primals_404 = add_356 = None
    return [addmm, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_405, convolution, squeeze_1, where, convolution_1, squeeze_4, where_1, convolution_2, squeeze_7, getitem_9, convolution_3, squeeze_10, where_3, convolution_4, squeeze_13, add_25, convolution_5, squeeze_16, cat, convolution_6, squeeze_19, where_6, convolution_7, squeeze_22, where_7, convolution_8, squeeze_25, getitem_27, convolution_9, squeeze_28, where_9, convolution_10, squeeze_31, add_56, convolution_11, squeeze_34, where_11, convolution_12, squeeze_37, add_67, convolution_13, squeeze_40, cat_1, convolution_14, squeeze_43, where_14, convolution_15, squeeze_46, where_15, convolution_16, squeeze_49, getitem_49, convolution_17, squeeze_52, where_17, convolution_18, squeeze_55, add_98, convolution_19, squeeze_58, where_19, convolution_20, squeeze_61, add_109, convolution_21, squeeze_64, where_21, convolution_22, squeeze_67, add_120, convolution_23, squeeze_70, where_23, convolution_24, squeeze_73, add_131, convolution_25, squeeze_76, where_25, convolution_26, squeeze_79, add_142, convolution_27, squeeze_82, where_27, convolution_28, squeeze_85, add_153, convolution_29, squeeze_88, where_29, convolution_30, squeeze_91, add_164, convolution_31, squeeze_94, where_31, convolution_32, squeeze_97, add_175, convolution_33, squeeze_100, cat_2, convolution_34, squeeze_103, where_34, convolution_35, squeeze_106, where_35, convolution_36, squeeze_109, getitem_95, convolution_37, squeeze_112, where_37, convolution_38, squeeze_115, add_206, convolution_39, squeeze_118, where_39, convolution_40, squeeze_121, add_217, convolution_41, squeeze_124, where_41, convolution_42, squeeze_127, add_228, convolution_43, squeeze_130, where_43, convolution_44, squeeze_133, add_239, convolution_45, squeeze_136, where_45, convolution_46, squeeze_139, add_250, convolution_47, squeeze_142, where_47, convolution_48, squeeze_145, add_261, convolution_49, squeeze_148, where_49, convolution_50, squeeze_151, add_272, convolution_51, squeeze_154, where_51, convolution_52, squeeze_157, add_283, convolution_53, squeeze_160, cat_3, convolution_54, squeeze_163, where_54, convolution_55, squeeze_166, where_55, convolution_56, squeeze_169, getitem_141, convolution_57, squeeze_172, where_57, convolution_58, squeeze_175, add_314, convolution_59, squeeze_178, where_59, convolution_60, squeeze_181, add_325, convolution_61, squeeze_184, where_61, convolution_62, squeeze_187, add_336, convolution_63, squeeze_190, where_63, convolution_64, squeeze_193, add_347, convolution_65, squeeze_196, cat_4, convolution_66, squeeze_199, view, permute_1, gt_67, unsqueeze_270, gt_68, unsqueeze_282, gt_69, unsqueeze_294, unsqueeze_306, gt_71, unsqueeze_318, unsqueeze_330, gt_73, unsqueeze_342, unsqueeze_354, gt_75, unsqueeze_366, unsqueeze_378, gt_77, unsqueeze_390, unsqueeze_402, unsqueeze_414, gt_80, unsqueeze_426, gt_81, unsqueeze_438, unsqueeze_450, gt_83, unsqueeze_462, unsqueeze_474, gt_85, unsqueeze_486, unsqueeze_498, gt_87, unsqueeze_510, unsqueeze_522, gt_89, unsqueeze_534, unsqueeze_546, gt_91, unsqueeze_558, unsqueeze_570, gt_93, unsqueeze_582, unsqueeze_594, gt_95, unsqueeze_606, unsqueeze_618, gt_97, unsqueeze_630, unsqueeze_642, unsqueeze_654, gt_100, unsqueeze_666, gt_101, unsqueeze_678, unsqueeze_690, gt_103, unsqueeze_702, unsqueeze_714, gt_105, unsqueeze_726, unsqueeze_738, gt_107, unsqueeze_750, unsqueeze_762, gt_109, unsqueeze_774, unsqueeze_786, gt_111, unsqueeze_798, unsqueeze_810, gt_113, unsqueeze_822, unsqueeze_834, gt_115, unsqueeze_846, unsqueeze_858, gt_117, unsqueeze_870, unsqueeze_882, unsqueeze_894, gt_120, unsqueeze_906, gt_121, unsqueeze_918, unsqueeze_930, gt_123, unsqueeze_942, unsqueeze_954, gt_125, unsqueeze_966, unsqueeze_978, unsqueeze_990, gt_128, unsqueeze_1002, gt_129, unsqueeze_1014, unsqueeze_1026, gt_131, unsqueeze_1038, unsqueeze_1050, unsqueeze_1062]
    