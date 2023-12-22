from __future__ import annotations



def forward(self, primals_1: "f32[32]", primals_2: "f32[32]", primals_3: "f32[32]", primals_4: "f32[32]", primals_5: "f32[16]", primals_6: "f32[16]", primals_7: "f32[96]", primals_8: "f32[96]", primals_9: "f32[96]", primals_10: "f32[96]", primals_11: "f32[24]", primals_12: "f32[24]", primals_13: "f32[144]", primals_14: "f32[144]", primals_15: "f32[144]", primals_16: "f32[144]", primals_17: "f32[24]", primals_18: "f32[24]", primals_19: "f32[144]", primals_20: "f32[144]", primals_21: "f32[144]", primals_22: "f32[144]", primals_23: "f32[40]", primals_24: "f32[40]", primals_25: "f32[240]", primals_26: "f32[240]", primals_27: "f32[240]", primals_28: "f32[240]", primals_29: "f32[40]", primals_30: "f32[40]", primals_31: "f32[240]", primals_32: "f32[240]", primals_33: "f32[240]", primals_34: "f32[240]", primals_35: "f32[80]", primals_36: "f32[80]", primals_37: "f32[480]", primals_38: "f32[480]", primals_39: "f32[480]", primals_40: "f32[480]", primals_41: "f32[80]", primals_42: "f32[80]", primals_43: "f32[480]", primals_44: "f32[480]", primals_45: "f32[480]", primals_46: "f32[480]", primals_47: "f32[80]", primals_48: "f32[80]", primals_49: "f32[480]", primals_50: "f32[480]", primals_51: "f32[480]", primals_52: "f32[480]", primals_53: "f32[80]", primals_54: "f32[80]", primals_55: "f32[480]", primals_56: "f32[480]", primals_57: "f32[480]", primals_58: "f32[480]", primals_59: "f32[112]", primals_60: "f32[112]", primals_61: "f32[672]", primals_62: "f32[672]", primals_63: "f32[672]", primals_64: "f32[672]", primals_65: "f32[112]", primals_66: "f32[112]", primals_67: "f32[672]", primals_68: "f32[672]", primals_69: "f32[672]", primals_70: "f32[672]", primals_71: "f32[112]", primals_72: "f32[112]", primals_73: "f32[672]", primals_74: "f32[672]", primals_75: "f32[672]", primals_76: "f32[672]", primals_77: "f32[112]", primals_78: "f32[112]", primals_79: "f32[672]", primals_80: "f32[672]", primals_81: "f32[672]", primals_82: "f32[672]", primals_83: "f32[192]", primals_84: "f32[192]", primals_85: "f32[1152]", primals_86: "f32[1152]", primals_87: "f32[1152]", primals_88: "f32[1152]", primals_89: "f32[192]", primals_90: "f32[192]", primals_91: "f32[1152]", primals_92: "f32[1152]", primals_93: "f32[1152]", primals_94: "f32[1152]", primals_95: "f32[192]", primals_96: "f32[192]", primals_97: "f32[1152]", primals_98: "f32[1152]", primals_99: "f32[1152]", primals_100: "f32[1152]", primals_101: "f32[192]", primals_102: "f32[192]", primals_103: "f32[1152]", primals_104: "f32[1152]", primals_105: "f32[1152]", primals_106: "f32[1152]", primals_107: "f32[192]", primals_108: "f32[192]", primals_109: "f32[1152]", primals_110: "f32[1152]", primals_111: "f32[1152]", primals_112: "f32[1152]", primals_113: "f32[320]", primals_114: "f32[320]", primals_115: "f32[1280]", primals_116: "f32[1280]", primals_117: "f32[32, 3, 3, 3]", primals_118: "f32[32, 1, 3, 3]", primals_119: "f32[8, 32, 1, 1]", primals_120: "f32[8]", primals_121: "f32[32, 8, 1, 1]", primals_122: "f32[32]", primals_123: "f32[16, 32, 1, 1]", primals_124: "f32[96, 16, 1, 1]", primals_125: "f32[96, 1, 3, 3]", primals_126: "f32[4, 96, 1, 1]", primals_127: "f32[4]", primals_128: "f32[96, 4, 1, 1]", primals_129: "f32[96]", primals_130: "f32[24, 96, 1, 1]", primals_131: "f32[144, 24, 1, 1]", primals_132: "f32[144, 1, 3, 3]", primals_133: "f32[6, 144, 1, 1]", primals_134: "f32[6]", primals_135: "f32[144, 6, 1, 1]", primals_136: "f32[144]", primals_137: "f32[24, 144, 1, 1]", primals_138: "f32[144, 24, 1, 1]", primals_139: "f32[144, 1, 5, 5]", primals_140: "f32[6, 144, 1, 1]", primals_141: "f32[6]", primals_142: "f32[144, 6, 1, 1]", primals_143: "f32[144]", primals_144: "f32[40, 144, 1, 1]", primals_145: "f32[240, 40, 1, 1]", primals_146: "f32[240, 1, 5, 5]", primals_147: "f32[10, 240, 1, 1]", primals_148: "f32[10]", primals_149: "f32[240, 10, 1, 1]", primals_150: "f32[240]", primals_151: "f32[40, 240, 1, 1]", primals_152: "f32[240, 40, 1, 1]", primals_153: "f32[240, 1, 3, 3]", primals_154: "f32[10, 240, 1, 1]", primals_155: "f32[10]", primals_156: "f32[240, 10, 1, 1]", primals_157: "f32[240]", primals_158: "f32[80, 240, 1, 1]", primals_159: "f32[480, 80, 1, 1]", primals_160: "f32[480, 1, 3, 3]", primals_161: "f32[20, 480, 1, 1]", primals_162: "f32[20]", primals_163: "f32[480, 20, 1, 1]", primals_164: "f32[480]", primals_165: "f32[80, 480, 1, 1]", primals_166: "f32[480, 80, 1, 1]", primals_167: "f32[480, 1, 3, 3]", primals_168: "f32[20, 480, 1, 1]", primals_169: "f32[20]", primals_170: "f32[480, 20, 1, 1]", primals_171: "f32[480]", primals_172: "f32[80, 480, 1, 1]", primals_173: "f32[480, 80, 1, 1]", primals_174: "f32[480, 1, 3, 3]", primals_175: "f32[20, 480, 1, 1]", primals_176: "f32[20]", primals_177: "f32[480, 20, 1, 1]", primals_178: "f32[480]", primals_179: "f32[80, 480, 1, 1]", primals_180: "f32[480, 80, 1, 1]", primals_181: "f32[480, 1, 5, 5]", primals_182: "f32[20, 480, 1, 1]", primals_183: "f32[20]", primals_184: "f32[480, 20, 1, 1]", primals_185: "f32[480]", primals_186: "f32[112, 480, 1, 1]", primals_187: "f32[672, 112, 1, 1]", primals_188: "f32[672, 1, 5, 5]", primals_189: "f32[28, 672, 1, 1]", primals_190: "f32[28]", primals_191: "f32[672, 28, 1, 1]", primals_192: "f32[672]", primals_193: "f32[112, 672, 1, 1]", primals_194: "f32[672, 112, 1, 1]", primals_195: "f32[672, 1, 5, 5]", primals_196: "f32[28, 672, 1, 1]", primals_197: "f32[28]", primals_198: "f32[672, 28, 1, 1]", primals_199: "f32[672]", primals_200: "f32[112, 672, 1, 1]", primals_201: "f32[672, 112, 1, 1]", primals_202: "f32[672, 1, 5, 5]", primals_203: "f32[28, 672, 1, 1]", primals_204: "f32[28]", primals_205: "f32[672, 28, 1, 1]", primals_206: "f32[672]", primals_207: "f32[112, 672, 1, 1]", primals_208: "f32[672, 112, 1, 1]", primals_209: "f32[672, 1, 5, 5]", primals_210: "f32[28, 672, 1, 1]", primals_211: "f32[28]", primals_212: "f32[672, 28, 1, 1]", primals_213: "f32[672]", primals_214: "f32[192, 672, 1, 1]", primals_215: "f32[1152, 192, 1, 1]", primals_216: "f32[1152, 1, 5, 5]", primals_217: "f32[48, 1152, 1, 1]", primals_218: "f32[48]", primals_219: "f32[1152, 48, 1, 1]", primals_220: "f32[1152]", primals_221: "f32[192, 1152, 1, 1]", primals_222: "f32[1152, 192, 1, 1]", primals_223: "f32[1152, 1, 5, 5]", primals_224: "f32[48, 1152, 1, 1]", primals_225: "f32[48]", primals_226: "f32[1152, 48, 1, 1]", primals_227: "f32[1152]", primals_228: "f32[192, 1152, 1, 1]", primals_229: "f32[1152, 192, 1, 1]", primals_230: "f32[1152, 1, 5, 5]", primals_231: "f32[48, 1152, 1, 1]", primals_232: "f32[48]", primals_233: "f32[1152, 48, 1, 1]", primals_234: "f32[1152]", primals_235: "f32[192, 1152, 1, 1]", primals_236: "f32[1152, 192, 1, 1]", primals_237: "f32[1152, 1, 5, 5]", primals_238: "f32[48, 1152, 1, 1]", primals_239: "f32[48]", primals_240: "f32[1152, 48, 1, 1]", primals_241: "f32[1152]", primals_242: "f32[192, 1152, 1, 1]", primals_243: "f32[1152, 192, 1, 1]", primals_244: "f32[1152, 1, 3, 3]", primals_245: "f32[48, 1152, 1, 1]", primals_246: "f32[48]", primals_247: "f32[1152, 48, 1, 1]", primals_248: "f32[1152]", primals_249: "f32[320, 1152, 1, 1]", primals_250: "f32[1280, 320, 1, 1]", primals_251: "f32[1000, 1280]", primals_252: "f32[1000]", primals_253: "i64[]", primals_254: "f32[32]", primals_255: "f32[32]", primals_256: "i64[]", primals_257: "f32[32]", primals_258: "f32[32]", primals_259: "i64[]", primals_260: "f32[16]", primals_261: "f32[16]", primals_262: "i64[]", primals_263: "f32[96]", primals_264: "f32[96]", primals_265: "i64[]", primals_266: "f32[96]", primals_267: "f32[96]", primals_268: "i64[]", primals_269: "f32[24]", primals_270: "f32[24]", primals_271: "i64[]", primals_272: "f32[144]", primals_273: "f32[144]", primals_274: "i64[]", primals_275: "f32[144]", primals_276: "f32[144]", primals_277: "i64[]", primals_278: "f32[24]", primals_279: "f32[24]", primals_280: "i64[]", primals_281: "f32[144]", primals_282: "f32[144]", primals_283: "i64[]", primals_284: "f32[144]", primals_285: "f32[144]", primals_286: "i64[]", primals_287: "f32[40]", primals_288: "f32[40]", primals_289: "i64[]", primals_290: "f32[240]", primals_291: "f32[240]", primals_292: "i64[]", primals_293: "f32[240]", primals_294: "f32[240]", primals_295: "i64[]", primals_296: "f32[40]", primals_297: "f32[40]", primals_298: "i64[]", primals_299: "f32[240]", primals_300: "f32[240]", primals_301: "i64[]", primals_302: "f32[240]", primals_303: "f32[240]", primals_304: "i64[]", primals_305: "f32[80]", primals_306: "f32[80]", primals_307: "i64[]", primals_308: "f32[480]", primals_309: "f32[480]", primals_310: "i64[]", primals_311: "f32[480]", primals_312: "f32[480]", primals_313: "i64[]", primals_314: "f32[80]", primals_315: "f32[80]", primals_316: "i64[]", primals_317: "f32[480]", primals_318: "f32[480]", primals_319: "i64[]", primals_320: "f32[480]", primals_321: "f32[480]", primals_322: "i64[]", primals_323: "f32[80]", primals_324: "f32[80]", primals_325: "i64[]", primals_326: "f32[480]", primals_327: "f32[480]", primals_328: "i64[]", primals_329: "f32[480]", primals_330: "f32[480]", primals_331: "i64[]", primals_332: "f32[80]", primals_333: "f32[80]", primals_334: "i64[]", primals_335: "f32[480]", primals_336: "f32[480]", primals_337: "i64[]", primals_338: "f32[480]", primals_339: "f32[480]", primals_340: "i64[]", primals_341: "f32[112]", primals_342: "f32[112]", primals_343: "i64[]", primals_344: "f32[672]", primals_345: "f32[672]", primals_346: "i64[]", primals_347: "f32[672]", primals_348: "f32[672]", primals_349: "i64[]", primals_350: "f32[112]", primals_351: "f32[112]", primals_352: "i64[]", primals_353: "f32[672]", primals_354: "f32[672]", primals_355: "i64[]", primals_356: "f32[672]", primals_357: "f32[672]", primals_358: "i64[]", primals_359: "f32[112]", primals_360: "f32[112]", primals_361: "i64[]", primals_362: "f32[672]", primals_363: "f32[672]", primals_364: "i64[]", primals_365: "f32[672]", primals_366: "f32[672]", primals_367: "i64[]", primals_368: "f32[112]", primals_369: "f32[112]", primals_370: "i64[]", primals_371: "f32[672]", primals_372: "f32[672]", primals_373: "i64[]", primals_374: "f32[672]", primals_375: "f32[672]", primals_376: "i64[]", primals_377: "f32[192]", primals_378: "f32[192]", primals_379: "i64[]", primals_380: "f32[1152]", primals_381: "f32[1152]", primals_382: "i64[]", primals_383: "f32[1152]", primals_384: "f32[1152]", primals_385: "i64[]", primals_386: "f32[192]", primals_387: "f32[192]", primals_388: "i64[]", primals_389: "f32[1152]", primals_390: "f32[1152]", primals_391: "i64[]", primals_392: "f32[1152]", primals_393: "f32[1152]", primals_394: "i64[]", primals_395: "f32[192]", primals_396: "f32[192]", primals_397: "i64[]", primals_398: "f32[1152]", primals_399: "f32[1152]", primals_400: "i64[]", primals_401: "f32[1152]", primals_402: "f32[1152]", primals_403: "i64[]", primals_404: "f32[192]", primals_405: "f32[192]", primals_406: "i64[]", primals_407: "f32[1152]", primals_408: "f32[1152]", primals_409: "i64[]", primals_410: "f32[1152]", primals_411: "f32[1152]", primals_412: "i64[]", primals_413: "f32[192]", primals_414: "f32[192]", primals_415: "i64[]", primals_416: "f32[1152]", primals_417: "f32[1152]", primals_418: "i64[]", primals_419: "f32[1152]", primals_420: "f32[1152]", primals_421: "i64[]", primals_422: "f32[320]", primals_423: "f32[320]", primals_424: "i64[]", primals_425: "f32[1280]", primals_426: "f32[1280]", primals_427: "f32[8, 3, 192, 192]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:162, code: x = self.conv_stem(x)
    convolution: "f32[8, 32, 96, 96]" = torch.ops.aten.convolution.default(primals_427, primals_117, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_253, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
    getitem: "f32[1, 32, 1, 1]" = var_mean[0]
    getitem_1: "f32[1, 32, 1, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
    rsqrt: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 32, 96, 96]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
    mul: "f32[8, 32, 96, 96]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    squeeze: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
    squeeze_1: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
    mul_1: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
    mul_2: "f32[32]" = torch.ops.aten.mul.Tensor(primals_254, 0.9)
    add_2: "f32[32]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[32]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.000013563552023);  squeeze_2 = None
    mul_4: "f32[32]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[32]" = torch.ops.aten.mul.Tensor(primals_255, 0.9)
    add_3: "f32[32]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    unsqueeze: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_1, -1)
    unsqueeze_1: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    mul_6: "f32[8, 32, 96, 96]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
    unsqueeze_2: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1);  primals_2 = None
    unsqueeze_3: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    add_4: "f32[8, 32, 96, 96]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone: "f32[8, 32, 96, 96]" = torch.ops.aten.clone.default(add_4)
    sigmoid: "f32[8, 32, 96, 96]" = torch.ops.aten.sigmoid.default(add_4)
    mul_7: "f32[8, 32, 96, 96]" = torch.ops.aten.mul.Tensor(add_4, sigmoid);  add_4 = sigmoid = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_1: "f32[8, 32, 96, 96]" = torch.ops.aten.convolution.default(mul_7, primals_118, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_256, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_2: "f32[1, 32, 1, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 32, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
    rsqrt_1: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_1: "f32[8, 32, 96, 96]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
    mul_8: "f32[8, 32, 96, 96]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_3: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
    squeeze_4: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
    mul_9: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
    mul_10: "f32[32]" = torch.ops.aten.mul.Tensor(primals_257, 0.9)
    add_7: "f32[32]" = torch.ops.aten.add.Tensor(mul_9, mul_10);  mul_9 = mul_10 = None
    squeeze_5: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
    mul_11: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.000013563552023);  squeeze_5 = None
    mul_12: "f32[32]" = torch.ops.aten.mul.Tensor(mul_11, 0.1);  mul_11 = None
    mul_13: "f32[32]" = torch.ops.aten.mul.Tensor(primals_258, 0.9)
    add_8: "f32[32]" = torch.ops.aten.add.Tensor(mul_12, mul_13);  mul_12 = mul_13 = None
    unsqueeze_4: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1)
    unsqueeze_5: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_14: "f32[8, 32, 96, 96]" = torch.ops.aten.mul.Tensor(mul_8, unsqueeze_5);  mul_8 = unsqueeze_5 = None
    unsqueeze_6: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_4, -1);  primals_4 = None
    unsqueeze_7: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_9: "f32[8, 32, 96, 96]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_7);  mul_14 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_1: "f32[8, 32, 96, 96]" = torch.ops.aten.sigmoid.default(add_9)
    mul_15: "f32[8, 32, 96, 96]" = torch.ops.aten.mul.Tensor(add_9, sigmoid_1);  sigmoid_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean: "f32[8, 32, 1, 1]" = torch.ops.aten.mean.dim(mul_15, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_2: "f32[8, 8, 1, 1]" = torch.ops.aten.convolution.default(mean, primals_119, primals_120, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_2: "f32[8, 8, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_2)
    mul_16: "f32[8, 8, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_2, sigmoid_2);  sigmoid_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_3: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mul_16, primals_121, primals_122, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_3: "f32[8, 32, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_3)
    mul_17: "f32[8, 32, 96, 96]" = torch.ops.aten.mul.Tensor(mul_15, sigmoid_3);  mul_15 = sigmoid_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_4: "f32[8, 16, 96, 96]" = torch.ops.aten.convolution.default(mul_17, primals_123, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_259, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_2 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 16, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 16, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
    rsqrt_2: "f32[1, 16, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_2: "f32[8, 16, 96, 96]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_5)
    mul_18: "f32[8, 16, 96, 96]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_7: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_19: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_20: "f32[16]" = torch.ops.aten.mul.Tensor(primals_260, 0.9)
    add_12: "f32[16]" = torch.ops.aten.add.Tensor(mul_19, mul_20);  mul_19 = mul_20 = None
    squeeze_8: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_21: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.000013563552023);  squeeze_8 = None
    mul_22: "f32[16]" = torch.ops.aten.mul.Tensor(mul_21, 0.1);  mul_21 = None
    mul_23: "f32[16]" = torch.ops.aten.mul.Tensor(primals_261, 0.9)
    add_13: "f32[16]" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
    unsqueeze_8: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_9: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_24: "f32[8, 16, 96, 96]" = torch.ops.aten.mul.Tensor(mul_18, unsqueeze_9);  mul_18 = unsqueeze_9 = None
    unsqueeze_10: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_11: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_14: "f32[8, 16, 96, 96]" = torch.ops.aten.add.Tensor(mul_24, unsqueeze_11);  mul_24 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_5: "f32[8, 96, 96, 96]" = torch.ops.aten.convolution.default(add_14, primals_124, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_15: "i64[]" = torch.ops.aten.add.Tensor(primals_262, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_3 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
    getitem_6: "f32[1, 96, 1, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 96, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_16: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
    rsqrt_3: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_3: "f32[8, 96, 96, 96]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_7)
    mul_25: "f32[8, 96, 96, 96]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_9: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
    squeeze_10: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
    mul_26: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
    mul_27: "f32[96]" = torch.ops.aten.mul.Tensor(primals_263, 0.9)
    add_17: "f32[96]" = torch.ops.aten.add.Tensor(mul_26, mul_27);  mul_26 = mul_27 = None
    squeeze_11: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
    mul_28: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.000013563552023);  squeeze_11 = None
    mul_29: "f32[96]" = torch.ops.aten.mul.Tensor(mul_28, 0.1);  mul_28 = None
    mul_30: "f32[96]" = torch.ops.aten.mul.Tensor(primals_264, 0.9)
    add_18: "f32[96]" = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
    unsqueeze_12: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1)
    unsqueeze_13: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_31: "f32[8, 96, 96, 96]" = torch.ops.aten.mul.Tensor(mul_25, unsqueeze_13);  mul_25 = unsqueeze_13 = None
    unsqueeze_14: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1);  primals_8 = None
    unsqueeze_15: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_19: "f32[8, 96, 96, 96]" = torch.ops.aten.add.Tensor(mul_31, unsqueeze_15);  mul_31 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_3: "f32[8, 96, 96, 96]" = torch.ops.aten.clone.default(add_19)
    sigmoid_4: "f32[8, 96, 96, 96]" = torch.ops.aten.sigmoid.default(add_19)
    mul_32: "f32[8, 96, 96, 96]" = torch.ops.aten.mul.Tensor(add_19, sigmoid_4);  add_19 = sigmoid_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_6: "f32[8, 96, 48, 48]" = torch.ops.aten.convolution.default(mul_32, primals_125, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 96)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_20: "i64[]" = torch.ops.aten.add.Tensor(primals_265, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_4 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
    getitem_8: "f32[1, 96, 1, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 96, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_21: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
    rsqrt_4: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_4: "f32[8, 96, 48, 48]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_9)
    mul_33: "f32[8, 96, 48, 48]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_12: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
    squeeze_13: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
    mul_34: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
    mul_35: "f32[96]" = torch.ops.aten.mul.Tensor(primals_266, 0.9)
    add_22: "f32[96]" = torch.ops.aten.add.Tensor(mul_34, mul_35);  mul_34 = mul_35 = None
    squeeze_14: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
    mul_36: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0000542564158212);  squeeze_14 = None
    mul_37: "f32[96]" = torch.ops.aten.mul.Tensor(mul_36, 0.1);  mul_36 = None
    mul_38: "f32[96]" = torch.ops.aten.mul.Tensor(primals_267, 0.9)
    add_23: "f32[96]" = torch.ops.aten.add.Tensor(mul_37, mul_38);  mul_37 = mul_38 = None
    unsqueeze_16: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1)
    unsqueeze_17: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    mul_39: "f32[8, 96, 48, 48]" = torch.ops.aten.mul.Tensor(mul_33, unsqueeze_17);  mul_33 = unsqueeze_17 = None
    unsqueeze_18: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_10, -1);  primals_10 = None
    unsqueeze_19: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    add_24: "f32[8, 96, 48, 48]" = torch.ops.aten.add.Tensor(mul_39, unsqueeze_19);  mul_39 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_5: "f32[8, 96, 48, 48]" = torch.ops.aten.sigmoid.default(add_24)
    mul_40: "f32[8, 96, 48, 48]" = torch.ops.aten.mul.Tensor(add_24, sigmoid_5);  sigmoid_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_1: "f32[8, 96, 1, 1]" = torch.ops.aten.mean.dim(mul_40, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_7: "f32[8, 4, 1, 1]" = torch.ops.aten.convolution.default(mean_1, primals_126, primals_127, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_6: "f32[8, 4, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_7)
    mul_41: "f32[8, 4, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_7, sigmoid_6);  sigmoid_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_8: "f32[8, 96, 1, 1]" = torch.ops.aten.convolution.default(mul_41, primals_128, primals_129, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_7: "f32[8, 96, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_8)
    mul_42: "f32[8, 96, 48, 48]" = torch.ops.aten.mul.Tensor(mul_40, sigmoid_7);  mul_40 = sigmoid_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_9: "f32[8, 24, 48, 48]" = torch.ops.aten.convolution.default(mul_42, primals_130, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_25: "i64[]" = torch.ops.aten.add.Tensor(primals_268, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_5 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
    getitem_10: "f32[1, 24, 1, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 24, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_26: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
    rsqrt_5: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    sub_5: "f32[8, 24, 48, 48]" = torch.ops.aten.sub.Tensor(convolution_9, getitem_11)
    mul_43: "f32[8, 24, 48, 48]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    squeeze_15: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
    squeeze_16: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
    mul_44: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
    mul_45: "f32[24]" = torch.ops.aten.mul.Tensor(primals_269, 0.9)
    add_27: "f32[24]" = torch.ops.aten.add.Tensor(mul_44, mul_45);  mul_44 = mul_45 = None
    squeeze_17: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
    mul_46: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0000542564158212);  squeeze_17 = None
    mul_47: "f32[24]" = torch.ops.aten.mul.Tensor(mul_46, 0.1);  mul_46 = None
    mul_48: "f32[24]" = torch.ops.aten.mul.Tensor(primals_270, 0.9)
    add_28: "f32[24]" = torch.ops.aten.add.Tensor(mul_47, mul_48);  mul_47 = mul_48 = None
    unsqueeze_20: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_21: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_49: "f32[8, 24, 48, 48]" = torch.ops.aten.mul.Tensor(mul_43, unsqueeze_21);  mul_43 = unsqueeze_21 = None
    unsqueeze_22: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_23: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_29: "f32[8, 24, 48, 48]" = torch.ops.aten.add.Tensor(mul_49, unsqueeze_23);  mul_49 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_10: "f32[8, 144, 48, 48]" = torch.ops.aten.convolution.default(add_29, primals_131, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_30: "i64[]" = torch.ops.aten.add.Tensor(primals_271, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_6 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
    getitem_12: "f32[1, 144, 1, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 144, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_31: "f32[1, 144, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
    rsqrt_6: "f32[1, 144, 1, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    sub_6: "f32[8, 144, 48, 48]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_13)
    mul_50: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    squeeze_18: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
    squeeze_19: "f32[144]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
    mul_51: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
    mul_52: "f32[144]" = torch.ops.aten.mul.Tensor(primals_272, 0.9)
    add_32: "f32[144]" = torch.ops.aten.add.Tensor(mul_51, mul_52);  mul_51 = mul_52 = None
    squeeze_20: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
    mul_53: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0000542564158212);  squeeze_20 = None
    mul_54: "f32[144]" = torch.ops.aten.mul.Tensor(mul_53, 0.1);  mul_53 = None
    mul_55: "f32[144]" = torch.ops.aten.mul.Tensor(primals_273, 0.9)
    add_33: "f32[144]" = torch.ops.aten.add.Tensor(mul_54, mul_55);  mul_54 = mul_55 = None
    unsqueeze_24: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1)
    unsqueeze_25: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    mul_56: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(mul_50, unsqueeze_25);  mul_50 = unsqueeze_25 = None
    unsqueeze_26: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1);  primals_14 = None
    unsqueeze_27: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    add_34: "f32[8, 144, 48, 48]" = torch.ops.aten.add.Tensor(mul_56, unsqueeze_27);  mul_56 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_6: "f32[8, 144, 48, 48]" = torch.ops.aten.clone.default(add_34)
    sigmoid_8: "f32[8, 144, 48, 48]" = torch.ops.aten.sigmoid.default(add_34)
    mul_57: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(add_34, sigmoid_8);  add_34 = sigmoid_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_11: "f32[8, 144, 48, 48]" = torch.ops.aten.convolution.default(mul_57, primals_132, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 144)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_35: "i64[]" = torch.ops.aten.add.Tensor(primals_274, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_7 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
    getitem_14: "f32[1, 144, 1, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 144, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_36: "f32[1, 144, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
    rsqrt_7: "f32[1, 144, 1, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_7: "f32[8, 144, 48, 48]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_15)
    mul_58: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    squeeze_21: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
    squeeze_22: "f32[144]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
    mul_59: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
    mul_60: "f32[144]" = torch.ops.aten.mul.Tensor(primals_275, 0.9)
    add_37: "f32[144]" = torch.ops.aten.add.Tensor(mul_59, mul_60);  mul_59 = mul_60 = None
    squeeze_23: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
    mul_61: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0000542564158212);  squeeze_23 = None
    mul_62: "f32[144]" = torch.ops.aten.mul.Tensor(mul_61, 0.1);  mul_61 = None
    mul_63: "f32[144]" = torch.ops.aten.mul.Tensor(primals_276, 0.9)
    add_38: "f32[144]" = torch.ops.aten.add.Tensor(mul_62, mul_63);  mul_62 = mul_63 = None
    unsqueeze_28: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1)
    unsqueeze_29: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_64: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(mul_58, unsqueeze_29);  mul_58 = unsqueeze_29 = None
    unsqueeze_30: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_16, -1);  primals_16 = None
    unsqueeze_31: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_39: "f32[8, 144, 48, 48]" = torch.ops.aten.add.Tensor(mul_64, unsqueeze_31);  mul_64 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_9: "f32[8, 144, 48, 48]" = torch.ops.aten.sigmoid.default(add_39)
    mul_65: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(add_39, sigmoid_9);  sigmoid_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_2: "f32[8, 144, 1, 1]" = torch.ops.aten.mean.dim(mul_65, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_12: "f32[8, 6, 1, 1]" = torch.ops.aten.convolution.default(mean_2, primals_133, primals_134, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_10: "f32[8, 6, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_12)
    mul_66: "f32[8, 6, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_12, sigmoid_10);  sigmoid_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_13: "f32[8, 144, 1, 1]" = torch.ops.aten.convolution.default(mul_66, primals_135, primals_136, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_11: "f32[8, 144, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_13)
    mul_67: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(mul_65, sigmoid_11);  mul_65 = sigmoid_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_14: "f32[8, 24, 48, 48]" = torch.ops.aten.convolution.default(mul_67, primals_137, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_40: "i64[]" = torch.ops.aten.add.Tensor(primals_277, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_8 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
    getitem_16: "f32[1, 24, 1, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 24, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_41: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
    rsqrt_8: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
    sub_8: "f32[8, 24, 48, 48]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_17)
    mul_68: "f32[8, 24, 48, 48]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    squeeze_24: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
    squeeze_25: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
    mul_69: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
    mul_70: "f32[24]" = torch.ops.aten.mul.Tensor(primals_278, 0.9)
    add_42: "f32[24]" = torch.ops.aten.add.Tensor(mul_69, mul_70);  mul_69 = mul_70 = None
    squeeze_26: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
    mul_71: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0000542564158212);  squeeze_26 = None
    mul_72: "f32[24]" = torch.ops.aten.mul.Tensor(mul_71, 0.1);  mul_71 = None
    mul_73: "f32[24]" = torch.ops.aten.mul.Tensor(primals_279, 0.9)
    add_43: "f32[24]" = torch.ops.aten.add.Tensor(mul_72, mul_73);  mul_72 = mul_73 = None
    unsqueeze_32: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_33: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    mul_74: "f32[8, 24, 48, 48]" = torch.ops.aten.mul.Tensor(mul_68, unsqueeze_33);  mul_68 = unsqueeze_33 = None
    unsqueeze_34: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_35: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    add_44: "f32[8, 24, 48, 48]" = torch.ops.aten.add.Tensor(mul_74, unsqueeze_35);  mul_74 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_45: "f32[8, 24, 48, 48]" = torch.ops.aten.add.Tensor(add_44, add_29);  add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_15: "f32[8, 144, 48, 48]" = torch.ops.aten.convolution.default(add_45, primals_138, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_46: "i64[]" = torch.ops.aten.add.Tensor(primals_280, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_9 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
    getitem_18: "f32[1, 144, 1, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 144, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_47: "f32[1, 144, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
    rsqrt_9: "f32[1, 144, 1, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    sub_9: "f32[8, 144, 48, 48]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_19)
    mul_75: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    squeeze_27: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
    squeeze_28: "f32[144]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
    mul_76: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
    mul_77: "f32[144]" = torch.ops.aten.mul.Tensor(primals_281, 0.9)
    add_48: "f32[144]" = torch.ops.aten.add.Tensor(mul_76, mul_77);  mul_76 = mul_77 = None
    squeeze_29: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
    mul_78: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0000542564158212);  squeeze_29 = None
    mul_79: "f32[144]" = torch.ops.aten.mul.Tensor(mul_78, 0.1);  mul_78 = None
    mul_80: "f32[144]" = torch.ops.aten.mul.Tensor(primals_282, 0.9)
    add_49: "f32[144]" = torch.ops.aten.add.Tensor(mul_79, mul_80);  mul_79 = mul_80 = None
    unsqueeze_36: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1)
    unsqueeze_37: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_81: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(mul_75, unsqueeze_37);  mul_75 = unsqueeze_37 = None
    unsqueeze_38: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1);  primals_20 = None
    unsqueeze_39: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_50: "f32[8, 144, 48, 48]" = torch.ops.aten.add.Tensor(mul_81, unsqueeze_39);  mul_81 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_9: "f32[8, 144, 48, 48]" = torch.ops.aten.clone.default(add_50)
    sigmoid_12: "f32[8, 144, 48, 48]" = torch.ops.aten.sigmoid.default(add_50)
    mul_82: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(add_50, sigmoid_12);  add_50 = sigmoid_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_16: "f32[8, 144, 24, 24]" = torch.ops.aten.convolution.default(mul_82, primals_139, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 144)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_51: "i64[]" = torch.ops.aten.add.Tensor(primals_283, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_10 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
    getitem_20: "f32[1, 144, 1, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 144, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_52: "f32[1, 144, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
    rsqrt_10: "f32[1, 144, 1, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_10: "f32[8, 144, 24, 24]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_21)
    mul_83: "f32[8, 144, 24, 24]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    squeeze_30: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
    squeeze_31: "f32[144]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
    mul_84: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
    mul_85: "f32[144]" = torch.ops.aten.mul.Tensor(primals_284, 0.9)
    add_53: "f32[144]" = torch.ops.aten.add.Tensor(mul_84, mul_85);  mul_84 = mul_85 = None
    squeeze_32: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
    mul_86: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0002170609941394);  squeeze_32 = None
    mul_87: "f32[144]" = torch.ops.aten.mul.Tensor(mul_86, 0.1);  mul_86 = None
    mul_88: "f32[144]" = torch.ops.aten.mul.Tensor(primals_285, 0.9)
    add_54: "f32[144]" = torch.ops.aten.add.Tensor(mul_87, mul_88);  mul_87 = mul_88 = None
    unsqueeze_40: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1)
    unsqueeze_41: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    mul_89: "f32[8, 144, 24, 24]" = torch.ops.aten.mul.Tensor(mul_83, unsqueeze_41);  mul_83 = unsqueeze_41 = None
    unsqueeze_42: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_22, -1);  primals_22 = None
    unsqueeze_43: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    add_55: "f32[8, 144, 24, 24]" = torch.ops.aten.add.Tensor(mul_89, unsqueeze_43);  mul_89 = unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_13: "f32[8, 144, 24, 24]" = torch.ops.aten.sigmoid.default(add_55)
    mul_90: "f32[8, 144, 24, 24]" = torch.ops.aten.mul.Tensor(add_55, sigmoid_13);  sigmoid_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_3: "f32[8, 144, 1, 1]" = torch.ops.aten.mean.dim(mul_90, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_17: "f32[8, 6, 1, 1]" = torch.ops.aten.convolution.default(mean_3, primals_140, primals_141, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_14: "f32[8, 6, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_17)
    mul_91: "f32[8, 6, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_17, sigmoid_14);  sigmoid_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_18: "f32[8, 144, 1, 1]" = torch.ops.aten.convolution.default(mul_91, primals_142, primals_143, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_15: "f32[8, 144, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_18)
    mul_92: "f32[8, 144, 24, 24]" = torch.ops.aten.mul.Tensor(mul_90, sigmoid_15);  mul_90 = sigmoid_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_19: "f32[8, 40, 24, 24]" = torch.ops.aten.convolution.default(mul_92, primals_144, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_56: "i64[]" = torch.ops.aten.add.Tensor(primals_286, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_11 = torch.ops.aten.var_mean.correction(convolution_19, [0, 2, 3], correction = 0, keepdim = True)
    getitem_22: "f32[1, 40, 1, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 40, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_57: "f32[1, 40, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
    rsqrt_11: "f32[1, 40, 1, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
    sub_11: "f32[8, 40, 24, 24]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_23)
    mul_93: "f32[8, 40, 24, 24]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    squeeze_33: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
    squeeze_34: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
    mul_94: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
    mul_95: "f32[40]" = torch.ops.aten.mul.Tensor(primals_287, 0.9)
    add_58: "f32[40]" = torch.ops.aten.add.Tensor(mul_94, mul_95);  mul_94 = mul_95 = None
    squeeze_35: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
    mul_96: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0002170609941394);  squeeze_35 = None
    mul_97: "f32[40]" = torch.ops.aten.mul.Tensor(mul_96, 0.1);  mul_96 = None
    mul_98: "f32[40]" = torch.ops.aten.mul.Tensor(primals_288, 0.9)
    add_59: "f32[40]" = torch.ops.aten.add.Tensor(mul_97, mul_98);  mul_97 = mul_98 = None
    unsqueeze_44: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_45: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_99: "f32[8, 40, 24, 24]" = torch.ops.aten.mul.Tensor(mul_93, unsqueeze_45);  mul_93 = unsqueeze_45 = None
    unsqueeze_46: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_47: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_60: "f32[8, 40, 24, 24]" = torch.ops.aten.add.Tensor(mul_99, unsqueeze_47);  mul_99 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_20: "f32[8, 240, 24, 24]" = torch.ops.aten.convolution.default(add_60, primals_145, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_61: "i64[]" = torch.ops.aten.add.Tensor(primals_289, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_12 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
    getitem_24: "f32[1, 240, 1, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 240, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_62: "f32[1, 240, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
    rsqrt_12: "f32[1, 240, 1, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    sub_12: "f32[8, 240, 24, 24]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_25)
    mul_100: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    squeeze_36: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
    squeeze_37: "f32[240]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
    mul_101: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
    mul_102: "f32[240]" = torch.ops.aten.mul.Tensor(primals_290, 0.9)
    add_63: "f32[240]" = torch.ops.aten.add.Tensor(mul_101, mul_102);  mul_101 = mul_102 = None
    squeeze_38: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
    mul_103: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0002170609941394);  squeeze_38 = None
    mul_104: "f32[240]" = torch.ops.aten.mul.Tensor(mul_103, 0.1);  mul_103 = None
    mul_105: "f32[240]" = torch.ops.aten.mul.Tensor(primals_291, 0.9)
    add_64: "f32[240]" = torch.ops.aten.add.Tensor(mul_104, mul_105);  mul_104 = mul_105 = None
    unsqueeze_48: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_25, -1)
    unsqueeze_49: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    mul_106: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(mul_100, unsqueeze_49);  mul_100 = unsqueeze_49 = None
    unsqueeze_50: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1);  primals_26 = None
    unsqueeze_51: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    add_65: "f32[8, 240, 24, 24]" = torch.ops.aten.add.Tensor(mul_106, unsqueeze_51);  mul_106 = unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_12: "f32[8, 240, 24, 24]" = torch.ops.aten.clone.default(add_65)
    sigmoid_16: "f32[8, 240, 24, 24]" = torch.ops.aten.sigmoid.default(add_65)
    mul_107: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(add_65, sigmoid_16);  add_65 = sigmoid_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_21: "f32[8, 240, 24, 24]" = torch.ops.aten.convolution.default(mul_107, primals_146, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 240)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_66: "i64[]" = torch.ops.aten.add.Tensor(primals_292, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_13 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
    getitem_26: "f32[1, 240, 1, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 240, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_67: "f32[1, 240, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
    rsqrt_13: "f32[1, 240, 1, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    sub_13: "f32[8, 240, 24, 24]" = torch.ops.aten.sub.Tensor(convolution_21, getitem_27)
    mul_108: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    squeeze_39: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
    squeeze_40: "f32[240]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
    mul_109: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
    mul_110: "f32[240]" = torch.ops.aten.mul.Tensor(primals_293, 0.9)
    add_68: "f32[240]" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
    squeeze_41: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
    mul_111: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0002170609941394);  squeeze_41 = None
    mul_112: "f32[240]" = torch.ops.aten.mul.Tensor(mul_111, 0.1);  mul_111 = None
    mul_113: "f32[240]" = torch.ops.aten.mul.Tensor(primals_294, 0.9)
    add_69: "f32[240]" = torch.ops.aten.add.Tensor(mul_112, mul_113);  mul_112 = mul_113 = None
    unsqueeze_52: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1)
    unsqueeze_53: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_114: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(mul_108, unsqueeze_53);  mul_108 = unsqueeze_53 = None
    unsqueeze_54: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_28, -1);  primals_28 = None
    unsqueeze_55: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_70: "f32[8, 240, 24, 24]" = torch.ops.aten.add.Tensor(mul_114, unsqueeze_55);  mul_114 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_17: "f32[8, 240, 24, 24]" = torch.ops.aten.sigmoid.default(add_70)
    mul_115: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(add_70, sigmoid_17);  sigmoid_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_4: "f32[8, 240, 1, 1]" = torch.ops.aten.mean.dim(mul_115, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_22: "f32[8, 10, 1, 1]" = torch.ops.aten.convolution.default(mean_4, primals_147, primals_148, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_18: "f32[8, 10, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_22)
    mul_116: "f32[8, 10, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_22, sigmoid_18);  sigmoid_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_23: "f32[8, 240, 1, 1]" = torch.ops.aten.convolution.default(mul_116, primals_149, primals_150, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_19: "f32[8, 240, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_23)
    mul_117: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(mul_115, sigmoid_19);  mul_115 = sigmoid_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_24: "f32[8, 40, 24, 24]" = torch.ops.aten.convolution.default(mul_117, primals_151, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_71: "i64[]" = torch.ops.aten.add.Tensor(primals_295, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_14 = torch.ops.aten.var_mean.correction(convolution_24, [0, 2, 3], correction = 0, keepdim = True)
    getitem_28: "f32[1, 40, 1, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 40, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_72: "f32[1, 40, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
    rsqrt_14: "f32[1, 40, 1, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
    sub_14: "f32[8, 40, 24, 24]" = torch.ops.aten.sub.Tensor(convolution_24, getitem_29)
    mul_118: "f32[8, 40, 24, 24]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    squeeze_42: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
    squeeze_43: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
    mul_119: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
    mul_120: "f32[40]" = torch.ops.aten.mul.Tensor(primals_296, 0.9)
    add_73: "f32[40]" = torch.ops.aten.add.Tensor(mul_119, mul_120);  mul_119 = mul_120 = None
    squeeze_44: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
    mul_121: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0002170609941394);  squeeze_44 = None
    mul_122: "f32[40]" = torch.ops.aten.mul.Tensor(mul_121, 0.1);  mul_121 = None
    mul_123: "f32[40]" = torch.ops.aten.mul.Tensor(primals_297, 0.9)
    add_74: "f32[40]" = torch.ops.aten.add.Tensor(mul_122, mul_123);  mul_122 = mul_123 = None
    unsqueeze_56: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_57: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    mul_124: "f32[8, 40, 24, 24]" = torch.ops.aten.mul.Tensor(mul_118, unsqueeze_57);  mul_118 = unsqueeze_57 = None
    unsqueeze_58: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_59: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    add_75: "f32[8, 40, 24, 24]" = torch.ops.aten.add.Tensor(mul_124, unsqueeze_59);  mul_124 = unsqueeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_76: "f32[8, 40, 24, 24]" = torch.ops.aten.add.Tensor(add_75, add_60);  add_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_25: "f32[8, 240, 24, 24]" = torch.ops.aten.convolution.default(add_76, primals_152, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_77: "i64[]" = torch.ops.aten.add.Tensor(primals_298, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_15 = torch.ops.aten.var_mean.correction(convolution_25, [0, 2, 3], correction = 0, keepdim = True)
    getitem_30: "f32[1, 240, 1, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 240, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_78: "f32[1, 240, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
    rsqrt_15: "f32[1, 240, 1, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    sub_15: "f32[8, 240, 24, 24]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_31)
    mul_125: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    squeeze_45: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
    squeeze_46: "f32[240]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
    mul_126: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
    mul_127: "f32[240]" = torch.ops.aten.mul.Tensor(primals_299, 0.9)
    add_79: "f32[240]" = torch.ops.aten.add.Tensor(mul_126, mul_127);  mul_126 = mul_127 = None
    squeeze_47: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
    mul_128: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0002170609941394);  squeeze_47 = None
    mul_129: "f32[240]" = torch.ops.aten.mul.Tensor(mul_128, 0.1);  mul_128 = None
    mul_130: "f32[240]" = torch.ops.aten.mul.Tensor(primals_300, 0.9)
    add_80: "f32[240]" = torch.ops.aten.add.Tensor(mul_129, mul_130);  mul_129 = mul_130 = None
    unsqueeze_60: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1)
    unsqueeze_61: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_131: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(mul_125, unsqueeze_61);  mul_125 = unsqueeze_61 = None
    unsqueeze_62: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1);  primals_32 = None
    unsqueeze_63: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_81: "f32[8, 240, 24, 24]" = torch.ops.aten.add.Tensor(mul_131, unsqueeze_63);  mul_131 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_15: "f32[8, 240, 24, 24]" = torch.ops.aten.clone.default(add_81)
    sigmoid_20: "f32[8, 240, 24, 24]" = torch.ops.aten.sigmoid.default(add_81)
    mul_132: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(add_81, sigmoid_20);  add_81 = sigmoid_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_26: "f32[8, 240, 12, 12]" = torch.ops.aten.convolution.default(mul_132, primals_153, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 240)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_82: "i64[]" = torch.ops.aten.add.Tensor(primals_301, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_16 = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
    getitem_32: "f32[1, 240, 1, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 240, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_83: "f32[1, 240, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
    rsqrt_16: "f32[1, 240, 1, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
    sub_16: "f32[8, 240, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_33)
    mul_133: "f32[8, 240, 12, 12]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    squeeze_48: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
    squeeze_49: "f32[240]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
    mul_134: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
    mul_135: "f32[240]" = torch.ops.aten.mul.Tensor(primals_302, 0.9)
    add_84: "f32[240]" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
    squeeze_50: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
    mul_136: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.000868809730669);  squeeze_50 = None
    mul_137: "f32[240]" = torch.ops.aten.mul.Tensor(mul_136, 0.1);  mul_136 = None
    mul_138: "f32[240]" = torch.ops.aten.mul.Tensor(primals_303, 0.9)
    add_85: "f32[240]" = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
    unsqueeze_64: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1)
    unsqueeze_65: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    mul_139: "f32[8, 240, 12, 12]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_65);  mul_133 = unsqueeze_65 = None
    unsqueeze_66: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_34, -1);  primals_34 = None
    unsqueeze_67: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    add_86: "f32[8, 240, 12, 12]" = torch.ops.aten.add.Tensor(mul_139, unsqueeze_67);  mul_139 = unsqueeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_21: "f32[8, 240, 12, 12]" = torch.ops.aten.sigmoid.default(add_86)
    mul_140: "f32[8, 240, 12, 12]" = torch.ops.aten.mul.Tensor(add_86, sigmoid_21);  sigmoid_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_5: "f32[8, 240, 1, 1]" = torch.ops.aten.mean.dim(mul_140, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_27: "f32[8, 10, 1, 1]" = torch.ops.aten.convolution.default(mean_5, primals_154, primals_155, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_22: "f32[8, 10, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_27)
    mul_141: "f32[8, 10, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_27, sigmoid_22);  sigmoid_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_28: "f32[8, 240, 1, 1]" = torch.ops.aten.convolution.default(mul_141, primals_156, primals_157, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_23: "f32[8, 240, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_28)
    mul_142: "f32[8, 240, 12, 12]" = torch.ops.aten.mul.Tensor(mul_140, sigmoid_23);  mul_140 = sigmoid_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_29: "f32[8, 80, 12, 12]" = torch.ops.aten.convolution.default(mul_142, primals_158, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_87: "i64[]" = torch.ops.aten.add.Tensor(primals_304, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_17 = torch.ops.aten.var_mean.correction(convolution_29, [0, 2, 3], correction = 0, keepdim = True)
    getitem_34: "f32[1, 80, 1, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 80, 1, 1]" = var_mean_17[1];  var_mean_17 = None
    add_88: "f32[1, 80, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
    rsqrt_17: "f32[1, 80, 1, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
    sub_17: "f32[8, 80, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_29, getitem_35)
    mul_143: "f32[8, 80, 12, 12]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    squeeze_51: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
    squeeze_52: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
    mul_144: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
    mul_145: "f32[80]" = torch.ops.aten.mul.Tensor(primals_305, 0.9)
    add_89: "f32[80]" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
    squeeze_53: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
    mul_146: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.000868809730669);  squeeze_53 = None
    mul_147: "f32[80]" = torch.ops.aten.mul.Tensor(mul_146, 0.1);  mul_146 = None
    mul_148: "f32[80]" = torch.ops.aten.mul.Tensor(primals_306, 0.9)
    add_90: "f32[80]" = torch.ops.aten.add.Tensor(mul_147, mul_148);  mul_147 = mul_148 = None
    unsqueeze_68: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_69: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_149: "f32[8, 80, 12, 12]" = torch.ops.aten.mul.Tensor(mul_143, unsqueeze_69);  mul_143 = unsqueeze_69 = None
    unsqueeze_70: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_71: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_91: "f32[8, 80, 12, 12]" = torch.ops.aten.add.Tensor(mul_149, unsqueeze_71);  mul_149 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_30: "f32[8, 480, 12, 12]" = torch.ops.aten.convolution.default(add_91, primals_159, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_92: "i64[]" = torch.ops.aten.add.Tensor(primals_307, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_18 = torch.ops.aten.var_mean.correction(convolution_30, [0, 2, 3], correction = 0, keepdim = True)
    getitem_36: "f32[1, 480, 1, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 480, 1, 1]" = var_mean_18[1];  var_mean_18 = None
    add_93: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
    rsqrt_18: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
    sub_18: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_30, getitem_37)
    mul_150: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    squeeze_54: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
    squeeze_55: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
    mul_151: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
    mul_152: "f32[480]" = torch.ops.aten.mul.Tensor(primals_308, 0.9)
    add_94: "f32[480]" = torch.ops.aten.add.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
    squeeze_56: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
    mul_153: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.000868809730669);  squeeze_56 = None
    mul_154: "f32[480]" = torch.ops.aten.mul.Tensor(mul_153, 0.1);  mul_153 = None
    mul_155: "f32[480]" = torch.ops.aten.mul.Tensor(primals_309, 0.9)
    add_95: "f32[480]" = torch.ops.aten.add.Tensor(mul_154, mul_155);  mul_154 = mul_155 = None
    unsqueeze_72: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1)
    unsqueeze_73: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    mul_156: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(mul_150, unsqueeze_73);  mul_150 = unsqueeze_73 = None
    unsqueeze_74: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1);  primals_38 = None
    unsqueeze_75: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    add_96: "f32[8, 480, 12, 12]" = torch.ops.aten.add.Tensor(mul_156, unsqueeze_75);  mul_156 = unsqueeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_18: "f32[8, 480, 12, 12]" = torch.ops.aten.clone.default(add_96)
    sigmoid_24: "f32[8, 480, 12, 12]" = torch.ops.aten.sigmoid.default(add_96)
    mul_157: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(add_96, sigmoid_24);  add_96 = sigmoid_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_31: "f32[8, 480, 12, 12]" = torch.ops.aten.convolution.default(mul_157, primals_160, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_97: "i64[]" = torch.ops.aten.add.Tensor(primals_310, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_19 = torch.ops.aten.var_mean.correction(convolution_31, [0, 2, 3], correction = 0, keepdim = True)
    getitem_38: "f32[1, 480, 1, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 480, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_98: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05)
    rsqrt_19: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    sub_19: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_31, getitem_39)
    mul_158: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    squeeze_57: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
    squeeze_58: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
    mul_159: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
    mul_160: "f32[480]" = torch.ops.aten.mul.Tensor(primals_311, 0.9)
    add_99: "f32[480]" = torch.ops.aten.add.Tensor(mul_159, mul_160);  mul_159 = mul_160 = None
    squeeze_59: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
    mul_161: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.000868809730669);  squeeze_59 = None
    mul_162: "f32[480]" = torch.ops.aten.mul.Tensor(mul_161, 0.1);  mul_161 = None
    mul_163: "f32[480]" = torch.ops.aten.mul.Tensor(primals_312, 0.9)
    add_100: "f32[480]" = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
    unsqueeze_76: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1)
    unsqueeze_77: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_164: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(mul_158, unsqueeze_77);  mul_158 = unsqueeze_77 = None
    unsqueeze_78: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_40, -1);  primals_40 = None
    unsqueeze_79: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_101: "f32[8, 480, 12, 12]" = torch.ops.aten.add.Tensor(mul_164, unsqueeze_79);  mul_164 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_25: "f32[8, 480, 12, 12]" = torch.ops.aten.sigmoid.default(add_101)
    mul_165: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(add_101, sigmoid_25);  sigmoid_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_6: "f32[8, 480, 1, 1]" = torch.ops.aten.mean.dim(mul_165, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_32: "f32[8, 20, 1, 1]" = torch.ops.aten.convolution.default(mean_6, primals_161, primals_162, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_26: "f32[8, 20, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_32)
    mul_166: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_32, sigmoid_26);  sigmoid_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_33: "f32[8, 480, 1, 1]" = torch.ops.aten.convolution.default(mul_166, primals_163, primals_164, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_27: "f32[8, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_33)
    mul_167: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(mul_165, sigmoid_27);  mul_165 = sigmoid_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_34: "f32[8, 80, 12, 12]" = torch.ops.aten.convolution.default(mul_167, primals_165, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_102: "i64[]" = torch.ops.aten.add.Tensor(primals_313, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_20 = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
    getitem_40: "f32[1, 80, 1, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 80, 1, 1]" = var_mean_20[1];  var_mean_20 = None
    add_103: "f32[1, 80, 1, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05)
    rsqrt_20: "f32[1, 80, 1, 1]" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
    sub_20: "f32[8, 80, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_34, getitem_41)
    mul_168: "f32[8, 80, 12, 12]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    squeeze_60: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
    squeeze_61: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
    mul_169: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
    mul_170: "f32[80]" = torch.ops.aten.mul.Tensor(primals_314, 0.9)
    add_104: "f32[80]" = torch.ops.aten.add.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None
    squeeze_62: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
    mul_171: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.000868809730669);  squeeze_62 = None
    mul_172: "f32[80]" = torch.ops.aten.mul.Tensor(mul_171, 0.1);  mul_171 = None
    mul_173: "f32[80]" = torch.ops.aten.mul.Tensor(primals_315, 0.9)
    add_105: "f32[80]" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
    unsqueeze_80: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_81: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    mul_174: "f32[8, 80, 12, 12]" = torch.ops.aten.mul.Tensor(mul_168, unsqueeze_81);  mul_168 = unsqueeze_81 = None
    unsqueeze_82: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_83: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    add_106: "f32[8, 80, 12, 12]" = torch.ops.aten.add.Tensor(mul_174, unsqueeze_83);  mul_174 = unsqueeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_107: "f32[8, 80, 12, 12]" = torch.ops.aten.add.Tensor(add_106, add_91);  add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_35: "f32[8, 480, 12, 12]" = torch.ops.aten.convolution.default(add_107, primals_166, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_108: "i64[]" = torch.ops.aten.add.Tensor(primals_316, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_21 = torch.ops.aten.var_mean.correction(convolution_35, [0, 2, 3], correction = 0, keepdim = True)
    getitem_42: "f32[1, 480, 1, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 480, 1, 1]" = var_mean_21[1];  var_mean_21 = None
    add_109: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05)
    rsqrt_21: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
    sub_21: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_35, getitem_43)
    mul_175: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    squeeze_63: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
    squeeze_64: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
    mul_176: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
    mul_177: "f32[480]" = torch.ops.aten.mul.Tensor(primals_317, 0.9)
    add_110: "f32[480]" = torch.ops.aten.add.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None
    squeeze_65: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
    mul_178: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.000868809730669);  squeeze_65 = None
    mul_179: "f32[480]" = torch.ops.aten.mul.Tensor(mul_178, 0.1);  mul_178 = None
    mul_180: "f32[480]" = torch.ops.aten.mul.Tensor(primals_318, 0.9)
    add_111: "f32[480]" = torch.ops.aten.add.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
    unsqueeze_84: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1)
    unsqueeze_85: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_181: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_85);  mul_175 = unsqueeze_85 = None
    unsqueeze_86: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1);  primals_44 = None
    unsqueeze_87: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_112: "f32[8, 480, 12, 12]" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_87);  mul_181 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_21: "f32[8, 480, 12, 12]" = torch.ops.aten.clone.default(add_112)
    sigmoid_28: "f32[8, 480, 12, 12]" = torch.ops.aten.sigmoid.default(add_112)
    mul_182: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(add_112, sigmoid_28);  add_112 = sigmoid_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_36: "f32[8, 480, 12, 12]" = torch.ops.aten.convolution.default(mul_182, primals_167, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_113: "i64[]" = torch.ops.aten.add.Tensor(primals_319, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_22 = torch.ops.aten.var_mean.correction(convolution_36, [0, 2, 3], correction = 0, keepdim = True)
    getitem_44: "f32[1, 480, 1, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 480, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_114: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05)
    rsqrt_22: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
    sub_22: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_36, getitem_45)
    mul_183: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    squeeze_66: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
    squeeze_67: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
    mul_184: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
    mul_185: "f32[480]" = torch.ops.aten.mul.Tensor(primals_320, 0.9)
    add_115: "f32[480]" = torch.ops.aten.add.Tensor(mul_184, mul_185);  mul_184 = mul_185 = None
    squeeze_68: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
    mul_186: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.000868809730669);  squeeze_68 = None
    mul_187: "f32[480]" = torch.ops.aten.mul.Tensor(mul_186, 0.1);  mul_186 = None
    mul_188: "f32[480]" = torch.ops.aten.mul.Tensor(primals_321, 0.9)
    add_116: "f32[480]" = torch.ops.aten.add.Tensor(mul_187, mul_188);  mul_187 = mul_188 = None
    unsqueeze_88: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1)
    unsqueeze_89: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    mul_189: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(mul_183, unsqueeze_89);  mul_183 = unsqueeze_89 = None
    unsqueeze_90: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_46, -1);  primals_46 = None
    unsqueeze_91: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    add_117: "f32[8, 480, 12, 12]" = torch.ops.aten.add.Tensor(mul_189, unsqueeze_91);  mul_189 = unsqueeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_29: "f32[8, 480, 12, 12]" = torch.ops.aten.sigmoid.default(add_117)
    mul_190: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(add_117, sigmoid_29);  sigmoid_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_7: "f32[8, 480, 1, 1]" = torch.ops.aten.mean.dim(mul_190, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_37: "f32[8, 20, 1, 1]" = torch.ops.aten.convolution.default(mean_7, primals_168, primals_169, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_30: "f32[8, 20, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_37)
    mul_191: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_37, sigmoid_30);  sigmoid_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_38: "f32[8, 480, 1, 1]" = torch.ops.aten.convolution.default(mul_191, primals_170, primals_171, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_31: "f32[8, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_38)
    mul_192: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(mul_190, sigmoid_31);  mul_190 = sigmoid_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_39: "f32[8, 80, 12, 12]" = torch.ops.aten.convolution.default(mul_192, primals_172, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_118: "i64[]" = torch.ops.aten.add.Tensor(primals_322, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_23 = torch.ops.aten.var_mean.correction(convolution_39, [0, 2, 3], correction = 0, keepdim = True)
    getitem_46: "f32[1, 80, 1, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 80, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_119: "f32[1, 80, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05)
    rsqrt_23: "f32[1, 80, 1, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
    sub_23: "f32[8, 80, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_39, getitem_47)
    mul_193: "f32[8, 80, 12, 12]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    squeeze_69: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
    squeeze_70: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
    mul_194: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
    mul_195: "f32[80]" = torch.ops.aten.mul.Tensor(primals_323, 0.9)
    add_120: "f32[80]" = torch.ops.aten.add.Tensor(mul_194, mul_195);  mul_194 = mul_195 = None
    squeeze_71: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
    mul_196: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.000868809730669);  squeeze_71 = None
    mul_197: "f32[80]" = torch.ops.aten.mul.Tensor(mul_196, 0.1);  mul_196 = None
    mul_198: "f32[80]" = torch.ops.aten.mul.Tensor(primals_324, 0.9)
    add_121: "f32[80]" = torch.ops.aten.add.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
    unsqueeze_92: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_93: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_199: "f32[8, 80, 12, 12]" = torch.ops.aten.mul.Tensor(mul_193, unsqueeze_93);  mul_193 = unsqueeze_93 = None
    unsqueeze_94: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_95: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_122: "f32[8, 80, 12, 12]" = torch.ops.aten.add.Tensor(mul_199, unsqueeze_95);  mul_199 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_123: "f32[8, 80, 12, 12]" = torch.ops.aten.add.Tensor(add_122, add_107);  add_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_40: "f32[8, 480, 12, 12]" = torch.ops.aten.convolution.default(add_123, primals_173, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_124: "i64[]" = torch.ops.aten.add.Tensor(primals_325, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_24 = torch.ops.aten.var_mean.correction(convolution_40, [0, 2, 3], correction = 0, keepdim = True)
    getitem_48: "f32[1, 480, 1, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 480, 1, 1]" = var_mean_24[1];  var_mean_24 = None
    add_125: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05)
    rsqrt_24: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_125);  add_125 = None
    sub_24: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_40, getitem_49)
    mul_200: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    squeeze_72: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
    squeeze_73: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
    mul_201: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
    mul_202: "f32[480]" = torch.ops.aten.mul.Tensor(primals_326, 0.9)
    add_126: "f32[480]" = torch.ops.aten.add.Tensor(mul_201, mul_202);  mul_201 = mul_202 = None
    squeeze_74: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
    mul_203: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.000868809730669);  squeeze_74 = None
    mul_204: "f32[480]" = torch.ops.aten.mul.Tensor(mul_203, 0.1);  mul_203 = None
    mul_205: "f32[480]" = torch.ops.aten.mul.Tensor(primals_327, 0.9)
    add_127: "f32[480]" = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None
    unsqueeze_96: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1)
    unsqueeze_97: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    mul_206: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(mul_200, unsqueeze_97);  mul_200 = unsqueeze_97 = None
    unsqueeze_98: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1);  primals_50 = None
    unsqueeze_99: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    add_128: "f32[8, 480, 12, 12]" = torch.ops.aten.add.Tensor(mul_206, unsqueeze_99);  mul_206 = unsqueeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_24: "f32[8, 480, 12, 12]" = torch.ops.aten.clone.default(add_128)
    sigmoid_32: "f32[8, 480, 12, 12]" = torch.ops.aten.sigmoid.default(add_128)
    mul_207: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(add_128, sigmoid_32);  add_128 = sigmoid_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_41: "f32[8, 480, 12, 12]" = torch.ops.aten.convolution.default(mul_207, primals_174, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_129: "i64[]" = torch.ops.aten.add.Tensor(primals_328, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_25 = torch.ops.aten.var_mean.correction(convolution_41, [0, 2, 3], correction = 0, keepdim = True)
    getitem_50: "f32[1, 480, 1, 1]" = var_mean_25[0]
    getitem_51: "f32[1, 480, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_130: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05)
    rsqrt_25: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
    sub_25: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_41, getitem_51)
    mul_208: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    squeeze_75: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
    squeeze_76: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
    mul_209: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
    mul_210: "f32[480]" = torch.ops.aten.mul.Tensor(primals_329, 0.9)
    add_131: "f32[480]" = torch.ops.aten.add.Tensor(mul_209, mul_210);  mul_209 = mul_210 = None
    squeeze_77: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
    mul_211: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.000868809730669);  squeeze_77 = None
    mul_212: "f32[480]" = torch.ops.aten.mul.Tensor(mul_211, 0.1);  mul_211 = None
    mul_213: "f32[480]" = torch.ops.aten.mul.Tensor(primals_330, 0.9)
    add_132: "f32[480]" = torch.ops.aten.add.Tensor(mul_212, mul_213);  mul_212 = mul_213 = None
    unsqueeze_100: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1)
    unsqueeze_101: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_214: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(mul_208, unsqueeze_101);  mul_208 = unsqueeze_101 = None
    unsqueeze_102: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_52, -1);  primals_52 = None
    unsqueeze_103: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_133: "f32[8, 480, 12, 12]" = torch.ops.aten.add.Tensor(mul_214, unsqueeze_103);  mul_214 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_33: "f32[8, 480, 12, 12]" = torch.ops.aten.sigmoid.default(add_133)
    mul_215: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(add_133, sigmoid_33);  sigmoid_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_8: "f32[8, 480, 1, 1]" = torch.ops.aten.mean.dim(mul_215, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_42: "f32[8, 20, 1, 1]" = torch.ops.aten.convolution.default(mean_8, primals_175, primals_176, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_34: "f32[8, 20, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_42)
    mul_216: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_42, sigmoid_34);  sigmoid_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_43: "f32[8, 480, 1, 1]" = torch.ops.aten.convolution.default(mul_216, primals_177, primals_178, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_35: "f32[8, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_43)
    mul_217: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(mul_215, sigmoid_35);  mul_215 = sigmoid_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_44: "f32[8, 80, 12, 12]" = torch.ops.aten.convolution.default(mul_217, primals_179, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_134: "i64[]" = torch.ops.aten.add.Tensor(primals_331, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_26 = torch.ops.aten.var_mean.correction(convolution_44, [0, 2, 3], correction = 0, keepdim = True)
    getitem_52: "f32[1, 80, 1, 1]" = var_mean_26[0]
    getitem_53: "f32[1, 80, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_135: "f32[1, 80, 1, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05)
    rsqrt_26: "f32[1, 80, 1, 1]" = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
    sub_26: "f32[8, 80, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_44, getitem_53)
    mul_218: "f32[8, 80, 12, 12]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    squeeze_78: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
    squeeze_79: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
    mul_219: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
    mul_220: "f32[80]" = torch.ops.aten.mul.Tensor(primals_332, 0.9)
    add_136: "f32[80]" = torch.ops.aten.add.Tensor(mul_219, mul_220);  mul_219 = mul_220 = None
    squeeze_80: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
    mul_221: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.000868809730669);  squeeze_80 = None
    mul_222: "f32[80]" = torch.ops.aten.mul.Tensor(mul_221, 0.1);  mul_221 = None
    mul_223: "f32[80]" = torch.ops.aten.mul.Tensor(primals_333, 0.9)
    add_137: "f32[80]" = torch.ops.aten.add.Tensor(mul_222, mul_223);  mul_222 = mul_223 = None
    unsqueeze_104: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
    unsqueeze_105: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    mul_224: "f32[8, 80, 12, 12]" = torch.ops.aten.mul.Tensor(mul_218, unsqueeze_105);  mul_218 = unsqueeze_105 = None
    unsqueeze_106: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
    unsqueeze_107: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    add_138: "f32[8, 80, 12, 12]" = torch.ops.aten.add.Tensor(mul_224, unsqueeze_107);  mul_224 = unsqueeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_139: "f32[8, 80, 12, 12]" = torch.ops.aten.add.Tensor(add_138, add_123);  add_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_45: "f32[8, 480, 12, 12]" = torch.ops.aten.convolution.default(add_139, primals_180, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_140: "i64[]" = torch.ops.aten.add.Tensor(primals_334, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_27 = torch.ops.aten.var_mean.correction(convolution_45, [0, 2, 3], correction = 0, keepdim = True)
    getitem_54: "f32[1, 480, 1, 1]" = var_mean_27[0]
    getitem_55: "f32[1, 480, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_141: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05)
    rsqrt_27: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
    sub_27: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_45, getitem_55)
    mul_225: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
    squeeze_81: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
    squeeze_82: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
    mul_226: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
    mul_227: "f32[480]" = torch.ops.aten.mul.Tensor(primals_335, 0.9)
    add_142: "f32[480]" = torch.ops.aten.add.Tensor(mul_226, mul_227);  mul_226 = mul_227 = None
    squeeze_83: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
    mul_228: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.000868809730669);  squeeze_83 = None
    mul_229: "f32[480]" = torch.ops.aten.mul.Tensor(mul_228, 0.1);  mul_228 = None
    mul_230: "f32[480]" = torch.ops.aten.mul.Tensor(primals_336, 0.9)
    add_143: "f32[480]" = torch.ops.aten.add.Tensor(mul_229, mul_230);  mul_229 = mul_230 = None
    unsqueeze_108: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1)
    unsqueeze_109: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_231: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(mul_225, unsqueeze_109);  mul_225 = unsqueeze_109 = None
    unsqueeze_110: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1);  primals_56 = None
    unsqueeze_111: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_144: "f32[8, 480, 12, 12]" = torch.ops.aten.add.Tensor(mul_231, unsqueeze_111);  mul_231 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_27: "f32[8, 480, 12, 12]" = torch.ops.aten.clone.default(add_144)
    sigmoid_36: "f32[8, 480, 12, 12]" = torch.ops.aten.sigmoid.default(add_144)
    mul_232: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(add_144, sigmoid_36);  add_144 = sigmoid_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_46: "f32[8, 480, 12, 12]" = torch.ops.aten.convolution.default(mul_232, primals_181, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 480)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_145: "i64[]" = torch.ops.aten.add.Tensor(primals_337, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_28 = torch.ops.aten.var_mean.correction(convolution_46, [0, 2, 3], correction = 0, keepdim = True)
    getitem_56: "f32[1, 480, 1, 1]" = var_mean_28[0]
    getitem_57: "f32[1, 480, 1, 1]" = var_mean_28[1];  var_mean_28 = None
    add_146: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05)
    rsqrt_28: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
    sub_28: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_46, getitem_57)
    mul_233: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
    squeeze_84: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
    squeeze_85: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
    mul_234: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
    mul_235: "f32[480]" = torch.ops.aten.mul.Tensor(primals_338, 0.9)
    add_147: "f32[480]" = torch.ops.aten.add.Tensor(mul_234, mul_235);  mul_234 = mul_235 = None
    squeeze_86: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
    mul_236: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.000868809730669);  squeeze_86 = None
    mul_237: "f32[480]" = torch.ops.aten.mul.Tensor(mul_236, 0.1);  mul_236 = None
    mul_238: "f32[480]" = torch.ops.aten.mul.Tensor(primals_339, 0.9)
    add_148: "f32[480]" = torch.ops.aten.add.Tensor(mul_237, mul_238);  mul_237 = mul_238 = None
    unsqueeze_112: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1)
    unsqueeze_113: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    mul_239: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(mul_233, unsqueeze_113);  mul_233 = unsqueeze_113 = None
    unsqueeze_114: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_58, -1);  primals_58 = None
    unsqueeze_115: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    add_149: "f32[8, 480, 12, 12]" = torch.ops.aten.add.Tensor(mul_239, unsqueeze_115);  mul_239 = unsqueeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_37: "f32[8, 480, 12, 12]" = torch.ops.aten.sigmoid.default(add_149)
    mul_240: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(add_149, sigmoid_37);  sigmoid_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_9: "f32[8, 480, 1, 1]" = torch.ops.aten.mean.dim(mul_240, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_47: "f32[8, 20, 1, 1]" = torch.ops.aten.convolution.default(mean_9, primals_182, primals_183, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_38: "f32[8, 20, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_47)
    mul_241: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_47, sigmoid_38);  sigmoid_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_48: "f32[8, 480, 1, 1]" = torch.ops.aten.convolution.default(mul_241, primals_184, primals_185, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_39: "f32[8, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_48)
    mul_242: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(mul_240, sigmoid_39);  mul_240 = sigmoid_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_49: "f32[8, 112, 12, 12]" = torch.ops.aten.convolution.default(mul_242, primals_186, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_150: "i64[]" = torch.ops.aten.add.Tensor(primals_340, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_29 = torch.ops.aten.var_mean.correction(convolution_49, [0, 2, 3], correction = 0, keepdim = True)
    getitem_58: "f32[1, 112, 1, 1]" = var_mean_29[0]
    getitem_59: "f32[1, 112, 1, 1]" = var_mean_29[1];  var_mean_29 = None
    add_151: "f32[1, 112, 1, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05)
    rsqrt_29: "f32[1, 112, 1, 1]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
    sub_29: "f32[8, 112, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_49, getitem_59)
    mul_243: "f32[8, 112, 12, 12]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
    squeeze_87: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
    squeeze_88: "f32[112]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
    mul_244: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
    mul_245: "f32[112]" = torch.ops.aten.mul.Tensor(primals_341, 0.9)
    add_152: "f32[112]" = torch.ops.aten.add.Tensor(mul_244, mul_245);  mul_244 = mul_245 = None
    squeeze_89: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
    mul_246: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.000868809730669);  squeeze_89 = None
    mul_247: "f32[112]" = torch.ops.aten.mul.Tensor(mul_246, 0.1);  mul_246 = None
    mul_248: "f32[112]" = torch.ops.aten.mul.Tensor(primals_342, 0.9)
    add_153: "f32[112]" = torch.ops.aten.add.Tensor(mul_247, mul_248);  mul_247 = mul_248 = None
    unsqueeze_116: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
    unsqueeze_117: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_249: "f32[8, 112, 12, 12]" = torch.ops.aten.mul.Tensor(mul_243, unsqueeze_117);  mul_243 = unsqueeze_117 = None
    unsqueeze_118: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
    unsqueeze_119: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_154: "f32[8, 112, 12, 12]" = torch.ops.aten.add.Tensor(mul_249, unsqueeze_119);  mul_249 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_50: "f32[8, 672, 12, 12]" = torch.ops.aten.convolution.default(add_154, primals_187, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_155: "i64[]" = torch.ops.aten.add.Tensor(primals_343, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_30 = torch.ops.aten.var_mean.correction(convolution_50, [0, 2, 3], correction = 0, keepdim = True)
    getitem_60: "f32[1, 672, 1, 1]" = var_mean_30[0]
    getitem_61: "f32[1, 672, 1, 1]" = var_mean_30[1];  var_mean_30 = None
    add_156: "f32[1, 672, 1, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05)
    rsqrt_30: "f32[1, 672, 1, 1]" = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
    sub_30: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_50, getitem_61)
    mul_250: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
    squeeze_90: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
    squeeze_91: "f32[672]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
    mul_251: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
    mul_252: "f32[672]" = torch.ops.aten.mul.Tensor(primals_344, 0.9)
    add_157: "f32[672]" = torch.ops.aten.add.Tensor(mul_251, mul_252);  mul_251 = mul_252 = None
    squeeze_92: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
    mul_253: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.000868809730669);  squeeze_92 = None
    mul_254: "f32[672]" = torch.ops.aten.mul.Tensor(mul_253, 0.1);  mul_253 = None
    mul_255: "f32[672]" = torch.ops.aten.mul.Tensor(primals_345, 0.9)
    add_158: "f32[672]" = torch.ops.aten.add.Tensor(mul_254, mul_255);  mul_254 = mul_255 = None
    unsqueeze_120: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_61, -1)
    unsqueeze_121: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    mul_256: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(mul_250, unsqueeze_121);  mul_250 = unsqueeze_121 = None
    unsqueeze_122: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1);  primals_62 = None
    unsqueeze_123: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    add_159: "f32[8, 672, 12, 12]" = torch.ops.aten.add.Tensor(mul_256, unsqueeze_123);  mul_256 = unsqueeze_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_30: "f32[8, 672, 12, 12]" = torch.ops.aten.clone.default(add_159)
    sigmoid_40: "f32[8, 672, 12, 12]" = torch.ops.aten.sigmoid.default(add_159)
    mul_257: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(add_159, sigmoid_40);  add_159 = sigmoid_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_51: "f32[8, 672, 12, 12]" = torch.ops.aten.convolution.default(mul_257, primals_188, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 672)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_160: "i64[]" = torch.ops.aten.add.Tensor(primals_346, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_31 = torch.ops.aten.var_mean.correction(convolution_51, [0, 2, 3], correction = 0, keepdim = True)
    getitem_62: "f32[1, 672, 1, 1]" = var_mean_31[0]
    getitem_63: "f32[1, 672, 1, 1]" = var_mean_31[1];  var_mean_31 = None
    add_161: "f32[1, 672, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
    rsqrt_31: "f32[1, 672, 1, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    sub_31: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_51, getitem_63)
    mul_258: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
    squeeze_93: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
    squeeze_94: "f32[672]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
    mul_259: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
    mul_260: "f32[672]" = torch.ops.aten.mul.Tensor(primals_347, 0.9)
    add_162: "f32[672]" = torch.ops.aten.add.Tensor(mul_259, mul_260);  mul_259 = mul_260 = None
    squeeze_95: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
    mul_261: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.000868809730669);  squeeze_95 = None
    mul_262: "f32[672]" = torch.ops.aten.mul.Tensor(mul_261, 0.1);  mul_261 = None
    mul_263: "f32[672]" = torch.ops.aten.mul.Tensor(primals_348, 0.9)
    add_163: "f32[672]" = torch.ops.aten.add.Tensor(mul_262, mul_263);  mul_262 = mul_263 = None
    unsqueeze_124: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1)
    unsqueeze_125: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_264: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(mul_258, unsqueeze_125);  mul_258 = unsqueeze_125 = None
    unsqueeze_126: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_64, -1);  primals_64 = None
    unsqueeze_127: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_164: "f32[8, 672, 12, 12]" = torch.ops.aten.add.Tensor(mul_264, unsqueeze_127);  mul_264 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_41: "f32[8, 672, 12, 12]" = torch.ops.aten.sigmoid.default(add_164)
    mul_265: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(add_164, sigmoid_41);  sigmoid_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_10: "f32[8, 672, 1, 1]" = torch.ops.aten.mean.dim(mul_265, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_52: "f32[8, 28, 1, 1]" = torch.ops.aten.convolution.default(mean_10, primals_189, primals_190, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_42: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_52)
    mul_266: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_52, sigmoid_42);  sigmoid_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_53: "f32[8, 672, 1, 1]" = torch.ops.aten.convolution.default(mul_266, primals_191, primals_192, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_43: "f32[8, 672, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_53)
    mul_267: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(mul_265, sigmoid_43);  mul_265 = sigmoid_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_54: "f32[8, 112, 12, 12]" = torch.ops.aten.convolution.default(mul_267, primals_193, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_165: "i64[]" = torch.ops.aten.add.Tensor(primals_349, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_32 = torch.ops.aten.var_mean.correction(convolution_54, [0, 2, 3], correction = 0, keepdim = True)
    getitem_64: "f32[1, 112, 1, 1]" = var_mean_32[0]
    getitem_65: "f32[1, 112, 1, 1]" = var_mean_32[1];  var_mean_32 = None
    add_166: "f32[1, 112, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
    rsqrt_32: "f32[1, 112, 1, 1]" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
    sub_32: "f32[8, 112, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_54, getitem_65)
    mul_268: "f32[8, 112, 12, 12]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
    squeeze_96: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
    squeeze_97: "f32[112]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
    mul_269: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
    mul_270: "f32[112]" = torch.ops.aten.mul.Tensor(primals_350, 0.9)
    add_167: "f32[112]" = torch.ops.aten.add.Tensor(mul_269, mul_270);  mul_269 = mul_270 = None
    squeeze_98: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
    mul_271: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_98, 1.000868809730669);  squeeze_98 = None
    mul_272: "f32[112]" = torch.ops.aten.mul.Tensor(mul_271, 0.1);  mul_271 = None
    mul_273: "f32[112]" = torch.ops.aten.mul.Tensor(primals_351, 0.9)
    add_168: "f32[112]" = torch.ops.aten.add.Tensor(mul_272, mul_273);  mul_272 = mul_273 = None
    unsqueeze_128: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1)
    unsqueeze_129: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    mul_274: "f32[8, 112, 12, 12]" = torch.ops.aten.mul.Tensor(mul_268, unsqueeze_129);  mul_268 = unsqueeze_129 = None
    unsqueeze_130: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
    unsqueeze_131: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    add_169: "f32[8, 112, 12, 12]" = torch.ops.aten.add.Tensor(mul_274, unsqueeze_131);  mul_274 = unsqueeze_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_170: "f32[8, 112, 12, 12]" = torch.ops.aten.add.Tensor(add_169, add_154);  add_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_55: "f32[8, 672, 12, 12]" = torch.ops.aten.convolution.default(add_170, primals_194, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_171: "i64[]" = torch.ops.aten.add.Tensor(primals_352, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_33 = torch.ops.aten.var_mean.correction(convolution_55, [0, 2, 3], correction = 0, keepdim = True)
    getitem_66: "f32[1, 672, 1, 1]" = var_mean_33[0]
    getitem_67: "f32[1, 672, 1, 1]" = var_mean_33[1];  var_mean_33 = None
    add_172: "f32[1, 672, 1, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05)
    rsqrt_33: "f32[1, 672, 1, 1]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
    sub_33: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_55, getitem_67)
    mul_275: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
    squeeze_99: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
    squeeze_100: "f32[672]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
    mul_276: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
    mul_277: "f32[672]" = torch.ops.aten.mul.Tensor(primals_353, 0.9)
    add_173: "f32[672]" = torch.ops.aten.add.Tensor(mul_276, mul_277);  mul_276 = mul_277 = None
    squeeze_101: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
    mul_278: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_101, 1.000868809730669);  squeeze_101 = None
    mul_279: "f32[672]" = torch.ops.aten.mul.Tensor(mul_278, 0.1);  mul_278 = None
    mul_280: "f32[672]" = torch.ops.aten.mul.Tensor(primals_354, 0.9)
    add_174: "f32[672]" = torch.ops.aten.add.Tensor(mul_279, mul_280);  mul_279 = mul_280 = None
    unsqueeze_132: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_67, -1)
    unsqueeze_133: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_281: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(mul_275, unsqueeze_133);  mul_275 = unsqueeze_133 = None
    unsqueeze_134: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1);  primals_68 = None
    unsqueeze_135: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_175: "f32[8, 672, 12, 12]" = torch.ops.aten.add.Tensor(mul_281, unsqueeze_135);  mul_281 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_33: "f32[8, 672, 12, 12]" = torch.ops.aten.clone.default(add_175)
    sigmoid_44: "f32[8, 672, 12, 12]" = torch.ops.aten.sigmoid.default(add_175)
    mul_282: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(add_175, sigmoid_44);  add_175 = sigmoid_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_56: "f32[8, 672, 12, 12]" = torch.ops.aten.convolution.default(mul_282, primals_195, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 672)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_176: "i64[]" = torch.ops.aten.add.Tensor(primals_355, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_34 = torch.ops.aten.var_mean.correction(convolution_56, [0, 2, 3], correction = 0, keepdim = True)
    getitem_68: "f32[1, 672, 1, 1]" = var_mean_34[0]
    getitem_69: "f32[1, 672, 1, 1]" = var_mean_34[1];  var_mean_34 = None
    add_177: "f32[1, 672, 1, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05)
    rsqrt_34: "f32[1, 672, 1, 1]" = torch.ops.aten.rsqrt.default(add_177);  add_177 = None
    sub_34: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_56, getitem_69)
    mul_283: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
    squeeze_102: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
    squeeze_103: "f32[672]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
    mul_284: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
    mul_285: "f32[672]" = torch.ops.aten.mul.Tensor(primals_356, 0.9)
    add_178: "f32[672]" = torch.ops.aten.add.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None
    squeeze_104: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
    mul_286: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_104, 1.000868809730669);  squeeze_104 = None
    mul_287: "f32[672]" = torch.ops.aten.mul.Tensor(mul_286, 0.1);  mul_286 = None
    mul_288: "f32[672]" = torch.ops.aten.mul.Tensor(primals_357, 0.9)
    add_179: "f32[672]" = torch.ops.aten.add.Tensor(mul_287, mul_288);  mul_287 = mul_288 = None
    unsqueeze_136: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1)
    unsqueeze_137: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    mul_289: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(mul_283, unsqueeze_137);  mul_283 = unsqueeze_137 = None
    unsqueeze_138: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_70, -1);  primals_70 = None
    unsqueeze_139: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    add_180: "f32[8, 672, 12, 12]" = torch.ops.aten.add.Tensor(mul_289, unsqueeze_139);  mul_289 = unsqueeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_45: "f32[8, 672, 12, 12]" = torch.ops.aten.sigmoid.default(add_180)
    mul_290: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(add_180, sigmoid_45);  sigmoid_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_11: "f32[8, 672, 1, 1]" = torch.ops.aten.mean.dim(mul_290, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_57: "f32[8, 28, 1, 1]" = torch.ops.aten.convolution.default(mean_11, primals_196, primals_197, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_46: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_57)
    mul_291: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_57, sigmoid_46);  sigmoid_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_58: "f32[8, 672, 1, 1]" = torch.ops.aten.convolution.default(mul_291, primals_198, primals_199, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_47: "f32[8, 672, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_58)
    mul_292: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(mul_290, sigmoid_47);  mul_290 = sigmoid_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_59: "f32[8, 112, 12, 12]" = torch.ops.aten.convolution.default(mul_292, primals_200, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_181: "i64[]" = torch.ops.aten.add.Tensor(primals_358, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_35 = torch.ops.aten.var_mean.correction(convolution_59, [0, 2, 3], correction = 0, keepdim = True)
    getitem_70: "f32[1, 112, 1, 1]" = var_mean_35[0]
    getitem_71: "f32[1, 112, 1, 1]" = var_mean_35[1];  var_mean_35 = None
    add_182: "f32[1, 112, 1, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05)
    rsqrt_35: "f32[1, 112, 1, 1]" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
    sub_35: "f32[8, 112, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_59, getitem_71)
    mul_293: "f32[8, 112, 12, 12]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
    squeeze_105: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
    squeeze_106: "f32[112]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
    mul_294: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
    mul_295: "f32[112]" = torch.ops.aten.mul.Tensor(primals_359, 0.9)
    add_183: "f32[112]" = torch.ops.aten.add.Tensor(mul_294, mul_295);  mul_294 = mul_295 = None
    squeeze_107: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
    mul_296: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_107, 1.000868809730669);  squeeze_107 = None
    mul_297: "f32[112]" = torch.ops.aten.mul.Tensor(mul_296, 0.1);  mul_296 = None
    mul_298: "f32[112]" = torch.ops.aten.mul.Tensor(primals_360, 0.9)
    add_184: "f32[112]" = torch.ops.aten.add.Tensor(mul_297, mul_298);  mul_297 = mul_298 = None
    unsqueeze_140: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
    unsqueeze_141: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_299: "f32[8, 112, 12, 12]" = torch.ops.aten.mul.Tensor(mul_293, unsqueeze_141);  mul_293 = unsqueeze_141 = None
    unsqueeze_142: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
    unsqueeze_143: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_185: "f32[8, 112, 12, 12]" = torch.ops.aten.add.Tensor(mul_299, unsqueeze_143);  mul_299 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_186: "f32[8, 112, 12, 12]" = torch.ops.aten.add.Tensor(add_185, add_170);  add_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_60: "f32[8, 672, 12, 12]" = torch.ops.aten.convolution.default(add_186, primals_201, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_187: "i64[]" = torch.ops.aten.add.Tensor(primals_361, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_36 = torch.ops.aten.var_mean.correction(convolution_60, [0, 2, 3], correction = 0, keepdim = True)
    getitem_72: "f32[1, 672, 1, 1]" = var_mean_36[0]
    getitem_73: "f32[1, 672, 1, 1]" = var_mean_36[1];  var_mean_36 = None
    add_188: "f32[1, 672, 1, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05)
    rsqrt_36: "f32[1, 672, 1, 1]" = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
    sub_36: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_60, getitem_73)
    mul_300: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
    squeeze_108: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
    squeeze_109: "f32[672]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
    mul_301: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
    mul_302: "f32[672]" = torch.ops.aten.mul.Tensor(primals_362, 0.9)
    add_189: "f32[672]" = torch.ops.aten.add.Tensor(mul_301, mul_302);  mul_301 = mul_302 = None
    squeeze_110: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
    mul_303: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_110, 1.000868809730669);  squeeze_110 = None
    mul_304: "f32[672]" = torch.ops.aten.mul.Tensor(mul_303, 0.1);  mul_303 = None
    mul_305: "f32[672]" = torch.ops.aten.mul.Tensor(primals_363, 0.9)
    add_190: "f32[672]" = torch.ops.aten.add.Tensor(mul_304, mul_305);  mul_304 = mul_305 = None
    unsqueeze_144: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_73, -1)
    unsqueeze_145: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    mul_306: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(mul_300, unsqueeze_145);  mul_300 = unsqueeze_145 = None
    unsqueeze_146: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1);  primals_74 = None
    unsqueeze_147: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    add_191: "f32[8, 672, 12, 12]" = torch.ops.aten.add.Tensor(mul_306, unsqueeze_147);  mul_306 = unsqueeze_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_36: "f32[8, 672, 12, 12]" = torch.ops.aten.clone.default(add_191)
    sigmoid_48: "f32[8, 672, 12, 12]" = torch.ops.aten.sigmoid.default(add_191)
    mul_307: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(add_191, sigmoid_48);  add_191 = sigmoid_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_61: "f32[8, 672, 12, 12]" = torch.ops.aten.convolution.default(mul_307, primals_202, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 672)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_192: "i64[]" = torch.ops.aten.add.Tensor(primals_364, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_37 = torch.ops.aten.var_mean.correction(convolution_61, [0, 2, 3], correction = 0, keepdim = True)
    getitem_74: "f32[1, 672, 1, 1]" = var_mean_37[0]
    getitem_75: "f32[1, 672, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    add_193: "f32[1, 672, 1, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05)
    rsqrt_37: "f32[1, 672, 1, 1]" = torch.ops.aten.rsqrt.default(add_193);  add_193 = None
    sub_37: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_61, getitem_75)
    mul_308: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
    squeeze_111: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
    squeeze_112: "f32[672]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
    mul_309: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
    mul_310: "f32[672]" = torch.ops.aten.mul.Tensor(primals_365, 0.9)
    add_194: "f32[672]" = torch.ops.aten.add.Tensor(mul_309, mul_310);  mul_309 = mul_310 = None
    squeeze_113: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
    mul_311: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_113, 1.000868809730669);  squeeze_113 = None
    mul_312: "f32[672]" = torch.ops.aten.mul.Tensor(mul_311, 0.1);  mul_311 = None
    mul_313: "f32[672]" = torch.ops.aten.mul.Tensor(primals_366, 0.9)
    add_195: "f32[672]" = torch.ops.aten.add.Tensor(mul_312, mul_313);  mul_312 = mul_313 = None
    unsqueeze_148: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1)
    unsqueeze_149: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_314: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(mul_308, unsqueeze_149);  mul_308 = unsqueeze_149 = None
    unsqueeze_150: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_76, -1);  primals_76 = None
    unsqueeze_151: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_196: "f32[8, 672, 12, 12]" = torch.ops.aten.add.Tensor(mul_314, unsqueeze_151);  mul_314 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_49: "f32[8, 672, 12, 12]" = torch.ops.aten.sigmoid.default(add_196)
    mul_315: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(add_196, sigmoid_49);  sigmoid_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_12: "f32[8, 672, 1, 1]" = torch.ops.aten.mean.dim(mul_315, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_62: "f32[8, 28, 1, 1]" = torch.ops.aten.convolution.default(mean_12, primals_203, primals_204, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_50: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_62)
    mul_316: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_62, sigmoid_50);  sigmoid_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_63: "f32[8, 672, 1, 1]" = torch.ops.aten.convolution.default(mul_316, primals_205, primals_206, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_51: "f32[8, 672, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_63)
    mul_317: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(mul_315, sigmoid_51);  mul_315 = sigmoid_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_64: "f32[8, 112, 12, 12]" = torch.ops.aten.convolution.default(mul_317, primals_207, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_197: "i64[]" = torch.ops.aten.add.Tensor(primals_367, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_38 = torch.ops.aten.var_mean.correction(convolution_64, [0, 2, 3], correction = 0, keepdim = True)
    getitem_76: "f32[1, 112, 1, 1]" = var_mean_38[0]
    getitem_77: "f32[1, 112, 1, 1]" = var_mean_38[1];  var_mean_38 = None
    add_198: "f32[1, 112, 1, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05)
    rsqrt_38: "f32[1, 112, 1, 1]" = torch.ops.aten.rsqrt.default(add_198);  add_198 = None
    sub_38: "f32[8, 112, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_64, getitem_77)
    mul_318: "f32[8, 112, 12, 12]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
    squeeze_114: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
    squeeze_115: "f32[112]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
    mul_319: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
    mul_320: "f32[112]" = torch.ops.aten.mul.Tensor(primals_368, 0.9)
    add_199: "f32[112]" = torch.ops.aten.add.Tensor(mul_319, mul_320);  mul_319 = mul_320 = None
    squeeze_116: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
    mul_321: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_116, 1.000868809730669);  squeeze_116 = None
    mul_322: "f32[112]" = torch.ops.aten.mul.Tensor(mul_321, 0.1);  mul_321 = None
    mul_323: "f32[112]" = torch.ops.aten.mul.Tensor(primals_369, 0.9)
    add_200: "f32[112]" = torch.ops.aten.add.Tensor(mul_322, mul_323);  mul_322 = mul_323 = None
    unsqueeze_152: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1)
    unsqueeze_153: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    mul_324: "f32[8, 112, 12, 12]" = torch.ops.aten.mul.Tensor(mul_318, unsqueeze_153);  mul_318 = unsqueeze_153 = None
    unsqueeze_154: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
    unsqueeze_155: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    add_201: "f32[8, 112, 12, 12]" = torch.ops.aten.add.Tensor(mul_324, unsqueeze_155);  mul_324 = unsqueeze_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_202: "f32[8, 112, 12, 12]" = torch.ops.aten.add.Tensor(add_201, add_186);  add_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_65: "f32[8, 672, 12, 12]" = torch.ops.aten.convolution.default(add_202, primals_208, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_203: "i64[]" = torch.ops.aten.add.Tensor(primals_370, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_39 = torch.ops.aten.var_mean.correction(convolution_65, [0, 2, 3], correction = 0, keepdim = True)
    getitem_78: "f32[1, 672, 1, 1]" = var_mean_39[0]
    getitem_79: "f32[1, 672, 1, 1]" = var_mean_39[1];  var_mean_39 = None
    add_204: "f32[1, 672, 1, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05)
    rsqrt_39: "f32[1, 672, 1, 1]" = torch.ops.aten.rsqrt.default(add_204);  add_204 = None
    sub_39: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_65, getitem_79)
    mul_325: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
    squeeze_117: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
    squeeze_118: "f32[672]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
    mul_326: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1)
    mul_327: "f32[672]" = torch.ops.aten.mul.Tensor(primals_371, 0.9)
    add_205: "f32[672]" = torch.ops.aten.add.Tensor(mul_326, mul_327);  mul_326 = mul_327 = None
    squeeze_119: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
    mul_328: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_119, 1.000868809730669);  squeeze_119 = None
    mul_329: "f32[672]" = torch.ops.aten.mul.Tensor(mul_328, 0.1);  mul_328 = None
    mul_330: "f32[672]" = torch.ops.aten.mul.Tensor(primals_372, 0.9)
    add_206: "f32[672]" = torch.ops.aten.add.Tensor(mul_329, mul_330);  mul_329 = mul_330 = None
    unsqueeze_156: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_79, -1)
    unsqueeze_157: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_331: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(mul_325, unsqueeze_157);  mul_325 = unsqueeze_157 = None
    unsqueeze_158: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1);  primals_80 = None
    unsqueeze_159: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_207: "f32[8, 672, 12, 12]" = torch.ops.aten.add.Tensor(mul_331, unsqueeze_159);  mul_331 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_39: "f32[8, 672, 12, 12]" = torch.ops.aten.clone.default(add_207)
    sigmoid_52: "f32[8, 672, 12, 12]" = torch.ops.aten.sigmoid.default(add_207)
    mul_332: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(add_207, sigmoid_52);  add_207 = sigmoid_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_66: "f32[8, 672, 6, 6]" = torch.ops.aten.convolution.default(mul_332, primals_209, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 672)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_208: "i64[]" = torch.ops.aten.add.Tensor(primals_373, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_40 = torch.ops.aten.var_mean.correction(convolution_66, [0, 2, 3], correction = 0, keepdim = True)
    getitem_80: "f32[1, 672, 1, 1]" = var_mean_40[0]
    getitem_81: "f32[1, 672, 1, 1]" = var_mean_40[1];  var_mean_40 = None
    add_209: "f32[1, 672, 1, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05)
    rsqrt_40: "f32[1, 672, 1, 1]" = torch.ops.aten.rsqrt.default(add_209);  add_209 = None
    sub_40: "f32[8, 672, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_66, getitem_81)
    mul_333: "f32[8, 672, 6, 6]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
    squeeze_120: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
    squeeze_121: "f32[672]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
    mul_334: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1)
    mul_335: "f32[672]" = torch.ops.aten.mul.Tensor(primals_374, 0.9)
    add_210: "f32[672]" = torch.ops.aten.add.Tensor(mul_334, mul_335);  mul_334 = mul_335 = None
    squeeze_122: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
    mul_336: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_122, 1.0034843205574913);  squeeze_122 = None
    mul_337: "f32[672]" = torch.ops.aten.mul.Tensor(mul_336, 0.1);  mul_336 = None
    mul_338: "f32[672]" = torch.ops.aten.mul.Tensor(primals_375, 0.9)
    add_211: "f32[672]" = torch.ops.aten.add.Tensor(mul_337, mul_338);  mul_337 = mul_338 = None
    unsqueeze_160: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1)
    unsqueeze_161: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    mul_339: "f32[8, 672, 6, 6]" = torch.ops.aten.mul.Tensor(mul_333, unsqueeze_161);  mul_333 = unsqueeze_161 = None
    unsqueeze_162: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_82, -1);  primals_82 = None
    unsqueeze_163: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    add_212: "f32[8, 672, 6, 6]" = torch.ops.aten.add.Tensor(mul_339, unsqueeze_163);  mul_339 = unsqueeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_53: "f32[8, 672, 6, 6]" = torch.ops.aten.sigmoid.default(add_212)
    mul_340: "f32[8, 672, 6, 6]" = torch.ops.aten.mul.Tensor(add_212, sigmoid_53);  sigmoid_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_13: "f32[8, 672, 1, 1]" = torch.ops.aten.mean.dim(mul_340, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_67: "f32[8, 28, 1, 1]" = torch.ops.aten.convolution.default(mean_13, primals_210, primals_211, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_54: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_67)
    mul_341: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_67, sigmoid_54);  sigmoid_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_68: "f32[8, 672, 1, 1]" = torch.ops.aten.convolution.default(mul_341, primals_212, primals_213, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_55: "f32[8, 672, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_68)
    mul_342: "f32[8, 672, 6, 6]" = torch.ops.aten.mul.Tensor(mul_340, sigmoid_55);  mul_340 = sigmoid_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_69: "f32[8, 192, 6, 6]" = torch.ops.aten.convolution.default(mul_342, primals_214, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_213: "i64[]" = torch.ops.aten.add.Tensor(primals_376, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_41 = torch.ops.aten.var_mean.correction(convolution_69, [0, 2, 3], correction = 0, keepdim = True)
    getitem_82: "f32[1, 192, 1, 1]" = var_mean_41[0]
    getitem_83: "f32[1, 192, 1, 1]" = var_mean_41[1];  var_mean_41 = None
    add_214: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05)
    rsqrt_41: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_214);  add_214 = None
    sub_41: "f32[8, 192, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_69, getitem_83)
    mul_343: "f32[8, 192, 6, 6]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
    squeeze_123: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
    squeeze_124: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
    mul_344: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
    mul_345: "f32[192]" = torch.ops.aten.mul.Tensor(primals_377, 0.9)
    add_215: "f32[192]" = torch.ops.aten.add.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
    squeeze_125: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
    mul_346: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_125, 1.0034843205574913);  squeeze_125 = None
    mul_347: "f32[192]" = torch.ops.aten.mul.Tensor(mul_346, 0.1);  mul_346 = None
    mul_348: "f32[192]" = torch.ops.aten.mul.Tensor(primals_378, 0.9)
    add_216: "f32[192]" = torch.ops.aten.add.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
    unsqueeze_164: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1)
    unsqueeze_165: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_349: "f32[8, 192, 6, 6]" = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_165);  mul_343 = unsqueeze_165 = None
    unsqueeze_166: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1);  primals_84 = None
    unsqueeze_167: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_217: "f32[8, 192, 6, 6]" = torch.ops.aten.add.Tensor(mul_349, unsqueeze_167);  mul_349 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_70: "f32[8, 1152, 6, 6]" = torch.ops.aten.convolution.default(add_217, primals_215, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_218: "i64[]" = torch.ops.aten.add.Tensor(primals_379, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_42 = torch.ops.aten.var_mean.correction(convolution_70, [0, 2, 3], correction = 0, keepdim = True)
    getitem_84: "f32[1, 1152, 1, 1]" = var_mean_42[0]
    getitem_85: "f32[1, 1152, 1, 1]" = var_mean_42[1];  var_mean_42 = None
    add_219: "f32[1, 1152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05)
    rsqrt_42: "f32[1, 1152, 1, 1]" = torch.ops.aten.rsqrt.default(add_219);  add_219 = None
    sub_42: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_70, getitem_85)
    mul_350: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
    squeeze_126: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
    squeeze_127: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
    mul_351: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_126, 0.1)
    mul_352: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_380, 0.9)
    add_220: "f32[1152]" = torch.ops.aten.add.Tensor(mul_351, mul_352);  mul_351 = mul_352 = None
    squeeze_128: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
    mul_353: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_128, 1.0034843205574913);  squeeze_128 = None
    mul_354: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_353, 0.1);  mul_353 = None
    mul_355: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_381, 0.9)
    add_221: "f32[1152]" = torch.ops.aten.add.Tensor(mul_354, mul_355);  mul_354 = mul_355 = None
    unsqueeze_168: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_85, -1)
    unsqueeze_169: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    mul_356: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_350, unsqueeze_169);  mul_350 = unsqueeze_169 = None
    unsqueeze_170: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_86, -1);  primals_86 = None
    unsqueeze_171: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    add_222: "f32[8, 1152, 6, 6]" = torch.ops.aten.add.Tensor(mul_356, unsqueeze_171);  mul_356 = unsqueeze_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_42: "f32[8, 1152, 6, 6]" = torch.ops.aten.clone.default(add_222)
    sigmoid_56: "f32[8, 1152, 6, 6]" = torch.ops.aten.sigmoid.default(add_222)
    mul_357: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(add_222, sigmoid_56);  add_222 = sigmoid_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_71: "f32[8, 1152, 6, 6]" = torch.ops.aten.convolution.default(mul_357, primals_216, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_223: "i64[]" = torch.ops.aten.add.Tensor(primals_382, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_43 = torch.ops.aten.var_mean.correction(convolution_71, [0, 2, 3], correction = 0, keepdim = True)
    getitem_86: "f32[1, 1152, 1, 1]" = var_mean_43[0]
    getitem_87: "f32[1, 1152, 1, 1]" = var_mean_43[1];  var_mean_43 = None
    add_224: "f32[1, 1152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05)
    rsqrt_43: "f32[1, 1152, 1, 1]" = torch.ops.aten.rsqrt.default(add_224);  add_224 = None
    sub_43: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_71, getitem_87)
    mul_358: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
    squeeze_129: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
    squeeze_130: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
    mul_359: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_129, 0.1)
    mul_360: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_383, 0.9)
    add_225: "f32[1152]" = torch.ops.aten.add.Tensor(mul_359, mul_360);  mul_359 = mul_360 = None
    squeeze_131: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
    mul_361: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_131, 1.0034843205574913);  squeeze_131 = None
    mul_362: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_361, 0.1);  mul_361 = None
    mul_363: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_384, 0.9)
    add_226: "f32[1152]" = torch.ops.aten.add.Tensor(mul_362, mul_363);  mul_362 = mul_363 = None
    unsqueeze_172: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_87, -1)
    unsqueeze_173: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_364: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_358, unsqueeze_173);  mul_358 = unsqueeze_173 = None
    unsqueeze_174: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_88, -1);  primals_88 = None
    unsqueeze_175: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_227: "f32[8, 1152, 6, 6]" = torch.ops.aten.add.Tensor(mul_364, unsqueeze_175);  mul_364 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_57: "f32[8, 1152, 6, 6]" = torch.ops.aten.sigmoid.default(add_227)
    mul_365: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(add_227, sigmoid_57);  sigmoid_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_14: "f32[8, 1152, 1, 1]" = torch.ops.aten.mean.dim(mul_365, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_72: "f32[8, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_14, primals_217, primals_218, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_58: "f32[8, 48, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_72)
    mul_366: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_72, sigmoid_58);  sigmoid_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_73: "f32[8, 1152, 1, 1]" = torch.ops.aten.convolution.default(mul_366, primals_219, primals_220, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_59: "f32[8, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_73)
    mul_367: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_365, sigmoid_59);  mul_365 = sigmoid_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_74: "f32[8, 192, 6, 6]" = torch.ops.aten.convolution.default(mul_367, primals_221, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_228: "i64[]" = torch.ops.aten.add.Tensor(primals_385, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_44 = torch.ops.aten.var_mean.correction(convolution_74, [0, 2, 3], correction = 0, keepdim = True)
    getitem_88: "f32[1, 192, 1, 1]" = var_mean_44[0]
    getitem_89: "f32[1, 192, 1, 1]" = var_mean_44[1];  var_mean_44 = None
    add_229: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05)
    rsqrt_44: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
    sub_44: "f32[8, 192, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_74, getitem_89)
    mul_368: "f32[8, 192, 6, 6]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
    squeeze_132: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
    squeeze_133: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
    mul_369: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_132, 0.1)
    mul_370: "f32[192]" = torch.ops.aten.mul.Tensor(primals_386, 0.9)
    add_230: "f32[192]" = torch.ops.aten.add.Tensor(mul_369, mul_370);  mul_369 = mul_370 = None
    squeeze_134: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
    mul_371: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_134, 1.0034843205574913);  squeeze_134 = None
    mul_372: "f32[192]" = torch.ops.aten.mul.Tensor(mul_371, 0.1);  mul_371 = None
    mul_373: "f32[192]" = torch.ops.aten.mul.Tensor(primals_387, 0.9)
    add_231: "f32[192]" = torch.ops.aten.add.Tensor(mul_372, mul_373);  mul_372 = mul_373 = None
    unsqueeze_176: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_89, -1)
    unsqueeze_177: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    mul_374: "f32[8, 192, 6, 6]" = torch.ops.aten.mul.Tensor(mul_368, unsqueeze_177);  mul_368 = unsqueeze_177 = None
    unsqueeze_178: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1);  primals_90 = None
    unsqueeze_179: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    add_232: "f32[8, 192, 6, 6]" = torch.ops.aten.add.Tensor(mul_374, unsqueeze_179);  mul_374 = unsqueeze_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_233: "f32[8, 192, 6, 6]" = torch.ops.aten.add.Tensor(add_232, add_217);  add_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_75: "f32[8, 1152, 6, 6]" = torch.ops.aten.convolution.default(add_233, primals_222, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_234: "i64[]" = torch.ops.aten.add.Tensor(primals_388, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_45 = torch.ops.aten.var_mean.correction(convolution_75, [0, 2, 3], correction = 0, keepdim = True)
    getitem_90: "f32[1, 1152, 1, 1]" = var_mean_45[0]
    getitem_91: "f32[1, 1152, 1, 1]" = var_mean_45[1];  var_mean_45 = None
    add_235: "f32[1, 1152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05)
    rsqrt_45: "f32[1, 1152, 1, 1]" = torch.ops.aten.rsqrt.default(add_235);  add_235 = None
    sub_45: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_75, getitem_91)
    mul_375: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
    squeeze_135: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
    squeeze_136: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
    mul_376: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_135, 0.1)
    mul_377: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_389, 0.9)
    add_236: "f32[1152]" = torch.ops.aten.add.Tensor(mul_376, mul_377);  mul_376 = mul_377 = None
    squeeze_137: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
    mul_378: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_137, 1.0034843205574913);  squeeze_137 = None
    mul_379: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_378, 0.1);  mul_378 = None
    mul_380: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_390, 0.9)
    add_237: "f32[1152]" = torch.ops.aten.add.Tensor(mul_379, mul_380);  mul_379 = mul_380 = None
    unsqueeze_180: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_91, -1)
    unsqueeze_181: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_381: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_375, unsqueeze_181);  mul_375 = unsqueeze_181 = None
    unsqueeze_182: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1);  primals_92 = None
    unsqueeze_183: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_238: "f32[8, 1152, 6, 6]" = torch.ops.aten.add.Tensor(mul_381, unsqueeze_183);  mul_381 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_45: "f32[8, 1152, 6, 6]" = torch.ops.aten.clone.default(add_238)
    sigmoid_60: "f32[8, 1152, 6, 6]" = torch.ops.aten.sigmoid.default(add_238)
    mul_382: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(add_238, sigmoid_60);  add_238 = sigmoid_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_76: "f32[8, 1152, 6, 6]" = torch.ops.aten.convolution.default(mul_382, primals_223, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_239: "i64[]" = torch.ops.aten.add.Tensor(primals_391, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_46 = torch.ops.aten.var_mean.correction(convolution_76, [0, 2, 3], correction = 0, keepdim = True)
    getitem_92: "f32[1, 1152, 1, 1]" = var_mean_46[0]
    getitem_93: "f32[1, 1152, 1, 1]" = var_mean_46[1];  var_mean_46 = None
    add_240: "f32[1, 1152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05)
    rsqrt_46: "f32[1, 1152, 1, 1]" = torch.ops.aten.rsqrt.default(add_240);  add_240 = None
    sub_46: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_76, getitem_93)
    mul_383: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
    squeeze_138: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3]);  getitem_93 = None
    squeeze_139: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
    mul_384: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_138, 0.1)
    mul_385: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_392, 0.9)
    add_241: "f32[1152]" = torch.ops.aten.add.Tensor(mul_384, mul_385);  mul_384 = mul_385 = None
    squeeze_140: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_92, [0, 2, 3]);  getitem_92 = None
    mul_386: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_140, 1.0034843205574913);  squeeze_140 = None
    mul_387: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_386, 0.1);  mul_386 = None
    mul_388: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_393, 0.9)
    add_242: "f32[1152]" = torch.ops.aten.add.Tensor(mul_387, mul_388);  mul_387 = mul_388 = None
    unsqueeze_184: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_93, -1)
    unsqueeze_185: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    mul_389: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_383, unsqueeze_185);  mul_383 = unsqueeze_185 = None
    unsqueeze_186: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_94, -1);  primals_94 = None
    unsqueeze_187: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    add_243: "f32[8, 1152, 6, 6]" = torch.ops.aten.add.Tensor(mul_389, unsqueeze_187);  mul_389 = unsqueeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_61: "f32[8, 1152, 6, 6]" = torch.ops.aten.sigmoid.default(add_243)
    mul_390: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(add_243, sigmoid_61);  sigmoid_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_15: "f32[8, 1152, 1, 1]" = torch.ops.aten.mean.dim(mul_390, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_77: "f32[8, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_15, primals_224, primals_225, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_62: "f32[8, 48, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_77)
    mul_391: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_77, sigmoid_62);  sigmoid_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_78: "f32[8, 1152, 1, 1]" = torch.ops.aten.convolution.default(mul_391, primals_226, primals_227, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_63: "f32[8, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_78)
    mul_392: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_390, sigmoid_63);  mul_390 = sigmoid_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_79: "f32[8, 192, 6, 6]" = torch.ops.aten.convolution.default(mul_392, primals_228, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_244: "i64[]" = torch.ops.aten.add.Tensor(primals_394, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_47 = torch.ops.aten.var_mean.correction(convolution_79, [0, 2, 3], correction = 0, keepdim = True)
    getitem_94: "f32[1, 192, 1, 1]" = var_mean_47[0]
    getitem_95: "f32[1, 192, 1, 1]" = var_mean_47[1];  var_mean_47 = None
    add_245: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05)
    rsqrt_47: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_245);  add_245 = None
    sub_47: "f32[8, 192, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_79, getitem_95)
    mul_393: "f32[8, 192, 6, 6]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
    squeeze_141: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
    squeeze_142: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
    mul_394: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_141, 0.1)
    mul_395: "f32[192]" = torch.ops.aten.mul.Tensor(primals_395, 0.9)
    add_246: "f32[192]" = torch.ops.aten.add.Tensor(mul_394, mul_395);  mul_394 = mul_395 = None
    squeeze_143: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_94, [0, 2, 3]);  getitem_94 = None
    mul_396: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_143, 1.0034843205574913);  squeeze_143 = None
    mul_397: "f32[192]" = torch.ops.aten.mul.Tensor(mul_396, 0.1);  mul_396 = None
    mul_398: "f32[192]" = torch.ops.aten.mul.Tensor(primals_396, 0.9)
    add_247: "f32[192]" = torch.ops.aten.add.Tensor(mul_397, mul_398);  mul_397 = mul_398 = None
    unsqueeze_188: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1)
    unsqueeze_189: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_399: "f32[8, 192, 6, 6]" = torch.ops.aten.mul.Tensor(mul_393, unsqueeze_189);  mul_393 = unsqueeze_189 = None
    unsqueeze_190: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1);  primals_96 = None
    unsqueeze_191: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_248: "f32[8, 192, 6, 6]" = torch.ops.aten.add.Tensor(mul_399, unsqueeze_191);  mul_399 = unsqueeze_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_249: "f32[8, 192, 6, 6]" = torch.ops.aten.add.Tensor(add_248, add_233);  add_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_80: "f32[8, 1152, 6, 6]" = torch.ops.aten.convolution.default(add_249, primals_229, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_250: "i64[]" = torch.ops.aten.add.Tensor(primals_397, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_48 = torch.ops.aten.var_mean.correction(convolution_80, [0, 2, 3], correction = 0, keepdim = True)
    getitem_96: "f32[1, 1152, 1, 1]" = var_mean_48[0]
    getitem_97: "f32[1, 1152, 1, 1]" = var_mean_48[1];  var_mean_48 = None
    add_251: "f32[1, 1152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05)
    rsqrt_48: "f32[1, 1152, 1, 1]" = torch.ops.aten.rsqrt.default(add_251);  add_251 = None
    sub_48: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_80, getitem_97)
    mul_400: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
    squeeze_144: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
    squeeze_145: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
    mul_401: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_144, 0.1)
    mul_402: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_398, 0.9)
    add_252: "f32[1152]" = torch.ops.aten.add.Tensor(mul_401, mul_402);  mul_401 = mul_402 = None
    squeeze_146: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
    mul_403: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_146, 1.0034843205574913);  squeeze_146 = None
    mul_404: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_403, 0.1);  mul_403 = None
    mul_405: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_399, 0.9)
    add_253: "f32[1152]" = torch.ops.aten.add.Tensor(mul_404, mul_405);  mul_404 = mul_405 = None
    unsqueeze_192: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_97, -1)
    unsqueeze_193: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    mul_406: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_400, unsqueeze_193);  mul_400 = unsqueeze_193 = None
    unsqueeze_194: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1);  primals_98 = None
    unsqueeze_195: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    add_254: "f32[8, 1152, 6, 6]" = torch.ops.aten.add.Tensor(mul_406, unsqueeze_195);  mul_406 = unsqueeze_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_48: "f32[8, 1152, 6, 6]" = torch.ops.aten.clone.default(add_254)
    sigmoid_64: "f32[8, 1152, 6, 6]" = torch.ops.aten.sigmoid.default(add_254)
    mul_407: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(add_254, sigmoid_64);  add_254 = sigmoid_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_81: "f32[8, 1152, 6, 6]" = torch.ops.aten.convolution.default(mul_407, primals_230, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_255: "i64[]" = torch.ops.aten.add.Tensor(primals_400, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_49 = torch.ops.aten.var_mean.correction(convolution_81, [0, 2, 3], correction = 0, keepdim = True)
    getitem_98: "f32[1, 1152, 1, 1]" = var_mean_49[0]
    getitem_99: "f32[1, 1152, 1, 1]" = var_mean_49[1];  var_mean_49 = None
    add_256: "f32[1, 1152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-05)
    rsqrt_49: "f32[1, 1152, 1, 1]" = torch.ops.aten.rsqrt.default(add_256);  add_256 = None
    sub_49: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_81, getitem_99)
    mul_408: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
    squeeze_147: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
    squeeze_148: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
    mul_409: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_147, 0.1)
    mul_410: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_401, 0.9)
    add_257: "f32[1152]" = torch.ops.aten.add.Tensor(mul_409, mul_410);  mul_409 = mul_410 = None
    squeeze_149: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_98, [0, 2, 3]);  getitem_98 = None
    mul_411: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_149, 1.0034843205574913);  squeeze_149 = None
    mul_412: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_411, 0.1);  mul_411 = None
    mul_413: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_402, 0.9)
    add_258: "f32[1152]" = torch.ops.aten.add.Tensor(mul_412, mul_413);  mul_412 = mul_413 = None
    unsqueeze_196: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_99, -1)
    unsqueeze_197: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_414: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_408, unsqueeze_197);  mul_408 = unsqueeze_197 = None
    unsqueeze_198: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_100, -1);  primals_100 = None
    unsqueeze_199: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_259: "f32[8, 1152, 6, 6]" = torch.ops.aten.add.Tensor(mul_414, unsqueeze_199);  mul_414 = unsqueeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_65: "f32[8, 1152, 6, 6]" = torch.ops.aten.sigmoid.default(add_259)
    mul_415: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(add_259, sigmoid_65);  sigmoid_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_16: "f32[8, 1152, 1, 1]" = torch.ops.aten.mean.dim(mul_415, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_82: "f32[8, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_16, primals_231, primals_232, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_66: "f32[8, 48, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_82)
    mul_416: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_82, sigmoid_66);  sigmoid_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_83: "f32[8, 1152, 1, 1]" = torch.ops.aten.convolution.default(mul_416, primals_233, primals_234, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_67: "f32[8, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_83)
    mul_417: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_415, sigmoid_67);  mul_415 = sigmoid_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_84: "f32[8, 192, 6, 6]" = torch.ops.aten.convolution.default(mul_417, primals_235, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_260: "i64[]" = torch.ops.aten.add.Tensor(primals_403, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_50 = torch.ops.aten.var_mean.correction(convolution_84, [0, 2, 3], correction = 0, keepdim = True)
    getitem_100: "f32[1, 192, 1, 1]" = var_mean_50[0]
    getitem_101: "f32[1, 192, 1, 1]" = var_mean_50[1];  var_mean_50 = None
    add_261: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05)
    rsqrt_50: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_261);  add_261 = None
    sub_50: "f32[8, 192, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_84, getitem_101)
    mul_418: "f32[8, 192, 6, 6]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
    squeeze_150: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2, 3]);  getitem_101 = None
    squeeze_151: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
    mul_419: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_150, 0.1)
    mul_420: "f32[192]" = torch.ops.aten.mul.Tensor(primals_404, 0.9)
    add_262: "f32[192]" = torch.ops.aten.add.Tensor(mul_419, mul_420);  mul_419 = mul_420 = None
    squeeze_152: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_100, [0, 2, 3]);  getitem_100 = None
    mul_421: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_152, 1.0034843205574913);  squeeze_152 = None
    mul_422: "f32[192]" = torch.ops.aten.mul.Tensor(mul_421, 0.1);  mul_421 = None
    mul_423: "f32[192]" = torch.ops.aten.mul.Tensor(primals_405, 0.9)
    add_263: "f32[192]" = torch.ops.aten.add.Tensor(mul_422, mul_423);  mul_422 = mul_423 = None
    unsqueeze_200: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_101, -1)
    unsqueeze_201: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    mul_424: "f32[8, 192, 6, 6]" = torch.ops.aten.mul.Tensor(mul_418, unsqueeze_201);  mul_418 = unsqueeze_201 = None
    unsqueeze_202: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_102, -1);  primals_102 = None
    unsqueeze_203: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    add_264: "f32[8, 192, 6, 6]" = torch.ops.aten.add.Tensor(mul_424, unsqueeze_203);  mul_424 = unsqueeze_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_265: "f32[8, 192, 6, 6]" = torch.ops.aten.add.Tensor(add_264, add_249);  add_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_85: "f32[8, 1152, 6, 6]" = torch.ops.aten.convolution.default(add_265, primals_236, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_266: "i64[]" = torch.ops.aten.add.Tensor(primals_406, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_51 = torch.ops.aten.var_mean.correction(convolution_85, [0, 2, 3], correction = 0, keepdim = True)
    getitem_102: "f32[1, 1152, 1, 1]" = var_mean_51[0]
    getitem_103: "f32[1, 1152, 1, 1]" = var_mean_51[1];  var_mean_51 = None
    add_267: "f32[1, 1152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05)
    rsqrt_51: "f32[1, 1152, 1, 1]" = torch.ops.aten.rsqrt.default(add_267);  add_267 = None
    sub_51: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_85, getitem_103)
    mul_425: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
    squeeze_153: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
    squeeze_154: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
    mul_426: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_153, 0.1)
    mul_427: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_407, 0.9)
    add_268: "f32[1152]" = torch.ops.aten.add.Tensor(mul_426, mul_427);  mul_426 = mul_427 = None
    squeeze_155: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_102, [0, 2, 3]);  getitem_102 = None
    mul_428: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_155, 1.0034843205574913);  squeeze_155 = None
    mul_429: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_428, 0.1);  mul_428 = None
    mul_430: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_408, 0.9)
    add_269: "f32[1152]" = torch.ops.aten.add.Tensor(mul_429, mul_430);  mul_429 = mul_430 = None
    unsqueeze_204: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_103, -1)
    unsqueeze_205: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_431: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_425, unsqueeze_205);  mul_425 = unsqueeze_205 = None
    unsqueeze_206: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_104, -1);  primals_104 = None
    unsqueeze_207: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_270: "f32[8, 1152, 6, 6]" = torch.ops.aten.add.Tensor(mul_431, unsqueeze_207);  mul_431 = unsqueeze_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_51: "f32[8, 1152, 6, 6]" = torch.ops.aten.clone.default(add_270)
    sigmoid_68: "f32[8, 1152, 6, 6]" = torch.ops.aten.sigmoid.default(add_270)
    mul_432: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(add_270, sigmoid_68);  add_270 = sigmoid_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_86: "f32[8, 1152, 6, 6]" = torch.ops.aten.convolution.default(mul_432, primals_237, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_271: "i64[]" = torch.ops.aten.add.Tensor(primals_409, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_52 = torch.ops.aten.var_mean.correction(convolution_86, [0, 2, 3], correction = 0, keepdim = True)
    getitem_104: "f32[1, 1152, 1, 1]" = var_mean_52[0]
    getitem_105: "f32[1, 1152, 1, 1]" = var_mean_52[1];  var_mean_52 = None
    add_272: "f32[1, 1152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05)
    rsqrt_52: "f32[1, 1152, 1, 1]" = torch.ops.aten.rsqrt.default(add_272);  add_272 = None
    sub_52: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_86, getitem_105)
    mul_433: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
    squeeze_156: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_105, [0, 2, 3]);  getitem_105 = None
    squeeze_157: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
    mul_434: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_156, 0.1)
    mul_435: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_410, 0.9)
    add_273: "f32[1152]" = torch.ops.aten.add.Tensor(mul_434, mul_435);  mul_434 = mul_435 = None
    squeeze_158: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_104, [0, 2, 3]);  getitem_104 = None
    mul_436: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_158, 1.0034843205574913);  squeeze_158 = None
    mul_437: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_436, 0.1);  mul_436 = None
    mul_438: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_411, 0.9)
    add_274: "f32[1152]" = torch.ops.aten.add.Tensor(mul_437, mul_438);  mul_437 = mul_438 = None
    unsqueeze_208: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_105, -1)
    unsqueeze_209: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    mul_439: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_433, unsqueeze_209);  mul_433 = unsqueeze_209 = None
    unsqueeze_210: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_106, -1);  primals_106 = None
    unsqueeze_211: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    add_275: "f32[8, 1152, 6, 6]" = torch.ops.aten.add.Tensor(mul_439, unsqueeze_211);  mul_439 = unsqueeze_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_69: "f32[8, 1152, 6, 6]" = torch.ops.aten.sigmoid.default(add_275)
    mul_440: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(add_275, sigmoid_69);  sigmoid_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_17: "f32[8, 1152, 1, 1]" = torch.ops.aten.mean.dim(mul_440, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_87: "f32[8, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_17, primals_238, primals_239, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_70: "f32[8, 48, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_87)
    mul_441: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_87, sigmoid_70);  sigmoid_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_88: "f32[8, 1152, 1, 1]" = torch.ops.aten.convolution.default(mul_441, primals_240, primals_241, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_71: "f32[8, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_88)
    mul_442: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_440, sigmoid_71);  mul_440 = sigmoid_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_89: "f32[8, 192, 6, 6]" = torch.ops.aten.convolution.default(mul_442, primals_242, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_276: "i64[]" = torch.ops.aten.add.Tensor(primals_412, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_53 = torch.ops.aten.var_mean.correction(convolution_89, [0, 2, 3], correction = 0, keepdim = True)
    getitem_106: "f32[1, 192, 1, 1]" = var_mean_53[0]
    getitem_107: "f32[1, 192, 1, 1]" = var_mean_53[1];  var_mean_53 = None
    add_277: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05)
    rsqrt_53: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_277);  add_277 = None
    sub_53: "f32[8, 192, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_89, getitem_107)
    mul_443: "f32[8, 192, 6, 6]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = None
    squeeze_159: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
    squeeze_160: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_53, [0, 2, 3]);  rsqrt_53 = None
    mul_444: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_159, 0.1)
    mul_445: "f32[192]" = torch.ops.aten.mul.Tensor(primals_413, 0.9)
    add_278: "f32[192]" = torch.ops.aten.add.Tensor(mul_444, mul_445);  mul_444 = mul_445 = None
    squeeze_161: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_106, [0, 2, 3]);  getitem_106 = None
    mul_446: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_161, 1.0034843205574913);  squeeze_161 = None
    mul_447: "f32[192]" = torch.ops.aten.mul.Tensor(mul_446, 0.1);  mul_446 = None
    mul_448: "f32[192]" = torch.ops.aten.mul.Tensor(primals_414, 0.9)
    add_279: "f32[192]" = torch.ops.aten.add.Tensor(mul_447, mul_448);  mul_447 = mul_448 = None
    unsqueeze_212: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_107, -1)
    unsqueeze_213: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_449: "f32[8, 192, 6, 6]" = torch.ops.aten.mul.Tensor(mul_443, unsqueeze_213);  mul_443 = unsqueeze_213 = None
    unsqueeze_214: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_108, -1);  primals_108 = None
    unsqueeze_215: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_280: "f32[8, 192, 6, 6]" = torch.ops.aten.add.Tensor(mul_449, unsqueeze_215);  mul_449 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_281: "f32[8, 192, 6, 6]" = torch.ops.aten.add.Tensor(add_280, add_265);  add_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_90: "f32[8, 1152, 6, 6]" = torch.ops.aten.convolution.default(add_281, primals_243, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_282: "i64[]" = torch.ops.aten.add.Tensor(primals_415, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_54 = torch.ops.aten.var_mean.correction(convolution_90, [0, 2, 3], correction = 0, keepdim = True)
    getitem_108: "f32[1, 1152, 1, 1]" = var_mean_54[0]
    getitem_109: "f32[1, 1152, 1, 1]" = var_mean_54[1];  var_mean_54 = None
    add_283: "f32[1, 1152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05)
    rsqrt_54: "f32[1, 1152, 1, 1]" = torch.ops.aten.rsqrt.default(add_283);  add_283 = None
    sub_54: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_90, getitem_109)
    mul_450: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = None
    squeeze_162: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_109, [0, 2, 3]);  getitem_109 = None
    squeeze_163: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2, 3]);  rsqrt_54 = None
    mul_451: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_162, 0.1)
    mul_452: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_416, 0.9)
    add_284: "f32[1152]" = torch.ops.aten.add.Tensor(mul_451, mul_452);  mul_451 = mul_452 = None
    squeeze_164: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_108, [0, 2, 3]);  getitem_108 = None
    mul_453: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_164, 1.0034843205574913);  squeeze_164 = None
    mul_454: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_453, 0.1);  mul_453 = None
    mul_455: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_417, 0.9)
    add_285: "f32[1152]" = torch.ops.aten.add.Tensor(mul_454, mul_455);  mul_454 = mul_455 = None
    unsqueeze_216: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_109, -1)
    unsqueeze_217: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    mul_456: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_450, unsqueeze_217);  mul_450 = unsqueeze_217 = None
    unsqueeze_218: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_110, -1);  primals_110 = None
    unsqueeze_219: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    add_286: "f32[8, 1152, 6, 6]" = torch.ops.aten.add.Tensor(mul_456, unsqueeze_219);  mul_456 = unsqueeze_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_54: "f32[8, 1152, 6, 6]" = torch.ops.aten.clone.default(add_286)
    sigmoid_72: "f32[8, 1152, 6, 6]" = torch.ops.aten.sigmoid.default(add_286)
    mul_457: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(add_286, sigmoid_72);  add_286 = sigmoid_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_91: "f32[8, 1152, 6, 6]" = torch.ops.aten.convolution.default(mul_457, primals_244, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1152)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_287: "i64[]" = torch.ops.aten.add.Tensor(primals_418, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_55 = torch.ops.aten.var_mean.correction(convolution_91, [0, 2, 3], correction = 0, keepdim = True)
    getitem_110: "f32[1, 1152, 1, 1]" = var_mean_55[0]
    getitem_111: "f32[1, 1152, 1, 1]" = var_mean_55[1];  var_mean_55 = None
    add_288: "f32[1, 1152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05)
    rsqrt_55: "f32[1, 1152, 1, 1]" = torch.ops.aten.rsqrt.default(add_288);  add_288 = None
    sub_55: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_91, getitem_111)
    mul_458: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = None
    squeeze_165: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_111, [0, 2, 3]);  getitem_111 = None
    squeeze_166: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2, 3]);  rsqrt_55 = None
    mul_459: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_165, 0.1)
    mul_460: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_419, 0.9)
    add_289: "f32[1152]" = torch.ops.aten.add.Tensor(mul_459, mul_460);  mul_459 = mul_460 = None
    squeeze_167: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_110, [0, 2, 3]);  getitem_110 = None
    mul_461: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_167, 1.0034843205574913);  squeeze_167 = None
    mul_462: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_461, 0.1);  mul_461 = None
    mul_463: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_420, 0.9)
    add_290: "f32[1152]" = torch.ops.aten.add.Tensor(mul_462, mul_463);  mul_462 = mul_463 = None
    unsqueeze_220: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_111, -1)
    unsqueeze_221: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_464: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_458, unsqueeze_221);  mul_458 = unsqueeze_221 = None
    unsqueeze_222: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_112, -1);  primals_112 = None
    unsqueeze_223: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_291: "f32[8, 1152, 6, 6]" = torch.ops.aten.add.Tensor(mul_464, unsqueeze_223);  mul_464 = unsqueeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_73: "f32[8, 1152, 6, 6]" = torch.ops.aten.sigmoid.default(add_291)
    mul_465: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(add_291, sigmoid_73);  sigmoid_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_18: "f32[8, 1152, 1, 1]" = torch.ops.aten.mean.dim(mul_465, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_92: "f32[8, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_18, primals_245, primals_246, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_74: "f32[8, 48, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_92)
    mul_466: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_92, sigmoid_74);  sigmoid_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_93: "f32[8, 1152, 1, 1]" = torch.ops.aten.convolution.default(mul_466, primals_247, primals_248, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_75: "f32[8, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_93)
    mul_467: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_465, sigmoid_75);  mul_465 = sigmoid_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_94: "f32[8, 320, 6, 6]" = torch.ops.aten.convolution.default(mul_467, primals_249, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_292: "i64[]" = torch.ops.aten.add.Tensor(primals_421, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_56 = torch.ops.aten.var_mean.correction(convolution_94, [0, 2, 3], correction = 0, keepdim = True)
    getitem_112: "f32[1, 320, 1, 1]" = var_mean_56[0]
    getitem_113: "f32[1, 320, 1, 1]" = var_mean_56[1];  var_mean_56 = None
    add_293: "f32[1, 320, 1, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05)
    rsqrt_56: "f32[1, 320, 1, 1]" = torch.ops.aten.rsqrt.default(add_293);  add_293 = None
    sub_56: "f32[8, 320, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_94, getitem_113)
    mul_468: "f32[8, 320, 6, 6]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = None
    squeeze_168: "f32[320]" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2, 3]);  getitem_113 = None
    squeeze_169: "f32[320]" = torch.ops.aten.squeeze.dims(rsqrt_56, [0, 2, 3]);  rsqrt_56 = None
    mul_469: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_168, 0.1)
    mul_470: "f32[320]" = torch.ops.aten.mul.Tensor(primals_422, 0.9)
    add_294: "f32[320]" = torch.ops.aten.add.Tensor(mul_469, mul_470);  mul_469 = mul_470 = None
    squeeze_170: "f32[320]" = torch.ops.aten.squeeze.dims(getitem_112, [0, 2, 3]);  getitem_112 = None
    mul_471: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_170, 1.0034843205574913);  squeeze_170 = None
    mul_472: "f32[320]" = torch.ops.aten.mul.Tensor(mul_471, 0.1);  mul_471 = None
    mul_473: "f32[320]" = torch.ops.aten.mul.Tensor(primals_423, 0.9)
    add_295: "f32[320]" = torch.ops.aten.add.Tensor(mul_472, mul_473);  mul_472 = mul_473 = None
    unsqueeze_224: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_113, -1)
    unsqueeze_225: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    mul_474: "f32[8, 320, 6, 6]" = torch.ops.aten.mul.Tensor(mul_468, unsqueeze_225);  mul_468 = unsqueeze_225 = None
    unsqueeze_226: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_114, -1);  primals_114 = None
    unsqueeze_227: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    add_296: "f32[8, 320, 6, 6]" = torch.ops.aten.add.Tensor(mul_474, unsqueeze_227);  mul_474 = unsqueeze_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:168, code: x = self.conv_head(x)
    convolution_95: "f32[8, 1280, 6, 6]" = torch.ops.aten.convolution.default(add_296, primals_250, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_297: "i64[]" = torch.ops.aten.add.Tensor(primals_424, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_57 = torch.ops.aten.var_mean.correction(convolution_95, [0, 2, 3], correction = 0, keepdim = True)
    getitem_114: "f32[1, 1280, 1, 1]" = var_mean_57[0]
    getitem_115: "f32[1, 1280, 1, 1]" = var_mean_57[1];  var_mean_57 = None
    add_298: "f32[1, 1280, 1, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-05)
    rsqrt_57: "f32[1, 1280, 1, 1]" = torch.ops.aten.rsqrt.default(add_298);  add_298 = None
    sub_57: "f32[8, 1280, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_95, getitem_115)
    mul_475: "f32[8, 1280, 6, 6]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = None
    squeeze_171: "f32[1280]" = torch.ops.aten.squeeze.dims(getitem_115, [0, 2, 3]);  getitem_115 = None
    squeeze_172: "f32[1280]" = torch.ops.aten.squeeze.dims(rsqrt_57, [0, 2, 3]);  rsqrt_57 = None
    mul_476: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_171, 0.1)
    mul_477: "f32[1280]" = torch.ops.aten.mul.Tensor(primals_425, 0.9)
    add_299: "f32[1280]" = torch.ops.aten.add.Tensor(mul_476, mul_477);  mul_476 = mul_477 = None
    squeeze_173: "f32[1280]" = torch.ops.aten.squeeze.dims(getitem_114, [0, 2, 3]);  getitem_114 = None
    mul_478: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_173, 1.0034843205574913);  squeeze_173 = None
    mul_479: "f32[1280]" = torch.ops.aten.mul.Tensor(mul_478, 0.1);  mul_478 = None
    mul_480: "f32[1280]" = torch.ops.aten.mul.Tensor(primals_426, 0.9)
    add_300: "f32[1280]" = torch.ops.aten.add.Tensor(mul_479, mul_480);  mul_479 = mul_480 = None
    unsqueeze_228: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(primals_115, -1)
    unsqueeze_229: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_481: "f32[8, 1280, 6, 6]" = torch.ops.aten.mul.Tensor(mul_475, unsqueeze_229);  mul_475 = unsqueeze_229 = None
    unsqueeze_230: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(primals_116, -1);  primals_116 = None
    unsqueeze_231: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_301: "f32[8, 1280, 6, 6]" = torch.ops.aten.add.Tensor(mul_481, unsqueeze_231);  mul_481 = unsqueeze_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_57: "f32[8, 1280, 6, 6]" = torch.ops.aten.clone.default(add_301)
    sigmoid_76: "f32[8, 1280, 6, 6]" = torch.ops.aten.sigmoid.default(add_301)
    mul_482: "f32[8, 1280, 6, 6]" = torch.ops.aten.mul.Tensor(add_301, sigmoid_76);  add_301 = sigmoid_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean_19: "f32[8, 1280, 1, 1]" = torch.ops.aten.mean.dim(mul_482, [-1, -2], True);  mul_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view: "f32[8, 1280]" = torch.ops.aten.view.default(mean_19, [8, 1280]);  mean_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:176, code: return x if pre_logits else self.classifier(x)
    permute: "f32[1280, 1000]" = torch.ops.aten.permute.default(primals_251, [1, 0]);  primals_251 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_252, view, permute);  primals_252 = None
    permute_1: "f32[1000, 1280]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_77: "f32[8, 1280, 6, 6]" = torch.ops.aten.sigmoid.default(clone_57)
    full_default: "f32[8, 1280, 6, 6]" = torch.ops.aten.full.default([8, 1280, 6, 6], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_58: "f32[8, 1280, 6, 6]" = torch.ops.aten.sub.Tensor(full_default, sigmoid_77);  full_default = None
    mul_483: "f32[8, 1280, 6, 6]" = torch.ops.aten.mul.Tensor(clone_57, sub_58);  clone_57 = sub_58 = None
    add_302: "f32[8, 1280, 6, 6]" = torch.ops.aten.add.Scalar(mul_483, 1);  mul_483 = None
    mul_484: "f32[8, 1280, 6, 6]" = torch.ops.aten.mul.Tensor(sigmoid_77, add_302);  sigmoid_77 = add_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_232: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(squeeze_171, 0);  squeeze_171 = None
    unsqueeze_233: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 2);  unsqueeze_232 = None
    unsqueeze_234: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 3);  unsqueeze_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_244: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(squeeze_168, 0);  squeeze_168 = None
    unsqueeze_245: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 2);  unsqueeze_244 = None
    unsqueeze_246: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 3);  unsqueeze_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    full_default_2: "f32[8, 1152, 6, 6]" = torch.ops.aten.full.default([8, 1152, 6, 6], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_256: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_165, 0);  squeeze_165 = None
    unsqueeze_257: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, 2);  unsqueeze_256 = None
    unsqueeze_258: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 3);  unsqueeze_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_80: "f32[8, 1152, 6, 6]" = torch.ops.aten.sigmoid.default(clone_54)
    sub_74: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_80)
    mul_523: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(clone_54, sub_74);  clone_54 = sub_74 = None
    add_306: "f32[8, 1152, 6, 6]" = torch.ops.aten.add.Scalar(mul_523, 1);  mul_523 = None
    mul_524: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sigmoid_80, add_306);  sigmoid_80 = add_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_268: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_162, 0);  squeeze_162 = None
    unsqueeze_269: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, 2);  unsqueeze_268 = None
    unsqueeze_270: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 3);  unsqueeze_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_280: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_159, 0);  squeeze_159 = None
    unsqueeze_281: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 2);  unsqueeze_280 = None
    unsqueeze_282: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 3);  unsqueeze_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_292: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_156, 0);  squeeze_156 = None
    unsqueeze_293: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 2);  unsqueeze_292 = None
    unsqueeze_294: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 3);  unsqueeze_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_83: "f32[8, 1152, 6, 6]" = torch.ops.aten.sigmoid.default(clone_51)
    sub_90: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_83)
    mul_563: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(clone_51, sub_90);  clone_51 = sub_90 = None
    add_310: "f32[8, 1152, 6, 6]" = torch.ops.aten.add.Scalar(mul_563, 1);  mul_563 = None
    mul_564: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sigmoid_83, add_310);  sigmoid_83 = add_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_304: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_153, 0);  squeeze_153 = None
    unsqueeze_305: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, 2);  unsqueeze_304 = None
    unsqueeze_306: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 3);  unsqueeze_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_316: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_150, 0);  squeeze_150 = None
    unsqueeze_317: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, 2);  unsqueeze_316 = None
    unsqueeze_318: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 3);  unsqueeze_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_328: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_147, 0);  squeeze_147 = None
    unsqueeze_329: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 2);  unsqueeze_328 = None
    unsqueeze_330: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 3);  unsqueeze_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_86: "f32[8, 1152, 6, 6]" = torch.ops.aten.sigmoid.default(clone_48)
    sub_106: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_86)
    mul_603: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(clone_48, sub_106);  clone_48 = sub_106 = None
    add_315: "f32[8, 1152, 6, 6]" = torch.ops.aten.add.Scalar(mul_603, 1);  mul_603 = None
    mul_604: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sigmoid_86, add_315);  sigmoid_86 = add_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_340: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
    unsqueeze_341: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, 2);  unsqueeze_340 = None
    unsqueeze_342: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 3);  unsqueeze_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_352: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_141, 0);  squeeze_141 = None
    unsqueeze_353: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, 2);  unsqueeze_352 = None
    unsqueeze_354: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 3);  unsqueeze_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_364: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_138, 0);  squeeze_138 = None
    unsqueeze_365: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, 2);  unsqueeze_364 = None
    unsqueeze_366: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 3);  unsqueeze_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_89: "f32[8, 1152, 6, 6]" = torch.ops.aten.sigmoid.default(clone_45)
    sub_122: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_89)
    mul_643: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(clone_45, sub_122);  clone_45 = sub_122 = None
    add_320: "f32[8, 1152, 6, 6]" = torch.ops.aten.add.Scalar(mul_643, 1);  mul_643 = None
    mul_644: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sigmoid_89, add_320);  sigmoid_89 = add_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_376: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
    unsqueeze_377: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 2);  unsqueeze_376 = None
    unsqueeze_378: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 3);  unsqueeze_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_388: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_132, 0);  squeeze_132 = None
    unsqueeze_389: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 2);  unsqueeze_388 = None
    unsqueeze_390: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 3);  unsqueeze_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_400: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
    unsqueeze_401: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, 2);  unsqueeze_400 = None
    unsqueeze_402: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 3);  unsqueeze_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_92: "f32[8, 1152, 6, 6]" = torch.ops.aten.sigmoid.default(clone_42)
    sub_138: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_92);  full_default_2 = None
    mul_683: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(clone_42, sub_138);  clone_42 = sub_138 = None
    add_325: "f32[8, 1152, 6, 6]" = torch.ops.aten.add.Scalar(mul_683, 1);  mul_683 = None
    mul_684: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sigmoid_92, add_325);  sigmoid_92 = add_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_412: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
    unsqueeze_413: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, 2);  unsqueeze_412 = None
    unsqueeze_414: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 3);  unsqueeze_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_424: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
    unsqueeze_425: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 2);  unsqueeze_424 = None
    unsqueeze_426: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 3);  unsqueeze_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_436: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
    unsqueeze_437: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 2);  unsqueeze_436 = None
    unsqueeze_438: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 3);  unsqueeze_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_95: "f32[8, 672, 12, 12]" = torch.ops.aten.sigmoid.default(clone_39)
    full_default_18: "f32[8, 672, 12, 12]" = torch.ops.aten.full.default([8, 672, 12, 12], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_154: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(full_default_18, sigmoid_95)
    mul_723: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(clone_39, sub_154);  clone_39 = sub_154 = None
    add_330: "f32[8, 672, 12, 12]" = torch.ops.aten.add.Scalar(mul_723, 1);  mul_723 = None
    mul_724: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(sigmoid_95, add_330);  sigmoid_95 = add_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_448: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
    unsqueeze_449: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, 2);  unsqueeze_448 = None
    unsqueeze_450: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 3);  unsqueeze_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_460: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
    unsqueeze_461: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, 2);  unsqueeze_460 = None
    unsqueeze_462: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 3);  unsqueeze_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_472: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
    unsqueeze_473: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, 2);  unsqueeze_472 = None
    unsqueeze_474: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 3);  unsqueeze_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_98: "f32[8, 672, 12, 12]" = torch.ops.aten.sigmoid.default(clone_36)
    sub_170: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(full_default_18, sigmoid_98)
    mul_763: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(clone_36, sub_170);  clone_36 = sub_170 = None
    add_334: "f32[8, 672, 12, 12]" = torch.ops.aten.add.Scalar(mul_763, 1);  mul_763 = None
    mul_764: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(sigmoid_98, add_334);  sigmoid_98 = add_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_484: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
    unsqueeze_485: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 2);  unsqueeze_484 = None
    unsqueeze_486: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 3);  unsqueeze_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_496: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
    unsqueeze_497: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, 2);  unsqueeze_496 = None
    unsqueeze_498: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 3);  unsqueeze_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_508: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
    unsqueeze_509: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, 2);  unsqueeze_508 = None
    unsqueeze_510: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 3);  unsqueeze_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_101: "f32[8, 672, 12, 12]" = torch.ops.aten.sigmoid.default(clone_33)
    sub_186: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(full_default_18, sigmoid_101)
    mul_803: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(clone_33, sub_186);  clone_33 = sub_186 = None
    add_339: "f32[8, 672, 12, 12]" = torch.ops.aten.add.Scalar(mul_803, 1);  mul_803 = None
    mul_804: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(sigmoid_101, add_339);  sigmoid_101 = add_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_520: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
    unsqueeze_521: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, 2);  unsqueeze_520 = None
    unsqueeze_522: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 3);  unsqueeze_521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_532: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
    unsqueeze_533: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, 2);  unsqueeze_532 = None
    unsqueeze_534: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 3);  unsqueeze_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_544: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
    unsqueeze_545: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, 2);  unsqueeze_544 = None
    unsqueeze_546: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 3);  unsqueeze_545 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_104: "f32[8, 672, 12, 12]" = torch.ops.aten.sigmoid.default(clone_30)
    sub_202: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(full_default_18, sigmoid_104);  full_default_18 = None
    mul_843: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(clone_30, sub_202);  clone_30 = sub_202 = None
    add_344: "f32[8, 672, 12, 12]" = torch.ops.aten.add.Scalar(mul_843, 1);  mul_843 = None
    mul_844: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(sigmoid_104, add_344);  sigmoid_104 = add_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_556: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    unsqueeze_557: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, 2);  unsqueeze_556 = None
    unsqueeze_558: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 3);  unsqueeze_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_568: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
    unsqueeze_569: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, 2);  unsqueeze_568 = None
    unsqueeze_570: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 3);  unsqueeze_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    full_default_29: "f32[8, 480, 12, 12]" = torch.ops.aten.full.default([8, 480, 12, 12], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_580: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    unsqueeze_581: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, 2);  unsqueeze_580 = None
    unsqueeze_582: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 3);  unsqueeze_581 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_107: "f32[8, 480, 12, 12]" = torch.ops.aten.sigmoid.default(clone_27)
    sub_218: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(full_default_29, sigmoid_107)
    mul_883: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(clone_27, sub_218);  clone_27 = sub_218 = None
    add_349: "f32[8, 480, 12, 12]" = torch.ops.aten.add.Scalar(mul_883, 1);  mul_883 = None
    mul_884: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sigmoid_107, add_349);  sigmoid_107 = add_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_592: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
    unsqueeze_593: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 2);  unsqueeze_592 = None
    unsqueeze_594: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 3);  unsqueeze_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_604: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_605: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, 2);  unsqueeze_604 = None
    unsqueeze_606: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 3);  unsqueeze_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_616: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_617: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, 2);  unsqueeze_616 = None
    unsqueeze_618: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 3);  unsqueeze_617 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_110: "f32[8, 480, 12, 12]" = torch.ops.aten.sigmoid.default(clone_24)
    sub_234: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(full_default_29, sigmoid_110)
    mul_923: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(clone_24, sub_234);  clone_24 = sub_234 = None
    add_353: "f32[8, 480, 12, 12]" = torch.ops.aten.add.Scalar(mul_923, 1);  mul_923 = None
    mul_924: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sigmoid_110, add_353);  sigmoid_110 = add_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_628: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_629: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, 2);  unsqueeze_628 = None
    unsqueeze_630: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 3);  unsqueeze_629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_640: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_641: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, 2);  unsqueeze_640 = None
    unsqueeze_642: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 3);  unsqueeze_641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_652: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_653: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, 2);  unsqueeze_652 = None
    unsqueeze_654: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 3);  unsqueeze_653 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_113: "f32[8, 480, 12, 12]" = torch.ops.aten.sigmoid.default(clone_21)
    sub_250: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(full_default_29, sigmoid_113)
    mul_963: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(clone_21, sub_250);  clone_21 = sub_250 = None
    add_358: "f32[8, 480, 12, 12]" = torch.ops.aten.add.Scalar(mul_963, 1);  mul_963 = None
    mul_964: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sigmoid_113, add_358);  sigmoid_113 = add_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_664: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_665: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, 2);  unsqueeze_664 = None
    unsqueeze_666: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 3);  unsqueeze_665 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_676: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_677: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, 2);  unsqueeze_676 = None
    unsqueeze_678: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 3);  unsqueeze_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_688: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_689: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, 2);  unsqueeze_688 = None
    unsqueeze_690: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 3);  unsqueeze_689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_116: "f32[8, 480, 12, 12]" = torch.ops.aten.sigmoid.default(clone_18)
    sub_266: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(full_default_29, sigmoid_116);  full_default_29 = None
    mul_1003: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(clone_18, sub_266);  clone_18 = sub_266 = None
    add_363: "f32[8, 480, 12, 12]" = torch.ops.aten.add.Scalar(mul_1003, 1);  mul_1003 = None
    mul_1004: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sigmoid_116, add_363);  sigmoid_116 = add_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_700: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_701: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, 2);  unsqueeze_700 = None
    unsqueeze_702: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 3);  unsqueeze_701 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_712: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_713: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, 2);  unsqueeze_712 = None
    unsqueeze_714: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 3);  unsqueeze_713 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_724: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_725: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, 2);  unsqueeze_724 = None
    unsqueeze_726: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 3);  unsqueeze_725 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_119: "f32[8, 240, 24, 24]" = torch.ops.aten.sigmoid.default(clone_15)
    full_default_42: "f32[8, 240, 24, 24]" = torch.ops.aten.full.default([8, 240, 24, 24], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_282: "f32[8, 240, 24, 24]" = torch.ops.aten.sub.Tensor(full_default_42, sigmoid_119)
    mul_1043: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(clone_15, sub_282);  clone_15 = sub_282 = None
    add_368: "f32[8, 240, 24, 24]" = torch.ops.aten.add.Scalar(mul_1043, 1);  mul_1043 = None
    mul_1044: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(sigmoid_119, add_368);  sigmoid_119 = add_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_736: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_737: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, 2);  unsqueeze_736 = None
    unsqueeze_738: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 3);  unsqueeze_737 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_748: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_749: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, 2);  unsqueeze_748 = None
    unsqueeze_750: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 3);  unsqueeze_749 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_760: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_761: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, 2);  unsqueeze_760 = None
    unsqueeze_762: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 3);  unsqueeze_761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_122: "f32[8, 240, 24, 24]" = torch.ops.aten.sigmoid.default(clone_12)
    sub_298: "f32[8, 240, 24, 24]" = torch.ops.aten.sub.Tensor(full_default_42, sigmoid_122);  full_default_42 = None
    mul_1083: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(clone_12, sub_298);  clone_12 = sub_298 = None
    add_372: "f32[8, 240, 24, 24]" = torch.ops.aten.add.Scalar(mul_1083, 1);  mul_1083 = None
    mul_1084: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(sigmoid_122, add_372);  sigmoid_122 = add_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_772: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_773: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, 2);  unsqueeze_772 = None
    unsqueeze_774: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 3);  unsqueeze_773 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_784: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_785: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_784, 2);  unsqueeze_784 = None
    unsqueeze_786: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 3);  unsqueeze_785 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_796: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_797: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_796, 2);  unsqueeze_796 = None
    unsqueeze_798: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_797, 3);  unsqueeze_797 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_125: "f32[8, 144, 48, 48]" = torch.ops.aten.sigmoid.default(clone_9)
    full_default_48: "f32[8, 144, 48, 48]" = torch.ops.aten.full.default([8, 144, 48, 48], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_314: "f32[8, 144, 48, 48]" = torch.ops.aten.sub.Tensor(full_default_48, sigmoid_125)
    mul_1123: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(clone_9, sub_314);  clone_9 = sub_314 = None
    add_377: "f32[8, 144, 48, 48]" = torch.ops.aten.add.Scalar(mul_1123, 1);  mul_1123 = None
    mul_1124: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(sigmoid_125, add_377);  sigmoid_125 = add_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_808: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_809: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_808, 2);  unsqueeze_808 = None
    unsqueeze_810: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 3);  unsqueeze_809 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_820: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_821: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_820, 2);  unsqueeze_820 = None
    unsqueeze_822: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 3);  unsqueeze_821 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_832: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_833: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_832, 2);  unsqueeze_832 = None
    unsqueeze_834: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_833, 3);  unsqueeze_833 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_128: "f32[8, 144, 48, 48]" = torch.ops.aten.sigmoid.default(clone_6)
    sub_330: "f32[8, 144, 48, 48]" = torch.ops.aten.sub.Tensor(full_default_48, sigmoid_128);  full_default_48 = None
    mul_1163: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(clone_6, sub_330);  clone_6 = sub_330 = None
    add_381: "f32[8, 144, 48, 48]" = torch.ops.aten.add.Scalar(mul_1163, 1);  mul_1163 = None
    mul_1164: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(sigmoid_128, add_381);  sigmoid_128 = add_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_844: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_845: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_844, 2);  unsqueeze_844 = None
    unsqueeze_846: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_845, 3);  unsqueeze_845 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_856: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_857: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_856, 2);  unsqueeze_856 = None
    unsqueeze_858: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_857, 3);  unsqueeze_857 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_868: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_869: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_868, 2);  unsqueeze_868 = None
    unsqueeze_870: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_869, 3);  unsqueeze_869 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_131: "f32[8, 96, 96, 96]" = torch.ops.aten.sigmoid.default(clone_3)
    full_default_54: "f32[8, 96, 96, 96]" = torch.ops.aten.full.default([8, 96, 96, 96], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_346: "f32[8, 96, 96, 96]" = torch.ops.aten.sub.Tensor(full_default_54, sigmoid_131);  full_default_54 = None
    mul_1203: "f32[8, 96, 96, 96]" = torch.ops.aten.mul.Tensor(clone_3, sub_346);  clone_3 = sub_346 = None
    add_386: "f32[8, 96, 96, 96]" = torch.ops.aten.add.Scalar(mul_1203, 1);  mul_1203 = None
    mul_1204: "f32[8, 96, 96, 96]" = torch.ops.aten.mul.Tensor(sigmoid_131, add_386);  sigmoid_131 = add_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_880: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_881: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_880, 2);  unsqueeze_880 = None
    unsqueeze_882: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_881, 3);  unsqueeze_881 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_892: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_893: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_892, 2);  unsqueeze_892 = None
    unsqueeze_894: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_893, 3);  unsqueeze_893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    full_default_56: "f32[8, 32, 96, 96]" = torch.ops.aten.full.default([8, 32, 96, 96], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_904: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_905: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_904, 2);  unsqueeze_904 = None
    unsqueeze_906: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_905, 3);  unsqueeze_905 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_134: "f32[8, 32, 96, 96]" = torch.ops.aten.sigmoid.default(clone)
    sub_362: "f32[8, 32, 96, 96]" = torch.ops.aten.sub.Tensor(full_default_56, sigmoid_134);  full_default_56 = None
    mul_1243: "f32[8, 32, 96, 96]" = torch.ops.aten.mul.Tensor(clone, sub_362);  clone = sub_362 = None
    add_390: "f32[8, 32, 96, 96]" = torch.ops.aten.add.Scalar(mul_1243, 1);  mul_1243 = None
    mul_1244: "f32[8, 32, 96, 96]" = torch.ops.aten.mul.Tensor(sigmoid_134, add_390);  sigmoid_134 = add_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_916: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_917: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_916, 2);  unsqueeze_916 = None
    unsqueeze_918: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_917, 3);  unsqueeze_917 = None
    
    # No stacktrace found for following nodes
    copy_: "i64[]" = torch.ops.aten.copy_.default(primals_253, add);  primals_253 = add = None
    copy__1: "f32[32]" = torch.ops.aten.copy_.default(primals_254, add_2);  primals_254 = add_2 = None
    copy__2: "f32[32]" = torch.ops.aten.copy_.default(primals_255, add_3);  primals_255 = add_3 = None
    copy__3: "i64[]" = torch.ops.aten.copy_.default(primals_256, add_5);  primals_256 = add_5 = None
    copy__4: "f32[32]" = torch.ops.aten.copy_.default(primals_257, add_7);  primals_257 = add_7 = None
    copy__5: "f32[32]" = torch.ops.aten.copy_.default(primals_258, add_8);  primals_258 = add_8 = None
    copy__6: "i64[]" = torch.ops.aten.copy_.default(primals_259, add_10);  primals_259 = add_10 = None
    copy__7: "f32[16]" = torch.ops.aten.copy_.default(primals_260, add_12);  primals_260 = add_12 = None
    copy__8: "f32[16]" = torch.ops.aten.copy_.default(primals_261, add_13);  primals_261 = add_13 = None
    copy__9: "i64[]" = torch.ops.aten.copy_.default(primals_262, add_15);  primals_262 = add_15 = None
    copy__10: "f32[96]" = torch.ops.aten.copy_.default(primals_263, add_17);  primals_263 = add_17 = None
    copy__11: "f32[96]" = torch.ops.aten.copy_.default(primals_264, add_18);  primals_264 = add_18 = None
    copy__12: "i64[]" = torch.ops.aten.copy_.default(primals_265, add_20);  primals_265 = add_20 = None
    copy__13: "f32[96]" = torch.ops.aten.copy_.default(primals_266, add_22);  primals_266 = add_22 = None
    copy__14: "f32[96]" = torch.ops.aten.copy_.default(primals_267, add_23);  primals_267 = add_23 = None
    copy__15: "i64[]" = torch.ops.aten.copy_.default(primals_268, add_25);  primals_268 = add_25 = None
    copy__16: "f32[24]" = torch.ops.aten.copy_.default(primals_269, add_27);  primals_269 = add_27 = None
    copy__17: "f32[24]" = torch.ops.aten.copy_.default(primals_270, add_28);  primals_270 = add_28 = None
    copy__18: "i64[]" = torch.ops.aten.copy_.default(primals_271, add_30);  primals_271 = add_30 = None
    copy__19: "f32[144]" = torch.ops.aten.copy_.default(primals_272, add_32);  primals_272 = add_32 = None
    copy__20: "f32[144]" = torch.ops.aten.copy_.default(primals_273, add_33);  primals_273 = add_33 = None
    copy__21: "i64[]" = torch.ops.aten.copy_.default(primals_274, add_35);  primals_274 = add_35 = None
    copy__22: "f32[144]" = torch.ops.aten.copy_.default(primals_275, add_37);  primals_275 = add_37 = None
    copy__23: "f32[144]" = torch.ops.aten.copy_.default(primals_276, add_38);  primals_276 = add_38 = None
    copy__24: "i64[]" = torch.ops.aten.copy_.default(primals_277, add_40);  primals_277 = add_40 = None
    copy__25: "f32[24]" = torch.ops.aten.copy_.default(primals_278, add_42);  primals_278 = add_42 = None
    copy__26: "f32[24]" = torch.ops.aten.copy_.default(primals_279, add_43);  primals_279 = add_43 = None
    copy__27: "i64[]" = torch.ops.aten.copy_.default(primals_280, add_46);  primals_280 = add_46 = None
    copy__28: "f32[144]" = torch.ops.aten.copy_.default(primals_281, add_48);  primals_281 = add_48 = None
    copy__29: "f32[144]" = torch.ops.aten.copy_.default(primals_282, add_49);  primals_282 = add_49 = None
    copy__30: "i64[]" = torch.ops.aten.copy_.default(primals_283, add_51);  primals_283 = add_51 = None
    copy__31: "f32[144]" = torch.ops.aten.copy_.default(primals_284, add_53);  primals_284 = add_53 = None
    copy__32: "f32[144]" = torch.ops.aten.copy_.default(primals_285, add_54);  primals_285 = add_54 = None
    copy__33: "i64[]" = torch.ops.aten.copy_.default(primals_286, add_56);  primals_286 = add_56 = None
    copy__34: "f32[40]" = torch.ops.aten.copy_.default(primals_287, add_58);  primals_287 = add_58 = None
    copy__35: "f32[40]" = torch.ops.aten.copy_.default(primals_288, add_59);  primals_288 = add_59 = None
    copy__36: "i64[]" = torch.ops.aten.copy_.default(primals_289, add_61);  primals_289 = add_61 = None
    copy__37: "f32[240]" = torch.ops.aten.copy_.default(primals_290, add_63);  primals_290 = add_63 = None
    copy__38: "f32[240]" = torch.ops.aten.copy_.default(primals_291, add_64);  primals_291 = add_64 = None
    copy__39: "i64[]" = torch.ops.aten.copy_.default(primals_292, add_66);  primals_292 = add_66 = None
    copy__40: "f32[240]" = torch.ops.aten.copy_.default(primals_293, add_68);  primals_293 = add_68 = None
    copy__41: "f32[240]" = torch.ops.aten.copy_.default(primals_294, add_69);  primals_294 = add_69 = None
    copy__42: "i64[]" = torch.ops.aten.copy_.default(primals_295, add_71);  primals_295 = add_71 = None
    copy__43: "f32[40]" = torch.ops.aten.copy_.default(primals_296, add_73);  primals_296 = add_73 = None
    copy__44: "f32[40]" = torch.ops.aten.copy_.default(primals_297, add_74);  primals_297 = add_74 = None
    copy__45: "i64[]" = torch.ops.aten.copy_.default(primals_298, add_77);  primals_298 = add_77 = None
    copy__46: "f32[240]" = torch.ops.aten.copy_.default(primals_299, add_79);  primals_299 = add_79 = None
    copy__47: "f32[240]" = torch.ops.aten.copy_.default(primals_300, add_80);  primals_300 = add_80 = None
    copy__48: "i64[]" = torch.ops.aten.copy_.default(primals_301, add_82);  primals_301 = add_82 = None
    copy__49: "f32[240]" = torch.ops.aten.copy_.default(primals_302, add_84);  primals_302 = add_84 = None
    copy__50: "f32[240]" = torch.ops.aten.copy_.default(primals_303, add_85);  primals_303 = add_85 = None
    copy__51: "i64[]" = torch.ops.aten.copy_.default(primals_304, add_87);  primals_304 = add_87 = None
    copy__52: "f32[80]" = torch.ops.aten.copy_.default(primals_305, add_89);  primals_305 = add_89 = None
    copy__53: "f32[80]" = torch.ops.aten.copy_.default(primals_306, add_90);  primals_306 = add_90 = None
    copy__54: "i64[]" = torch.ops.aten.copy_.default(primals_307, add_92);  primals_307 = add_92 = None
    copy__55: "f32[480]" = torch.ops.aten.copy_.default(primals_308, add_94);  primals_308 = add_94 = None
    copy__56: "f32[480]" = torch.ops.aten.copy_.default(primals_309, add_95);  primals_309 = add_95 = None
    copy__57: "i64[]" = torch.ops.aten.copy_.default(primals_310, add_97);  primals_310 = add_97 = None
    copy__58: "f32[480]" = torch.ops.aten.copy_.default(primals_311, add_99);  primals_311 = add_99 = None
    copy__59: "f32[480]" = torch.ops.aten.copy_.default(primals_312, add_100);  primals_312 = add_100 = None
    copy__60: "i64[]" = torch.ops.aten.copy_.default(primals_313, add_102);  primals_313 = add_102 = None
    copy__61: "f32[80]" = torch.ops.aten.copy_.default(primals_314, add_104);  primals_314 = add_104 = None
    copy__62: "f32[80]" = torch.ops.aten.copy_.default(primals_315, add_105);  primals_315 = add_105 = None
    copy__63: "i64[]" = torch.ops.aten.copy_.default(primals_316, add_108);  primals_316 = add_108 = None
    copy__64: "f32[480]" = torch.ops.aten.copy_.default(primals_317, add_110);  primals_317 = add_110 = None
    copy__65: "f32[480]" = torch.ops.aten.copy_.default(primals_318, add_111);  primals_318 = add_111 = None
    copy__66: "i64[]" = torch.ops.aten.copy_.default(primals_319, add_113);  primals_319 = add_113 = None
    copy__67: "f32[480]" = torch.ops.aten.copy_.default(primals_320, add_115);  primals_320 = add_115 = None
    copy__68: "f32[480]" = torch.ops.aten.copy_.default(primals_321, add_116);  primals_321 = add_116 = None
    copy__69: "i64[]" = torch.ops.aten.copy_.default(primals_322, add_118);  primals_322 = add_118 = None
    copy__70: "f32[80]" = torch.ops.aten.copy_.default(primals_323, add_120);  primals_323 = add_120 = None
    copy__71: "f32[80]" = torch.ops.aten.copy_.default(primals_324, add_121);  primals_324 = add_121 = None
    copy__72: "i64[]" = torch.ops.aten.copy_.default(primals_325, add_124);  primals_325 = add_124 = None
    copy__73: "f32[480]" = torch.ops.aten.copy_.default(primals_326, add_126);  primals_326 = add_126 = None
    copy__74: "f32[480]" = torch.ops.aten.copy_.default(primals_327, add_127);  primals_327 = add_127 = None
    copy__75: "i64[]" = torch.ops.aten.copy_.default(primals_328, add_129);  primals_328 = add_129 = None
    copy__76: "f32[480]" = torch.ops.aten.copy_.default(primals_329, add_131);  primals_329 = add_131 = None
    copy__77: "f32[480]" = torch.ops.aten.copy_.default(primals_330, add_132);  primals_330 = add_132 = None
    copy__78: "i64[]" = torch.ops.aten.copy_.default(primals_331, add_134);  primals_331 = add_134 = None
    copy__79: "f32[80]" = torch.ops.aten.copy_.default(primals_332, add_136);  primals_332 = add_136 = None
    copy__80: "f32[80]" = torch.ops.aten.copy_.default(primals_333, add_137);  primals_333 = add_137 = None
    copy__81: "i64[]" = torch.ops.aten.copy_.default(primals_334, add_140);  primals_334 = add_140 = None
    copy__82: "f32[480]" = torch.ops.aten.copy_.default(primals_335, add_142);  primals_335 = add_142 = None
    copy__83: "f32[480]" = torch.ops.aten.copy_.default(primals_336, add_143);  primals_336 = add_143 = None
    copy__84: "i64[]" = torch.ops.aten.copy_.default(primals_337, add_145);  primals_337 = add_145 = None
    copy__85: "f32[480]" = torch.ops.aten.copy_.default(primals_338, add_147);  primals_338 = add_147 = None
    copy__86: "f32[480]" = torch.ops.aten.copy_.default(primals_339, add_148);  primals_339 = add_148 = None
    copy__87: "i64[]" = torch.ops.aten.copy_.default(primals_340, add_150);  primals_340 = add_150 = None
    copy__88: "f32[112]" = torch.ops.aten.copy_.default(primals_341, add_152);  primals_341 = add_152 = None
    copy__89: "f32[112]" = torch.ops.aten.copy_.default(primals_342, add_153);  primals_342 = add_153 = None
    copy__90: "i64[]" = torch.ops.aten.copy_.default(primals_343, add_155);  primals_343 = add_155 = None
    copy__91: "f32[672]" = torch.ops.aten.copy_.default(primals_344, add_157);  primals_344 = add_157 = None
    copy__92: "f32[672]" = torch.ops.aten.copy_.default(primals_345, add_158);  primals_345 = add_158 = None
    copy__93: "i64[]" = torch.ops.aten.copy_.default(primals_346, add_160);  primals_346 = add_160 = None
    copy__94: "f32[672]" = torch.ops.aten.copy_.default(primals_347, add_162);  primals_347 = add_162 = None
    copy__95: "f32[672]" = torch.ops.aten.copy_.default(primals_348, add_163);  primals_348 = add_163 = None
    copy__96: "i64[]" = torch.ops.aten.copy_.default(primals_349, add_165);  primals_349 = add_165 = None
    copy__97: "f32[112]" = torch.ops.aten.copy_.default(primals_350, add_167);  primals_350 = add_167 = None
    copy__98: "f32[112]" = torch.ops.aten.copy_.default(primals_351, add_168);  primals_351 = add_168 = None
    copy__99: "i64[]" = torch.ops.aten.copy_.default(primals_352, add_171);  primals_352 = add_171 = None
    copy__100: "f32[672]" = torch.ops.aten.copy_.default(primals_353, add_173);  primals_353 = add_173 = None
    copy__101: "f32[672]" = torch.ops.aten.copy_.default(primals_354, add_174);  primals_354 = add_174 = None
    copy__102: "i64[]" = torch.ops.aten.copy_.default(primals_355, add_176);  primals_355 = add_176 = None
    copy__103: "f32[672]" = torch.ops.aten.copy_.default(primals_356, add_178);  primals_356 = add_178 = None
    copy__104: "f32[672]" = torch.ops.aten.copy_.default(primals_357, add_179);  primals_357 = add_179 = None
    copy__105: "i64[]" = torch.ops.aten.copy_.default(primals_358, add_181);  primals_358 = add_181 = None
    copy__106: "f32[112]" = torch.ops.aten.copy_.default(primals_359, add_183);  primals_359 = add_183 = None
    copy__107: "f32[112]" = torch.ops.aten.copy_.default(primals_360, add_184);  primals_360 = add_184 = None
    copy__108: "i64[]" = torch.ops.aten.copy_.default(primals_361, add_187);  primals_361 = add_187 = None
    copy__109: "f32[672]" = torch.ops.aten.copy_.default(primals_362, add_189);  primals_362 = add_189 = None
    copy__110: "f32[672]" = torch.ops.aten.copy_.default(primals_363, add_190);  primals_363 = add_190 = None
    copy__111: "i64[]" = torch.ops.aten.copy_.default(primals_364, add_192);  primals_364 = add_192 = None
    copy__112: "f32[672]" = torch.ops.aten.copy_.default(primals_365, add_194);  primals_365 = add_194 = None
    copy__113: "f32[672]" = torch.ops.aten.copy_.default(primals_366, add_195);  primals_366 = add_195 = None
    copy__114: "i64[]" = torch.ops.aten.copy_.default(primals_367, add_197);  primals_367 = add_197 = None
    copy__115: "f32[112]" = torch.ops.aten.copy_.default(primals_368, add_199);  primals_368 = add_199 = None
    copy__116: "f32[112]" = torch.ops.aten.copy_.default(primals_369, add_200);  primals_369 = add_200 = None
    copy__117: "i64[]" = torch.ops.aten.copy_.default(primals_370, add_203);  primals_370 = add_203 = None
    copy__118: "f32[672]" = torch.ops.aten.copy_.default(primals_371, add_205);  primals_371 = add_205 = None
    copy__119: "f32[672]" = torch.ops.aten.copy_.default(primals_372, add_206);  primals_372 = add_206 = None
    copy__120: "i64[]" = torch.ops.aten.copy_.default(primals_373, add_208);  primals_373 = add_208 = None
    copy__121: "f32[672]" = torch.ops.aten.copy_.default(primals_374, add_210);  primals_374 = add_210 = None
    copy__122: "f32[672]" = torch.ops.aten.copy_.default(primals_375, add_211);  primals_375 = add_211 = None
    copy__123: "i64[]" = torch.ops.aten.copy_.default(primals_376, add_213);  primals_376 = add_213 = None
    copy__124: "f32[192]" = torch.ops.aten.copy_.default(primals_377, add_215);  primals_377 = add_215 = None
    copy__125: "f32[192]" = torch.ops.aten.copy_.default(primals_378, add_216);  primals_378 = add_216 = None
    copy__126: "i64[]" = torch.ops.aten.copy_.default(primals_379, add_218);  primals_379 = add_218 = None
    copy__127: "f32[1152]" = torch.ops.aten.copy_.default(primals_380, add_220);  primals_380 = add_220 = None
    copy__128: "f32[1152]" = torch.ops.aten.copy_.default(primals_381, add_221);  primals_381 = add_221 = None
    copy__129: "i64[]" = torch.ops.aten.copy_.default(primals_382, add_223);  primals_382 = add_223 = None
    copy__130: "f32[1152]" = torch.ops.aten.copy_.default(primals_383, add_225);  primals_383 = add_225 = None
    copy__131: "f32[1152]" = torch.ops.aten.copy_.default(primals_384, add_226);  primals_384 = add_226 = None
    copy__132: "i64[]" = torch.ops.aten.copy_.default(primals_385, add_228);  primals_385 = add_228 = None
    copy__133: "f32[192]" = torch.ops.aten.copy_.default(primals_386, add_230);  primals_386 = add_230 = None
    copy__134: "f32[192]" = torch.ops.aten.copy_.default(primals_387, add_231);  primals_387 = add_231 = None
    copy__135: "i64[]" = torch.ops.aten.copy_.default(primals_388, add_234);  primals_388 = add_234 = None
    copy__136: "f32[1152]" = torch.ops.aten.copy_.default(primals_389, add_236);  primals_389 = add_236 = None
    copy__137: "f32[1152]" = torch.ops.aten.copy_.default(primals_390, add_237);  primals_390 = add_237 = None
    copy__138: "i64[]" = torch.ops.aten.copy_.default(primals_391, add_239);  primals_391 = add_239 = None
    copy__139: "f32[1152]" = torch.ops.aten.copy_.default(primals_392, add_241);  primals_392 = add_241 = None
    copy__140: "f32[1152]" = torch.ops.aten.copy_.default(primals_393, add_242);  primals_393 = add_242 = None
    copy__141: "i64[]" = torch.ops.aten.copy_.default(primals_394, add_244);  primals_394 = add_244 = None
    copy__142: "f32[192]" = torch.ops.aten.copy_.default(primals_395, add_246);  primals_395 = add_246 = None
    copy__143: "f32[192]" = torch.ops.aten.copy_.default(primals_396, add_247);  primals_396 = add_247 = None
    copy__144: "i64[]" = torch.ops.aten.copy_.default(primals_397, add_250);  primals_397 = add_250 = None
    copy__145: "f32[1152]" = torch.ops.aten.copy_.default(primals_398, add_252);  primals_398 = add_252 = None
    copy__146: "f32[1152]" = torch.ops.aten.copy_.default(primals_399, add_253);  primals_399 = add_253 = None
    copy__147: "i64[]" = torch.ops.aten.copy_.default(primals_400, add_255);  primals_400 = add_255 = None
    copy__148: "f32[1152]" = torch.ops.aten.copy_.default(primals_401, add_257);  primals_401 = add_257 = None
    copy__149: "f32[1152]" = torch.ops.aten.copy_.default(primals_402, add_258);  primals_402 = add_258 = None
    copy__150: "i64[]" = torch.ops.aten.copy_.default(primals_403, add_260);  primals_403 = add_260 = None
    copy__151: "f32[192]" = torch.ops.aten.copy_.default(primals_404, add_262);  primals_404 = add_262 = None
    copy__152: "f32[192]" = torch.ops.aten.copy_.default(primals_405, add_263);  primals_405 = add_263 = None
    copy__153: "i64[]" = torch.ops.aten.copy_.default(primals_406, add_266);  primals_406 = add_266 = None
    copy__154: "f32[1152]" = torch.ops.aten.copy_.default(primals_407, add_268);  primals_407 = add_268 = None
    copy__155: "f32[1152]" = torch.ops.aten.copy_.default(primals_408, add_269);  primals_408 = add_269 = None
    copy__156: "i64[]" = torch.ops.aten.copy_.default(primals_409, add_271);  primals_409 = add_271 = None
    copy__157: "f32[1152]" = torch.ops.aten.copy_.default(primals_410, add_273);  primals_410 = add_273 = None
    copy__158: "f32[1152]" = torch.ops.aten.copy_.default(primals_411, add_274);  primals_411 = add_274 = None
    copy__159: "i64[]" = torch.ops.aten.copy_.default(primals_412, add_276);  primals_412 = add_276 = None
    copy__160: "f32[192]" = torch.ops.aten.copy_.default(primals_413, add_278);  primals_413 = add_278 = None
    copy__161: "f32[192]" = torch.ops.aten.copy_.default(primals_414, add_279);  primals_414 = add_279 = None
    copy__162: "i64[]" = torch.ops.aten.copy_.default(primals_415, add_282);  primals_415 = add_282 = None
    copy__163: "f32[1152]" = torch.ops.aten.copy_.default(primals_416, add_284);  primals_416 = add_284 = None
    copy__164: "f32[1152]" = torch.ops.aten.copy_.default(primals_417, add_285);  primals_417 = add_285 = None
    copy__165: "i64[]" = torch.ops.aten.copy_.default(primals_418, add_287);  primals_418 = add_287 = None
    copy__166: "f32[1152]" = torch.ops.aten.copy_.default(primals_419, add_289);  primals_419 = add_289 = None
    copy__167: "f32[1152]" = torch.ops.aten.copy_.default(primals_420, add_290);  primals_420 = add_290 = None
    copy__168: "i64[]" = torch.ops.aten.copy_.default(primals_421, add_292);  primals_421 = add_292 = None
    copy__169: "f32[320]" = torch.ops.aten.copy_.default(primals_422, add_294);  primals_422 = add_294 = None
    copy__170: "f32[320]" = torch.ops.aten.copy_.default(primals_423, add_295);  primals_423 = add_295 = None
    copy__171: "i64[]" = torch.ops.aten.copy_.default(primals_424, add_297);  primals_424 = add_297 = None
    copy__172: "f32[1280]" = torch.ops.aten.copy_.default(primals_425, add_299);  primals_425 = add_299 = None
    copy__173: "f32[1280]" = torch.ops.aten.copy_.default(primals_426, add_300);  primals_426 = add_300 = None
    return [addmm, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_118, primals_119, primals_121, primals_123, primals_124, primals_125, primals_126, primals_128, primals_130, primals_131, primals_132, primals_133, primals_135, primals_137, primals_138, primals_139, primals_140, primals_142, primals_144, primals_145, primals_146, primals_147, primals_149, primals_151, primals_152, primals_153, primals_154, primals_156, primals_158, primals_159, primals_160, primals_161, primals_163, primals_165, primals_166, primals_167, primals_168, primals_170, primals_172, primals_173, primals_174, primals_175, primals_177, primals_179, primals_180, primals_181, primals_182, primals_184, primals_186, primals_187, primals_188, primals_189, primals_191, primals_193, primals_194, primals_195, primals_196, primals_198, primals_200, primals_201, primals_202, primals_203, primals_205, primals_207, primals_208, primals_209, primals_210, primals_212, primals_214, primals_215, primals_216, primals_217, primals_219, primals_221, primals_222, primals_223, primals_224, primals_226, primals_228, primals_229, primals_230, primals_231, primals_233, primals_235, primals_236, primals_237, primals_238, primals_240, primals_242, primals_243, primals_244, primals_245, primals_247, primals_249, primals_250, primals_427, convolution, squeeze_1, mul_7, convolution_1, squeeze_4, add_9, mean, convolution_2, mul_16, convolution_3, mul_17, convolution_4, squeeze_7, add_14, convolution_5, squeeze_10, mul_32, convolution_6, squeeze_13, add_24, mean_1, convolution_7, mul_41, convolution_8, mul_42, convolution_9, squeeze_16, add_29, convolution_10, squeeze_19, mul_57, convolution_11, squeeze_22, add_39, mean_2, convolution_12, mul_66, convolution_13, mul_67, convolution_14, squeeze_25, add_45, convolution_15, squeeze_28, mul_82, convolution_16, squeeze_31, add_55, mean_3, convolution_17, mul_91, convolution_18, mul_92, convolution_19, squeeze_34, add_60, convolution_20, squeeze_37, mul_107, convolution_21, squeeze_40, add_70, mean_4, convolution_22, mul_116, convolution_23, mul_117, convolution_24, squeeze_43, add_76, convolution_25, squeeze_46, mul_132, convolution_26, squeeze_49, add_86, mean_5, convolution_27, mul_141, convolution_28, mul_142, convolution_29, squeeze_52, add_91, convolution_30, squeeze_55, mul_157, convolution_31, squeeze_58, add_101, mean_6, convolution_32, mul_166, convolution_33, mul_167, convolution_34, squeeze_61, add_107, convolution_35, squeeze_64, mul_182, convolution_36, squeeze_67, add_117, mean_7, convolution_37, mul_191, convolution_38, mul_192, convolution_39, squeeze_70, add_123, convolution_40, squeeze_73, mul_207, convolution_41, squeeze_76, add_133, mean_8, convolution_42, mul_216, convolution_43, mul_217, convolution_44, squeeze_79, add_139, convolution_45, squeeze_82, mul_232, convolution_46, squeeze_85, add_149, mean_9, convolution_47, mul_241, convolution_48, mul_242, convolution_49, squeeze_88, add_154, convolution_50, squeeze_91, mul_257, convolution_51, squeeze_94, add_164, mean_10, convolution_52, mul_266, convolution_53, mul_267, convolution_54, squeeze_97, add_170, convolution_55, squeeze_100, mul_282, convolution_56, squeeze_103, add_180, mean_11, convolution_57, mul_291, convolution_58, mul_292, convolution_59, squeeze_106, add_186, convolution_60, squeeze_109, mul_307, convolution_61, squeeze_112, add_196, mean_12, convolution_62, mul_316, convolution_63, mul_317, convolution_64, squeeze_115, add_202, convolution_65, squeeze_118, mul_332, convolution_66, squeeze_121, add_212, mean_13, convolution_67, mul_341, convolution_68, mul_342, convolution_69, squeeze_124, add_217, convolution_70, squeeze_127, mul_357, convolution_71, squeeze_130, add_227, mean_14, convolution_72, mul_366, convolution_73, mul_367, convolution_74, squeeze_133, add_233, convolution_75, squeeze_136, mul_382, convolution_76, squeeze_139, add_243, mean_15, convolution_77, mul_391, convolution_78, mul_392, convolution_79, squeeze_142, add_249, convolution_80, squeeze_145, mul_407, convolution_81, squeeze_148, add_259, mean_16, convolution_82, mul_416, convolution_83, mul_417, convolution_84, squeeze_151, add_265, convolution_85, squeeze_154, mul_432, convolution_86, squeeze_157, add_275, mean_17, convolution_87, mul_441, convolution_88, mul_442, convolution_89, squeeze_160, add_281, convolution_90, squeeze_163, mul_457, convolution_91, squeeze_166, add_291, mean_18, convolution_92, mul_466, convolution_93, mul_467, convolution_94, squeeze_169, add_296, convolution_95, squeeze_172, view, permute_1, mul_484, unsqueeze_234, unsqueeze_246, unsqueeze_258, mul_524, unsqueeze_270, unsqueeze_282, unsqueeze_294, mul_564, unsqueeze_306, unsqueeze_318, unsqueeze_330, mul_604, unsqueeze_342, unsqueeze_354, unsqueeze_366, mul_644, unsqueeze_378, unsqueeze_390, unsqueeze_402, mul_684, unsqueeze_414, unsqueeze_426, unsqueeze_438, mul_724, unsqueeze_450, unsqueeze_462, unsqueeze_474, mul_764, unsqueeze_486, unsqueeze_498, unsqueeze_510, mul_804, unsqueeze_522, unsqueeze_534, unsqueeze_546, mul_844, unsqueeze_558, unsqueeze_570, unsqueeze_582, mul_884, unsqueeze_594, unsqueeze_606, unsqueeze_618, mul_924, unsqueeze_630, unsqueeze_642, unsqueeze_654, mul_964, unsqueeze_666, unsqueeze_678, unsqueeze_690, mul_1004, unsqueeze_702, unsqueeze_714, unsqueeze_726, mul_1044, unsqueeze_738, unsqueeze_750, unsqueeze_762, mul_1084, unsqueeze_774, unsqueeze_786, unsqueeze_798, mul_1124, unsqueeze_810, unsqueeze_822, unsqueeze_834, mul_1164, unsqueeze_846, unsqueeze_858, unsqueeze_870, mul_1204, unsqueeze_882, unsqueeze_894, unsqueeze_906, mul_1244, unsqueeze_918]
    