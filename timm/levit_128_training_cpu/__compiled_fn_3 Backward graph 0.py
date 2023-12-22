from __future__ import annotations



def forward(self, primals_15: "f32[16, 3, 3, 3]", primals_16: "f32[16]", primals_18: "f32[32, 16, 3, 3]", primals_19: "f32[32]", primals_21: "f32[64, 32, 3, 3]", primals_22: "f32[64]", primals_24: "f32[128, 64, 3, 3]", primals_25: "f32[128]", primals_28: "f32[256]", primals_31: "f32[128]", primals_34: "f32[256]", primals_37: "f32[128]", primals_40: "f32[256]", primals_43: "f32[128]", primals_46: "f32[256]", primals_49: "f32[128]", primals_52: "f32[256]", primals_55: "f32[128]", primals_58: "f32[256]", primals_61: "f32[128]", primals_64: "f32[256]", primals_67: "f32[128]", primals_70: "f32[256]", primals_73: "f32[128]", primals_76: "f32[640]", primals_79: "f32[128]", primals_82: "f32[256]", primals_85: "f32[512]", primals_88: "f32[256]", primals_91: "f32[512]", primals_94: "f32[256]", primals_97: "f32[512]", primals_100: "f32[256]", primals_103: "f32[512]", primals_106: "f32[256]", primals_109: "f32[512]", primals_112: "f32[256]", primals_115: "f32[512]", primals_118: "f32[256]", primals_121: "f32[512]", primals_124: "f32[256]", primals_127: "f32[512]", primals_130: "f32[256]", primals_133: "f32[512]", primals_136: "f32[256]", primals_139: "f32[1280]", primals_142: "f32[256]", primals_145: "f32[384]", primals_148: "f32[768]", primals_151: "f32[384]", primals_154: "f32[768]", primals_157: "f32[384]", primals_160: "f32[768]", primals_163: "f32[384]", primals_166: "f32[768]", primals_169: "f32[384]", primals_172: "f32[768]", primals_175: "f32[384]", primals_178: "f32[768]", primals_181: "f32[384]", primals_184: "f32[768]", primals_187: "f32[384]", primals_190: "f32[768]", primals_193: "f32[384]", primals_196: "f32[768]", primals_199: "f32[384]", primals_201: "f32[384]", primals_205: "f32[384]", primals_209: "i64[196, 196]", primals_210: "i64[196, 196]", primals_211: "i64[196, 196]", primals_212: "i64[196, 196]", primals_213: "i64[49, 196]", primals_214: "i64[49, 49]", primals_215: "i64[49, 49]", primals_216: "i64[49, 49]", primals_217: "i64[49, 49]", primals_218: "i64[16, 49]", primals_219: "i64[16, 16]", primals_220: "i64[16, 16]", primals_221: "i64[16, 16]", primals_222: "i64[16, 16]", primals_415: "f32[8, 3, 224, 224]", convolution: "f32[8, 16, 112, 112]", squeeze_1: "f32[16]", add_4: "f32[8, 16, 112, 112]", div: "f32[8, 16, 112, 112]", convolution_1: "f32[8, 32, 56, 56]", squeeze_4: "f32[32]", add_10: "f32[8, 32, 56, 56]", div_1: "f32[8, 32, 56, 56]", convolution_2: "f32[8, 64, 28, 28]", squeeze_7: "f32[64]", add_16: "f32[8, 64, 28, 28]", div_2: "f32[8, 64, 28, 28]", convolution_3: "f32[8, 128, 14, 14]", squeeze_10: "f32[128]", view_1: "f32[1568, 128]", mm: "f32[1568, 256]", squeeze_13: "f32[256]", view_12: "f32[8, 196, 128]", view_13: "f32[1568, 128]", mm_1: "f32[1568, 128]", squeeze_16: "f32[128]", view_17: "f32[1568, 128]", mm_2: "f32[1568, 256]", squeeze_19: "f32[256]", view_20: "f32[8, 196, 256]", view_21: "f32[1568, 256]", mm_3: "f32[1568, 128]", squeeze_22: "f32[128]", view_25: "f32[1568, 128]", mm_4: "f32[1568, 256]", squeeze_25: "f32[256]", view_36: "f32[8, 196, 128]", view_37: "f32[1568, 128]", mm_5: "f32[1568, 128]", squeeze_28: "f32[128]", view_41: "f32[1568, 128]", mm_6: "f32[1568, 256]", squeeze_31: "f32[256]", view_44: "f32[8, 196, 256]", view_45: "f32[1568, 256]", mm_7: "f32[1568, 128]", squeeze_34: "f32[128]", view_49: "f32[1568, 128]", mm_8: "f32[1568, 256]", squeeze_37: "f32[256]", view_60: "f32[8, 196, 128]", view_61: "f32[1568, 128]", mm_9: "f32[1568, 128]", squeeze_40: "f32[128]", view_65: "f32[1568, 128]", mm_10: "f32[1568, 256]", squeeze_43: "f32[256]", view_68: "f32[8, 196, 256]", view_69: "f32[1568, 256]", mm_11: "f32[1568, 128]", squeeze_46: "f32[128]", view_73: "f32[1568, 128]", mm_12: "f32[1568, 256]", squeeze_49: "f32[256]", view_84: "f32[8, 196, 128]", view_85: "f32[1568, 128]", mm_13: "f32[1568, 128]", squeeze_52: "f32[128]", view_89: "f32[1568, 128]", mm_14: "f32[1568, 256]", squeeze_55: "f32[256]", view_92: "f32[8, 196, 256]", view_93: "f32[1568, 256]", mm_15: "f32[1568, 128]", squeeze_58: "f32[128]", view_97: "f32[1568, 128]", mm_16: "f32[1568, 640]", squeeze_61: "f32[640]", view_104: "f32[392, 128]", mm_17: "f32[392, 128]", squeeze_64: "f32[128]", view_115: "f32[8, 49, 512]", view_116: "f32[392, 512]", mm_18: "f32[392, 256]", squeeze_67: "f32[256]", view_120: "f32[392, 256]", mm_19: "f32[392, 512]", squeeze_70: "f32[512]", view_123: "f32[8, 49, 512]", view_124: "f32[392, 512]", mm_20: "f32[392, 256]", squeeze_73: "f32[256]", view_128: "f32[392, 256]", mm_21: "f32[392, 512]", squeeze_76: "f32[512]", view_139: "f32[8, 49, 256]", view_140: "f32[392, 256]", mm_22: "f32[392, 256]", squeeze_79: "f32[256]", view_144: "f32[392, 256]", mm_23: "f32[392, 512]", squeeze_82: "f32[512]", view_147: "f32[8, 49, 512]", view_148: "f32[392, 512]", mm_24: "f32[392, 256]", squeeze_85: "f32[256]", view_152: "f32[392, 256]", mm_25: "f32[392, 512]", squeeze_88: "f32[512]", view_163: "f32[8, 49, 256]", view_164: "f32[392, 256]", mm_26: "f32[392, 256]", squeeze_91: "f32[256]", view_168: "f32[392, 256]", mm_27: "f32[392, 512]", squeeze_94: "f32[512]", view_171: "f32[8, 49, 512]", view_172: "f32[392, 512]", mm_28: "f32[392, 256]", squeeze_97: "f32[256]", view_176: "f32[392, 256]", mm_29: "f32[392, 512]", squeeze_100: "f32[512]", view_187: "f32[8, 49, 256]", view_188: "f32[392, 256]", mm_30: "f32[392, 256]", squeeze_103: "f32[256]", view_192: "f32[392, 256]", mm_31: "f32[392, 512]", squeeze_106: "f32[512]", view_195: "f32[8, 49, 512]", view_196: "f32[392, 512]", mm_32: "f32[392, 256]", squeeze_109: "f32[256]", view_200: "f32[392, 256]", mm_33: "f32[392, 512]", squeeze_112: "f32[512]", view_211: "f32[8, 49, 256]", view_212: "f32[392, 256]", mm_34: "f32[392, 256]", squeeze_115: "f32[256]", view_216: "f32[392, 256]", mm_35: "f32[392, 512]", squeeze_118: "f32[512]", view_219: "f32[8, 49, 512]", view_220: "f32[392, 512]", mm_36: "f32[392, 256]", squeeze_121: "f32[256]", view_224: "f32[392, 256]", mm_37: "f32[392, 1280]", squeeze_124: "f32[1280]", view_231: "f32[128, 256]", mm_38: "f32[128, 256]", squeeze_127: "f32[256]", view_242: "f32[8, 16, 1024]", view_243: "f32[128, 1024]", mm_39: "f32[128, 384]", squeeze_130: "f32[384]", view_247: "f32[128, 384]", mm_40: "f32[128, 768]", squeeze_133: "f32[768]", view_250: "f32[8, 16, 768]", view_251: "f32[128, 768]", mm_41: "f32[128, 384]", squeeze_136: "f32[384]", view_255: "f32[128, 384]", mm_42: "f32[128, 768]", squeeze_139: "f32[768]", view_266: "f32[8, 16, 384]", view_267: "f32[128, 384]", mm_43: "f32[128, 384]", squeeze_142: "f32[384]", view_271: "f32[128, 384]", mm_44: "f32[128, 768]", squeeze_145: "f32[768]", view_274: "f32[8, 16, 768]", view_275: "f32[128, 768]", mm_45: "f32[128, 384]", squeeze_148: "f32[384]", view_279: "f32[128, 384]", mm_46: "f32[128, 768]", squeeze_151: "f32[768]", view_290: "f32[8, 16, 384]", view_291: "f32[128, 384]", mm_47: "f32[128, 384]", squeeze_154: "f32[384]", view_295: "f32[128, 384]", mm_48: "f32[128, 768]", squeeze_157: "f32[768]", view_298: "f32[8, 16, 768]", view_299: "f32[128, 768]", mm_49: "f32[128, 384]", squeeze_160: "f32[384]", view_303: "f32[128, 384]", mm_50: "f32[128, 768]", squeeze_163: "f32[768]", view_314: "f32[8, 16, 384]", view_315: "f32[128, 384]", mm_51: "f32[128, 384]", squeeze_166: "f32[384]", view_319: "f32[128, 384]", mm_52: "f32[128, 768]", squeeze_169: "f32[768]", view_322: "f32[8, 16, 768]", view_323: "f32[128, 768]", mm_53: "f32[128, 384]", squeeze_172: "f32[384]", view_327: "f32[128, 384]", mm_54: "f32[128, 768]", squeeze_175: "f32[768]", view_338: "f32[8, 16, 384]", view_339: "f32[128, 384]", mm_55: "f32[128, 384]", squeeze_178: "f32[384]", view_343: "f32[128, 384]", mm_56: "f32[128, 768]", squeeze_181: "f32[768]", view_346: "f32[8, 16, 768]", view_347: "f32[128, 768]", mm_57: "f32[128, 384]", squeeze_184: "f32[384]", mean: "f32[8, 384]", clone_81: "f32[8, 384]", clone_82: "f32[8, 384]", permute_117: "f32[1000, 384]", permute_121: "f32[1000, 384]", unsqueeze_25: "f32[1, 384]", permute_127: "f32[384, 768]", unsqueeze_29: "f32[1, 768]", permute_131: "f32[768, 384]", unsqueeze_33: "f32[1, 384]", permute_135: "f32[384, 384]", permute_138: "f32[96, 16, 16]", permute_139: "f32[96, 32, 16]", alias_14: "f32[8, 12, 16, 16]", permute_140: "f32[96, 16, 16]", permute_141: "f32[96, 16, 16]", unsqueeze_37: "f32[1, 768]", permute_147: "f32[768, 384]", unsqueeze_41: "f32[1, 384]", permute_151: "f32[384, 768]", unsqueeze_45: "f32[1, 768]", permute_155: "f32[768, 384]", unsqueeze_49: "f32[1, 384]", permute_159: "f32[384, 384]", permute_162: "f32[96, 16, 16]", permute_163: "f32[96, 32, 16]", alias_15: "f32[8, 12, 16, 16]", permute_164: "f32[96, 16, 16]", permute_165: "f32[96, 16, 16]", unsqueeze_53: "f32[1, 768]", permute_171: "f32[768, 384]", unsqueeze_57: "f32[1, 384]", permute_175: "f32[384, 768]", unsqueeze_61: "f32[1, 768]", permute_179: "f32[768, 384]", unsqueeze_65: "f32[1, 384]", permute_183: "f32[384, 384]", permute_186: "f32[96, 16, 16]", permute_187: "f32[96, 32, 16]", alias_16: "f32[8, 12, 16, 16]", permute_188: "f32[96, 16, 16]", permute_189: "f32[96, 16, 16]", unsqueeze_69: "f32[1, 768]", permute_195: "f32[768, 384]", unsqueeze_73: "f32[1, 384]", permute_199: "f32[384, 768]", unsqueeze_77: "f32[1, 768]", permute_203: "f32[768, 384]", unsqueeze_81: "f32[1, 384]", permute_207: "f32[384, 384]", permute_210: "f32[96, 16, 16]", permute_211: "f32[96, 32, 16]", alias_17: "f32[8, 12, 16, 16]", permute_212: "f32[96, 16, 16]", permute_213: "f32[96, 16, 16]", unsqueeze_85: "f32[1, 768]", permute_219: "f32[768, 384]", unsqueeze_89: "f32[1, 384]", permute_223: "f32[384, 768]", unsqueeze_93: "f32[1, 768]", permute_227: "f32[768, 384]", unsqueeze_97: "f32[1, 384]", permute_231: "f32[384, 1024]", permute_234: "f32[128, 49, 16]", permute_235: "f32[128, 64, 49]", alias_18: "f32[8, 16, 16, 49]", permute_236: "f32[128, 16, 16]", permute_237: "f32[128, 49, 16]", unsqueeze_101: "f32[1, 256]", permute_241: "f32[256, 256]", unsqueeze_105: "f32[1, 1280]", permute_247: "f32[1280, 256]", unsqueeze_109: "f32[1, 256]", permute_251: "f32[256, 512]", unsqueeze_113: "f32[1, 512]", permute_255: "f32[512, 256]", unsqueeze_117: "f32[1, 256]", permute_259: "f32[256, 256]", permute_262: "f32[64, 49, 49]", permute_263: "f32[64, 32, 49]", alias_19: "f32[8, 8, 49, 49]", permute_264: "f32[64, 16, 49]", permute_265: "f32[64, 49, 16]", unsqueeze_121: "f32[1, 512]", permute_271: "f32[512, 256]", unsqueeze_125: "f32[1, 256]", permute_275: "f32[256, 512]", unsqueeze_129: "f32[1, 512]", permute_279: "f32[512, 256]", unsqueeze_133: "f32[1, 256]", permute_283: "f32[256, 256]", permute_286: "f32[64, 49, 49]", permute_287: "f32[64, 32, 49]", alias_20: "f32[8, 8, 49, 49]", permute_288: "f32[64, 16, 49]", permute_289: "f32[64, 49, 16]", unsqueeze_137: "f32[1, 512]", permute_295: "f32[512, 256]", unsqueeze_141: "f32[1, 256]", permute_299: "f32[256, 512]", unsqueeze_145: "f32[1, 512]", permute_303: "f32[512, 256]", unsqueeze_149: "f32[1, 256]", permute_307: "f32[256, 256]", permute_310: "f32[64, 49, 49]", permute_311: "f32[64, 32, 49]", alias_21: "f32[8, 8, 49, 49]", permute_312: "f32[64, 16, 49]", permute_313: "f32[64, 49, 16]", unsqueeze_153: "f32[1, 512]", permute_319: "f32[512, 256]", unsqueeze_157: "f32[1, 256]", permute_323: "f32[256, 512]", unsqueeze_161: "f32[1, 512]", permute_327: "f32[512, 256]", unsqueeze_165: "f32[1, 256]", permute_331: "f32[256, 256]", permute_334: "f32[64, 49, 49]", permute_335: "f32[64, 32, 49]", alias_22: "f32[8, 8, 49, 49]", permute_336: "f32[64, 16, 49]", permute_337: "f32[64, 49, 16]", unsqueeze_169: "f32[1, 512]", permute_343: "f32[512, 256]", unsqueeze_173: "f32[1, 256]", permute_347: "f32[256, 512]", unsqueeze_177: "f32[1, 512]", permute_351: "f32[512, 256]", unsqueeze_181: "f32[1, 256]", permute_355: "f32[256, 512]", permute_358: "f32[64, 196, 49]", permute_359: "f32[64, 64, 196]", alias_23: "f32[8, 8, 49, 196]", permute_360: "f32[64, 16, 49]", permute_361: "f32[64, 196, 16]", unsqueeze_185: "f32[1, 128]", permute_365: "f32[128, 128]", unsqueeze_189: "f32[1, 640]", permute_371: "f32[640, 128]", unsqueeze_193: "f32[1, 128]", permute_375: "f32[128, 256]", unsqueeze_197: "f32[1, 256]", permute_379: "f32[256, 128]", unsqueeze_201: "f32[1, 128]", permute_383: "f32[128, 128]", permute_386: "f32[32, 196, 196]", permute_387: "f32[32, 32, 196]", alias_24: "f32[8, 4, 196, 196]", permute_388: "f32[32, 16, 196]", permute_389: "f32[32, 196, 16]", unsqueeze_205: "f32[1, 256]", permute_395: "f32[256, 128]", unsqueeze_209: "f32[1, 128]", permute_399: "f32[128, 256]", unsqueeze_213: "f32[1, 256]", permute_403: "f32[256, 128]", unsqueeze_217: "f32[1, 128]", permute_407: "f32[128, 128]", permute_410: "f32[32, 196, 196]", permute_411: "f32[32, 32, 196]", alias_25: "f32[8, 4, 196, 196]", permute_412: "f32[32, 16, 196]", permute_413: "f32[32, 196, 16]", unsqueeze_221: "f32[1, 256]", permute_419: "f32[256, 128]", unsqueeze_225: "f32[1, 128]", permute_423: "f32[128, 256]", unsqueeze_229: "f32[1, 256]", permute_427: "f32[256, 128]", unsqueeze_233: "f32[1, 128]", permute_431: "f32[128, 128]", permute_434: "f32[32, 196, 196]", permute_435: "f32[32, 32, 196]", alias_26: "f32[8, 4, 196, 196]", permute_436: "f32[32, 16, 196]", permute_437: "f32[32, 196, 16]", unsqueeze_237: "f32[1, 256]", permute_443: "f32[256, 128]", unsqueeze_241: "f32[1, 128]", permute_447: "f32[128, 256]", unsqueeze_245: "f32[1, 256]", permute_451: "f32[256, 128]", unsqueeze_249: "f32[1, 128]", permute_455: "f32[128, 128]", permute_458: "f32[32, 196, 196]", permute_459: "f32[32, 32, 196]", alias_27: "f32[8, 4, 196, 196]", permute_460: "f32[32, 16, 196]", permute_461: "f32[32, 196, 16]", unsqueeze_253: "f32[1, 256]", permute_467: "f32[256, 128]", unsqueeze_259: "f32[1, 128, 1, 1]", unsqueeze_271: "f32[1, 64, 1, 1]", unsqueeze_283: "f32[1, 32, 1, 1]", unsqueeze_295: "f32[1, 16, 1, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_2: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm, [8, 196, 256]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_3: "f32[1568, 256]" = torch.ops.aten.view.default(view_2, [1568, 256]);  view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_14: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_1, [8, 196, 128]);  mm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_15: "f32[1568, 128]" = torch.ops.aten.view.default(view_14, [1568, 128]);  view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_18: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_2, [8, 196, 256]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_19: "f32[1568, 256]" = torch.ops.aten.view.default(view_18, [1568, 256]);  view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_22: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_3, [8, 196, 128]);  mm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_23: "f32[1568, 128]" = torch.ops.aten.view.default(view_22, [1568, 128]);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_26: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_4, [8, 196, 256]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_27: "f32[1568, 256]" = torch.ops.aten.view.default(view_26, [1568, 256]);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_38: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_5, [8, 196, 128]);  mm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_39: "f32[1568, 128]" = torch.ops.aten.view.default(view_38, [1568, 128]);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_42: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_6, [8, 196, 256]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_43: "f32[1568, 256]" = torch.ops.aten.view.default(view_42, [1568, 256]);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_46: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_7, [8, 196, 128]);  mm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_47: "f32[1568, 128]" = torch.ops.aten.view.default(view_46, [1568, 128]);  view_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_50: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_8, [8, 196, 256]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_51: "f32[1568, 256]" = torch.ops.aten.view.default(view_50, [1568, 256]);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_62: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_9, [8, 196, 128]);  mm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_63: "f32[1568, 128]" = torch.ops.aten.view.default(view_62, [1568, 128]);  view_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_66: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_10, [8, 196, 256]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_67: "f32[1568, 256]" = torch.ops.aten.view.default(view_66, [1568, 256]);  view_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_70: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_11, [8, 196, 128]);  mm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_71: "f32[1568, 128]" = torch.ops.aten.view.default(view_70, [1568, 128]);  view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_74: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_12, [8, 196, 256]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_75: "f32[1568, 256]" = torch.ops.aten.view.default(view_74, [1568, 256]);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_86: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_13, [8, 196, 128]);  mm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_87: "f32[1568, 128]" = torch.ops.aten.view.default(view_86, [1568, 128]);  view_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_90: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_14, [8, 196, 256]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_91: "f32[1568, 256]" = torch.ops.aten.view.default(view_90, [1568, 256]);  view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_94: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_15, [8, 196, 128]);  mm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_95: "f32[1568, 128]" = torch.ops.aten.view.default(view_94, [1568, 128]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_98: "f32[8, 196, 640]" = torch.ops.aten.view.default(mm_16, [8, 196, 640]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_99: "f32[1568, 640]" = torch.ops.aten.view.default(view_98, [1568, 640]);  view_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_105: "f32[8, 49, 128]" = torch.ops.aten.view.default(mm_17, [8, 49, 128]);  mm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_106: "f32[392, 128]" = torch.ops.aten.view.default(view_105, [392, 128]);  view_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_117: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_18, [8, 49, 256]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_118: "f32[392, 256]" = torch.ops.aten.view.default(view_117, [392, 256]);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_121: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_19, [8, 49, 512]);  mm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_122: "f32[392, 512]" = torch.ops.aten.view.default(view_121, [392, 512]);  view_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_125: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_20, [8, 49, 256]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_126: "f32[392, 256]" = torch.ops.aten.view.default(view_125, [392, 256]);  view_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_129: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_21, [8, 49, 512]);  mm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_130: "f32[392, 512]" = torch.ops.aten.view.default(view_129, [392, 512]);  view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_141: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_22, [8, 49, 256]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_142: "f32[392, 256]" = torch.ops.aten.view.default(view_141, [392, 256]);  view_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_145: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_23, [8, 49, 512]);  mm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_146: "f32[392, 512]" = torch.ops.aten.view.default(view_145, [392, 512]);  view_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_149: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_24, [8, 49, 256]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_150: "f32[392, 256]" = torch.ops.aten.view.default(view_149, [392, 256]);  view_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_153: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_25, [8, 49, 512]);  mm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_154: "f32[392, 512]" = torch.ops.aten.view.default(view_153, [392, 512]);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_165: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_26, [8, 49, 256]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_166: "f32[392, 256]" = torch.ops.aten.view.default(view_165, [392, 256]);  view_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_169: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_27, [8, 49, 512]);  mm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_170: "f32[392, 512]" = torch.ops.aten.view.default(view_169, [392, 512]);  view_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_173: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_28, [8, 49, 256]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_174: "f32[392, 256]" = torch.ops.aten.view.default(view_173, [392, 256]);  view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_177: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_29, [8, 49, 512]);  mm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_178: "f32[392, 512]" = torch.ops.aten.view.default(view_177, [392, 512]);  view_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_189: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_30, [8, 49, 256]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_190: "f32[392, 256]" = torch.ops.aten.view.default(view_189, [392, 256]);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_193: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_31, [8, 49, 512]);  mm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_194: "f32[392, 512]" = torch.ops.aten.view.default(view_193, [392, 512]);  view_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_197: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_32, [8, 49, 256]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_198: "f32[392, 256]" = torch.ops.aten.view.default(view_197, [392, 256]);  view_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_201: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_33, [8, 49, 512]);  mm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_202: "f32[392, 512]" = torch.ops.aten.view.default(view_201, [392, 512]);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_213: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_34, [8, 49, 256]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_214: "f32[392, 256]" = torch.ops.aten.view.default(view_213, [392, 256]);  view_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_217: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_35, [8, 49, 512]);  mm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_218: "f32[392, 512]" = torch.ops.aten.view.default(view_217, [392, 512]);  view_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_221: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_36, [8, 49, 256]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_222: "f32[392, 256]" = torch.ops.aten.view.default(view_221, [392, 256]);  view_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_225: "f32[8, 49, 1280]" = torch.ops.aten.view.default(mm_37, [8, 49, 1280]);  mm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_226: "f32[392, 1280]" = torch.ops.aten.view.default(view_225, [392, 1280]);  view_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_232: "f32[8, 16, 256]" = torch.ops.aten.view.default(mm_38, [8, 16, 256]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_233: "f32[128, 256]" = torch.ops.aten.view.default(view_232, [128, 256]);  view_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_244: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_39, [8, 16, 384]);  mm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_245: "f32[128, 384]" = torch.ops.aten.view.default(view_244, [128, 384]);  view_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_248: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_40, [8, 16, 768]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_249: "f32[128, 768]" = torch.ops.aten.view.default(view_248, [128, 768]);  view_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_252: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_41, [8, 16, 384]);  mm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_253: "f32[128, 384]" = torch.ops.aten.view.default(view_252, [128, 384]);  view_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_256: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_42, [8, 16, 768]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_257: "f32[128, 768]" = torch.ops.aten.view.default(view_256, [128, 768]);  view_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_268: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_43, [8, 16, 384]);  mm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_269: "f32[128, 384]" = torch.ops.aten.view.default(view_268, [128, 384]);  view_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_272: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_44, [8, 16, 768]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_273: "f32[128, 768]" = torch.ops.aten.view.default(view_272, [128, 768]);  view_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_276: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_45, [8, 16, 384]);  mm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_277: "f32[128, 384]" = torch.ops.aten.view.default(view_276, [128, 384]);  view_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_280: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_46, [8, 16, 768]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_281: "f32[128, 768]" = torch.ops.aten.view.default(view_280, [128, 768]);  view_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_292: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_47, [8, 16, 384]);  mm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_293: "f32[128, 384]" = torch.ops.aten.view.default(view_292, [128, 384]);  view_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_296: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_48, [8, 16, 768]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_297: "f32[128, 768]" = torch.ops.aten.view.default(view_296, [128, 768]);  view_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_300: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_49, [8, 16, 384]);  mm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_301: "f32[128, 384]" = torch.ops.aten.view.default(view_300, [128, 384]);  view_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_304: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_50, [8, 16, 768]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_305: "f32[128, 768]" = torch.ops.aten.view.default(view_304, [128, 768]);  view_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_316: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_51, [8, 16, 384]);  mm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_317: "f32[128, 384]" = torch.ops.aten.view.default(view_316, [128, 384]);  view_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_320: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_52, [8, 16, 768]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_321: "f32[128, 768]" = torch.ops.aten.view.default(view_320, [128, 768]);  view_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_324: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_53, [8, 16, 384]);  mm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_325: "f32[128, 384]" = torch.ops.aten.view.default(view_324, [128, 384]);  view_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_328: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_54, [8, 16, 768]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_329: "f32[128, 768]" = torch.ops.aten.view.default(view_328, [128, 768]);  view_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_340: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_55, [8, 16, 384]);  mm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_341: "f32[128, 384]" = torch.ops.aten.view.default(view_340, [128, 384]);  view_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_344: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_56, [8, 16, 768]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_345: "f32[128, 768]" = torch.ops.aten.view.default(view_344, [128, 768]);  view_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_348: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_57, [8, 16, 384]);  mm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_349: "f32[128, 384]" = torch.ops.aten.view.default(view_348, [128, 384]);  view_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:119, code: return self.linear(self.drop(self.bn(x)))
    var_mean_62 = torch.ops.aten.var_mean.correction(mean, [0], correction = 0, keepdim = True)
    getitem_164: "f32[1, 384]" = var_mean_62[0]
    getitem_165: "f32[1, 384]" = var_mean_62[1];  var_mean_62 = None
    add_382: "f32[1, 384]" = torch.ops.aten.add.Tensor(getitem_164, 1e-05);  getitem_164 = None
    rsqrt_62: "f32[1, 384]" = torch.ops.aten.rsqrt.default(add_382);  add_382 = None
    squeeze_186: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_165, [0]);  getitem_165 = None
    squeeze_187: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_62, [0]);  rsqrt_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:690, code: return (x + x_dist) / 2
    div_46: "f32[8, 1000]" = torch.ops.aten.div.Tensor(tangents_1, 2);  tangents_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:119, code: return self.linear(self.drop(self.bn(x)))
    mm_58: "f32[8, 384]" = torch.ops.aten.mm.default(div_46, permute_117);  permute_117 = None
    permute_118: "f32[1000, 8]" = torch.ops.aten.permute.default(div_46, [1, 0])
    mm_59: "f32[1000, 384]" = torch.ops.aten.mm.default(permute_118, clone_82);  clone_82 = None
    permute_119: "f32[384, 1000]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_15: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(div_46, [0], True)
    view_351: "f32[1000]" = torch.ops.aten.view.default(sum_15, [1000]);  sum_15 = None
    permute_120: "f32[1000, 384]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    unsqueeze_16: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_186, 0);  squeeze_186 = None
    sum_16: "f32[384]" = torch.ops.aten.sum.dim_IntList(mm_58, [0])
    sub_78: "f32[8, 384]" = torch.ops.aten.sub.Tensor(mean, unsqueeze_16);  mean = unsqueeze_16 = None
    mul_493: "f32[8, 384]" = torch.ops.aten.mul.Tensor(mm_58, sub_78)
    sum_17: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_493, [0]);  mul_493 = None
    mul_494: "f32[384]" = torch.ops.aten.mul.Tensor(sum_16, 0.125)
    unsqueeze_17: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_494, 0);  mul_494 = None
    mul_495: "f32[384]" = torch.ops.aten.mul.Tensor(sum_17, 0.125)
    mul_496: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_187, squeeze_187)
    mul_497: "f32[384]" = torch.ops.aten.mul.Tensor(mul_495, mul_496);  mul_495 = None
    unsqueeze_18: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_497, 0);  mul_497 = None
    mul_498: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_187, primals_205);  primals_205 = None
    unsqueeze_19: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_498, 0);  mul_498 = None
    mul_499: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_18);  unsqueeze_18 = None
    sub_80: "f32[8, 384]" = torch.ops.aten.sub.Tensor(mm_58, mul_499);  mm_58 = mul_499 = None
    sub_81: "f32[8, 384]" = torch.ops.aten.sub.Tensor(sub_80, unsqueeze_17);  sub_80 = unsqueeze_17 = None
    mul_500: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_19);  sub_81 = unsqueeze_19 = None
    mul_501: "f32[384]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_187);  sum_17 = None
    mm_60: "f32[8, 384]" = torch.ops.aten.mm.default(div_46, permute_121);  div_46 = permute_121 = None
    mm_61: "f32[1000, 384]" = torch.ops.aten.mm.default(permute_118, clone_81);  permute_118 = clone_81 = None
    permute_123: "f32[384, 1000]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    permute_124: "f32[1000, 384]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
    sum_19: "f32[384]" = torch.ops.aten.sum.dim_IntList(mm_60, [0])
    mul_502: "f32[8, 384]" = torch.ops.aten.mul.Tensor(mm_60, sub_78)
    sum_20: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_502, [0]);  mul_502 = None
    mul_503: "f32[384]" = torch.ops.aten.mul.Tensor(sum_19, 0.125)
    unsqueeze_21: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_503, 0);  mul_503 = None
    mul_504: "f32[384]" = torch.ops.aten.mul.Tensor(sum_20, 0.125)
    mul_506: "f32[384]" = torch.ops.aten.mul.Tensor(mul_504, mul_496);  mul_504 = mul_496 = None
    unsqueeze_22: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_506, 0);  mul_506 = None
    mul_507: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_187, primals_201);  primals_201 = None
    unsqueeze_23: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_507, 0);  mul_507 = None
    mul_508: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_22);  sub_78 = unsqueeze_22 = None
    sub_84: "f32[8, 384]" = torch.ops.aten.sub.Tensor(mm_60, mul_508);  mm_60 = mul_508 = None
    sub_85: "f32[8, 384]" = torch.ops.aten.sub.Tensor(sub_84, unsqueeze_21);  sub_84 = unsqueeze_21 = None
    mul_509: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_23);  sub_85 = unsqueeze_23 = None
    mul_510: "f32[384]" = torch.ops.aten.mul.Tensor(sum_20, squeeze_187);  sum_20 = squeeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:119, code: return self.linear(self.drop(self.bn(x)))
    add_392: "f32[8, 384]" = torch.ops.aten.add.Tensor(mul_500, mul_509);  mul_500 = mul_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:681, code: x = x.mean(dim=(-2, -1)) if self.use_conv else x.mean(dim=1)
    unsqueeze_24: "f32[8, 1, 384]" = torch.ops.aten.unsqueeze.default(add_392, 1);  add_392 = None
    expand_56: "f32[8, 16, 384]" = torch.ops.aten.expand.default(unsqueeze_24, [8, 16, 384]);  unsqueeze_24 = None
    div_47: "f32[8, 16, 384]" = torch.ops.aten.div.Scalar(expand_56, 16);  expand_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_353: "f32[128, 384]" = torch.ops.aten.view.default(div_47, [128, 384])
    sum_21: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_353, [0])
    sub_86: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_349, unsqueeze_25);  view_349 = unsqueeze_25 = None
    mul_511: "f32[128, 384]" = torch.ops.aten.mul.Tensor(view_353, sub_86)
    sum_22: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_511, [0]);  mul_511 = None
    mul_512: "f32[384]" = torch.ops.aten.mul.Tensor(sum_21, 0.0078125)
    unsqueeze_26: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_512, 0);  mul_512 = None
    mul_513: "f32[384]" = torch.ops.aten.mul.Tensor(sum_22, 0.0078125)
    mul_514: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_184, squeeze_184)
    mul_515: "f32[384]" = torch.ops.aten.mul.Tensor(mul_513, mul_514);  mul_513 = mul_514 = None
    unsqueeze_27: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_515, 0);  mul_515 = None
    mul_516: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_184, primals_199);  primals_199 = None
    unsqueeze_28: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_516, 0);  mul_516 = None
    mul_517: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_27);  sub_86 = unsqueeze_27 = None
    sub_88: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_353, mul_517);  view_353 = mul_517 = None
    sub_89: "f32[128, 384]" = torch.ops.aten.sub.Tensor(sub_88, unsqueeze_26);  sub_88 = unsqueeze_26 = None
    mul_518: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_28);  sub_89 = unsqueeze_28 = None
    mul_519: "f32[384]" = torch.ops.aten.mul.Tensor(sum_22, squeeze_184);  sum_22 = squeeze_184 = None
    view_354: "f32[8, 16, 384]" = torch.ops.aten.view.default(mul_518, [8, 16, 384]);  mul_518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_355: "f32[128, 384]" = torch.ops.aten.view.default(view_354, [128, 384]);  view_354 = None
    permute_125: "f32[384, 128]" = torch.ops.aten.permute.default(view_355, [1, 0])
    mm_62: "f32[384, 768]" = torch.ops.aten.mm.default(permute_125, view_347);  permute_125 = view_347 = None
    permute_126: "f32[768, 384]" = torch.ops.aten.permute.default(mm_62, [1, 0]);  mm_62 = None
    mm_63: "f32[128, 768]" = torch.ops.aten.mm.default(view_355, permute_127);  view_355 = permute_127 = None
    view_356: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_63, [8, 16, 768]);  mm_63 = None
    permute_128: "f32[384, 768]" = torch.ops.aten.permute.default(permute_126, [1, 0]);  permute_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    lt: "b8[8, 16, 768]" = torch.ops.aten.lt.Scalar(view_346, -3)
    le: "b8[8, 16, 768]" = torch.ops.aten.le.Scalar(view_346, 3)
    div_48: "f32[8, 16, 768]" = torch.ops.aten.div.Tensor(view_346, 3);  view_346 = None
    add_393: "f32[8, 16, 768]" = torch.ops.aten.add.Tensor(div_48, 0.5);  div_48 = None
    mul_520: "f32[8, 16, 768]" = torch.ops.aten.mul.Tensor(view_356, add_393);  add_393 = None
    where: "f32[8, 16, 768]" = torch.ops.aten.where.self(le, mul_520, view_356);  le = mul_520 = view_356 = None
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_1: "f32[8, 16, 768]" = torch.ops.aten.where.self(lt, full_default, where);  lt = where = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_357: "f32[128, 768]" = torch.ops.aten.view.default(where_1, [128, 768]);  where_1 = None
    sum_23: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_357, [0])
    sub_90: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_345, unsqueeze_29);  view_345 = unsqueeze_29 = None
    mul_521: "f32[128, 768]" = torch.ops.aten.mul.Tensor(view_357, sub_90)
    sum_24: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_521, [0]);  mul_521 = None
    mul_522: "f32[768]" = torch.ops.aten.mul.Tensor(sum_23, 0.0078125)
    unsqueeze_30: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_522, 0);  mul_522 = None
    mul_523: "f32[768]" = torch.ops.aten.mul.Tensor(sum_24, 0.0078125)
    mul_524: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_181, squeeze_181)
    mul_525: "f32[768]" = torch.ops.aten.mul.Tensor(mul_523, mul_524);  mul_523 = mul_524 = None
    unsqueeze_31: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_525, 0);  mul_525 = None
    mul_526: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_181, primals_196);  primals_196 = None
    unsqueeze_32: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_526, 0);  mul_526 = None
    mul_527: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_31);  sub_90 = unsqueeze_31 = None
    sub_92: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_357, mul_527);  view_357 = mul_527 = None
    sub_93: "f32[128, 768]" = torch.ops.aten.sub.Tensor(sub_92, unsqueeze_30);  sub_92 = unsqueeze_30 = None
    mul_528: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_32);  sub_93 = unsqueeze_32 = None
    mul_529: "f32[768]" = torch.ops.aten.mul.Tensor(sum_24, squeeze_181);  sum_24 = squeeze_181 = None
    view_358: "f32[8, 16, 768]" = torch.ops.aten.view.default(mul_528, [8, 16, 768]);  mul_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_359: "f32[128, 768]" = torch.ops.aten.view.default(view_358, [128, 768]);  view_358 = None
    permute_129: "f32[768, 128]" = torch.ops.aten.permute.default(view_359, [1, 0])
    mm_64: "f32[768, 384]" = torch.ops.aten.mm.default(permute_129, view_343);  permute_129 = view_343 = None
    permute_130: "f32[384, 768]" = torch.ops.aten.permute.default(mm_64, [1, 0]);  mm_64 = None
    mm_65: "f32[128, 384]" = torch.ops.aten.mm.default(view_359, permute_131);  view_359 = permute_131 = None
    view_360: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_65, [8, 16, 384]);  mm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_394: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(div_47, view_360);  div_47 = view_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_132: "f32[768, 384]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_361: "f32[128, 384]" = torch.ops.aten.view.default(add_394, [128, 384])
    sum_25: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_361, [0])
    sub_94: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_341, unsqueeze_33);  view_341 = unsqueeze_33 = None
    mul_530: "f32[128, 384]" = torch.ops.aten.mul.Tensor(view_361, sub_94)
    sum_26: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_530, [0]);  mul_530 = None
    mul_531: "f32[384]" = torch.ops.aten.mul.Tensor(sum_25, 0.0078125)
    unsqueeze_34: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_531, 0);  mul_531 = None
    mul_532: "f32[384]" = torch.ops.aten.mul.Tensor(sum_26, 0.0078125)
    mul_533: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_178, squeeze_178)
    mul_534: "f32[384]" = torch.ops.aten.mul.Tensor(mul_532, mul_533);  mul_532 = mul_533 = None
    unsqueeze_35: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_534, 0);  mul_534 = None
    mul_535: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_178, primals_193);  primals_193 = None
    unsqueeze_36: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_535, 0);  mul_535 = None
    mul_536: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_35);  sub_94 = unsqueeze_35 = None
    sub_96: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_361, mul_536);  view_361 = mul_536 = None
    sub_97: "f32[128, 384]" = torch.ops.aten.sub.Tensor(sub_96, unsqueeze_34);  sub_96 = unsqueeze_34 = None
    mul_537: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_36);  sub_97 = unsqueeze_36 = None
    mul_538: "f32[384]" = torch.ops.aten.mul.Tensor(sum_26, squeeze_178);  sum_26 = squeeze_178 = None
    view_362: "f32[8, 16, 384]" = torch.ops.aten.view.default(mul_537, [8, 16, 384]);  mul_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_363: "f32[128, 384]" = torch.ops.aten.view.default(view_362, [128, 384]);  view_362 = None
    permute_133: "f32[384, 128]" = torch.ops.aten.permute.default(view_363, [1, 0])
    mm_66: "f32[384, 384]" = torch.ops.aten.mm.default(permute_133, view_339);  permute_133 = view_339 = None
    permute_134: "f32[384, 384]" = torch.ops.aten.permute.default(mm_66, [1, 0]);  mm_66 = None
    mm_67: "f32[128, 384]" = torch.ops.aten.mm.default(view_363, permute_135);  view_363 = permute_135 = None
    view_364: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_67, [8, 16, 384]);  mm_67 = None
    permute_136: "f32[384, 384]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    lt_1: "b8[8, 16, 384]" = torch.ops.aten.lt.Scalar(view_338, -3)
    le_1: "b8[8, 16, 384]" = torch.ops.aten.le.Scalar(view_338, 3)
    div_49: "f32[8, 16, 384]" = torch.ops.aten.div.Tensor(view_338, 3);  view_338 = None
    add_395: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(div_49, 0.5);  div_49 = None
    mul_539: "f32[8, 16, 384]" = torch.ops.aten.mul.Tensor(view_364, add_395);  add_395 = None
    where_2: "f32[8, 16, 384]" = torch.ops.aten.where.self(le_1, mul_539, view_364);  le_1 = mul_539 = view_364 = None
    where_3: "f32[8, 16, 384]" = torch.ops.aten.where.self(lt_1, full_default, where_2);  lt_1 = where_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    view_365: "f32[8, 16, 12, 32]" = torch.ops.aten.view.default(where_3, [8, 16, 12, 32]);  where_3 = None
    permute_137: "f32[8, 12, 16, 32]" = torch.ops.aten.permute.default(view_365, [0, 2, 1, 3]);  view_365 = None
    clone_83: "f32[8, 12, 16, 32]" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
    view_366: "f32[96, 16, 32]" = torch.ops.aten.view.default(clone_83, [96, 16, 32]);  clone_83 = None
    bmm_28: "f32[96, 16, 32]" = torch.ops.aten.bmm.default(permute_138, view_366);  permute_138 = None
    bmm_29: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(view_366, permute_139);  view_366 = permute_139 = None
    view_367: "f32[8, 12, 16, 32]" = torch.ops.aten.view.default(bmm_28, [8, 12, 16, 32]);  bmm_28 = None
    view_368: "f32[8, 12, 16, 16]" = torch.ops.aten.view.default(bmm_29, [8, 12, 16, 16]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    mul_540: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(view_368, alias_14);  view_368 = None
    sum_27: "f32[8, 12, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_540, [-1], True)
    mul_541: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(alias_14, sum_27);  alias_14 = sum_27 = None
    sub_98: "f32[8, 12, 16, 16]" = torch.ops.aten.sub.Tensor(mul_540, mul_541);  mul_540 = mul_541 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    sum_28: "f32[1, 12, 16, 16]" = torch.ops.aten.sum.dim_IntList(sub_98, [0], True)
    view_369: "f32[12, 16, 16]" = torch.ops.aten.view.default(sum_28, [12, 16, 16]);  sum_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    full_default_2: "f32[12, 16]" = torch.ops.aten.full.default([12, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    index_put: "f32[12, 16]" = torch.ops.aten.index_put.default(full_default_2, [None, primals_222], view_369, True);  primals_222 = view_369 = None
    slice_scatter: "f32[12, 16]" = torch.ops.aten.slice_scatter.default(full_default_2, index_put, 0, 0, 9223372036854775807);  index_put = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    mul_542: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(sub_98, 0.25);  sub_98 = None
    view_370: "f32[96, 16, 16]" = torch.ops.aten.view.default(mul_542, [96, 16, 16]);  mul_542 = None
    bmm_30: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(permute_140, view_370);  permute_140 = None
    bmm_31: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(view_370, permute_141);  view_370 = permute_141 = None
    view_371: "f32[8, 12, 16, 16]" = torch.ops.aten.view.default(bmm_30, [8, 12, 16, 16]);  bmm_30 = None
    view_372: "f32[8, 12, 16, 16]" = torch.ops.aten.view.default(bmm_31, [8, 12, 16, 16]);  bmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_142: "f32[8, 16, 12, 32]" = torch.ops.aten.permute.default(view_367, [0, 2, 1, 3]);  view_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_143: "f32[8, 16, 12, 16]" = torch.ops.aten.permute.default(view_371, [0, 3, 1, 2]);  view_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_144: "f32[8, 16, 12, 16]" = torch.ops.aten.permute.default(view_372, [0, 2, 1, 3]);  view_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    cat: "f32[8, 16, 12, 64]" = torch.ops.aten.cat.default([permute_144, permute_143, permute_142], 3);  permute_144 = permute_143 = permute_142 = None
    view_373: "f32[8, 16, 768]" = torch.ops.aten.view.default(cat, [8, 16, 768]);  cat = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_374: "f32[128, 768]" = torch.ops.aten.view.default(view_373, [128, 768]);  view_373 = None
    sum_29: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_374, [0])
    sub_99: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_329, unsqueeze_37);  view_329 = unsqueeze_37 = None
    mul_543: "f32[128, 768]" = torch.ops.aten.mul.Tensor(view_374, sub_99)
    sum_30: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_543, [0]);  mul_543 = None
    mul_544: "f32[768]" = torch.ops.aten.mul.Tensor(sum_29, 0.0078125)
    unsqueeze_38: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_544, 0);  mul_544 = None
    mul_545: "f32[768]" = torch.ops.aten.mul.Tensor(sum_30, 0.0078125)
    mul_546: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_175, squeeze_175)
    mul_547: "f32[768]" = torch.ops.aten.mul.Tensor(mul_545, mul_546);  mul_545 = mul_546 = None
    unsqueeze_39: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_547, 0);  mul_547 = None
    mul_548: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_175, primals_190);  primals_190 = None
    unsqueeze_40: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_548, 0);  mul_548 = None
    mul_549: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_39);  sub_99 = unsqueeze_39 = None
    sub_101: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_374, mul_549);  view_374 = mul_549 = None
    sub_102: "f32[128, 768]" = torch.ops.aten.sub.Tensor(sub_101, unsqueeze_38);  sub_101 = unsqueeze_38 = None
    mul_550: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_40);  sub_102 = unsqueeze_40 = None
    mul_551: "f32[768]" = torch.ops.aten.mul.Tensor(sum_30, squeeze_175);  sum_30 = squeeze_175 = None
    view_375: "f32[8, 16, 768]" = torch.ops.aten.view.default(mul_550, [8, 16, 768]);  mul_550 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_376: "f32[128, 768]" = torch.ops.aten.view.default(view_375, [128, 768]);  view_375 = None
    permute_145: "f32[768, 128]" = torch.ops.aten.permute.default(view_376, [1, 0])
    mm_68: "f32[768, 384]" = torch.ops.aten.mm.default(permute_145, view_327);  permute_145 = view_327 = None
    permute_146: "f32[384, 768]" = torch.ops.aten.permute.default(mm_68, [1, 0]);  mm_68 = None
    mm_69: "f32[128, 384]" = torch.ops.aten.mm.default(view_376, permute_147);  view_376 = permute_147 = None
    view_377: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_69, [8, 16, 384]);  mm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_396: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_394, view_377);  add_394 = view_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_148: "f32[768, 384]" = torch.ops.aten.permute.default(permute_146, [1, 0]);  permute_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_378: "f32[128, 384]" = torch.ops.aten.view.default(add_396, [128, 384])
    sum_31: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_378, [0])
    sub_103: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_325, unsqueeze_41);  view_325 = unsqueeze_41 = None
    mul_552: "f32[128, 384]" = torch.ops.aten.mul.Tensor(view_378, sub_103)
    sum_32: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_552, [0]);  mul_552 = None
    mul_553: "f32[384]" = torch.ops.aten.mul.Tensor(sum_31, 0.0078125)
    unsqueeze_42: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_553, 0);  mul_553 = None
    mul_554: "f32[384]" = torch.ops.aten.mul.Tensor(sum_32, 0.0078125)
    mul_555: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_172, squeeze_172)
    mul_556: "f32[384]" = torch.ops.aten.mul.Tensor(mul_554, mul_555);  mul_554 = mul_555 = None
    unsqueeze_43: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_556, 0);  mul_556 = None
    mul_557: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_172, primals_187);  primals_187 = None
    unsqueeze_44: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_557, 0);  mul_557 = None
    mul_558: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_43);  sub_103 = unsqueeze_43 = None
    sub_105: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_378, mul_558);  view_378 = mul_558 = None
    sub_106: "f32[128, 384]" = torch.ops.aten.sub.Tensor(sub_105, unsqueeze_42);  sub_105 = unsqueeze_42 = None
    mul_559: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_44);  sub_106 = unsqueeze_44 = None
    mul_560: "f32[384]" = torch.ops.aten.mul.Tensor(sum_32, squeeze_172);  sum_32 = squeeze_172 = None
    view_379: "f32[8, 16, 384]" = torch.ops.aten.view.default(mul_559, [8, 16, 384]);  mul_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_380: "f32[128, 384]" = torch.ops.aten.view.default(view_379, [128, 384]);  view_379 = None
    permute_149: "f32[384, 128]" = torch.ops.aten.permute.default(view_380, [1, 0])
    mm_70: "f32[384, 768]" = torch.ops.aten.mm.default(permute_149, view_323);  permute_149 = view_323 = None
    permute_150: "f32[768, 384]" = torch.ops.aten.permute.default(mm_70, [1, 0]);  mm_70 = None
    mm_71: "f32[128, 768]" = torch.ops.aten.mm.default(view_380, permute_151);  view_380 = permute_151 = None
    view_381: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_71, [8, 16, 768]);  mm_71 = None
    permute_152: "f32[384, 768]" = torch.ops.aten.permute.default(permute_150, [1, 0]);  permute_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    lt_2: "b8[8, 16, 768]" = torch.ops.aten.lt.Scalar(view_322, -3)
    le_2: "b8[8, 16, 768]" = torch.ops.aten.le.Scalar(view_322, 3)
    div_50: "f32[8, 16, 768]" = torch.ops.aten.div.Tensor(view_322, 3);  view_322 = None
    add_397: "f32[8, 16, 768]" = torch.ops.aten.add.Tensor(div_50, 0.5);  div_50 = None
    mul_561: "f32[8, 16, 768]" = torch.ops.aten.mul.Tensor(view_381, add_397);  add_397 = None
    where_4: "f32[8, 16, 768]" = torch.ops.aten.where.self(le_2, mul_561, view_381);  le_2 = mul_561 = view_381 = None
    where_5: "f32[8, 16, 768]" = torch.ops.aten.where.self(lt_2, full_default, where_4);  lt_2 = where_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_382: "f32[128, 768]" = torch.ops.aten.view.default(where_5, [128, 768]);  where_5 = None
    sum_33: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_382, [0])
    sub_107: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_321, unsqueeze_45);  view_321 = unsqueeze_45 = None
    mul_562: "f32[128, 768]" = torch.ops.aten.mul.Tensor(view_382, sub_107)
    sum_34: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_562, [0]);  mul_562 = None
    mul_563: "f32[768]" = torch.ops.aten.mul.Tensor(sum_33, 0.0078125)
    unsqueeze_46: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_563, 0);  mul_563 = None
    mul_564: "f32[768]" = torch.ops.aten.mul.Tensor(sum_34, 0.0078125)
    mul_565: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
    mul_566: "f32[768]" = torch.ops.aten.mul.Tensor(mul_564, mul_565);  mul_564 = mul_565 = None
    unsqueeze_47: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_566, 0);  mul_566 = None
    mul_567: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_169, primals_184);  primals_184 = None
    unsqueeze_48: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_567, 0);  mul_567 = None
    mul_568: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_47);  sub_107 = unsqueeze_47 = None
    sub_109: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_382, mul_568);  view_382 = mul_568 = None
    sub_110: "f32[128, 768]" = torch.ops.aten.sub.Tensor(sub_109, unsqueeze_46);  sub_109 = unsqueeze_46 = None
    mul_569: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_48);  sub_110 = unsqueeze_48 = None
    mul_570: "f32[768]" = torch.ops.aten.mul.Tensor(sum_34, squeeze_169);  sum_34 = squeeze_169 = None
    view_383: "f32[8, 16, 768]" = torch.ops.aten.view.default(mul_569, [8, 16, 768]);  mul_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_384: "f32[128, 768]" = torch.ops.aten.view.default(view_383, [128, 768]);  view_383 = None
    permute_153: "f32[768, 128]" = torch.ops.aten.permute.default(view_384, [1, 0])
    mm_72: "f32[768, 384]" = torch.ops.aten.mm.default(permute_153, view_319);  permute_153 = view_319 = None
    permute_154: "f32[384, 768]" = torch.ops.aten.permute.default(mm_72, [1, 0]);  mm_72 = None
    mm_73: "f32[128, 384]" = torch.ops.aten.mm.default(view_384, permute_155);  view_384 = permute_155 = None
    view_385: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_73, [8, 16, 384]);  mm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_398: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_396, view_385);  add_396 = view_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_156: "f32[768, 384]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_386: "f32[128, 384]" = torch.ops.aten.view.default(add_398, [128, 384])
    sum_35: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_386, [0])
    sub_111: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_317, unsqueeze_49);  view_317 = unsqueeze_49 = None
    mul_571: "f32[128, 384]" = torch.ops.aten.mul.Tensor(view_386, sub_111)
    sum_36: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_571, [0]);  mul_571 = None
    mul_572: "f32[384]" = torch.ops.aten.mul.Tensor(sum_35, 0.0078125)
    unsqueeze_50: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_572, 0);  mul_572 = None
    mul_573: "f32[384]" = torch.ops.aten.mul.Tensor(sum_36, 0.0078125)
    mul_574: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
    mul_575: "f32[384]" = torch.ops.aten.mul.Tensor(mul_573, mul_574);  mul_573 = mul_574 = None
    unsqueeze_51: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_575, 0);  mul_575 = None
    mul_576: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_166, primals_181);  primals_181 = None
    unsqueeze_52: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_576, 0);  mul_576 = None
    mul_577: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_51);  sub_111 = unsqueeze_51 = None
    sub_113: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_386, mul_577);  view_386 = mul_577 = None
    sub_114: "f32[128, 384]" = torch.ops.aten.sub.Tensor(sub_113, unsqueeze_50);  sub_113 = unsqueeze_50 = None
    mul_578: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_52);  sub_114 = unsqueeze_52 = None
    mul_579: "f32[384]" = torch.ops.aten.mul.Tensor(sum_36, squeeze_166);  sum_36 = squeeze_166 = None
    view_387: "f32[8, 16, 384]" = torch.ops.aten.view.default(mul_578, [8, 16, 384]);  mul_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_388: "f32[128, 384]" = torch.ops.aten.view.default(view_387, [128, 384]);  view_387 = None
    permute_157: "f32[384, 128]" = torch.ops.aten.permute.default(view_388, [1, 0])
    mm_74: "f32[384, 384]" = torch.ops.aten.mm.default(permute_157, view_315);  permute_157 = view_315 = None
    permute_158: "f32[384, 384]" = torch.ops.aten.permute.default(mm_74, [1, 0]);  mm_74 = None
    mm_75: "f32[128, 384]" = torch.ops.aten.mm.default(view_388, permute_159);  view_388 = permute_159 = None
    view_389: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_75, [8, 16, 384]);  mm_75 = None
    permute_160: "f32[384, 384]" = torch.ops.aten.permute.default(permute_158, [1, 0]);  permute_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    lt_3: "b8[8, 16, 384]" = torch.ops.aten.lt.Scalar(view_314, -3)
    le_3: "b8[8, 16, 384]" = torch.ops.aten.le.Scalar(view_314, 3)
    div_51: "f32[8, 16, 384]" = torch.ops.aten.div.Tensor(view_314, 3);  view_314 = None
    add_399: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(div_51, 0.5);  div_51 = None
    mul_580: "f32[8, 16, 384]" = torch.ops.aten.mul.Tensor(view_389, add_399);  add_399 = None
    where_6: "f32[8, 16, 384]" = torch.ops.aten.where.self(le_3, mul_580, view_389);  le_3 = mul_580 = view_389 = None
    where_7: "f32[8, 16, 384]" = torch.ops.aten.where.self(lt_3, full_default, where_6);  lt_3 = where_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    view_390: "f32[8, 16, 12, 32]" = torch.ops.aten.view.default(where_7, [8, 16, 12, 32]);  where_7 = None
    permute_161: "f32[8, 12, 16, 32]" = torch.ops.aten.permute.default(view_390, [0, 2, 1, 3]);  view_390 = None
    clone_84: "f32[8, 12, 16, 32]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
    view_391: "f32[96, 16, 32]" = torch.ops.aten.view.default(clone_84, [96, 16, 32]);  clone_84 = None
    bmm_32: "f32[96, 16, 32]" = torch.ops.aten.bmm.default(permute_162, view_391);  permute_162 = None
    bmm_33: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(view_391, permute_163);  view_391 = permute_163 = None
    view_392: "f32[8, 12, 16, 32]" = torch.ops.aten.view.default(bmm_32, [8, 12, 16, 32]);  bmm_32 = None
    view_393: "f32[8, 12, 16, 16]" = torch.ops.aten.view.default(bmm_33, [8, 12, 16, 16]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    mul_581: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(view_393, alias_15);  view_393 = None
    sum_37: "f32[8, 12, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_581, [-1], True)
    mul_582: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(alias_15, sum_37);  alias_15 = sum_37 = None
    sub_115: "f32[8, 12, 16, 16]" = torch.ops.aten.sub.Tensor(mul_581, mul_582);  mul_581 = mul_582 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    sum_38: "f32[1, 12, 16, 16]" = torch.ops.aten.sum.dim_IntList(sub_115, [0], True)
    view_394: "f32[12, 16, 16]" = torch.ops.aten.view.default(sum_38, [12, 16, 16]);  sum_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    index_put_1: "f32[12, 16]" = torch.ops.aten.index_put.default(full_default_2, [None, primals_221], view_394, True);  primals_221 = view_394 = None
    slice_scatter_1: "f32[12, 16]" = torch.ops.aten.slice_scatter.default(full_default_2, index_put_1, 0, 0, 9223372036854775807);  index_put_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    mul_583: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(sub_115, 0.25);  sub_115 = None
    view_395: "f32[96, 16, 16]" = torch.ops.aten.view.default(mul_583, [96, 16, 16]);  mul_583 = None
    bmm_34: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(permute_164, view_395);  permute_164 = None
    bmm_35: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(view_395, permute_165);  view_395 = permute_165 = None
    view_396: "f32[8, 12, 16, 16]" = torch.ops.aten.view.default(bmm_34, [8, 12, 16, 16]);  bmm_34 = None
    view_397: "f32[8, 12, 16, 16]" = torch.ops.aten.view.default(bmm_35, [8, 12, 16, 16]);  bmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_166: "f32[8, 16, 12, 32]" = torch.ops.aten.permute.default(view_392, [0, 2, 1, 3]);  view_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_167: "f32[8, 16, 12, 16]" = torch.ops.aten.permute.default(view_396, [0, 3, 1, 2]);  view_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_168: "f32[8, 16, 12, 16]" = torch.ops.aten.permute.default(view_397, [0, 2, 1, 3]);  view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    cat_1: "f32[8, 16, 12, 64]" = torch.ops.aten.cat.default([permute_168, permute_167, permute_166], 3);  permute_168 = permute_167 = permute_166 = None
    view_398: "f32[8, 16, 768]" = torch.ops.aten.view.default(cat_1, [8, 16, 768]);  cat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_399: "f32[128, 768]" = torch.ops.aten.view.default(view_398, [128, 768]);  view_398 = None
    sum_39: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_399, [0])
    sub_116: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_305, unsqueeze_53);  view_305 = unsqueeze_53 = None
    mul_584: "f32[128, 768]" = torch.ops.aten.mul.Tensor(view_399, sub_116)
    sum_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_584, [0]);  mul_584 = None
    mul_585: "f32[768]" = torch.ops.aten.mul.Tensor(sum_39, 0.0078125)
    unsqueeze_54: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_585, 0);  mul_585 = None
    mul_586: "f32[768]" = torch.ops.aten.mul.Tensor(sum_40, 0.0078125)
    mul_587: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
    mul_588: "f32[768]" = torch.ops.aten.mul.Tensor(mul_586, mul_587);  mul_586 = mul_587 = None
    unsqueeze_55: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_588, 0);  mul_588 = None
    mul_589: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_163, primals_178);  primals_178 = None
    unsqueeze_56: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_589, 0);  mul_589 = None
    mul_590: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_55);  sub_116 = unsqueeze_55 = None
    sub_118: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_399, mul_590);  view_399 = mul_590 = None
    sub_119: "f32[128, 768]" = torch.ops.aten.sub.Tensor(sub_118, unsqueeze_54);  sub_118 = unsqueeze_54 = None
    mul_591: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_56);  sub_119 = unsqueeze_56 = None
    mul_592: "f32[768]" = torch.ops.aten.mul.Tensor(sum_40, squeeze_163);  sum_40 = squeeze_163 = None
    view_400: "f32[8, 16, 768]" = torch.ops.aten.view.default(mul_591, [8, 16, 768]);  mul_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_401: "f32[128, 768]" = torch.ops.aten.view.default(view_400, [128, 768]);  view_400 = None
    permute_169: "f32[768, 128]" = torch.ops.aten.permute.default(view_401, [1, 0])
    mm_76: "f32[768, 384]" = torch.ops.aten.mm.default(permute_169, view_303);  permute_169 = view_303 = None
    permute_170: "f32[384, 768]" = torch.ops.aten.permute.default(mm_76, [1, 0]);  mm_76 = None
    mm_77: "f32[128, 384]" = torch.ops.aten.mm.default(view_401, permute_171);  view_401 = permute_171 = None
    view_402: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_77, [8, 16, 384]);  mm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_400: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_398, view_402);  add_398 = view_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_172: "f32[768, 384]" = torch.ops.aten.permute.default(permute_170, [1, 0]);  permute_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_403: "f32[128, 384]" = torch.ops.aten.view.default(add_400, [128, 384])
    sum_41: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_403, [0])
    sub_120: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_301, unsqueeze_57);  view_301 = unsqueeze_57 = None
    mul_593: "f32[128, 384]" = torch.ops.aten.mul.Tensor(view_403, sub_120)
    sum_42: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_593, [0]);  mul_593 = None
    mul_594: "f32[384]" = torch.ops.aten.mul.Tensor(sum_41, 0.0078125)
    unsqueeze_58: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_594, 0);  mul_594 = None
    mul_595: "f32[384]" = torch.ops.aten.mul.Tensor(sum_42, 0.0078125)
    mul_596: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
    mul_597: "f32[384]" = torch.ops.aten.mul.Tensor(mul_595, mul_596);  mul_595 = mul_596 = None
    unsqueeze_59: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_597, 0);  mul_597 = None
    mul_598: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_160, primals_175);  primals_175 = None
    unsqueeze_60: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_598, 0);  mul_598 = None
    mul_599: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_59);  sub_120 = unsqueeze_59 = None
    sub_122: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_403, mul_599);  view_403 = mul_599 = None
    sub_123: "f32[128, 384]" = torch.ops.aten.sub.Tensor(sub_122, unsqueeze_58);  sub_122 = unsqueeze_58 = None
    mul_600: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_60);  sub_123 = unsqueeze_60 = None
    mul_601: "f32[384]" = torch.ops.aten.mul.Tensor(sum_42, squeeze_160);  sum_42 = squeeze_160 = None
    view_404: "f32[8, 16, 384]" = torch.ops.aten.view.default(mul_600, [8, 16, 384]);  mul_600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_405: "f32[128, 384]" = torch.ops.aten.view.default(view_404, [128, 384]);  view_404 = None
    permute_173: "f32[384, 128]" = torch.ops.aten.permute.default(view_405, [1, 0])
    mm_78: "f32[384, 768]" = torch.ops.aten.mm.default(permute_173, view_299);  permute_173 = view_299 = None
    permute_174: "f32[768, 384]" = torch.ops.aten.permute.default(mm_78, [1, 0]);  mm_78 = None
    mm_79: "f32[128, 768]" = torch.ops.aten.mm.default(view_405, permute_175);  view_405 = permute_175 = None
    view_406: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_79, [8, 16, 768]);  mm_79 = None
    permute_176: "f32[384, 768]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    lt_4: "b8[8, 16, 768]" = torch.ops.aten.lt.Scalar(view_298, -3)
    le_4: "b8[8, 16, 768]" = torch.ops.aten.le.Scalar(view_298, 3)
    div_52: "f32[8, 16, 768]" = torch.ops.aten.div.Tensor(view_298, 3);  view_298 = None
    add_401: "f32[8, 16, 768]" = torch.ops.aten.add.Tensor(div_52, 0.5);  div_52 = None
    mul_602: "f32[8, 16, 768]" = torch.ops.aten.mul.Tensor(view_406, add_401);  add_401 = None
    where_8: "f32[8, 16, 768]" = torch.ops.aten.where.self(le_4, mul_602, view_406);  le_4 = mul_602 = view_406 = None
    where_9: "f32[8, 16, 768]" = torch.ops.aten.where.self(lt_4, full_default, where_8);  lt_4 = where_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_407: "f32[128, 768]" = torch.ops.aten.view.default(where_9, [128, 768]);  where_9 = None
    sum_43: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_407, [0])
    sub_124: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_297, unsqueeze_61);  view_297 = unsqueeze_61 = None
    mul_603: "f32[128, 768]" = torch.ops.aten.mul.Tensor(view_407, sub_124)
    sum_44: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_603, [0]);  mul_603 = None
    mul_604: "f32[768]" = torch.ops.aten.mul.Tensor(sum_43, 0.0078125)
    unsqueeze_62: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_604, 0);  mul_604 = None
    mul_605: "f32[768]" = torch.ops.aten.mul.Tensor(sum_44, 0.0078125)
    mul_606: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
    mul_607: "f32[768]" = torch.ops.aten.mul.Tensor(mul_605, mul_606);  mul_605 = mul_606 = None
    unsqueeze_63: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_607, 0);  mul_607 = None
    mul_608: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_157, primals_172);  primals_172 = None
    unsqueeze_64: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_608, 0);  mul_608 = None
    mul_609: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_63);  sub_124 = unsqueeze_63 = None
    sub_126: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_407, mul_609);  view_407 = mul_609 = None
    sub_127: "f32[128, 768]" = torch.ops.aten.sub.Tensor(sub_126, unsqueeze_62);  sub_126 = unsqueeze_62 = None
    mul_610: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_64);  sub_127 = unsqueeze_64 = None
    mul_611: "f32[768]" = torch.ops.aten.mul.Tensor(sum_44, squeeze_157);  sum_44 = squeeze_157 = None
    view_408: "f32[8, 16, 768]" = torch.ops.aten.view.default(mul_610, [8, 16, 768]);  mul_610 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_409: "f32[128, 768]" = torch.ops.aten.view.default(view_408, [128, 768]);  view_408 = None
    permute_177: "f32[768, 128]" = torch.ops.aten.permute.default(view_409, [1, 0])
    mm_80: "f32[768, 384]" = torch.ops.aten.mm.default(permute_177, view_295);  permute_177 = view_295 = None
    permute_178: "f32[384, 768]" = torch.ops.aten.permute.default(mm_80, [1, 0]);  mm_80 = None
    mm_81: "f32[128, 384]" = torch.ops.aten.mm.default(view_409, permute_179);  view_409 = permute_179 = None
    view_410: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_81, [8, 16, 384]);  mm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_402: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_400, view_410);  add_400 = view_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_180: "f32[768, 384]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_411: "f32[128, 384]" = torch.ops.aten.view.default(add_402, [128, 384])
    sum_45: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_411, [0])
    sub_128: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_293, unsqueeze_65);  view_293 = unsqueeze_65 = None
    mul_612: "f32[128, 384]" = torch.ops.aten.mul.Tensor(view_411, sub_128)
    sum_46: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_612, [0]);  mul_612 = None
    mul_613: "f32[384]" = torch.ops.aten.mul.Tensor(sum_45, 0.0078125)
    unsqueeze_66: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_613, 0);  mul_613 = None
    mul_614: "f32[384]" = torch.ops.aten.mul.Tensor(sum_46, 0.0078125)
    mul_615: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
    mul_616: "f32[384]" = torch.ops.aten.mul.Tensor(mul_614, mul_615);  mul_614 = mul_615 = None
    unsqueeze_67: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_616, 0);  mul_616 = None
    mul_617: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_154, primals_169);  primals_169 = None
    unsqueeze_68: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_617, 0);  mul_617 = None
    mul_618: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_67);  sub_128 = unsqueeze_67 = None
    sub_130: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_411, mul_618);  view_411 = mul_618 = None
    sub_131: "f32[128, 384]" = torch.ops.aten.sub.Tensor(sub_130, unsqueeze_66);  sub_130 = unsqueeze_66 = None
    mul_619: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_68);  sub_131 = unsqueeze_68 = None
    mul_620: "f32[384]" = torch.ops.aten.mul.Tensor(sum_46, squeeze_154);  sum_46 = squeeze_154 = None
    view_412: "f32[8, 16, 384]" = torch.ops.aten.view.default(mul_619, [8, 16, 384]);  mul_619 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_413: "f32[128, 384]" = torch.ops.aten.view.default(view_412, [128, 384]);  view_412 = None
    permute_181: "f32[384, 128]" = torch.ops.aten.permute.default(view_413, [1, 0])
    mm_82: "f32[384, 384]" = torch.ops.aten.mm.default(permute_181, view_291);  permute_181 = view_291 = None
    permute_182: "f32[384, 384]" = torch.ops.aten.permute.default(mm_82, [1, 0]);  mm_82 = None
    mm_83: "f32[128, 384]" = torch.ops.aten.mm.default(view_413, permute_183);  view_413 = permute_183 = None
    view_414: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_83, [8, 16, 384]);  mm_83 = None
    permute_184: "f32[384, 384]" = torch.ops.aten.permute.default(permute_182, [1, 0]);  permute_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    lt_5: "b8[8, 16, 384]" = torch.ops.aten.lt.Scalar(view_290, -3)
    le_5: "b8[8, 16, 384]" = torch.ops.aten.le.Scalar(view_290, 3)
    div_53: "f32[8, 16, 384]" = torch.ops.aten.div.Tensor(view_290, 3);  view_290 = None
    add_403: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(div_53, 0.5);  div_53 = None
    mul_621: "f32[8, 16, 384]" = torch.ops.aten.mul.Tensor(view_414, add_403);  add_403 = None
    where_10: "f32[8, 16, 384]" = torch.ops.aten.where.self(le_5, mul_621, view_414);  le_5 = mul_621 = view_414 = None
    where_11: "f32[8, 16, 384]" = torch.ops.aten.where.self(lt_5, full_default, where_10);  lt_5 = where_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    view_415: "f32[8, 16, 12, 32]" = torch.ops.aten.view.default(where_11, [8, 16, 12, 32]);  where_11 = None
    permute_185: "f32[8, 12, 16, 32]" = torch.ops.aten.permute.default(view_415, [0, 2, 1, 3]);  view_415 = None
    clone_85: "f32[8, 12, 16, 32]" = torch.ops.aten.clone.default(permute_185, memory_format = torch.contiguous_format);  permute_185 = None
    view_416: "f32[96, 16, 32]" = torch.ops.aten.view.default(clone_85, [96, 16, 32]);  clone_85 = None
    bmm_36: "f32[96, 16, 32]" = torch.ops.aten.bmm.default(permute_186, view_416);  permute_186 = None
    bmm_37: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(view_416, permute_187);  view_416 = permute_187 = None
    view_417: "f32[8, 12, 16, 32]" = torch.ops.aten.view.default(bmm_36, [8, 12, 16, 32]);  bmm_36 = None
    view_418: "f32[8, 12, 16, 16]" = torch.ops.aten.view.default(bmm_37, [8, 12, 16, 16]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    mul_622: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(view_418, alias_16);  view_418 = None
    sum_47: "f32[8, 12, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_622, [-1], True)
    mul_623: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(alias_16, sum_47);  alias_16 = sum_47 = None
    sub_132: "f32[8, 12, 16, 16]" = torch.ops.aten.sub.Tensor(mul_622, mul_623);  mul_622 = mul_623 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    sum_48: "f32[1, 12, 16, 16]" = torch.ops.aten.sum.dim_IntList(sub_132, [0], True)
    view_419: "f32[12, 16, 16]" = torch.ops.aten.view.default(sum_48, [12, 16, 16]);  sum_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    index_put_2: "f32[12, 16]" = torch.ops.aten.index_put.default(full_default_2, [None, primals_220], view_419, True);  primals_220 = view_419 = None
    slice_scatter_2: "f32[12, 16]" = torch.ops.aten.slice_scatter.default(full_default_2, index_put_2, 0, 0, 9223372036854775807);  index_put_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    mul_624: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(sub_132, 0.25);  sub_132 = None
    view_420: "f32[96, 16, 16]" = torch.ops.aten.view.default(mul_624, [96, 16, 16]);  mul_624 = None
    bmm_38: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(permute_188, view_420);  permute_188 = None
    bmm_39: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(view_420, permute_189);  view_420 = permute_189 = None
    view_421: "f32[8, 12, 16, 16]" = torch.ops.aten.view.default(bmm_38, [8, 12, 16, 16]);  bmm_38 = None
    view_422: "f32[8, 12, 16, 16]" = torch.ops.aten.view.default(bmm_39, [8, 12, 16, 16]);  bmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_190: "f32[8, 16, 12, 32]" = torch.ops.aten.permute.default(view_417, [0, 2, 1, 3]);  view_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_191: "f32[8, 16, 12, 16]" = torch.ops.aten.permute.default(view_421, [0, 3, 1, 2]);  view_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_192: "f32[8, 16, 12, 16]" = torch.ops.aten.permute.default(view_422, [0, 2, 1, 3]);  view_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    cat_2: "f32[8, 16, 12, 64]" = torch.ops.aten.cat.default([permute_192, permute_191, permute_190], 3);  permute_192 = permute_191 = permute_190 = None
    view_423: "f32[8, 16, 768]" = torch.ops.aten.view.default(cat_2, [8, 16, 768]);  cat_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_424: "f32[128, 768]" = torch.ops.aten.view.default(view_423, [128, 768]);  view_423 = None
    sum_49: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_424, [0])
    sub_133: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_281, unsqueeze_69);  view_281 = unsqueeze_69 = None
    mul_625: "f32[128, 768]" = torch.ops.aten.mul.Tensor(view_424, sub_133)
    sum_50: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_625, [0]);  mul_625 = None
    mul_626: "f32[768]" = torch.ops.aten.mul.Tensor(sum_49, 0.0078125)
    unsqueeze_70: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_626, 0);  mul_626 = None
    mul_627: "f32[768]" = torch.ops.aten.mul.Tensor(sum_50, 0.0078125)
    mul_628: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
    mul_629: "f32[768]" = torch.ops.aten.mul.Tensor(mul_627, mul_628);  mul_627 = mul_628 = None
    unsqueeze_71: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_629, 0);  mul_629 = None
    mul_630: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_151, primals_166);  primals_166 = None
    unsqueeze_72: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_630, 0);  mul_630 = None
    mul_631: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_71);  sub_133 = unsqueeze_71 = None
    sub_135: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_424, mul_631);  view_424 = mul_631 = None
    sub_136: "f32[128, 768]" = torch.ops.aten.sub.Tensor(sub_135, unsqueeze_70);  sub_135 = unsqueeze_70 = None
    mul_632: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_72);  sub_136 = unsqueeze_72 = None
    mul_633: "f32[768]" = torch.ops.aten.mul.Tensor(sum_50, squeeze_151);  sum_50 = squeeze_151 = None
    view_425: "f32[8, 16, 768]" = torch.ops.aten.view.default(mul_632, [8, 16, 768]);  mul_632 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_426: "f32[128, 768]" = torch.ops.aten.view.default(view_425, [128, 768]);  view_425 = None
    permute_193: "f32[768, 128]" = torch.ops.aten.permute.default(view_426, [1, 0])
    mm_84: "f32[768, 384]" = torch.ops.aten.mm.default(permute_193, view_279);  permute_193 = view_279 = None
    permute_194: "f32[384, 768]" = torch.ops.aten.permute.default(mm_84, [1, 0]);  mm_84 = None
    mm_85: "f32[128, 384]" = torch.ops.aten.mm.default(view_426, permute_195);  view_426 = permute_195 = None
    view_427: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_85, [8, 16, 384]);  mm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_404: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_402, view_427);  add_402 = view_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_196: "f32[768, 384]" = torch.ops.aten.permute.default(permute_194, [1, 0]);  permute_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_428: "f32[128, 384]" = torch.ops.aten.view.default(add_404, [128, 384])
    sum_51: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_428, [0])
    sub_137: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_277, unsqueeze_73);  view_277 = unsqueeze_73 = None
    mul_634: "f32[128, 384]" = torch.ops.aten.mul.Tensor(view_428, sub_137)
    sum_52: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_634, [0]);  mul_634 = None
    mul_635: "f32[384]" = torch.ops.aten.mul.Tensor(sum_51, 0.0078125)
    unsqueeze_74: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_635, 0);  mul_635 = None
    mul_636: "f32[384]" = torch.ops.aten.mul.Tensor(sum_52, 0.0078125)
    mul_637: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
    mul_638: "f32[384]" = torch.ops.aten.mul.Tensor(mul_636, mul_637);  mul_636 = mul_637 = None
    unsqueeze_75: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_638, 0);  mul_638 = None
    mul_639: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_148, primals_163);  primals_163 = None
    unsqueeze_76: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_639, 0);  mul_639 = None
    mul_640: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_75);  sub_137 = unsqueeze_75 = None
    sub_139: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_428, mul_640);  view_428 = mul_640 = None
    sub_140: "f32[128, 384]" = torch.ops.aten.sub.Tensor(sub_139, unsqueeze_74);  sub_139 = unsqueeze_74 = None
    mul_641: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_76);  sub_140 = unsqueeze_76 = None
    mul_642: "f32[384]" = torch.ops.aten.mul.Tensor(sum_52, squeeze_148);  sum_52 = squeeze_148 = None
    view_429: "f32[8, 16, 384]" = torch.ops.aten.view.default(mul_641, [8, 16, 384]);  mul_641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_430: "f32[128, 384]" = torch.ops.aten.view.default(view_429, [128, 384]);  view_429 = None
    permute_197: "f32[384, 128]" = torch.ops.aten.permute.default(view_430, [1, 0])
    mm_86: "f32[384, 768]" = torch.ops.aten.mm.default(permute_197, view_275);  permute_197 = view_275 = None
    permute_198: "f32[768, 384]" = torch.ops.aten.permute.default(mm_86, [1, 0]);  mm_86 = None
    mm_87: "f32[128, 768]" = torch.ops.aten.mm.default(view_430, permute_199);  view_430 = permute_199 = None
    view_431: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_87, [8, 16, 768]);  mm_87 = None
    permute_200: "f32[384, 768]" = torch.ops.aten.permute.default(permute_198, [1, 0]);  permute_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    lt_6: "b8[8, 16, 768]" = torch.ops.aten.lt.Scalar(view_274, -3)
    le_6: "b8[8, 16, 768]" = torch.ops.aten.le.Scalar(view_274, 3)
    div_54: "f32[8, 16, 768]" = torch.ops.aten.div.Tensor(view_274, 3);  view_274 = None
    add_405: "f32[8, 16, 768]" = torch.ops.aten.add.Tensor(div_54, 0.5);  div_54 = None
    mul_643: "f32[8, 16, 768]" = torch.ops.aten.mul.Tensor(view_431, add_405);  add_405 = None
    where_12: "f32[8, 16, 768]" = torch.ops.aten.where.self(le_6, mul_643, view_431);  le_6 = mul_643 = view_431 = None
    where_13: "f32[8, 16, 768]" = torch.ops.aten.where.self(lt_6, full_default, where_12);  lt_6 = where_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_432: "f32[128, 768]" = torch.ops.aten.view.default(where_13, [128, 768]);  where_13 = None
    sum_53: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_432, [0])
    sub_141: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_273, unsqueeze_77);  view_273 = unsqueeze_77 = None
    mul_644: "f32[128, 768]" = torch.ops.aten.mul.Tensor(view_432, sub_141)
    sum_54: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_644, [0]);  mul_644 = None
    mul_645: "f32[768]" = torch.ops.aten.mul.Tensor(sum_53, 0.0078125)
    unsqueeze_78: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_645, 0);  mul_645 = None
    mul_646: "f32[768]" = torch.ops.aten.mul.Tensor(sum_54, 0.0078125)
    mul_647: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
    mul_648: "f32[768]" = torch.ops.aten.mul.Tensor(mul_646, mul_647);  mul_646 = mul_647 = None
    unsqueeze_79: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_648, 0);  mul_648 = None
    mul_649: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_145, primals_160);  primals_160 = None
    unsqueeze_80: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_649, 0);  mul_649 = None
    mul_650: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_79);  sub_141 = unsqueeze_79 = None
    sub_143: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_432, mul_650);  view_432 = mul_650 = None
    sub_144: "f32[128, 768]" = torch.ops.aten.sub.Tensor(sub_143, unsqueeze_78);  sub_143 = unsqueeze_78 = None
    mul_651: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_80);  sub_144 = unsqueeze_80 = None
    mul_652: "f32[768]" = torch.ops.aten.mul.Tensor(sum_54, squeeze_145);  sum_54 = squeeze_145 = None
    view_433: "f32[8, 16, 768]" = torch.ops.aten.view.default(mul_651, [8, 16, 768]);  mul_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_434: "f32[128, 768]" = torch.ops.aten.view.default(view_433, [128, 768]);  view_433 = None
    permute_201: "f32[768, 128]" = torch.ops.aten.permute.default(view_434, [1, 0])
    mm_88: "f32[768, 384]" = torch.ops.aten.mm.default(permute_201, view_271);  permute_201 = view_271 = None
    permute_202: "f32[384, 768]" = torch.ops.aten.permute.default(mm_88, [1, 0]);  mm_88 = None
    mm_89: "f32[128, 384]" = torch.ops.aten.mm.default(view_434, permute_203);  view_434 = permute_203 = None
    view_435: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_89, [8, 16, 384]);  mm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_406: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_404, view_435);  add_404 = view_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_204: "f32[768, 384]" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_436: "f32[128, 384]" = torch.ops.aten.view.default(add_406, [128, 384])
    sum_55: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_436, [0])
    sub_145: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_269, unsqueeze_81);  view_269 = unsqueeze_81 = None
    mul_653: "f32[128, 384]" = torch.ops.aten.mul.Tensor(view_436, sub_145)
    sum_56: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_653, [0]);  mul_653 = None
    mul_654: "f32[384]" = torch.ops.aten.mul.Tensor(sum_55, 0.0078125)
    unsqueeze_82: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_654, 0);  mul_654 = None
    mul_655: "f32[384]" = torch.ops.aten.mul.Tensor(sum_56, 0.0078125)
    mul_656: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
    mul_657: "f32[384]" = torch.ops.aten.mul.Tensor(mul_655, mul_656);  mul_655 = mul_656 = None
    unsqueeze_83: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_657, 0);  mul_657 = None
    mul_658: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_157);  primals_157 = None
    unsqueeze_84: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_658, 0);  mul_658 = None
    mul_659: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_83);  sub_145 = unsqueeze_83 = None
    sub_147: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_436, mul_659);  view_436 = mul_659 = None
    sub_148: "f32[128, 384]" = torch.ops.aten.sub.Tensor(sub_147, unsqueeze_82);  sub_147 = unsqueeze_82 = None
    mul_660: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_84);  sub_148 = unsqueeze_84 = None
    mul_661: "f32[384]" = torch.ops.aten.mul.Tensor(sum_56, squeeze_142);  sum_56 = squeeze_142 = None
    view_437: "f32[8, 16, 384]" = torch.ops.aten.view.default(mul_660, [8, 16, 384]);  mul_660 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_438: "f32[128, 384]" = torch.ops.aten.view.default(view_437, [128, 384]);  view_437 = None
    permute_205: "f32[384, 128]" = torch.ops.aten.permute.default(view_438, [1, 0])
    mm_90: "f32[384, 384]" = torch.ops.aten.mm.default(permute_205, view_267);  permute_205 = view_267 = None
    permute_206: "f32[384, 384]" = torch.ops.aten.permute.default(mm_90, [1, 0]);  mm_90 = None
    mm_91: "f32[128, 384]" = torch.ops.aten.mm.default(view_438, permute_207);  view_438 = permute_207 = None
    view_439: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_91, [8, 16, 384]);  mm_91 = None
    permute_208: "f32[384, 384]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    lt_7: "b8[8, 16, 384]" = torch.ops.aten.lt.Scalar(view_266, -3)
    le_7: "b8[8, 16, 384]" = torch.ops.aten.le.Scalar(view_266, 3)
    div_55: "f32[8, 16, 384]" = torch.ops.aten.div.Tensor(view_266, 3);  view_266 = None
    add_407: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(div_55, 0.5);  div_55 = None
    mul_662: "f32[8, 16, 384]" = torch.ops.aten.mul.Tensor(view_439, add_407);  add_407 = None
    where_14: "f32[8, 16, 384]" = torch.ops.aten.where.self(le_7, mul_662, view_439);  le_7 = mul_662 = view_439 = None
    where_15: "f32[8, 16, 384]" = torch.ops.aten.where.self(lt_7, full_default, where_14);  lt_7 = where_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    view_440: "f32[8, 16, 12, 32]" = torch.ops.aten.view.default(where_15, [8, 16, 12, 32]);  where_15 = None
    permute_209: "f32[8, 12, 16, 32]" = torch.ops.aten.permute.default(view_440, [0, 2, 1, 3]);  view_440 = None
    clone_86: "f32[8, 12, 16, 32]" = torch.ops.aten.clone.default(permute_209, memory_format = torch.contiguous_format);  permute_209 = None
    view_441: "f32[96, 16, 32]" = torch.ops.aten.view.default(clone_86, [96, 16, 32]);  clone_86 = None
    bmm_40: "f32[96, 16, 32]" = torch.ops.aten.bmm.default(permute_210, view_441);  permute_210 = None
    bmm_41: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(view_441, permute_211);  view_441 = permute_211 = None
    view_442: "f32[8, 12, 16, 32]" = torch.ops.aten.view.default(bmm_40, [8, 12, 16, 32]);  bmm_40 = None
    view_443: "f32[8, 12, 16, 16]" = torch.ops.aten.view.default(bmm_41, [8, 12, 16, 16]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    mul_663: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(view_443, alias_17);  view_443 = None
    sum_57: "f32[8, 12, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_663, [-1], True)
    mul_664: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(alias_17, sum_57);  alias_17 = sum_57 = None
    sub_149: "f32[8, 12, 16, 16]" = torch.ops.aten.sub.Tensor(mul_663, mul_664);  mul_663 = mul_664 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    sum_58: "f32[1, 12, 16, 16]" = torch.ops.aten.sum.dim_IntList(sub_149, [0], True)
    view_444: "f32[12, 16, 16]" = torch.ops.aten.view.default(sum_58, [12, 16, 16]);  sum_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    index_put_3: "f32[12, 16]" = torch.ops.aten.index_put.default(full_default_2, [None, primals_219], view_444, True);  primals_219 = view_444 = None
    slice_scatter_3: "f32[12, 16]" = torch.ops.aten.slice_scatter.default(full_default_2, index_put_3, 0, 0, 9223372036854775807);  full_default_2 = index_put_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    mul_665: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(sub_149, 0.25);  sub_149 = None
    view_445: "f32[96, 16, 16]" = torch.ops.aten.view.default(mul_665, [96, 16, 16]);  mul_665 = None
    bmm_42: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(permute_212, view_445);  permute_212 = None
    bmm_43: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(view_445, permute_213);  view_445 = permute_213 = None
    view_446: "f32[8, 12, 16, 16]" = torch.ops.aten.view.default(bmm_42, [8, 12, 16, 16]);  bmm_42 = None
    view_447: "f32[8, 12, 16, 16]" = torch.ops.aten.view.default(bmm_43, [8, 12, 16, 16]);  bmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_214: "f32[8, 16, 12, 32]" = torch.ops.aten.permute.default(view_442, [0, 2, 1, 3]);  view_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_215: "f32[8, 16, 12, 16]" = torch.ops.aten.permute.default(view_446, [0, 3, 1, 2]);  view_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_216: "f32[8, 16, 12, 16]" = torch.ops.aten.permute.default(view_447, [0, 2, 1, 3]);  view_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    cat_3: "f32[8, 16, 12, 64]" = torch.ops.aten.cat.default([permute_216, permute_215, permute_214], 3);  permute_216 = permute_215 = permute_214 = None
    view_448: "f32[8, 16, 768]" = torch.ops.aten.view.default(cat_3, [8, 16, 768]);  cat_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_449: "f32[128, 768]" = torch.ops.aten.view.default(view_448, [128, 768]);  view_448 = None
    sum_59: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_449, [0])
    sub_150: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_257, unsqueeze_85);  view_257 = unsqueeze_85 = None
    mul_666: "f32[128, 768]" = torch.ops.aten.mul.Tensor(view_449, sub_150)
    sum_60: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_666, [0]);  mul_666 = None
    mul_667: "f32[768]" = torch.ops.aten.mul.Tensor(sum_59, 0.0078125)
    unsqueeze_86: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_667, 0);  mul_667 = None
    mul_668: "f32[768]" = torch.ops.aten.mul.Tensor(sum_60, 0.0078125)
    mul_669: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
    mul_670: "f32[768]" = torch.ops.aten.mul.Tensor(mul_668, mul_669);  mul_668 = mul_669 = None
    unsqueeze_87: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_670, 0);  mul_670 = None
    mul_671: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_154);  primals_154 = None
    unsqueeze_88: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_671, 0);  mul_671 = None
    mul_672: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_150, unsqueeze_87);  sub_150 = unsqueeze_87 = None
    sub_152: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_449, mul_672);  view_449 = mul_672 = None
    sub_153: "f32[128, 768]" = torch.ops.aten.sub.Tensor(sub_152, unsqueeze_86);  sub_152 = unsqueeze_86 = None
    mul_673: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_88);  sub_153 = unsqueeze_88 = None
    mul_674: "f32[768]" = torch.ops.aten.mul.Tensor(sum_60, squeeze_139);  sum_60 = squeeze_139 = None
    view_450: "f32[8, 16, 768]" = torch.ops.aten.view.default(mul_673, [8, 16, 768]);  mul_673 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_451: "f32[128, 768]" = torch.ops.aten.view.default(view_450, [128, 768]);  view_450 = None
    permute_217: "f32[768, 128]" = torch.ops.aten.permute.default(view_451, [1, 0])
    mm_92: "f32[768, 384]" = torch.ops.aten.mm.default(permute_217, view_255);  permute_217 = view_255 = None
    permute_218: "f32[384, 768]" = torch.ops.aten.permute.default(mm_92, [1, 0]);  mm_92 = None
    mm_93: "f32[128, 384]" = torch.ops.aten.mm.default(view_451, permute_219);  view_451 = permute_219 = None
    view_452: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_93, [8, 16, 384]);  mm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_408: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_406, view_452);  add_406 = view_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_220: "f32[768, 384]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_453: "f32[128, 384]" = torch.ops.aten.view.default(add_408, [128, 384])
    sum_61: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_453, [0])
    sub_154: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_253, unsqueeze_89);  view_253 = unsqueeze_89 = None
    mul_675: "f32[128, 384]" = torch.ops.aten.mul.Tensor(view_453, sub_154)
    sum_62: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_675, [0]);  mul_675 = None
    mul_676: "f32[384]" = torch.ops.aten.mul.Tensor(sum_61, 0.0078125)
    unsqueeze_90: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_676, 0);  mul_676 = None
    mul_677: "f32[384]" = torch.ops.aten.mul.Tensor(sum_62, 0.0078125)
    mul_678: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
    mul_679: "f32[384]" = torch.ops.aten.mul.Tensor(mul_677, mul_678);  mul_677 = mul_678 = None
    unsqueeze_91: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_679, 0);  mul_679 = None
    mul_680: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_151);  primals_151 = None
    unsqueeze_92: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_680, 0);  mul_680 = None
    mul_681: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_91);  sub_154 = unsqueeze_91 = None
    sub_156: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_453, mul_681);  view_453 = mul_681 = None
    sub_157: "f32[128, 384]" = torch.ops.aten.sub.Tensor(sub_156, unsqueeze_90);  sub_156 = unsqueeze_90 = None
    mul_682: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_92);  sub_157 = unsqueeze_92 = None
    mul_683: "f32[384]" = torch.ops.aten.mul.Tensor(sum_62, squeeze_136);  sum_62 = squeeze_136 = None
    view_454: "f32[8, 16, 384]" = torch.ops.aten.view.default(mul_682, [8, 16, 384]);  mul_682 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_455: "f32[128, 384]" = torch.ops.aten.view.default(view_454, [128, 384]);  view_454 = None
    permute_221: "f32[384, 128]" = torch.ops.aten.permute.default(view_455, [1, 0])
    mm_94: "f32[384, 768]" = torch.ops.aten.mm.default(permute_221, view_251);  permute_221 = view_251 = None
    permute_222: "f32[768, 384]" = torch.ops.aten.permute.default(mm_94, [1, 0]);  mm_94 = None
    mm_95: "f32[128, 768]" = torch.ops.aten.mm.default(view_455, permute_223);  view_455 = permute_223 = None
    view_456: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_95, [8, 16, 768]);  mm_95 = None
    permute_224: "f32[384, 768]" = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    lt_8: "b8[8, 16, 768]" = torch.ops.aten.lt.Scalar(view_250, -3)
    le_8: "b8[8, 16, 768]" = torch.ops.aten.le.Scalar(view_250, 3)
    div_56: "f32[8, 16, 768]" = torch.ops.aten.div.Tensor(view_250, 3);  view_250 = None
    add_409: "f32[8, 16, 768]" = torch.ops.aten.add.Tensor(div_56, 0.5);  div_56 = None
    mul_684: "f32[8, 16, 768]" = torch.ops.aten.mul.Tensor(view_456, add_409);  add_409 = None
    where_16: "f32[8, 16, 768]" = torch.ops.aten.where.self(le_8, mul_684, view_456);  le_8 = mul_684 = view_456 = None
    where_17: "f32[8, 16, 768]" = torch.ops.aten.where.self(lt_8, full_default, where_16);  lt_8 = where_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_457: "f32[128, 768]" = torch.ops.aten.view.default(where_17, [128, 768]);  where_17 = None
    sum_63: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_457, [0])
    sub_158: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_249, unsqueeze_93);  view_249 = unsqueeze_93 = None
    mul_685: "f32[128, 768]" = torch.ops.aten.mul.Tensor(view_457, sub_158)
    sum_64: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_685, [0]);  mul_685 = None
    mul_686: "f32[768]" = torch.ops.aten.mul.Tensor(sum_63, 0.0078125)
    unsqueeze_94: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_686, 0);  mul_686 = None
    mul_687: "f32[768]" = torch.ops.aten.mul.Tensor(sum_64, 0.0078125)
    mul_688: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
    mul_689: "f32[768]" = torch.ops.aten.mul.Tensor(mul_687, mul_688);  mul_687 = mul_688 = None
    unsqueeze_95: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_689, 0);  mul_689 = None
    mul_690: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_148);  primals_148 = None
    unsqueeze_96: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_690, 0);  mul_690 = None
    mul_691: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_158, unsqueeze_95);  sub_158 = unsqueeze_95 = None
    sub_160: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_457, mul_691);  view_457 = mul_691 = None
    sub_161: "f32[128, 768]" = torch.ops.aten.sub.Tensor(sub_160, unsqueeze_94);  sub_160 = unsqueeze_94 = None
    mul_692: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_96);  sub_161 = unsqueeze_96 = None
    mul_693: "f32[768]" = torch.ops.aten.mul.Tensor(sum_64, squeeze_133);  sum_64 = squeeze_133 = None
    view_458: "f32[8, 16, 768]" = torch.ops.aten.view.default(mul_692, [8, 16, 768]);  mul_692 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_459: "f32[128, 768]" = torch.ops.aten.view.default(view_458, [128, 768]);  view_458 = None
    permute_225: "f32[768, 128]" = torch.ops.aten.permute.default(view_459, [1, 0])
    mm_96: "f32[768, 384]" = torch.ops.aten.mm.default(permute_225, view_247);  permute_225 = view_247 = None
    permute_226: "f32[384, 768]" = torch.ops.aten.permute.default(mm_96, [1, 0]);  mm_96 = None
    mm_97: "f32[128, 384]" = torch.ops.aten.mm.default(view_459, permute_227);  view_459 = permute_227 = None
    view_460: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_97, [8, 16, 384]);  mm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_410: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_408, view_460);  add_408 = view_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_228: "f32[768, 384]" = torch.ops.aten.permute.default(permute_226, [1, 0]);  permute_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_461: "f32[128, 384]" = torch.ops.aten.view.default(add_410, [128, 384]);  add_410 = None
    sum_65: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_461, [0])
    sub_162: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_245, unsqueeze_97);  view_245 = unsqueeze_97 = None
    mul_694: "f32[128, 384]" = torch.ops.aten.mul.Tensor(view_461, sub_162)
    sum_66: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_694, [0]);  mul_694 = None
    mul_695: "f32[384]" = torch.ops.aten.mul.Tensor(sum_65, 0.0078125)
    unsqueeze_98: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_695, 0);  mul_695 = None
    mul_696: "f32[384]" = torch.ops.aten.mul.Tensor(sum_66, 0.0078125)
    mul_697: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
    mul_698: "f32[384]" = torch.ops.aten.mul.Tensor(mul_696, mul_697);  mul_696 = mul_697 = None
    unsqueeze_99: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_698, 0);  mul_698 = None
    mul_699: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_145);  primals_145 = None
    unsqueeze_100: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_699, 0);  mul_699 = None
    mul_700: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_99);  sub_162 = unsqueeze_99 = None
    sub_164: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_461, mul_700);  view_461 = mul_700 = None
    sub_165: "f32[128, 384]" = torch.ops.aten.sub.Tensor(sub_164, unsqueeze_98);  sub_164 = unsqueeze_98 = None
    mul_701: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_100);  sub_165 = unsqueeze_100 = None
    mul_702: "f32[384]" = torch.ops.aten.mul.Tensor(sum_66, squeeze_130);  sum_66 = squeeze_130 = None
    view_462: "f32[8, 16, 384]" = torch.ops.aten.view.default(mul_701, [8, 16, 384]);  mul_701 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_463: "f32[128, 384]" = torch.ops.aten.view.default(view_462, [128, 384]);  view_462 = None
    permute_229: "f32[384, 128]" = torch.ops.aten.permute.default(view_463, [1, 0])
    mm_98: "f32[384, 1024]" = torch.ops.aten.mm.default(permute_229, view_243);  permute_229 = view_243 = None
    permute_230: "f32[1024, 384]" = torch.ops.aten.permute.default(mm_98, [1, 0]);  mm_98 = None
    mm_99: "f32[128, 1024]" = torch.ops.aten.mm.default(view_463, permute_231);  view_463 = permute_231 = None
    view_464: "f32[8, 16, 1024]" = torch.ops.aten.view.default(mm_99, [8, 16, 1024]);  mm_99 = None
    permute_232: "f32[384, 1024]" = torch.ops.aten.permute.default(permute_230, [1, 0]);  permute_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:340, code: x = self.proj(x)
    lt_9: "b8[8, 16, 1024]" = torch.ops.aten.lt.Scalar(view_242, -3)
    le_9: "b8[8, 16, 1024]" = torch.ops.aten.le.Scalar(view_242, 3)
    div_57: "f32[8, 16, 1024]" = torch.ops.aten.div.Tensor(view_242, 3);  view_242 = None
    add_411: "f32[8, 16, 1024]" = torch.ops.aten.add.Tensor(div_57, 0.5);  div_57 = None
    mul_703: "f32[8, 16, 1024]" = torch.ops.aten.mul.Tensor(view_464, add_411);  add_411 = None
    where_18: "f32[8, 16, 1024]" = torch.ops.aten.where.self(le_9, mul_703, view_464);  le_9 = mul_703 = view_464 = None
    where_19: "f32[8, 16, 1024]" = torch.ops.aten.where.self(lt_9, full_default, where_18);  lt_9 = where_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:339, code: x = (attn @ v).transpose(1, 2).reshape(B, -1, self.val_attn_dim)
    view_465: "f32[8, 16, 16, 64]" = torch.ops.aten.view.default(where_19, [8, 16, 16, 64]);  where_19 = None
    permute_233: "f32[8, 16, 16, 64]" = torch.ops.aten.permute.default(view_465, [0, 2, 1, 3]);  view_465 = None
    clone_87: "f32[8, 16, 16, 64]" = torch.ops.aten.clone.default(permute_233, memory_format = torch.contiguous_format);  permute_233 = None
    view_466: "f32[128, 16, 64]" = torch.ops.aten.view.default(clone_87, [128, 16, 64]);  clone_87 = None
    bmm_44: "f32[128, 49, 64]" = torch.ops.aten.bmm.default(permute_234, view_466);  permute_234 = None
    bmm_45: "f32[128, 16, 49]" = torch.ops.aten.bmm.default(view_466, permute_235);  view_466 = permute_235 = None
    view_467: "f32[8, 16, 49, 64]" = torch.ops.aten.view.default(bmm_44, [8, 16, 49, 64]);  bmm_44 = None
    view_468: "f32[8, 16, 16, 49]" = torch.ops.aten.view.default(bmm_45, [8, 16, 16, 49]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:337, code: attn = attn.softmax(dim=-1)
    mul_704: "f32[8, 16, 16, 49]" = torch.ops.aten.mul.Tensor(view_468, alias_18);  view_468 = None
    sum_67: "f32[8, 16, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_704, [-1], True)
    mul_705: "f32[8, 16, 16, 49]" = torch.ops.aten.mul.Tensor(alias_18, sum_67);  alias_18 = sum_67 = None
    sub_166: "f32[8, 16, 16, 49]" = torch.ops.aten.sub.Tensor(mul_704, mul_705);  mul_704 = mul_705 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:336, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    sum_68: "f32[1, 16, 16, 49]" = torch.ops.aten.sum.dim_IntList(sub_166, [0], True)
    view_469: "f32[16, 16, 49]" = torch.ops.aten.view.default(sum_68, [16, 16, 49]);  sum_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:311, code: return self.attention_biases[:, self.attention_bias_idxs]
    full_default_18: "f32[16, 49]" = torch.ops.aten.full.default([16, 49], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    index_put_4: "f32[16, 49]" = torch.ops.aten.index_put.default(full_default_18, [None, primals_218], view_469, True);  primals_218 = view_469 = None
    slice_scatter_4: "f32[16, 49]" = torch.ops.aten.slice_scatter.default(full_default_18, index_put_4, 0, 0, 9223372036854775807);  full_default_18 = index_put_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:336, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    mul_706: "f32[8, 16, 16, 49]" = torch.ops.aten.mul.Tensor(sub_166, 0.25);  sub_166 = None
    view_470: "f32[128, 16, 49]" = torch.ops.aten.view.default(mul_706, [128, 16, 49]);  mul_706 = None
    bmm_46: "f32[128, 16, 49]" = torch.ops.aten.bmm.default(permute_236, view_470);  permute_236 = None
    bmm_47: "f32[128, 16, 16]" = torch.ops.aten.bmm.default(view_470, permute_237);  view_470 = permute_237 = None
    view_471: "f32[8, 16, 16, 49]" = torch.ops.aten.view.default(bmm_46, [8, 16, 16, 49]);  bmm_46 = None
    view_472: "f32[8, 16, 16, 16]" = torch.ops.aten.view.default(bmm_47, [8, 16, 16, 16]);  bmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:334, code: q = self.q(x).view(B, -1, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
    permute_238: "f32[8, 16, 16, 16]" = torch.ops.aten.permute.default(view_472, [0, 2, 1, 3]);  view_472 = None
    clone_88: "f32[8, 16, 16, 16]" = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None
    view_473: "f32[8, 16, 256]" = torch.ops.aten.view.default(clone_88, [8, 16, 256]);  clone_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_474: "f32[128, 256]" = torch.ops.aten.view.default(view_473, [128, 256]);  view_473 = None
    sum_69: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_474, [0])
    sub_167: "f32[128, 256]" = torch.ops.aten.sub.Tensor(view_233, unsqueeze_101);  view_233 = unsqueeze_101 = None
    mul_707: "f32[128, 256]" = torch.ops.aten.mul.Tensor(view_474, sub_167)
    sum_70: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_707, [0]);  mul_707 = None
    mul_708: "f32[256]" = torch.ops.aten.mul.Tensor(sum_69, 0.0078125)
    unsqueeze_102: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_708, 0);  mul_708 = None
    mul_709: "f32[256]" = torch.ops.aten.mul.Tensor(sum_70, 0.0078125)
    mul_710: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
    mul_711: "f32[256]" = torch.ops.aten.mul.Tensor(mul_709, mul_710);  mul_709 = mul_710 = None
    unsqueeze_103: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_711, 0);  mul_711 = None
    mul_712: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_142);  primals_142 = None
    unsqueeze_104: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_712, 0);  mul_712 = None
    mul_713: "f32[128, 256]" = torch.ops.aten.mul.Tensor(sub_167, unsqueeze_103);  sub_167 = unsqueeze_103 = None
    sub_169: "f32[128, 256]" = torch.ops.aten.sub.Tensor(view_474, mul_713);  view_474 = mul_713 = None
    sub_170: "f32[128, 256]" = torch.ops.aten.sub.Tensor(sub_169, unsqueeze_102);  sub_169 = unsqueeze_102 = None
    mul_714: "f32[128, 256]" = torch.ops.aten.mul.Tensor(sub_170, unsqueeze_104);  sub_170 = unsqueeze_104 = None
    mul_715: "f32[256]" = torch.ops.aten.mul.Tensor(sum_70, squeeze_127);  sum_70 = squeeze_127 = None
    view_475: "f32[8, 16, 256]" = torch.ops.aten.view.default(mul_714, [8, 16, 256]);  mul_714 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_476: "f32[128, 256]" = torch.ops.aten.view.default(view_475, [128, 256]);  view_475 = None
    permute_239: "f32[256, 128]" = torch.ops.aten.permute.default(view_476, [1, 0])
    mm_100: "f32[256, 256]" = torch.ops.aten.mm.default(permute_239, view_231);  permute_239 = view_231 = None
    permute_240: "f32[256, 256]" = torch.ops.aten.permute.default(mm_100, [1, 0]);  mm_100 = None
    mm_101: "f32[128, 256]" = torch.ops.aten.mm.default(view_476, permute_241);  view_476 = permute_241 = None
    view_477: "f32[8, 16, 256]" = torch.ops.aten.view.default(mm_101, [8, 16, 256]);  mm_101 = None
    permute_242: "f32[256, 256]" = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:162, code: return x.reshape(B, -1, C)
    view_478: "f32[8, 4, 4, 256]" = torch.ops.aten.view.default(view_477, [8, 4, 4, 256]);  view_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:161, code: x = x[:, ::self.stride, ::self.stride]
    full_default_20: "f32[8, 4, 7, 256]" = torch.ops.aten.full.default([8, 4, 7, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_5: "f32[8, 4, 7, 256]" = torch.ops.aten.slice_scatter.default(full_default_20, view_478, 2, 0, 9223372036854775807, 2);  full_default_20 = view_478 = None
    full_default_21: "f32[8, 7, 7, 256]" = torch.ops.aten.full.default([8, 7, 7, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_6: "f32[8, 7, 7, 256]" = torch.ops.aten.slice_scatter.default(full_default_21, slice_scatter_5, 1, 0, 9223372036854775807, 2);  slice_scatter_5 = None
    slice_scatter_7: "f32[8, 7, 7, 256]" = torch.ops.aten.slice_scatter.default(full_default_21, slice_scatter_6, 0, 0, 9223372036854775807);  full_default_21 = slice_scatter_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:157, code: x = x.view(B, self.resolution[0], self.resolution[1], C)
    view_479: "f32[8, 49, 256]" = torch.ops.aten.view.default(slice_scatter_7, [8, 49, 256]);  slice_scatter_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:333, code: v = v.permute(0, 2, 1, 3)  # BHNC
    permute_243: "f32[8, 49, 16, 64]" = torch.ops.aten.permute.default(view_467, [0, 2, 1, 3]);  view_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:332, code: k = k.permute(0, 2, 3, 1)  # BHCN
    permute_244: "f32[8, 49, 16, 16]" = torch.ops.aten.permute.default(view_471, [0, 3, 1, 2]);  view_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:331, code: k, v = self.kv(x).view(B, N, self.num_heads, -1).split([self.key_dim, self.val_dim], dim=3)
    cat_4: "f32[8, 49, 16, 80]" = torch.ops.aten.cat.default([permute_244, permute_243], 3);  permute_244 = permute_243 = None
    view_480: "f32[8, 49, 1280]" = torch.ops.aten.view.default(cat_4, [8, 49, 1280]);  cat_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_481: "f32[392, 1280]" = torch.ops.aten.view.default(view_480, [392, 1280]);  view_480 = None
    sum_71: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_481, [0])
    sub_171: "f32[392, 1280]" = torch.ops.aten.sub.Tensor(view_226, unsqueeze_105);  view_226 = unsqueeze_105 = None
    mul_716: "f32[392, 1280]" = torch.ops.aten.mul.Tensor(view_481, sub_171)
    sum_72: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_716, [0]);  mul_716 = None
    mul_717: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_71, 0.002551020408163265)
    unsqueeze_106: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_717, 0);  mul_717 = None
    mul_718: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_72, 0.002551020408163265)
    mul_719: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
    mul_720: "f32[1280]" = torch.ops.aten.mul.Tensor(mul_718, mul_719);  mul_718 = mul_719 = None
    unsqueeze_107: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_720, 0);  mul_720 = None
    mul_721: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_139);  primals_139 = None
    unsqueeze_108: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_721, 0);  mul_721 = None
    mul_722: "f32[392, 1280]" = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_107);  sub_171 = unsqueeze_107 = None
    sub_173: "f32[392, 1280]" = torch.ops.aten.sub.Tensor(view_481, mul_722);  view_481 = mul_722 = None
    sub_174: "f32[392, 1280]" = torch.ops.aten.sub.Tensor(sub_173, unsqueeze_106);  sub_173 = unsqueeze_106 = None
    mul_723: "f32[392, 1280]" = torch.ops.aten.mul.Tensor(sub_174, unsqueeze_108);  sub_174 = unsqueeze_108 = None
    mul_724: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_72, squeeze_124);  sum_72 = squeeze_124 = None
    view_482: "f32[8, 49, 1280]" = torch.ops.aten.view.default(mul_723, [8, 49, 1280]);  mul_723 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_483: "f32[392, 1280]" = torch.ops.aten.view.default(view_482, [392, 1280]);  view_482 = None
    permute_245: "f32[1280, 392]" = torch.ops.aten.permute.default(view_483, [1, 0])
    mm_102: "f32[1280, 256]" = torch.ops.aten.mm.default(permute_245, view_224);  permute_245 = view_224 = None
    permute_246: "f32[256, 1280]" = torch.ops.aten.permute.default(mm_102, [1, 0]);  mm_102 = None
    mm_103: "f32[392, 256]" = torch.ops.aten.mm.default(view_483, permute_247);  view_483 = permute_247 = None
    view_484: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_103, [8, 49, 256]);  mm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_412: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(view_479, view_484);  view_479 = view_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_248: "f32[1280, 256]" = torch.ops.aten.permute.default(permute_246, [1, 0]);  permute_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_485: "f32[392, 256]" = torch.ops.aten.view.default(add_412, [392, 256])
    sum_73: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_485, [0])
    sub_175: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_222, unsqueeze_109);  view_222 = unsqueeze_109 = None
    mul_725: "f32[392, 256]" = torch.ops.aten.mul.Tensor(view_485, sub_175)
    sum_74: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_725, [0]);  mul_725 = None
    mul_726: "f32[256]" = torch.ops.aten.mul.Tensor(sum_73, 0.002551020408163265)
    unsqueeze_110: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_726, 0);  mul_726 = None
    mul_727: "f32[256]" = torch.ops.aten.mul.Tensor(sum_74, 0.002551020408163265)
    mul_728: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
    mul_729: "f32[256]" = torch.ops.aten.mul.Tensor(mul_727, mul_728);  mul_727 = mul_728 = None
    unsqueeze_111: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_729, 0);  mul_729 = None
    mul_730: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_136);  primals_136 = None
    unsqueeze_112: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_730, 0);  mul_730 = None
    mul_731: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_111);  sub_175 = unsqueeze_111 = None
    sub_177: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_485, mul_731);  view_485 = mul_731 = None
    sub_178: "f32[392, 256]" = torch.ops.aten.sub.Tensor(sub_177, unsqueeze_110);  sub_177 = unsqueeze_110 = None
    mul_732: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_178, unsqueeze_112);  sub_178 = unsqueeze_112 = None
    mul_733: "f32[256]" = torch.ops.aten.mul.Tensor(sum_74, squeeze_121);  sum_74 = squeeze_121 = None
    view_486: "f32[8, 49, 256]" = torch.ops.aten.view.default(mul_732, [8, 49, 256]);  mul_732 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_487: "f32[392, 256]" = torch.ops.aten.view.default(view_486, [392, 256]);  view_486 = None
    permute_249: "f32[256, 392]" = torch.ops.aten.permute.default(view_487, [1, 0])
    mm_104: "f32[256, 512]" = torch.ops.aten.mm.default(permute_249, view_220);  permute_249 = view_220 = None
    permute_250: "f32[512, 256]" = torch.ops.aten.permute.default(mm_104, [1, 0]);  mm_104 = None
    mm_105: "f32[392, 512]" = torch.ops.aten.mm.default(view_487, permute_251);  view_487 = permute_251 = None
    view_488: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_105, [8, 49, 512]);  mm_105 = None
    permute_252: "f32[256, 512]" = torch.ops.aten.permute.default(permute_250, [1, 0]);  permute_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    lt_10: "b8[8, 49, 512]" = torch.ops.aten.lt.Scalar(view_219, -3)
    le_10: "b8[8, 49, 512]" = torch.ops.aten.le.Scalar(view_219, 3)
    div_58: "f32[8, 49, 512]" = torch.ops.aten.div.Tensor(view_219, 3);  view_219 = None
    add_413: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(div_58, 0.5);  div_58 = None
    mul_734: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_488, add_413);  add_413 = None
    where_20: "f32[8, 49, 512]" = torch.ops.aten.where.self(le_10, mul_734, view_488);  le_10 = mul_734 = view_488 = None
    where_21: "f32[8, 49, 512]" = torch.ops.aten.where.self(lt_10, full_default, where_20);  lt_10 = where_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_489: "f32[392, 512]" = torch.ops.aten.view.default(where_21, [392, 512]);  where_21 = None
    sum_75: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_489, [0])
    sub_179: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_218, unsqueeze_113);  view_218 = unsqueeze_113 = None
    mul_735: "f32[392, 512]" = torch.ops.aten.mul.Tensor(view_489, sub_179)
    sum_76: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_735, [0]);  mul_735 = None
    mul_736: "f32[512]" = torch.ops.aten.mul.Tensor(sum_75, 0.002551020408163265)
    unsqueeze_114: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_736, 0);  mul_736 = None
    mul_737: "f32[512]" = torch.ops.aten.mul.Tensor(sum_76, 0.002551020408163265)
    mul_738: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
    mul_739: "f32[512]" = torch.ops.aten.mul.Tensor(mul_737, mul_738);  mul_737 = mul_738 = None
    unsqueeze_115: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_739, 0);  mul_739 = None
    mul_740: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_133);  primals_133 = None
    unsqueeze_116: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_740, 0);  mul_740 = None
    mul_741: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_115);  sub_179 = unsqueeze_115 = None
    sub_181: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_489, mul_741);  view_489 = mul_741 = None
    sub_182: "f32[392, 512]" = torch.ops.aten.sub.Tensor(sub_181, unsqueeze_114);  sub_181 = unsqueeze_114 = None
    mul_742: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_116);  sub_182 = unsqueeze_116 = None
    mul_743: "f32[512]" = torch.ops.aten.mul.Tensor(sum_76, squeeze_118);  sum_76 = squeeze_118 = None
    view_490: "f32[8, 49, 512]" = torch.ops.aten.view.default(mul_742, [8, 49, 512]);  mul_742 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_491: "f32[392, 512]" = torch.ops.aten.view.default(view_490, [392, 512]);  view_490 = None
    permute_253: "f32[512, 392]" = torch.ops.aten.permute.default(view_491, [1, 0])
    mm_106: "f32[512, 256]" = torch.ops.aten.mm.default(permute_253, view_216);  permute_253 = view_216 = None
    permute_254: "f32[256, 512]" = torch.ops.aten.permute.default(mm_106, [1, 0]);  mm_106 = None
    mm_107: "f32[392, 256]" = torch.ops.aten.mm.default(view_491, permute_255);  view_491 = permute_255 = None
    view_492: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_107, [8, 49, 256]);  mm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_414: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_412, view_492);  add_412 = view_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_256: "f32[512, 256]" = torch.ops.aten.permute.default(permute_254, [1, 0]);  permute_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_493: "f32[392, 256]" = torch.ops.aten.view.default(add_414, [392, 256])
    sum_77: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_493, [0])
    sub_183: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_214, unsqueeze_117);  view_214 = unsqueeze_117 = None
    mul_744: "f32[392, 256]" = torch.ops.aten.mul.Tensor(view_493, sub_183)
    sum_78: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_744, [0]);  mul_744 = None
    mul_745: "f32[256]" = torch.ops.aten.mul.Tensor(sum_77, 0.002551020408163265)
    unsqueeze_118: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_745, 0);  mul_745 = None
    mul_746: "f32[256]" = torch.ops.aten.mul.Tensor(sum_78, 0.002551020408163265)
    mul_747: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
    mul_748: "f32[256]" = torch.ops.aten.mul.Tensor(mul_746, mul_747);  mul_746 = mul_747 = None
    unsqueeze_119: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_748, 0);  mul_748 = None
    mul_749: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_130);  primals_130 = None
    unsqueeze_120: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_749, 0);  mul_749 = None
    mul_750: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_183, unsqueeze_119);  sub_183 = unsqueeze_119 = None
    sub_185: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_493, mul_750);  view_493 = mul_750 = None
    sub_186: "f32[392, 256]" = torch.ops.aten.sub.Tensor(sub_185, unsqueeze_118);  sub_185 = unsqueeze_118 = None
    mul_751: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_186, unsqueeze_120);  sub_186 = unsqueeze_120 = None
    mul_752: "f32[256]" = torch.ops.aten.mul.Tensor(sum_78, squeeze_115);  sum_78 = squeeze_115 = None
    view_494: "f32[8, 49, 256]" = torch.ops.aten.view.default(mul_751, [8, 49, 256]);  mul_751 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_495: "f32[392, 256]" = torch.ops.aten.view.default(view_494, [392, 256]);  view_494 = None
    permute_257: "f32[256, 392]" = torch.ops.aten.permute.default(view_495, [1, 0])
    mm_108: "f32[256, 256]" = torch.ops.aten.mm.default(permute_257, view_212);  permute_257 = view_212 = None
    permute_258: "f32[256, 256]" = torch.ops.aten.permute.default(mm_108, [1, 0]);  mm_108 = None
    mm_109: "f32[392, 256]" = torch.ops.aten.mm.default(view_495, permute_259);  view_495 = permute_259 = None
    view_496: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_109, [8, 49, 256]);  mm_109 = None
    permute_260: "f32[256, 256]" = torch.ops.aten.permute.default(permute_258, [1, 0]);  permute_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    lt_11: "b8[8, 49, 256]" = torch.ops.aten.lt.Scalar(view_211, -3)
    le_11: "b8[8, 49, 256]" = torch.ops.aten.le.Scalar(view_211, 3)
    div_59: "f32[8, 49, 256]" = torch.ops.aten.div.Tensor(view_211, 3);  view_211 = None
    add_415: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(div_59, 0.5);  div_59 = None
    mul_753: "f32[8, 49, 256]" = torch.ops.aten.mul.Tensor(view_496, add_415);  add_415 = None
    where_22: "f32[8, 49, 256]" = torch.ops.aten.where.self(le_11, mul_753, view_496);  le_11 = mul_753 = view_496 = None
    where_23: "f32[8, 49, 256]" = torch.ops.aten.where.self(lt_11, full_default, where_22);  lt_11 = where_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    view_497: "f32[8, 49, 8, 32]" = torch.ops.aten.view.default(where_23, [8, 49, 8, 32]);  where_23 = None
    permute_261: "f32[8, 8, 49, 32]" = torch.ops.aten.permute.default(view_497, [0, 2, 1, 3]);  view_497 = None
    clone_89: "f32[8, 8, 49, 32]" = torch.ops.aten.clone.default(permute_261, memory_format = torch.contiguous_format);  permute_261 = None
    view_498: "f32[64, 49, 32]" = torch.ops.aten.view.default(clone_89, [64, 49, 32]);  clone_89 = None
    bmm_48: "f32[64, 49, 32]" = torch.ops.aten.bmm.default(permute_262, view_498);  permute_262 = None
    bmm_49: "f32[64, 49, 49]" = torch.ops.aten.bmm.default(view_498, permute_263);  view_498 = permute_263 = None
    view_499: "f32[8, 8, 49, 32]" = torch.ops.aten.view.default(bmm_48, [8, 8, 49, 32]);  bmm_48 = None
    view_500: "f32[8, 8, 49, 49]" = torch.ops.aten.view.default(bmm_49, [8, 8, 49, 49]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    mul_754: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(view_500, alias_19);  view_500 = None
    sum_79: "f32[8, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_754, [-1], True)
    mul_755: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(alias_19, sum_79);  alias_19 = sum_79 = None
    sub_187: "f32[8, 8, 49, 49]" = torch.ops.aten.sub.Tensor(mul_754, mul_755);  mul_754 = mul_755 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    sum_80: "f32[1, 8, 49, 49]" = torch.ops.aten.sum.dim_IntList(sub_187, [0], True)
    view_501: "f32[8, 49, 49]" = torch.ops.aten.view.default(sum_80, [8, 49, 49]);  sum_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    full_default_25: "f32[8, 49]" = torch.ops.aten.full.default([8, 49], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    index_put_5: "f32[8, 49]" = torch.ops.aten.index_put.default(full_default_25, [None, primals_217], view_501, True);  primals_217 = view_501 = None
    slice_scatter_8: "f32[8, 49]" = torch.ops.aten.slice_scatter.default(full_default_25, index_put_5, 0, 0, 9223372036854775807);  index_put_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    mul_756: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(sub_187, 0.25);  sub_187 = None
    view_502: "f32[64, 49, 49]" = torch.ops.aten.view.default(mul_756, [64, 49, 49]);  mul_756 = None
    bmm_50: "f32[64, 16, 49]" = torch.ops.aten.bmm.default(permute_264, view_502);  permute_264 = None
    bmm_51: "f32[64, 49, 16]" = torch.ops.aten.bmm.default(view_502, permute_265);  view_502 = permute_265 = None
    view_503: "f32[8, 8, 16, 49]" = torch.ops.aten.view.default(bmm_50, [8, 8, 16, 49]);  bmm_50 = None
    view_504: "f32[8, 8, 49, 16]" = torch.ops.aten.view.default(bmm_51, [8, 8, 49, 16]);  bmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_266: "f32[8, 49, 8, 32]" = torch.ops.aten.permute.default(view_499, [0, 2, 1, 3]);  view_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_267: "f32[8, 49, 8, 16]" = torch.ops.aten.permute.default(view_503, [0, 3, 1, 2]);  view_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_268: "f32[8, 49, 8, 16]" = torch.ops.aten.permute.default(view_504, [0, 2, 1, 3]);  view_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    cat_5: "f32[8, 49, 8, 64]" = torch.ops.aten.cat.default([permute_268, permute_267, permute_266], 3);  permute_268 = permute_267 = permute_266 = None
    view_505: "f32[8, 49, 512]" = torch.ops.aten.view.default(cat_5, [8, 49, 512]);  cat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_506: "f32[392, 512]" = torch.ops.aten.view.default(view_505, [392, 512]);  view_505 = None
    sum_81: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_506, [0])
    sub_188: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_202, unsqueeze_121);  view_202 = unsqueeze_121 = None
    mul_757: "f32[392, 512]" = torch.ops.aten.mul.Tensor(view_506, sub_188)
    sum_82: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_757, [0]);  mul_757 = None
    mul_758: "f32[512]" = torch.ops.aten.mul.Tensor(sum_81, 0.002551020408163265)
    unsqueeze_122: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_758, 0);  mul_758 = None
    mul_759: "f32[512]" = torch.ops.aten.mul.Tensor(sum_82, 0.002551020408163265)
    mul_760: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_761: "f32[512]" = torch.ops.aten.mul.Tensor(mul_759, mul_760);  mul_759 = mul_760 = None
    unsqueeze_123: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_761, 0);  mul_761 = None
    mul_762: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_127);  primals_127 = None
    unsqueeze_124: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_762, 0);  mul_762 = None
    mul_763: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_123);  sub_188 = unsqueeze_123 = None
    sub_190: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_506, mul_763);  view_506 = mul_763 = None
    sub_191: "f32[392, 512]" = torch.ops.aten.sub.Tensor(sub_190, unsqueeze_122);  sub_190 = unsqueeze_122 = None
    mul_764: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_191, unsqueeze_124);  sub_191 = unsqueeze_124 = None
    mul_765: "f32[512]" = torch.ops.aten.mul.Tensor(sum_82, squeeze_112);  sum_82 = squeeze_112 = None
    view_507: "f32[8, 49, 512]" = torch.ops.aten.view.default(mul_764, [8, 49, 512]);  mul_764 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_508: "f32[392, 512]" = torch.ops.aten.view.default(view_507, [392, 512]);  view_507 = None
    permute_269: "f32[512, 392]" = torch.ops.aten.permute.default(view_508, [1, 0])
    mm_110: "f32[512, 256]" = torch.ops.aten.mm.default(permute_269, view_200);  permute_269 = view_200 = None
    permute_270: "f32[256, 512]" = torch.ops.aten.permute.default(mm_110, [1, 0]);  mm_110 = None
    mm_111: "f32[392, 256]" = torch.ops.aten.mm.default(view_508, permute_271);  view_508 = permute_271 = None
    view_509: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_111, [8, 49, 256]);  mm_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_416: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_414, view_509);  add_414 = view_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_272: "f32[512, 256]" = torch.ops.aten.permute.default(permute_270, [1, 0]);  permute_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_510: "f32[392, 256]" = torch.ops.aten.view.default(add_416, [392, 256])
    sum_83: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_510, [0])
    sub_192: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_198, unsqueeze_125);  view_198 = unsqueeze_125 = None
    mul_766: "f32[392, 256]" = torch.ops.aten.mul.Tensor(view_510, sub_192)
    sum_84: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_766, [0]);  mul_766 = None
    mul_767: "f32[256]" = torch.ops.aten.mul.Tensor(sum_83, 0.002551020408163265)
    unsqueeze_126: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_767, 0);  mul_767 = None
    mul_768: "f32[256]" = torch.ops.aten.mul.Tensor(sum_84, 0.002551020408163265)
    mul_769: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_770: "f32[256]" = torch.ops.aten.mul.Tensor(mul_768, mul_769);  mul_768 = mul_769 = None
    unsqueeze_127: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_770, 0);  mul_770 = None
    mul_771: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_124);  primals_124 = None
    unsqueeze_128: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_771, 0);  mul_771 = None
    mul_772: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_127);  sub_192 = unsqueeze_127 = None
    sub_194: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_510, mul_772);  view_510 = mul_772 = None
    sub_195: "f32[392, 256]" = torch.ops.aten.sub.Tensor(sub_194, unsqueeze_126);  sub_194 = unsqueeze_126 = None
    mul_773: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_195, unsqueeze_128);  sub_195 = unsqueeze_128 = None
    mul_774: "f32[256]" = torch.ops.aten.mul.Tensor(sum_84, squeeze_109);  sum_84 = squeeze_109 = None
    view_511: "f32[8, 49, 256]" = torch.ops.aten.view.default(mul_773, [8, 49, 256]);  mul_773 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_512: "f32[392, 256]" = torch.ops.aten.view.default(view_511, [392, 256]);  view_511 = None
    permute_273: "f32[256, 392]" = torch.ops.aten.permute.default(view_512, [1, 0])
    mm_112: "f32[256, 512]" = torch.ops.aten.mm.default(permute_273, view_196);  permute_273 = view_196 = None
    permute_274: "f32[512, 256]" = torch.ops.aten.permute.default(mm_112, [1, 0]);  mm_112 = None
    mm_113: "f32[392, 512]" = torch.ops.aten.mm.default(view_512, permute_275);  view_512 = permute_275 = None
    view_513: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_113, [8, 49, 512]);  mm_113 = None
    permute_276: "f32[256, 512]" = torch.ops.aten.permute.default(permute_274, [1, 0]);  permute_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    lt_12: "b8[8, 49, 512]" = torch.ops.aten.lt.Scalar(view_195, -3)
    le_12: "b8[8, 49, 512]" = torch.ops.aten.le.Scalar(view_195, 3)
    div_60: "f32[8, 49, 512]" = torch.ops.aten.div.Tensor(view_195, 3);  view_195 = None
    add_417: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(div_60, 0.5);  div_60 = None
    mul_775: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_513, add_417);  add_417 = None
    where_24: "f32[8, 49, 512]" = torch.ops.aten.where.self(le_12, mul_775, view_513);  le_12 = mul_775 = view_513 = None
    where_25: "f32[8, 49, 512]" = torch.ops.aten.where.self(lt_12, full_default, where_24);  lt_12 = where_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_514: "f32[392, 512]" = torch.ops.aten.view.default(where_25, [392, 512]);  where_25 = None
    sum_85: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_514, [0])
    sub_196: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_194, unsqueeze_129);  view_194 = unsqueeze_129 = None
    mul_776: "f32[392, 512]" = torch.ops.aten.mul.Tensor(view_514, sub_196)
    sum_86: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_776, [0]);  mul_776 = None
    mul_777: "f32[512]" = torch.ops.aten.mul.Tensor(sum_85, 0.002551020408163265)
    unsqueeze_130: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_777, 0);  mul_777 = None
    mul_778: "f32[512]" = torch.ops.aten.mul.Tensor(sum_86, 0.002551020408163265)
    mul_779: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_780: "f32[512]" = torch.ops.aten.mul.Tensor(mul_778, mul_779);  mul_778 = mul_779 = None
    unsqueeze_131: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_780, 0);  mul_780 = None
    mul_781: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_121);  primals_121 = None
    unsqueeze_132: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_781, 0);  mul_781 = None
    mul_782: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_131);  sub_196 = unsqueeze_131 = None
    sub_198: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_514, mul_782);  view_514 = mul_782 = None
    sub_199: "f32[392, 512]" = torch.ops.aten.sub.Tensor(sub_198, unsqueeze_130);  sub_198 = unsqueeze_130 = None
    mul_783: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_199, unsqueeze_132);  sub_199 = unsqueeze_132 = None
    mul_784: "f32[512]" = torch.ops.aten.mul.Tensor(sum_86, squeeze_106);  sum_86 = squeeze_106 = None
    view_515: "f32[8, 49, 512]" = torch.ops.aten.view.default(mul_783, [8, 49, 512]);  mul_783 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_516: "f32[392, 512]" = torch.ops.aten.view.default(view_515, [392, 512]);  view_515 = None
    permute_277: "f32[512, 392]" = torch.ops.aten.permute.default(view_516, [1, 0])
    mm_114: "f32[512, 256]" = torch.ops.aten.mm.default(permute_277, view_192);  permute_277 = view_192 = None
    permute_278: "f32[256, 512]" = torch.ops.aten.permute.default(mm_114, [1, 0]);  mm_114 = None
    mm_115: "f32[392, 256]" = torch.ops.aten.mm.default(view_516, permute_279);  view_516 = permute_279 = None
    view_517: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_115, [8, 49, 256]);  mm_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_418: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_416, view_517);  add_416 = view_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_280: "f32[512, 256]" = torch.ops.aten.permute.default(permute_278, [1, 0]);  permute_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_518: "f32[392, 256]" = torch.ops.aten.view.default(add_418, [392, 256])
    sum_87: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_518, [0])
    sub_200: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_190, unsqueeze_133);  view_190 = unsqueeze_133 = None
    mul_785: "f32[392, 256]" = torch.ops.aten.mul.Tensor(view_518, sub_200)
    sum_88: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_785, [0]);  mul_785 = None
    mul_786: "f32[256]" = torch.ops.aten.mul.Tensor(sum_87, 0.002551020408163265)
    unsqueeze_134: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_786, 0);  mul_786 = None
    mul_787: "f32[256]" = torch.ops.aten.mul.Tensor(sum_88, 0.002551020408163265)
    mul_788: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_789: "f32[256]" = torch.ops.aten.mul.Tensor(mul_787, mul_788);  mul_787 = mul_788 = None
    unsqueeze_135: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_789, 0);  mul_789 = None
    mul_790: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_118);  primals_118 = None
    unsqueeze_136: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_790, 0);  mul_790 = None
    mul_791: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_135);  sub_200 = unsqueeze_135 = None
    sub_202: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_518, mul_791);  view_518 = mul_791 = None
    sub_203: "f32[392, 256]" = torch.ops.aten.sub.Tensor(sub_202, unsqueeze_134);  sub_202 = unsqueeze_134 = None
    mul_792: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_136);  sub_203 = unsqueeze_136 = None
    mul_793: "f32[256]" = torch.ops.aten.mul.Tensor(sum_88, squeeze_103);  sum_88 = squeeze_103 = None
    view_519: "f32[8, 49, 256]" = torch.ops.aten.view.default(mul_792, [8, 49, 256]);  mul_792 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_520: "f32[392, 256]" = torch.ops.aten.view.default(view_519, [392, 256]);  view_519 = None
    permute_281: "f32[256, 392]" = torch.ops.aten.permute.default(view_520, [1, 0])
    mm_116: "f32[256, 256]" = torch.ops.aten.mm.default(permute_281, view_188);  permute_281 = view_188 = None
    permute_282: "f32[256, 256]" = torch.ops.aten.permute.default(mm_116, [1, 0]);  mm_116 = None
    mm_117: "f32[392, 256]" = torch.ops.aten.mm.default(view_520, permute_283);  view_520 = permute_283 = None
    view_521: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_117, [8, 49, 256]);  mm_117 = None
    permute_284: "f32[256, 256]" = torch.ops.aten.permute.default(permute_282, [1, 0]);  permute_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    lt_13: "b8[8, 49, 256]" = torch.ops.aten.lt.Scalar(view_187, -3)
    le_13: "b8[8, 49, 256]" = torch.ops.aten.le.Scalar(view_187, 3)
    div_61: "f32[8, 49, 256]" = torch.ops.aten.div.Tensor(view_187, 3);  view_187 = None
    add_419: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(div_61, 0.5);  div_61 = None
    mul_794: "f32[8, 49, 256]" = torch.ops.aten.mul.Tensor(view_521, add_419);  add_419 = None
    where_26: "f32[8, 49, 256]" = torch.ops.aten.where.self(le_13, mul_794, view_521);  le_13 = mul_794 = view_521 = None
    where_27: "f32[8, 49, 256]" = torch.ops.aten.where.self(lt_13, full_default, where_26);  lt_13 = where_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    view_522: "f32[8, 49, 8, 32]" = torch.ops.aten.view.default(where_27, [8, 49, 8, 32]);  where_27 = None
    permute_285: "f32[8, 8, 49, 32]" = torch.ops.aten.permute.default(view_522, [0, 2, 1, 3]);  view_522 = None
    clone_90: "f32[8, 8, 49, 32]" = torch.ops.aten.clone.default(permute_285, memory_format = torch.contiguous_format);  permute_285 = None
    view_523: "f32[64, 49, 32]" = torch.ops.aten.view.default(clone_90, [64, 49, 32]);  clone_90 = None
    bmm_52: "f32[64, 49, 32]" = torch.ops.aten.bmm.default(permute_286, view_523);  permute_286 = None
    bmm_53: "f32[64, 49, 49]" = torch.ops.aten.bmm.default(view_523, permute_287);  view_523 = permute_287 = None
    view_524: "f32[8, 8, 49, 32]" = torch.ops.aten.view.default(bmm_52, [8, 8, 49, 32]);  bmm_52 = None
    view_525: "f32[8, 8, 49, 49]" = torch.ops.aten.view.default(bmm_53, [8, 8, 49, 49]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    mul_795: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(view_525, alias_20);  view_525 = None
    sum_89: "f32[8, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_795, [-1], True)
    mul_796: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(alias_20, sum_89);  alias_20 = sum_89 = None
    sub_204: "f32[8, 8, 49, 49]" = torch.ops.aten.sub.Tensor(mul_795, mul_796);  mul_795 = mul_796 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    sum_90: "f32[1, 8, 49, 49]" = torch.ops.aten.sum.dim_IntList(sub_204, [0], True)
    view_526: "f32[8, 49, 49]" = torch.ops.aten.view.default(sum_90, [8, 49, 49]);  sum_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    index_put_6: "f32[8, 49]" = torch.ops.aten.index_put.default(full_default_25, [None, primals_216], view_526, True);  primals_216 = view_526 = None
    slice_scatter_9: "f32[8, 49]" = torch.ops.aten.slice_scatter.default(full_default_25, index_put_6, 0, 0, 9223372036854775807);  index_put_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    mul_797: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(sub_204, 0.25);  sub_204 = None
    view_527: "f32[64, 49, 49]" = torch.ops.aten.view.default(mul_797, [64, 49, 49]);  mul_797 = None
    bmm_54: "f32[64, 16, 49]" = torch.ops.aten.bmm.default(permute_288, view_527);  permute_288 = None
    bmm_55: "f32[64, 49, 16]" = torch.ops.aten.bmm.default(view_527, permute_289);  view_527 = permute_289 = None
    view_528: "f32[8, 8, 16, 49]" = torch.ops.aten.view.default(bmm_54, [8, 8, 16, 49]);  bmm_54 = None
    view_529: "f32[8, 8, 49, 16]" = torch.ops.aten.view.default(bmm_55, [8, 8, 49, 16]);  bmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_290: "f32[8, 49, 8, 32]" = torch.ops.aten.permute.default(view_524, [0, 2, 1, 3]);  view_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_291: "f32[8, 49, 8, 16]" = torch.ops.aten.permute.default(view_528, [0, 3, 1, 2]);  view_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_292: "f32[8, 49, 8, 16]" = torch.ops.aten.permute.default(view_529, [0, 2, 1, 3]);  view_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    cat_6: "f32[8, 49, 8, 64]" = torch.ops.aten.cat.default([permute_292, permute_291, permute_290], 3);  permute_292 = permute_291 = permute_290 = None
    view_530: "f32[8, 49, 512]" = torch.ops.aten.view.default(cat_6, [8, 49, 512]);  cat_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_531: "f32[392, 512]" = torch.ops.aten.view.default(view_530, [392, 512]);  view_530 = None
    sum_91: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_531, [0])
    sub_205: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_178, unsqueeze_137);  view_178 = unsqueeze_137 = None
    mul_798: "f32[392, 512]" = torch.ops.aten.mul.Tensor(view_531, sub_205)
    sum_92: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_798, [0]);  mul_798 = None
    mul_799: "f32[512]" = torch.ops.aten.mul.Tensor(sum_91, 0.002551020408163265)
    unsqueeze_138: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_799, 0);  mul_799 = None
    mul_800: "f32[512]" = torch.ops.aten.mul.Tensor(sum_92, 0.002551020408163265)
    mul_801: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_802: "f32[512]" = torch.ops.aten.mul.Tensor(mul_800, mul_801);  mul_800 = mul_801 = None
    unsqueeze_139: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_802, 0);  mul_802 = None
    mul_803: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_115);  primals_115 = None
    unsqueeze_140: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_803, 0);  mul_803 = None
    mul_804: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_139);  sub_205 = unsqueeze_139 = None
    sub_207: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_531, mul_804);  view_531 = mul_804 = None
    sub_208: "f32[392, 512]" = torch.ops.aten.sub.Tensor(sub_207, unsqueeze_138);  sub_207 = unsqueeze_138 = None
    mul_805: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_140);  sub_208 = unsqueeze_140 = None
    mul_806: "f32[512]" = torch.ops.aten.mul.Tensor(sum_92, squeeze_100);  sum_92 = squeeze_100 = None
    view_532: "f32[8, 49, 512]" = torch.ops.aten.view.default(mul_805, [8, 49, 512]);  mul_805 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_533: "f32[392, 512]" = torch.ops.aten.view.default(view_532, [392, 512]);  view_532 = None
    permute_293: "f32[512, 392]" = torch.ops.aten.permute.default(view_533, [1, 0])
    mm_118: "f32[512, 256]" = torch.ops.aten.mm.default(permute_293, view_176);  permute_293 = view_176 = None
    permute_294: "f32[256, 512]" = torch.ops.aten.permute.default(mm_118, [1, 0]);  mm_118 = None
    mm_119: "f32[392, 256]" = torch.ops.aten.mm.default(view_533, permute_295);  view_533 = permute_295 = None
    view_534: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_119, [8, 49, 256]);  mm_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_420: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_418, view_534);  add_418 = view_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_296: "f32[512, 256]" = torch.ops.aten.permute.default(permute_294, [1, 0]);  permute_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_535: "f32[392, 256]" = torch.ops.aten.view.default(add_420, [392, 256])
    sum_93: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_535, [0])
    sub_209: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_174, unsqueeze_141);  view_174 = unsqueeze_141 = None
    mul_807: "f32[392, 256]" = torch.ops.aten.mul.Tensor(view_535, sub_209)
    sum_94: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_807, [0]);  mul_807 = None
    mul_808: "f32[256]" = torch.ops.aten.mul.Tensor(sum_93, 0.002551020408163265)
    unsqueeze_142: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_808, 0);  mul_808 = None
    mul_809: "f32[256]" = torch.ops.aten.mul.Tensor(sum_94, 0.002551020408163265)
    mul_810: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_811: "f32[256]" = torch.ops.aten.mul.Tensor(mul_809, mul_810);  mul_809 = mul_810 = None
    unsqueeze_143: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_811, 0);  mul_811 = None
    mul_812: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_112);  primals_112 = None
    unsqueeze_144: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_812, 0);  mul_812 = None
    mul_813: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_143);  sub_209 = unsqueeze_143 = None
    sub_211: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_535, mul_813);  view_535 = mul_813 = None
    sub_212: "f32[392, 256]" = torch.ops.aten.sub.Tensor(sub_211, unsqueeze_142);  sub_211 = unsqueeze_142 = None
    mul_814: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_144);  sub_212 = unsqueeze_144 = None
    mul_815: "f32[256]" = torch.ops.aten.mul.Tensor(sum_94, squeeze_97);  sum_94 = squeeze_97 = None
    view_536: "f32[8, 49, 256]" = torch.ops.aten.view.default(mul_814, [8, 49, 256]);  mul_814 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_537: "f32[392, 256]" = torch.ops.aten.view.default(view_536, [392, 256]);  view_536 = None
    permute_297: "f32[256, 392]" = torch.ops.aten.permute.default(view_537, [1, 0])
    mm_120: "f32[256, 512]" = torch.ops.aten.mm.default(permute_297, view_172);  permute_297 = view_172 = None
    permute_298: "f32[512, 256]" = torch.ops.aten.permute.default(mm_120, [1, 0]);  mm_120 = None
    mm_121: "f32[392, 512]" = torch.ops.aten.mm.default(view_537, permute_299);  view_537 = permute_299 = None
    view_538: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_121, [8, 49, 512]);  mm_121 = None
    permute_300: "f32[256, 512]" = torch.ops.aten.permute.default(permute_298, [1, 0]);  permute_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    lt_14: "b8[8, 49, 512]" = torch.ops.aten.lt.Scalar(view_171, -3)
    le_14: "b8[8, 49, 512]" = torch.ops.aten.le.Scalar(view_171, 3)
    div_62: "f32[8, 49, 512]" = torch.ops.aten.div.Tensor(view_171, 3);  view_171 = None
    add_421: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(div_62, 0.5);  div_62 = None
    mul_816: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_538, add_421);  add_421 = None
    where_28: "f32[8, 49, 512]" = torch.ops.aten.where.self(le_14, mul_816, view_538);  le_14 = mul_816 = view_538 = None
    where_29: "f32[8, 49, 512]" = torch.ops.aten.where.self(lt_14, full_default, where_28);  lt_14 = where_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_539: "f32[392, 512]" = torch.ops.aten.view.default(where_29, [392, 512]);  where_29 = None
    sum_95: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_539, [0])
    sub_213: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_170, unsqueeze_145);  view_170 = unsqueeze_145 = None
    mul_817: "f32[392, 512]" = torch.ops.aten.mul.Tensor(view_539, sub_213)
    sum_96: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_817, [0]);  mul_817 = None
    mul_818: "f32[512]" = torch.ops.aten.mul.Tensor(sum_95, 0.002551020408163265)
    unsqueeze_146: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_818, 0);  mul_818 = None
    mul_819: "f32[512]" = torch.ops.aten.mul.Tensor(sum_96, 0.002551020408163265)
    mul_820: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_821: "f32[512]" = torch.ops.aten.mul.Tensor(mul_819, mul_820);  mul_819 = mul_820 = None
    unsqueeze_147: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_821, 0);  mul_821 = None
    mul_822: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_109);  primals_109 = None
    unsqueeze_148: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_822, 0);  mul_822 = None
    mul_823: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_147);  sub_213 = unsqueeze_147 = None
    sub_215: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_539, mul_823);  view_539 = mul_823 = None
    sub_216: "f32[392, 512]" = torch.ops.aten.sub.Tensor(sub_215, unsqueeze_146);  sub_215 = unsqueeze_146 = None
    mul_824: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_148);  sub_216 = unsqueeze_148 = None
    mul_825: "f32[512]" = torch.ops.aten.mul.Tensor(sum_96, squeeze_94);  sum_96 = squeeze_94 = None
    view_540: "f32[8, 49, 512]" = torch.ops.aten.view.default(mul_824, [8, 49, 512]);  mul_824 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_541: "f32[392, 512]" = torch.ops.aten.view.default(view_540, [392, 512]);  view_540 = None
    permute_301: "f32[512, 392]" = torch.ops.aten.permute.default(view_541, [1, 0])
    mm_122: "f32[512, 256]" = torch.ops.aten.mm.default(permute_301, view_168);  permute_301 = view_168 = None
    permute_302: "f32[256, 512]" = torch.ops.aten.permute.default(mm_122, [1, 0]);  mm_122 = None
    mm_123: "f32[392, 256]" = torch.ops.aten.mm.default(view_541, permute_303);  view_541 = permute_303 = None
    view_542: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_123, [8, 49, 256]);  mm_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_422: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_420, view_542);  add_420 = view_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_304: "f32[512, 256]" = torch.ops.aten.permute.default(permute_302, [1, 0]);  permute_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_543: "f32[392, 256]" = torch.ops.aten.view.default(add_422, [392, 256])
    sum_97: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_543, [0])
    sub_217: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_166, unsqueeze_149);  view_166 = unsqueeze_149 = None
    mul_826: "f32[392, 256]" = torch.ops.aten.mul.Tensor(view_543, sub_217)
    sum_98: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_826, [0]);  mul_826 = None
    mul_827: "f32[256]" = torch.ops.aten.mul.Tensor(sum_97, 0.002551020408163265)
    unsqueeze_150: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_827, 0);  mul_827 = None
    mul_828: "f32[256]" = torch.ops.aten.mul.Tensor(sum_98, 0.002551020408163265)
    mul_829: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_830: "f32[256]" = torch.ops.aten.mul.Tensor(mul_828, mul_829);  mul_828 = mul_829 = None
    unsqueeze_151: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_830, 0);  mul_830 = None
    mul_831: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_106);  primals_106 = None
    unsqueeze_152: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_831, 0);  mul_831 = None
    mul_832: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_151);  sub_217 = unsqueeze_151 = None
    sub_219: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_543, mul_832);  view_543 = mul_832 = None
    sub_220: "f32[392, 256]" = torch.ops.aten.sub.Tensor(sub_219, unsqueeze_150);  sub_219 = unsqueeze_150 = None
    mul_833: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_152);  sub_220 = unsqueeze_152 = None
    mul_834: "f32[256]" = torch.ops.aten.mul.Tensor(sum_98, squeeze_91);  sum_98 = squeeze_91 = None
    view_544: "f32[8, 49, 256]" = torch.ops.aten.view.default(mul_833, [8, 49, 256]);  mul_833 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_545: "f32[392, 256]" = torch.ops.aten.view.default(view_544, [392, 256]);  view_544 = None
    permute_305: "f32[256, 392]" = torch.ops.aten.permute.default(view_545, [1, 0])
    mm_124: "f32[256, 256]" = torch.ops.aten.mm.default(permute_305, view_164);  permute_305 = view_164 = None
    permute_306: "f32[256, 256]" = torch.ops.aten.permute.default(mm_124, [1, 0]);  mm_124 = None
    mm_125: "f32[392, 256]" = torch.ops.aten.mm.default(view_545, permute_307);  view_545 = permute_307 = None
    view_546: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_125, [8, 49, 256]);  mm_125 = None
    permute_308: "f32[256, 256]" = torch.ops.aten.permute.default(permute_306, [1, 0]);  permute_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    lt_15: "b8[8, 49, 256]" = torch.ops.aten.lt.Scalar(view_163, -3)
    le_15: "b8[8, 49, 256]" = torch.ops.aten.le.Scalar(view_163, 3)
    div_63: "f32[8, 49, 256]" = torch.ops.aten.div.Tensor(view_163, 3);  view_163 = None
    add_423: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(div_63, 0.5);  div_63 = None
    mul_835: "f32[8, 49, 256]" = torch.ops.aten.mul.Tensor(view_546, add_423);  add_423 = None
    where_30: "f32[8, 49, 256]" = torch.ops.aten.where.self(le_15, mul_835, view_546);  le_15 = mul_835 = view_546 = None
    where_31: "f32[8, 49, 256]" = torch.ops.aten.where.self(lt_15, full_default, where_30);  lt_15 = where_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    view_547: "f32[8, 49, 8, 32]" = torch.ops.aten.view.default(where_31, [8, 49, 8, 32]);  where_31 = None
    permute_309: "f32[8, 8, 49, 32]" = torch.ops.aten.permute.default(view_547, [0, 2, 1, 3]);  view_547 = None
    clone_91: "f32[8, 8, 49, 32]" = torch.ops.aten.clone.default(permute_309, memory_format = torch.contiguous_format);  permute_309 = None
    view_548: "f32[64, 49, 32]" = torch.ops.aten.view.default(clone_91, [64, 49, 32]);  clone_91 = None
    bmm_56: "f32[64, 49, 32]" = torch.ops.aten.bmm.default(permute_310, view_548);  permute_310 = None
    bmm_57: "f32[64, 49, 49]" = torch.ops.aten.bmm.default(view_548, permute_311);  view_548 = permute_311 = None
    view_549: "f32[8, 8, 49, 32]" = torch.ops.aten.view.default(bmm_56, [8, 8, 49, 32]);  bmm_56 = None
    view_550: "f32[8, 8, 49, 49]" = torch.ops.aten.view.default(bmm_57, [8, 8, 49, 49]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    mul_836: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(view_550, alias_21);  view_550 = None
    sum_99: "f32[8, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_836, [-1], True)
    mul_837: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(alias_21, sum_99);  alias_21 = sum_99 = None
    sub_221: "f32[8, 8, 49, 49]" = torch.ops.aten.sub.Tensor(mul_836, mul_837);  mul_836 = mul_837 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    sum_100: "f32[1, 8, 49, 49]" = torch.ops.aten.sum.dim_IntList(sub_221, [0], True)
    view_551: "f32[8, 49, 49]" = torch.ops.aten.view.default(sum_100, [8, 49, 49]);  sum_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    index_put_7: "f32[8, 49]" = torch.ops.aten.index_put.default(full_default_25, [None, primals_215], view_551, True);  primals_215 = view_551 = None
    slice_scatter_10: "f32[8, 49]" = torch.ops.aten.slice_scatter.default(full_default_25, index_put_7, 0, 0, 9223372036854775807);  index_put_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    mul_838: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(sub_221, 0.25);  sub_221 = None
    view_552: "f32[64, 49, 49]" = torch.ops.aten.view.default(mul_838, [64, 49, 49]);  mul_838 = None
    bmm_58: "f32[64, 16, 49]" = torch.ops.aten.bmm.default(permute_312, view_552);  permute_312 = None
    bmm_59: "f32[64, 49, 16]" = torch.ops.aten.bmm.default(view_552, permute_313);  view_552 = permute_313 = None
    view_553: "f32[8, 8, 16, 49]" = torch.ops.aten.view.default(bmm_58, [8, 8, 16, 49]);  bmm_58 = None
    view_554: "f32[8, 8, 49, 16]" = torch.ops.aten.view.default(bmm_59, [8, 8, 49, 16]);  bmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_314: "f32[8, 49, 8, 32]" = torch.ops.aten.permute.default(view_549, [0, 2, 1, 3]);  view_549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_315: "f32[8, 49, 8, 16]" = torch.ops.aten.permute.default(view_553, [0, 3, 1, 2]);  view_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_316: "f32[8, 49, 8, 16]" = torch.ops.aten.permute.default(view_554, [0, 2, 1, 3]);  view_554 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    cat_7: "f32[8, 49, 8, 64]" = torch.ops.aten.cat.default([permute_316, permute_315, permute_314], 3);  permute_316 = permute_315 = permute_314 = None
    view_555: "f32[8, 49, 512]" = torch.ops.aten.view.default(cat_7, [8, 49, 512]);  cat_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_556: "f32[392, 512]" = torch.ops.aten.view.default(view_555, [392, 512]);  view_555 = None
    sum_101: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_556, [0])
    sub_222: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_154, unsqueeze_153);  view_154 = unsqueeze_153 = None
    mul_839: "f32[392, 512]" = torch.ops.aten.mul.Tensor(view_556, sub_222)
    sum_102: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_839, [0]);  mul_839 = None
    mul_840: "f32[512]" = torch.ops.aten.mul.Tensor(sum_101, 0.002551020408163265)
    unsqueeze_154: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_840, 0);  mul_840 = None
    mul_841: "f32[512]" = torch.ops.aten.mul.Tensor(sum_102, 0.002551020408163265)
    mul_842: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_843: "f32[512]" = torch.ops.aten.mul.Tensor(mul_841, mul_842);  mul_841 = mul_842 = None
    unsqueeze_155: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_843, 0);  mul_843 = None
    mul_844: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_103);  primals_103 = None
    unsqueeze_156: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_844, 0);  mul_844 = None
    mul_845: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_222, unsqueeze_155);  sub_222 = unsqueeze_155 = None
    sub_224: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_556, mul_845);  view_556 = mul_845 = None
    sub_225: "f32[392, 512]" = torch.ops.aten.sub.Tensor(sub_224, unsqueeze_154);  sub_224 = unsqueeze_154 = None
    mul_846: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_156);  sub_225 = unsqueeze_156 = None
    mul_847: "f32[512]" = torch.ops.aten.mul.Tensor(sum_102, squeeze_88);  sum_102 = squeeze_88 = None
    view_557: "f32[8, 49, 512]" = torch.ops.aten.view.default(mul_846, [8, 49, 512]);  mul_846 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_558: "f32[392, 512]" = torch.ops.aten.view.default(view_557, [392, 512]);  view_557 = None
    permute_317: "f32[512, 392]" = torch.ops.aten.permute.default(view_558, [1, 0])
    mm_126: "f32[512, 256]" = torch.ops.aten.mm.default(permute_317, view_152);  permute_317 = view_152 = None
    permute_318: "f32[256, 512]" = torch.ops.aten.permute.default(mm_126, [1, 0]);  mm_126 = None
    mm_127: "f32[392, 256]" = torch.ops.aten.mm.default(view_558, permute_319);  view_558 = permute_319 = None
    view_559: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_127, [8, 49, 256]);  mm_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_424: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_422, view_559);  add_422 = view_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_320: "f32[512, 256]" = torch.ops.aten.permute.default(permute_318, [1, 0]);  permute_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_560: "f32[392, 256]" = torch.ops.aten.view.default(add_424, [392, 256])
    sum_103: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_560, [0])
    sub_226: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_150, unsqueeze_157);  view_150 = unsqueeze_157 = None
    mul_848: "f32[392, 256]" = torch.ops.aten.mul.Tensor(view_560, sub_226)
    sum_104: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_848, [0]);  mul_848 = None
    mul_849: "f32[256]" = torch.ops.aten.mul.Tensor(sum_103, 0.002551020408163265)
    unsqueeze_158: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_849, 0);  mul_849 = None
    mul_850: "f32[256]" = torch.ops.aten.mul.Tensor(sum_104, 0.002551020408163265)
    mul_851: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_852: "f32[256]" = torch.ops.aten.mul.Tensor(mul_850, mul_851);  mul_850 = mul_851 = None
    unsqueeze_159: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_852, 0);  mul_852 = None
    mul_853: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_100);  primals_100 = None
    unsqueeze_160: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_853, 0);  mul_853 = None
    mul_854: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_226, unsqueeze_159);  sub_226 = unsqueeze_159 = None
    sub_228: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_560, mul_854);  view_560 = mul_854 = None
    sub_229: "f32[392, 256]" = torch.ops.aten.sub.Tensor(sub_228, unsqueeze_158);  sub_228 = unsqueeze_158 = None
    mul_855: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_160);  sub_229 = unsqueeze_160 = None
    mul_856: "f32[256]" = torch.ops.aten.mul.Tensor(sum_104, squeeze_85);  sum_104 = squeeze_85 = None
    view_561: "f32[8, 49, 256]" = torch.ops.aten.view.default(mul_855, [8, 49, 256]);  mul_855 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_562: "f32[392, 256]" = torch.ops.aten.view.default(view_561, [392, 256]);  view_561 = None
    permute_321: "f32[256, 392]" = torch.ops.aten.permute.default(view_562, [1, 0])
    mm_128: "f32[256, 512]" = torch.ops.aten.mm.default(permute_321, view_148);  permute_321 = view_148 = None
    permute_322: "f32[512, 256]" = torch.ops.aten.permute.default(mm_128, [1, 0]);  mm_128 = None
    mm_129: "f32[392, 512]" = torch.ops.aten.mm.default(view_562, permute_323);  view_562 = permute_323 = None
    view_563: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_129, [8, 49, 512]);  mm_129 = None
    permute_324: "f32[256, 512]" = torch.ops.aten.permute.default(permute_322, [1, 0]);  permute_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    lt_16: "b8[8, 49, 512]" = torch.ops.aten.lt.Scalar(view_147, -3)
    le_16: "b8[8, 49, 512]" = torch.ops.aten.le.Scalar(view_147, 3)
    div_64: "f32[8, 49, 512]" = torch.ops.aten.div.Tensor(view_147, 3);  view_147 = None
    add_425: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(div_64, 0.5);  div_64 = None
    mul_857: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_563, add_425);  add_425 = None
    where_32: "f32[8, 49, 512]" = torch.ops.aten.where.self(le_16, mul_857, view_563);  le_16 = mul_857 = view_563 = None
    where_33: "f32[8, 49, 512]" = torch.ops.aten.where.self(lt_16, full_default, where_32);  lt_16 = where_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_564: "f32[392, 512]" = torch.ops.aten.view.default(where_33, [392, 512]);  where_33 = None
    sum_105: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_564, [0])
    sub_230: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_146, unsqueeze_161);  view_146 = unsqueeze_161 = None
    mul_858: "f32[392, 512]" = torch.ops.aten.mul.Tensor(view_564, sub_230)
    sum_106: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_858, [0]);  mul_858 = None
    mul_859: "f32[512]" = torch.ops.aten.mul.Tensor(sum_105, 0.002551020408163265)
    unsqueeze_162: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_859, 0);  mul_859 = None
    mul_860: "f32[512]" = torch.ops.aten.mul.Tensor(sum_106, 0.002551020408163265)
    mul_861: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_862: "f32[512]" = torch.ops.aten.mul.Tensor(mul_860, mul_861);  mul_860 = mul_861 = None
    unsqueeze_163: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_862, 0);  mul_862 = None
    mul_863: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_97);  primals_97 = None
    unsqueeze_164: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_863, 0);  mul_863 = None
    mul_864: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_230, unsqueeze_163);  sub_230 = unsqueeze_163 = None
    sub_232: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_564, mul_864);  view_564 = mul_864 = None
    sub_233: "f32[392, 512]" = torch.ops.aten.sub.Tensor(sub_232, unsqueeze_162);  sub_232 = unsqueeze_162 = None
    mul_865: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_233, unsqueeze_164);  sub_233 = unsqueeze_164 = None
    mul_866: "f32[512]" = torch.ops.aten.mul.Tensor(sum_106, squeeze_82);  sum_106 = squeeze_82 = None
    view_565: "f32[8, 49, 512]" = torch.ops.aten.view.default(mul_865, [8, 49, 512]);  mul_865 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_566: "f32[392, 512]" = torch.ops.aten.view.default(view_565, [392, 512]);  view_565 = None
    permute_325: "f32[512, 392]" = torch.ops.aten.permute.default(view_566, [1, 0])
    mm_130: "f32[512, 256]" = torch.ops.aten.mm.default(permute_325, view_144);  permute_325 = view_144 = None
    permute_326: "f32[256, 512]" = torch.ops.aten.permute.default(mm_130, [1, 0]);  mm_130 = None
    mm_131: "f32[392, 256]" = torch.ops.aten.mm.default(view_566, permute_327);  view_566 = permute_327 = None
    view_567: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_131, [8, 49, 256]);  mm_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_426: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_424, view_567);  add_424 = view_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_328: "f32[512, 256]" = torch.ops.aten.permute.default(permute_326, [1, 0]);  permute_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_568: "f32[392, 256]" = torch.ops.aten.view.default(add_426, [392, 256])
    sum_107: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_568, [0])
    sub_234: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_142, unsqueeze_165);  view_142 = unsqueeze_165 = None
    mul_867: "f32[392, 256]" = torch.ops.aten.mul.Tensor(view_568, sub_234)
    sum_108: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_867, [0]);  mul_867 = None
    mul_868: "f32[256]" = torch.ops.aten.mul.Tensor(sum_107, 0.002551020408163265)
    unsqueeze_166: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_868, 0);  mul_868 = None
    mul_869: "f32[256]" = torch.ops.aten.mul.Tensor(sum_108, 0.002551020408163265)
    mul_870: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_871: "f32[256]" = torch.ops.aten.mul.Tensor(mul_869, mul_870);  mul_869 = mul_870 = None
    unsqueeze_167: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_871, 0);  mul_871 = None
    mul_872: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_94);  primals_94 = None
    unsqueeze_168: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_872, 0);  mul_872 = None
    mul_873: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_234, unsqueeze_167);  sub_234 = unsqueeze_167 = None
    sub_236: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_568, mul_873);  view_568 = mul_873 = None
    sub_237: "f32[392, 256]" = torch.ops.aten.sub.Tensor(sub_236, unsqueeze_166);  sub_236 = unsqueeze_166 = None
    mul_874: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_237, unsqueeze_168);  sub_237 = unsqueeze_168 = None
    mul_875: "f32[256]" = torch.ops.aten.mul.Tensor(sum_108, squeeze_79);  sum_108 = squeeze_79 = None
    view_569: "f32[8, 49, 256]" = torch.ops.aten.view.default(mul_874, [8, 49, 256]);  mul_874 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_570: "f32[392, 256]" = torch.ops.aten.view.default(view_569, [392, 256]);  view_569 = None
    permute_329: "f32[256, 392]" = torch.ops.aten.permute.default(view_570, [1, 0])
    mm_132: "f32[256, 256]" = torch.ops.aten.mm.default(permute_329, view_140);  permute_329 = view_140 = None
    permute_330: "f32[256, 256]" = torch.ops.aten.permute.default(mm_132, [1, 0]);  mm_132 = None
    mm_133: "f32[392, 256]" = torch.ops.aten.mm.default(view_570, permute_331);  view_570 = permute_331 = None
    view_571: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_133, [8, 49, 256]);  mm_133 = None
    permute_332: "f32[256, 256]" = torch.ops.aten.permute.default(permute_330, [1, 0]);  permute_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    lt_17: "b8[8, 49, 256]" = torch.ops.aten.lt.Scalar(view_139, -3)
    le_17: "b8[8, 49, 256]" = torch.ops.aten.le.Scalar(view_139, 3)
    div_65: "f32[8, 49, 256]" = torch.ops.aten.div.Tensor(view_139, 3);  view_139 = None
    add_427: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(div_65, 0.5);  div_65 = None
    mul_876: "f32[8, 49, 256]" = torch.ops.aten.mul.Tensor(view_571, add_427);  add_427 = None
    where_34: "f32[8, 49, 256]" = torch.ops.aten.where.self(le_17, mul_876, view_571);  le_17 = mul_876 = view_571 = None
    where_35: "f32[8, 49, 256]" = torch.ops.aten.where.self(lt_17, full_default, where_34);  lt_17 = where_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    view_572: "f32[8, 49, 8, 32]" = torch.ops.aten.view.default(where_35, [8, 49, 8, 32]);  where_35 = None
    permute_333: "f32[8, 8, 49, 32]" = torch.ops.aten.permute.default(view_572, [0, 2, 1, 3]);  view_572 = None
    clone_92: "f32[8, 8, 49, 32]" = torch.ops.aten.clone.default(permute_333, memory_format = torch.contiguous_format);  permute_333 = None
    view_573: "f32[64, 49, 32]" = torch.ops.aten.view.default(clone_92, [64, 49, 32]);  clone_92 = None
    bmm_60: "f32[64, 49, 32]" = torch.ops.aten.bmm.default(permute_334, view_573);  permute_334 = None
    bmm_61: "f32[64, 49, 49]" = torch.ops.aten.bmm.default(view_573, permute_335);  view_573 = permute_335 = None
    view_574: "f32[8, 8, 49, 32]" = torch.ops.aten.view.default(bmm_60, [8, 8, 49, 32]);  bmm_60 = None
    view_575: "f32[8, 8, 49, 49]" = torch.ops.aten.view.default(bmm_61, [8, 8, 49, 49]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    mul_877: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(view_575, alias_22);  view_575 = None
    sum_109: "f32[8, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_877, [-1], True)
    mul_878: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(alias_22, sum_109);  alias_22 = sum_109 = None
    sub_238: "f32[8, 8, 49, 49]" = torch.ops.aten.sub.Tensor(mul_877, mul_878);  mul_877 = mul_878 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    sum_110: "f32[1, 8, 49, 49]" = torch.ops.aten.sum.dim_IntList(sub_238, [0], True)
    view_576: "f32[8, 49, 49]" = torch.ops.aten.view.default(sum_110, [8, 49, 49]);  sum_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    index_put_8: "f32[8, 49]" = torch.ops.aten.index_put.default(full_default_25, [None, primals_214], view_576, True);  primals_214 = view_576 = None
    slice_scatter_11: "f32[8, 49]" = torch.ops.aten.slice_scatter.default(full_default_25, index_put_8, 0, 0, 9223372036854775807);  full_default_25 = index_put_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    mul_879: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(sub_238, 0.25);  sub_238 = None
    view_577: "f32[64, 49, 49]" = torch.ops.aten.view.default(mul_879, [64, 49, 49]);  mul_879 = None
    bmm_62: "f32[64, 16, 49]" = torch.ops.aten.bmm.default(permute_336, view_577);  permute_336 = None
    bmm_63: "f32[64, 49, 16]" = torch.ops.aten.bmm.default(view_577, permute_337);  view_577 = permute_337 = None
    view_578: "f32[8, 8, 16, 49]" = torch.ops.aten.view.default(bmm_62, [8, 8, 16, 49]);  bmm_62 = None
    view_579: "f32[8, 8, 49, 16]" = torch.ops.aten.view.default(bmm_63, [8, 8, 49, 16]);  bmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_338: "f32[8, 49, 8, 32]" = torch.ops.aten.permute.default(view_574, [0, 2, 1, 3]);  view_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_339: "f32[8, 49, 8, 16]" = torch.ops.aten.permute.default(view_578, [0, 3, 1, 2]);  view_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_340: "f32[8, 49, 8, 16]" = torch.ops.aten.permute.default(view_579, [0, 2, 1, 3]);  view_579 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    cat_8: "f32[8, 49, 8, 64]" = torch.ops.aten.cat.default([permute_340, permute_339, permute_338], 3);  permute_340 = permute_339 = permute_338 = None
    view_580: "f32[8, 49, 512]" = torch.ops.aten.view.default(cat_8, [8, 49, 512]);  cat_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_581: "f32[392, 512]" = torch.ops.aten.view.default(view_580, [392, 512]);  view_580 = None
    sum_111: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_581, [0])
    sub_239: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_130, unsqueeze_169);  view_130 = unsqueeze_169 = None
    mul_880: "f32[392, 512]" = torch.ops.aten.mul.Tensor(view_581, sub_239)
    sum_112: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_880, [0]);  mul_880 = None
    mul_881: "f32[512]" = torch.ops.aten.mul.Tensor(sum_111, 0.002551020408163265)
    unsqueeze_170: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_881, 0);  mul_881 = None
    mul_882: "f32[512]" = torch.ops.aten.mul.Tensor(sum_112, 0.002551020408163265)
    mul_883: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_884: "f32[512]" = torch.ops.aten.mul.Tensor(mul_882, mul_883);  mul_882 = mul_883 = None
    unsqueeze_171: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_884, 0);  mul_884 = None
    mul_885: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_91);  primals_91 = None
    unsqueeze_172: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_885, 0);  mul_885 = None
    mul_886: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_239, unsqueeze_171);  sub_239 = unsqueeze_171 = None
    sub_241: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_581, mul_886);  view_581 = mul_886 = None
    sub_242: "f32[392, 512]" = torch.ops.aten.sub.Tensor(sub_241, unsqueeze_170);  sub_241 = unsqueeze_170 = None
    mul_887: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_242, unsqueeze_172);  sub_242 = unsqueeze_172 = None
    mul_888: "f32[512]" = torch.ops.aten.mul.Tensor(sum_112, squeeze_76);  sum_112 = squeeze_76 = None
    view_582: "f32[8, 49, 512]" = torch.ops.aten.view.default(mul_887, [8, 49, 512]);  mul_887 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_583: "f32[392, 512]" = torch.ops.aten.view.default(view_582, [392, 512]);  view_582 = None
    permute_341: "f32[512, 392]" = torch.ops.aten.permute.default(view_583, [1, 0])
    mm_134: "f32[512, 256]" = torch.ops.aten.mm.default(permute_341, view_128);  permute_341 = view_128 = None
    permute_342: "f32[256, 512]" = torch.ops.aten.permute.default(mm_134, [1, 0]);  mm_134 = None
    mm_135: "f32[392, 256]" = torch.ops.aten.mm.default(view_583, permute_343);  view_583 = permute_343 = None
    view_584: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_135, [8, 49, 256]);  mm_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_428: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_426, view_584);  add_426 = view_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_344: "f32[512, 256]" = torch.ops.aten.permute.default(permute_342, [1, 0]);  permute_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_585: "f32[392, 256]" = torch.ops.aten.view.default(add_428, [392, 256])
    sum_113: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_585, [0])
    sub_243: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_126, unsqueeze_173);  view_126 = unsqueeze_173 = None
    mul_889: "f32[392, 256]" = torch.ops.aten.mul.Tensor(view_585, sub_243)
    sum_114: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_889, [0]);  mul_889 = None
    mul_890: "f32[256]" = torch.ops.aten.mul.Tensor(sum_113, 0.002551020408163265)
    unsqueeze_174: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_890, 0);  mul_890 = None
    mul_891: "f32[256]" = torch.ops.aten.mul.Tensor(sum_114, 0.002551020408163265)
    mul_892: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_893: "f32[256]" = torch.ops.aten.mul.Tensor(mul_891, mul_892);  mul_891 = mul_892 = None
    unsqueeze_175: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_893, 0);  mul_893 = None
    mul_894: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_88);  primals_88 = None
    unsqueeze_176: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_894, 0);  mul_894 = None
    mul_895: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_243, unsqueeze_175);  sub_243 = unsqueeze_175 = None
    sub_245: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_585, mul_895);  view_585 = mul_895 = None
    sub_246: "f32[392, 256]" = torch.ops.aten.sub.Tensor(sub_245, unsqueeze_174);  sub_245 = unsqueeze_174 = None
    mul_896: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_246, unsqueeze_176);  sub_246 = unsqueeze_176 = None
    mul_897: "f32[256]" = torch.ops.aten.mul.Tensor(sum_114, squeeze_73);  sum_114 = squeeze_73 = None
    view_586: "f32[8, 49, 256]" = torch.ops.aten.view.default(mul_896, [8, 49, 256]);  mul_896 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_587: "f32[392, 256]" = torch.ops.aten.view.default(view_586, [392, 256]);  view_586 = None
    permute_345: "f32[256, 392]" = torch.ops.aten.permute.default(view_587, [1, 0])
    mm_136: "f32[256, 512]" = torch.ops.aten.mm.default(permute_345, view_124);  permute_345 = view_124 = None
    permute_346: "f32[512, 256]" = torch.ops.aten.permute.default(mm_136, [1, 0]);  mm_136 = None
    mm_137: "f32[392, 512]" = torch.ops.aten.mm.default(view_587, permute_347);  view_587 = permute_347 = None
    view_588: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_137, [8, 49, 512]);  mm_137 = None
    permute_348: "f32[256, 512]" = torch.ops.aten.permute.default(permute_346, [1, 0]);  permute_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    lt_18: "b8[8, 49, 512]" = torch.ops.aten.lt.Scalar(view_123, -3)
    le_18: "b8[8, 49, 512]" = torch.ops.aten.le.Scalar(view_123, 3)
    div_66: "f32[8, 49, 512]" = torch.ops.aten.div.Tensor(view_123, 3);  view_123 = None
    add_429: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(div_66, 0.5);  div_66 = None
    mul_898: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_588, add_429);  add_429 = None
    where_36: "f32[8, 49, 512]" = torch.ops.aten.where.self(le_18, mul_898, view_588);  le_18 = mul_898 = view_588 = None
    where_37: "f32[8, 49, 512]" = torch.ops.aten.where.self(lt_18, full_default, where_36);  lt_18 = where_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_589: "f32[392, 512]" = torch.ops.aten.view.default(where_37, [392, 512]);  where_37 = None
    sum_115: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_589, [0])
    sub_247: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_122, unsqueeze_177);  view_122 = unsqueeze_177 = None
    mul_899: "f32[392, 512]" = torch.ops.aten.mul.Tensor(view_589, sub_247)
    sum_116: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_899, [0]);  mul_899 = None
    mul_900: "f32[512]" = torch.ops.aten.mul.Tensor(sum_115, 0.002551020408163265)
    unsqueeze_178: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_900, 0);  mul_900 = None
    mul_901: "f32[512]" = torch.ops.aten.mul.Tensor(sum_116, 0.002551020408163265)
    mul_902: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_903: "f32[512]" = torch.ops.aten.mul.Tensor(mul_901, mul_902);  mul_901 = mul_902 = None
    unsqueeze_179: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_903, 0);  mul_903 = None
    mul_904: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_85);  primals_85 = None
    unsqueeze_180: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_904, 0);  mul_904 = None
    mul_905: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_247, unsqueeze_179);  sub_247 = unsqueeze_179 = None
    sub_249: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_589, mul_905);  view_589 = mul_905 = None
    sub_250: "f32[392, 512]" = torch.ops.aten.sub.Tensor(sub_249, unsqueeze_178);  sub_249 = unsqueeze_178 = None
    mul_906: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_250, unsqueeze_180);  sub_250 = unsqueeze_180 = None
    mul_907: "f32[512]" = torch.ops.aten.mul.Tensor(sum_116, squeeze_70);  sum_116 = squeeze_70 = None
    view_590: "f32[8, 49, 512]" = torch.ops.aten.view.default(mul_906, [8, 49, 512]);  mul_906 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_591: "f32[392, 512]" = torch.ops.aten.view.default(view_590, [392, 512]);  view_590 = None
    permute_349: "f32[512, 392]" = torch.ops.aten.permute.default(view_591, [1, 0])
    mm_138: "f32[512, 256]" = torch.ops.aten.mm.default(permute_349, view_120);  permute_349 = view_120 = None
    permute_350: "f32[256, 512]" = torch.ops.aten.permute.default(mm_138, [1, 0]);  mm_138 = None
    mm_139: "f32[392, 256]" = torch.ops.aten.mm.default(view_591, permute_351);  view_591 = permute_351 = None
    view_592: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_139, [8, 49, 256]);  mm_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_430: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_428, view_592);  add_428 = view_592 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_352: "f32[512, 256]" = torch.ops.aten.permute.default(permute_350, [1, 0]);  permute_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_593: "f32[392, 256]" = torch.ops.aten.view.default(add_430, [392, 256]);  add_430 = None
    sum_117: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_593, [0])
    sub_251: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_118, unsqueeze_181);  view_118 = unsqueeze_181 = None
    mul_908: "f32[392, 256]" = torch.ops.aten.mul.Tensor(view_593, sub_251)
    sum_118: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_908, [0]);  mul_908 = None
    mul_909: "f32[256]" = torch.ops.aten.mul.Tensor(sum_117, 0.002551020408163265)
    unsqueeze_182: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_909, 0);  mul_909 = None
    mul_910: "f32[256]" = torch.ops.aten.mul.Tensor(sum_118, 0.002551020408163265)
    mul_911: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_912: "f32[256]" = torch.ops.aten.mul.Tensor(mul_910, mul_911);  mul_910 = mul_911 = None
    unsqueeze_183: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_912, 0);  mul_912 = None
    mul_913: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_82);  primals_82 = None
    unsqueeze_184: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_913, 0);  mul_913 = None
    mul_914: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_251, unsqueeze_183);  sub_251 = unsqueeze_183 = None
    sub_253: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_593, mul_914);  view_593 = mul_914 = None
    sub_254: "f32[392, 256]" = torch.ops.aten.sub.Tensor(sub_253, unsqueeze_182);  sub_253 = unsqueeze_182 = None
    mul_915: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_254, unsqueeze_184);  sub_254 = unsqueeze_184 = None
    mul_916: "f32[256]" = torch.ops.aten.mul.Tensor(sum_118, squeeze_67);  sum_118 = squeeze_67 = None
    view_594: "f32[8, 49, 256]" = torch.ops.aten.view.default(mul_915, [8, 49, 256]);  mul_915 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_595: "f32[392, 256]" = torch.ops.aten.view.default(view_594, [392, 256]);  view_594 = None
    permute_353: "f32[256, 392]" = torch.ops.aten.permute.default(view_595, [1, 0])
    mm_140: "f32[256, 512]" = torch.ops.aten.mm.default(permute_353, view_116);  permute_353 = view_116 = None
    permute_354: "f32[512, 256]" = torch.ops.aten.permute.default(mm_140, [1, 0]);  mm_140 = None
    mm_141: "f32[392, 512]" = torch.ops.aten.mm.default(view_595, permute_355);  view_595 = permute_355 = None
    view_596: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_141, [8, 49, 512]);  mm_141 = None
    permute_356: "f32[256, 512]" = torch.ops.aten.permute.default(permute_354, [1, 0]);  permute_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:340, code: x = self.proj(x)
    lt_19: "b8[8, 49, 512]" = torch.ops.aten.lt.Scalar(view_115, -3)
    le_19: "b8[8, 49, 512]" = torch.ops.aten.le.Scalar(view_115, 3)
    div_67: "f32[8, 49, 512]" = torch.ops.aten.div.Tensor(view_115, 3);  view_115 = None
    add_431: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(div_67, 0.5);  div_67 = None
    mul_917: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_596, add_431);  add_431 = None
    where_38: "f32[8, 49, 512]" = torch.ops.aten.where.self(le_19, mul_917, view_596);  le_19 = mul_917 = view_596 = None
    where_39: "f32[8, 49, 512]" = torch.ops.aten.where.self(lt_19, full_default, where_38);  lt_19 = where_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:339, code: x = (attn @ v).transpose(1, 2).reshape(B, -1, self.val_attn_dim)
    view_597: "f32[8, 49, 8, 64]" = torch.ops.aten.view.default(where_39, [8, 49, 8, 64]);  where_39 = None
    permute_357: "f32[8, 8, 49, 64]" = torch.ops.aten.permute.default(view_597, [0, 2, 1, 3]);  view_597 = None
    clone_93: "f32[8, 8, 49, 64]" = torch.ops.aten.clone.default(permute_357, memory_format = torch.contiguous_format);  permute_357 = None
    view_598: "f32[64, 49, 64]" = torch.ops.aten.view.default(clone_93, [64, 49, 64]);  clone_93 = None
    bmm_64: "f32[64, 196, 64]" = torch.ops.aten.bmm.default(permute_358, view_598);  permute_358 = None
    bmm_65: "f32[64, 49, 196]" = torch.ops.aten.bmm.default(view_598, permute_359);  view_598 = permute_359 = None
    view_599: "f32[8, 8, 196, 64]" = torch.ops.aten.view.default(bmm_64, [8, 8, 196, 64]);  bmm_64 = None
    view_600: "f32[8, 8, 49, 196]" = torch.ops.aten.view.default(bmm_65, [8, 8, 49, 196]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:337, code: attn = attn.softmax(dim=-1)
    mul_918: "f32[8, 8, 49, 196]" = torch.ops.aten.mul.Tensor(view_600, alias_23);  view_600 = None
    sum_119: "f32[8, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_918, [-1], True)
    mul_919: "f32[8, 8, 49, 196]" = torch.ops.aten.mul.Tensor(alias_23, sum_119);  alias_23 = sum_119 = None
    sub_255: "f32[8, 8, 49, 196]" = torch.ops.aten.sub.Tensor(mul_918, mul_919);  mul_918 = mul_919 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:336, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    sum_120: "f32[1, 8, 49, 196]" = torch.ops.aten.sum.dim_IntList(sub_255, [0], True)
    view_601: "f32[8, 49, 196]" = torch.ops.aten.view.default(sum_120, [8, 49, 196]);  sum_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:311, code: return self.attention_biases[:, self.attention_bias_idxs]
    full_default_41: "f32[8, 196]" = torch.ops.aten.full.default([8, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    index_put_9: "f32[8, 196]" = torch.ops.aten.index_put.default(full_default_41, [None, primals_213], view_601, True);  primals_213 = view_601 = None
    slice_scatter_12: "f32[8, 196]" = torch.ops.aten.slice_scatter.default(full_default_41, index_put_9, 0, 0, 9223372036854775807);  full_default_41 = index_put_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:336, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    mul_920: "f32[8, 8, 49, 196]" = torch.ops.aten.mul.Tensor(sub_255, 0.25);  sub_255 = None
    view_602: "f32[64, 49, 196]" = torch.ops.aten.view.default(mul_920, [64, 49, 196]);  mul_920 = None
    bmm_66: "f32[64, 16, 196]" = torch.ops.aten.bmm.default(permute_360, view_602);  permute_360 = None
    bmm_67: "f32[64, 49, 16]" = torch.ops.aten.bmm.default(view_602, permute_361);  view_602 = permute_361 = None
    view_603: "f32[8, 8, 16, 196]" = torch.ops.aten.view.default(bmm_66, [8, 8, 16, 196]);  bmm_66 = None
    view_604: "f32[8, 8, 49, 16]" = torch.ops.aten.view.default(bmm_67, [8, 8, 49, 16]);  bmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:334, code: q = self.q(x).view(B, -1, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
    permute_362: "f32[8, 49, 8, 16]" = torch.ops.aten.permute.default(view_604, [0, 2, 1, 3]);  view_604 = None
    clone_94: "f32[8, 49, 8, 16]" = torch.ops.aten.clone.default(permute_362, memory_format = torch.contiguous_format);  permute_362 = None
    view_605: "f32[8, 49, 128]" = torch.ops.aten.view.default(clone_94, [8, 49, 128]);  clone_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_606: "f32[392, 128]" = torch.ops.aten.view.default(view_605, [392, 128]);  view_605 = None
    sum_121: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_606, [0])
    sub_256: "f32[392, 128]" = torch.ops.aten.sub.Tensor(view_106, unsqueeze_185);  view_106 = unsqueeze_185 = None
    mul_921: "f32[392, 128]" = torch.ops.aten.mul.Tensor(view_606, sub_256)
    sum_122: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_921, [0]);  mul_921 = None
    mul_922: "f32[128]" = torch.ops.aten.mul.Tensor(sum_121, 0.002551020408163265)
    unsqueeze_186: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_922, 0);  mul_922 = None
    mul_923: "f32[128]" = torch.ops.aten.mul.Tensor(sum_122, 0.002551020408163265)
    mul_924: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_925: "f32[128]" = torch.ops.aten.mul.Tensor(mul_923, mul_924);  mul_923 = mul_924 = None
    unsqueeze_187: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_925, 0);  mul_925 = None
    mul_926: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_79);  primals_79 = None
    unsqueeze_188: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_926, 0);  mul_926 = None
    mul_927: "f32[392, 128]" = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_187);  sub_256 = unsqueeze_187 = None
    sub_258: "f32[392, 128]" = torch.ops.aten.sub.Tensor(view_606, mul_927);  view_606 = mul_927 = None
    sub_259: "f32[392, 128]" = torch.ops.aten.sub.Tensor(sub_258, unsqueeze_186);  sub_258 = unsqueeze_186 = None
    mul_928: "f32[392, 128]" = torch.ops.aten.mul.Tensor(sub_259, unsqueeze_188);  sub_259 = unsqueeze_188 = None
    mul_929: "f32[128]" = torch.ops.aten.mul.Tensor(sum_122, squeeze_64);  sum_122 = squeeze_64 = None
    view_607: "f32[8, 49, 128]" = torch.ops.aten.view.default(mul_928, [8, 49, 128]);  mul_928 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_608: "f32[392, 128]" = torch.ops.aten.view.default(view_607, [392, 128]);  view_607 = None
    permute_363: "f32[128, 392]" = torch.ops.aten.permute.default(view_608, [1, 0])
    mm_142: "f32[128, 128]" = torch.ops.aten.mm.default(permute_363, view_104);  permute_363 = view_104 = None
    permute_364: "f32[128, 128]" = torch.ops.aten.permute.default(mm_142, [1, 0]);  mm_142 = None
    mm_143: "f32[392, 128]" = torch.ops.aten.mm.default(view_608, permute_365);  view_608 = permute_365 = None
    view_609: "f32[8, 49, 128]" = torch.ops.aten.view.default(mm_143, [8, 49, 128]);  mm_143 = None
    permute_366: "f32[128, 128]" = torch.ops.aten.permute.default(permute_364, [1, 0]);  permute_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:162, code: return x.reshape(B, -1, C)
    view_610: "f32[8, 7, 7, 128]" = torch.ops.aten.view.default(view_609, [8, 7, 7, 128]);  view_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:161, code: x = x[:, ::self.stride, ::self.stride]
    full_default_43: "f32[8, 7, 14, 128]" = torch.ops.aten.full.default([8, 7, 14, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_13: "f32[8, 7, 14, 128]" = torch.ops.aten.slice_scatter.default(full_default_43, view_610, 2, 0, 9223372036854775807, 2);  full_default_43 = view_610 = None
    full_default_44: "f32[8, 14, 14, 128]" = torch.ops.aten.full.default([8, 14, 14, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_14: "f32[8, 14, 14, 128]" = torch.ops.aten.slice_scatter.default(full_default_44, slice_scatter_13, 1, 0, 9223372036854775807, 2);  slice_scatter_13 = None
    slice_scatter_15: "f32[8, 14, 14, 128]" = torch.ops.aten.slice_scatter.default(full_default_44, slice_scatter_14, 0, 0, 9223372036854775807);  full_default_44 = slice_scatter_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:157, code: x = x.view(B, self.resolution[0], self.resolution[1], C)
    view_611: "f32[8, 196, 128]" = torch.ops.aten.view.default(slice_scatter_15, [8, 196, 128]);  slice_scatter_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:333, code: v = v.permute(0, 2, 1, 3)  # BHNC
    permute_367: "f32[8, 196, 8, 64]" = torch.ops.aten.permute.default(view_599, [0, 2, 1, 3]);  view_599 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:332, code: k = k.permute(0, 2, 3, 1)  # BHCN
    permute_368: "f32[8, 196, 8, 16]" = torch.ops.aten.permute.default(view_603, [0, 3, 1, 2]);  view_603 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:331, code: k, v = self.kv(x).view(B, N, self.num_heads, -1).split([self.key_dim, self.val_dim], dim=3)
    cat_9: "f32[8, 196, 8, 80]" = torch.ops.aten.cat.default([permute_368, permute_367], 3);  permute_368 = permute_367 = None
    view_612: "f32[8, 196, 640]" = torch.ops.aten.view.default(cat_9, [8, 196, 640]);  cat_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_613: "f32[1568, 640]" = torch.ops.aten.view.default(view_612, [1568, 640]);  view_612 = None
    sum_123: "f32[640]" = torch.ops.aten.sum.dim_IntList(view_613, [0])
    sub_260: "f32[1568, 640]" = torch.ops.aten.sub.Tensor(view_99, unsqueeze_189);  view_99 = unsqueeze_189 = None
    mul_930: "f32[1568, 640]" = torch.ops.aten.mul.Tensor(view_613, sub_260)
    sum_124: "f32[640]" = torch.ops.aten.sum.dim_IntList(mul_930, [0]);  mul_930 = None
    mul_931: "f32[640]" = torch.ops.aten.mul.Tensor(sum_123, 0.0006377551020408163)
    unsqueeze_190: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_931, 0);  mul_931 = None
    mul_932: "f32[640]" = torch.ops.aten.mul.Tensor(sum_124, 0.0006377551020408163)
    mul_933: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_934: "f32[640]" = torch.ops.aten.mul.Tensor(mul_932, mul_933);  mul_932 = mul_933 = None
    unsqueeze_191: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_934, 0);  mul_934 = None
    mul_935: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_76);  primals_76 = None
    unsqueeze_192: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_935, 0);  mul_935 = None
    mul_936: "f32[1568, 640]" = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_191);  sub_260 = unsqueeze_191 = None
    sub_262: "f32[1568, 640]" = torch.ops.aten.sub.Tensor(view_613, mul_936);  view_613 = mul_936 = None
    sub_263: "f32[1568, 640]" = torch.ops.aten.sub.Tensor(sub_262, unsqueeze_190);  sub_262 = unsqueeze_190 = None
    mul_937: "f32[1568, 640]" = torch.ops.aten.mul.Tensor(sub_263, unsqueeze_192);  sub_263 = unsqueeze_192 = None
    mul_938: "f32[640]" = torch.ops.aten.mul.Tensor(sum_124, squeeze_61);  sum_124 = squeeze_61 = None
    view_614: "f32[8, 196, 640]" = torch.ops.aten.view.default(mul_937, [8, 196, 640]);  mul_937 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_615: "f32[1568, 640]" = torch.ops.aten.view.default(view_614, [1568, 640]);  view_614 = None
    permute_369: "f32[640, 1568]" = torch.ops.aten.permute.default(view_615, [1, 0])
    mm_144: "f32[640, 128]" = torch.ops.aten.mm.default(permute_369, view_97);  permute_369 = view_97 = None
    permute_370: "f32[128, 640]" = torch.ops.aten.permute.default(mm_144, [1, 0]);  mm_144 = None
    mm_145: "f32[1568, 128]" = torch.ops.aten.mm.default(view_615, permute_371);  view_615 = permute_371 = None
    view_616: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_145, [8, 196, 128]);  mm_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_432: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(view_611, view_616);  view_611 = view_616 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_372: "f32[640, 128]" = torch.ops.aten.permute.default(permute_370, [1, 0]);  permute_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_617: "f32[1568, 128]" = torch.ops.aten.view.default(add_432, [1568, 128])
    sum_125: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_617, [0])
    sub_264: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_95, unsqueeze_193);  view_95 = unsqueeze_193 = None
    mul_939: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(view_617, sub_264)
    sum_126: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_939, [0]);  mul_939 = None
    mul_940: "f32[128]" = torch.ops.aten.mul.Tensor(sum_125, 0.0006377551020408163)
    unsqueeze_194: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_940, 0);  mul_940 = None
    mul_941: "f32[128]" = torch.ops.aten.mul.Tensor(sum_126, 0.0006377551020408163)
    mul_942: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_943: "f32[128]" = torch.ops.aten.mul.Tensor(mul_941, mul_942);  mul_941 = mul_942 = None
    unsqueeze_195: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_943, 0);  mul_943 = None
    mul_944: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_73);  primals_73 = None
    unsqueeze_196: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_944, 0);  mul_944 = None
    mul_945: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_195);  sub_264 = unsqueeze_195 = None
    sub_266: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_617, mul_945);  view_617 = mul_945 = None
    sub_267: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(sub_266, unsqueeze_194);  sub_266 = unsqueeze_194 = None
    mul_946: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_267, unsqueeze_196);  sub_267 = unsqueeze_196 = None
    mul_947: "f32[128]" = torch.ops.aten.mul.Tensor(sum_126, squeeze_58);  sum_126 = squeeze_58 = None
    view_618: "f32[8, 196, 128]" = torch.ops.aten.view.default(mul_946, [8, 196, 128]);  mul_946 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_619: "f32[1568, 128]" = torch.ops.aten.view.default(view_618, [1568, 128]);  view_618 = None
    permute_373: "f32[128, 1568]" = torch.ops.aten.permute.default(view_619, [1, 0])
    mm_146: "f32[128, 256]" = torch.ops.aten.mm.default(permute_373, view_93);  permute_373 = view_93 = None
    permute_374: "f32[256, 128]" = torch.ops.aten.permute.default(mm_146, [1, 0]);  mm_146 = None
    mm_147: "f32[1568, 256]" = torch.ops.aten.mm.default(view_619, permute_375);  view_619 = permute_375 = None
    view_620: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_147, [8, 196, 256]);  mm_147 = None
    permute_376: "f32[128, 256]" = torch.ops.aten.permute.default(permute_374, [1, 0]);  permute_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    lt_20: "b8[8, 196, 256]" = torch.ops.aten.lt.Scalar(view_92, -3)
    le_20: "b8[8, 196, 256]" = torch.ops.aten.le.Scalar(view_92, 3)
    div_68: "f32[8, 196, 256]" = torch.ops.aten.div.Tensor(view_92, 3);  view_92 = None
    add_433: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(div_68, 0.5);  div_68 = None
    mul_948: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_620, add_433);  add_433 = None
    where_40: "f32[8, 196, 256]" = torch.ops.aten.where.self(le_20, mul_948, view_620);  le_20 = mul_948 = view_620 = None
    where_41: "f32[8, 196, 256]" = torch.ops.aten.where.self(lt_20, full_default, where_40);  lt_20 = where_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_621: "f32[1568, 256]" = torch.ops.aten.view.default(where_41, [1568, 256]);  where_41 = None
    sum_127: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_621, [0])
    sub_268: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_91, unsqueeze_197);  view_91 = unsqueeze_197 = None
    mul_949: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(view_621, sub_268)
    sum_128: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_949, [0]);  mul_949 = None
    mul_950: "f32[256]" = torch.ops.aten.mul.Tensor(sum_127, 0.0006377551020408163)
    unsqueeze_198: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_950, 0);  mul_950 = None
    mul_951: "f32[256]" = torch.ops.aten.mul.Tensor(sum_128, 0.0006377551020408163)
    mul_952: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_953: "f32[256]" = torch.ops.aten.mul.Tensor(mul_951, mul_952);  mul_951 = mul_952 = None
    unsqueeze_199: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_953, 0);  mul_953 = None
    mul_954: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_70);  primals_70 = None
    unsqueeze_200: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_954, 0);  mul_954 = None
    mul_955: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_199);  sub_268 = unsqueeze_199 = None
    sub_270: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_621, mul_955);  view_621 = mul_955 = None
    sub_271: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(sub_270, unsqueeze_198);  sub_270 = unsqueeze_198 = None
    mul_956: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_271, unsqueeze_200);  sub_271 = unsqueeze_200 = None
    mul_957: "f32[256]" = torch.ops.aten.mul.Tensor(sum_128, squeeze_55);  sum_128 = squeeze_55 = None
    view_622: "f32[8, 196, 256]" = torch.ops.aten.view.default(mul_956, [8, 196, 256]);  mul_956 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_623: "f32[1568, 256]" = torch.ops.aten.view.default(view_622, [1568, 256]);  view_622 = None
    permute_377: "f32[256, 1568]" = torch.ops.aten.permute.default(view_623, [1, 0])
    mm_148: "f32[256, 128]" = torch.ops.aten.mm.default(permute_377, view_89);  permute_377 = view_89 = None
    permute_378: "f32[128, 256]" = torch.ops.aten.permute.default(mm_148, [1, 0]);  mm_148 = None
    mm_149: "f32[1568, 128]" = torch.ops.aten.mm.default(view_623, permute_379);  view_623 = permute_379 = None
    view_624: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_149, [8, 196, 128]);  mm_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_434: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_432, view_624);  add_432 = view_624 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_380: "f32[256, 128]" = torch.ops.aten.permute.default(permute_378, [1, 0]);  permute_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_625: "f32[1568, 128]" = torch.ops.aten.view.default(add_434, [1568, 128])
    sum_129: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_625, [0])
    sub_272: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_87, unsqueeze_201);  view_87 = unsqueeze_201 = None
    mul_958: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(view_625, sub_272)
    sum_130: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_958, [0]);  mul_958 = None
    mul_959: "f32[128]" = torch.ops.aten.mul.Tensor(sum_129, 0.0006377551020408163)
    unsqueeze_202: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_959, 0);  mul_959 = None
    mul_960: "f32[128]" = torch.ops.aten.mul.Tensor(sum_130, 0.0006377551020408163)
    mul_961: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_962: "f32[128]" = torch.ops.aten.mul.Tensor(mul_960, mul_961);  mul_960 = mul_961 = None
    unsqueeze_203: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_962, 0);  mul_962 = None
    mul_963: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_67);  primals_67 = None
    unsqueeze_204: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_963, 0);  mul_963 = None
    mul_964: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_203);  sub_272 = unsqueeze_203 = None
    sub_274: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_625, mul_964);  view_625 = mul_964 = None
    sub_275: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(sub_274, unsqueeze_202);  sub_274 = unsqueeze_202 = None
    mul_965: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_275, unsqueeze_204);  sub_275 = unsqueeze_204 = None
    mul_966: "f32[128]" = torch.ops.aten.mul.Tensor(sum_130, squeeze_52);  sum_130 = squeeze_52 = None
    view_626: "f32[8, 196, 128]" = torch.ops.aten.view.default(mul_965, [8, 196, 128]);  mul_965 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_627: "f32[1568, 128]" = torch.ops.aten.view.default(view_626, [1568, 128]);  view_626 = None
    permute_381: "f32[128, 1568]" = torch.ops.aten.permute.default(view_627, [1, 0])
    mm_150: "f32[128, 128]" = torch.ops.aten.mm.default(permute_381, view_85);  permute_381 = view_85 = None
    permute_382: "f32[128, 128]" = torch.ops.aten.permute.default(mm_150, [1, 0]);  mm_150 = None
    mm_151: "f32[1568, 128]" = torch.ops.aten.mm.default(view_627, permute_383);  view_627 = permute_383 = None
    view_628: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_151, [8, 196, 128]);  mm_151 = None
    permute_384: "f32[128, 128]" = torch.ops.aten.permute.default(permute_382, [1, 0]);  permute_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    lt_21: "b8[8, 196, 128]" = torch.ops.aten.lt.Scalar(view_84, -3)
    le_21: "b8[8, 196, 128]" = torch.ops.aten.le.Scalar(view_84, 3)
    div_69: "f32[8, 196, 128]" = torch.ops.aten.div.Tensor(view_84, 3);  view_84 = None
    add_435: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(div_69, 0.5);  div_69 = None
    mul_967: "f32[8, 196, 128]" = torch.ops.aten.mul.Tensor(view_628, add_435);  add_435 = None
    where_42: "f32[8, 196, 128]" = torch.ops.aten.where.self(le_21, mul_967, view_628);  le_21 = mul_967 = view_628 = None
    where_43: "f32[8, 196, 128]" = torch.ops.aten.where.self(lt_21, full_default, where_42);  lt_21 = where_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    view_629: "f32[8, 196, 4, 32]" = torch.ops.aten.view.default(where_43, [8, 196, 4, 32]);  where_43 = None
    permute_385: "f32[8, 4, 196, 32]" = torch.ops.aten.permute.default(view_629, [0, 2, 1, 3]);  view_629 = None
    clone_95: "f32[8, 4, 196, 32]" = torch.ops.aten.clone.default(permute_385, memory_format = torch.contiguous_format);  permute_385 = None
    view_630: "f32[32, 196, 32]" = torch.ops.aten.view.default(clone_95, [32, 196, 32]);  clone_95 = None
    bmm_68: "f32[32, 196, 32]" = torch.ops.aten.bmm.default(permute_386, view_630);  permute_386 = None
    bmm_69: "f32[32, 196, 196]" = torch.ops.aten.bmm.default(view_630, permute_387);  view_630 = permute_387 = None
    view_631: "f32[8, 4, 196, 32]" = torch.ops.aten.view.default(bmm_68, [8, 4, 196, 32]);  bmm_68 = None
    view_632: "f32[8, 4, 196, 196]" = torch.ops.aten.view.default(bmm_69, [8, 4, 196, 196]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    mul_968: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(view_632, alias_24);  view_632 = None
    sum_131: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_968, [-1], True)
    mul_969: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(alias_24, sum_131);  alias_24 = sum_131 = None
    sub_276: "f32[8, 4, 196, 196]" = torch.ops.aten.sub.Tensor(mul_968, mul_969);  mul_968 = mul_969 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    sum_132: "f32[1, 4, 196, 196]" = torch.ops.aten.sum.dim_IntList(sub_276, [0], True)
    view_633: "f32[4, 196, 196]" = torch.ops.aten.view.default(sum_132, [4, 196, 196]);  sum_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    full_default_48: "f32[4, 196]" = torch.ops.aten.full.default([4, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    index_put_10: "f32[4, 196]" = torch.ops.aten.index_put.default(full_default_48, [None, primals_212], view_633, True);  primals_212 = view_633 = None
    slice_scatter_16: "f32[4, 196]" = torch.ops.aten.slice_scatter.default(full_default_48, index_put_10, 0, 0, 9223372036854775807);  index_put_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    mul_970: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(sub_276, 0.25);  sub_276 = None
    view_634: "f32[32, 196, 196]" = torch.ops.aten.view.default(mul_970, [32, 196, 196]);  mul_970 = None
    bmm_70: "f32[32, 16, 196]" = torch.ops.aten.bmm.default(permute_388, view_634);  permute_388 = None
    bmm_71: "f32[32, 196, 16]" = torch.ops.aten.bmm.default(view_634, permute_389);  view_634 = permute_389 = None
    view_635: "f32[8, 4, 16, 196]" = torch.ops.aten.view.default(bmm_70, [8, 4, 16, 196]);  bmm_70 = None
    view_636: "f32[8, 4, 196, 16]" = torch.ops.aten.view.default(bmm_71, [8, 4, 196, 16]);  bmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_390: "f32[8, 196, 4, 32]" = torch.ops.aten.permute.default(view_631, [0, 2, 1, 3]);  view_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_391: "f32[8, 196, 4, 16]" = torch.ops.aten.permute.default(view_635, [0, 3, 1, 2]);  view_635 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_392: "f32[8, 196, 4, 16]" = torch.ops.aten.permute.default(view_636, [0, 2, 1, 3]);  view_636 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    cat_10: "f32[8, 196, 4, 64]" = torch.ops.aten.cat.default([permute_392, permute_391, permute_390], 3);  permute_392 = permute_391 = permute_390 = None
    view_637: "f32[8, 196, 256]" = torch.ops.aten.view.default(cat_10, [8, 196, 256]);  cat_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_638: "f32[1568, 256]" = torch.ops.aten.view.default(view_637, [1568, 256]);  view_637 = None
    sum_133: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_638, [0])
    sub_277: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_75, unsqueeze_205);  view_75 = unsqueeze_205 = None
    mul_971: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(view_638, sub_277)
    sum_134: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_971, [0]);  mul_971 = None
    mul_972: "f32[256]" = torch.ops.aten.mul.Tensor(sum_133, 0.0006377551020408163)
    unsqueeze_206: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_972, 0);  mul_972 = None
    mul_973: "f32[256]" = torch.ops.aten.mul.Tensor(sum_134, 0.0006377551020408163)
    mul_974: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_975: "f32[256]" = torch.ops.aten.mul.Tensor(mul_973, mul_974);  mul_973 = mul_974 = None
    unsqueeze_207: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_975, 0);  mul_975 = None
    mul_976: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_64);  primals_64 = None
    unsqueeze_208: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_976, 0);  mul_976 = None
    mul_977: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_277, unsqueeze_207);  sub_277 = unsqueeze_207 = None
    sub_279: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_638, mul_977);  view_638 = mul_977 = None
    sub_280: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(sub_279, unsqueeze_206);  sub_279 = unsqueeze_206 = None
    mul_978: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_280, unsqueeze_208);  sub_280 = unsqueeze_208 = None
    mul_979: "f32[256]" = torch.ops.aten.mul.Tensor(sum_134, squeeze_49);  sum_134 = squeeze_49 = None
    view_639: "f32[8, 196, 256]" = torch.ops.aten.view.default(mul_978, [8, 196, 256]);  mul_978 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_640: "f32[1568, 256]" = torch.ops.aten.view.default(view_639, [1568, 256]);  view_639 = None
    permute_393: "f32[256, 1568]" = torch.ops.aten.permute.default(view_640, [1, 0])
    mm_152: "f32[256, 128]" = torch.ops.aten.mm.default(permute_393, view_73);  permute_393 = view_73 = None
    permute_394: "f32[128, 256]" = torch.ops.aten.permute.default(mm_152, [1, 0]);  mm_152 = None
    mm_153: "f32[1568, 128]" = torch.ops.aten.mm.default(view_640, permute_395);  view_640 = permute_395 = None
    view_641: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_153, [8, 196, 128]);  mm_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_436: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_434, view_641);  add_434 = view_641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_396: "f32[256, 128]" = torch.ops.aten.permute.default(permute_394, [1, 0]);  permute_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_642: "f32[1568, 128]" = torch.ops.aten.view.default(add_436, [1568, 128])
    sum_135: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_642, [0])
    sub_281: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_71, unsqueeze_209);  view_71 = unsqueeze_209 = None
    mul_980: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(view_642, sub_281)
    sum_136: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_980, [0]);  mul_980 = None
    mul_981: "f32[128]" = torch.ops.aten.mul.Tensor(sum_135, 0.0006377551020408163)
    unsqueeze_210: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_981, 0);  mul_981 = None
    mul_982: "f32[128]" = torch.ops.aten.mul.Tensor(sum_136, 0.0006377551020408163)
    mul_983: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_984: "f32[128]" = torch.ops.aten.mul.Tensor(mul_982, mul_983);  mul_982 = mul_983 = None
    unsqueeze_211: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_984, 0);  mul_984 = None
    mul_985: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_61);  primals_61 = None
    unsqueeze_212: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_985, 0);  mul_985 = None
    mul_986: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_281, unsqueeze_211);  sub_281 = unsqueeze_211 = None
    sub_283: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_642, mul_986);  view_642 = mul_986 = None
    sub_284: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(sub_283, unsqueeze_210);  sub_283 = unsqueeze_210 = None
    mul_987: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_284, unsqueeze_212);  sub_284 = unsqueeze_212 = None
    mul_988: "f32[128]" = torch.ops.aten.mul.Tensor(sum_136, squeeze_46);  sum_136 = squeeze_46 = None
    view_643: "f32[8, 196, 128]" = torch.ops.aten.view.default(mul_987, [8, 196, 128]);  mul_987 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_644: "f32[1568, 128]" = torch.ops.aten.view.default(view_643, [1568, 128]);  view_643 = None
    permute_397: "f32[128, 1568]" = torch.ops.aten.permute.default(view_644, [1, 0])
    mm_154: "f32[128, 256]" = torch.ops.aten.mm.default(permute_397, view_69);  permute_397 = view_69 = None
    permute_398: "f32[256, 128]" = torch.ops.aten.permute.default(mm_154, [1, 0]);  mm_154 = None
    mm_155: "f32[1568, 256]" = torch.ops.aten.mm.default(view_644, permute_399);  view_644 = permute_399 = None
    view_645: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_155, [8, 196, 256]);  mm_155 = None
    permute_400: "f32[128, 256]" = torch.ops.aten.permute.default(permute_398, [1, 0]);  permute_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    lt_22: "b8[8, 196, 256]" = torch.ops.aten.lt.Scalar(view_68, -3)
    le_22: "b8[8, 196, 256]" = torch.ops.aten.le.Scalar(view_68, 3)
    div_70: "f32[8, 196, 256]" = torch.ops.aten.div.Tensor(view_68, 3);  view_68 = None
    add_437: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(div_70, 0.5);  div_70 = None
    mul_989: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_645, add_437);  add_437 = None
    where_44: "f32[8, 196, 256]" = torch.ops.aten.where.self(le_22, mul_989, view_645);  le_22 = mul_989 = view_645 = None
    where_45: "f32[8, 196, 256]" = torch.ops.aten.where.self(lt_22, full_default, where_44);  lt_22 = where_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_646: "f32[1568, 256]" = torch.ops.aten.view.default(where_45, [1568, 256]);  where_45 = None
    sum_137: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_646, [0])
    sub_285: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_67, unsqueeze_213);  view_67 = unsqueeze_213 = None
    mul_990: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(view_646, sub_285)
    sum_138: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_990, [0]);  mul_990 = None
    mul_991: "f32[256]" = torch.ops.aten.mul.Tensor(sum_137, 0.0006377551020408163)
    unsqueeze_214: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_991, 0);  mul_991 = None
    mul_992: "f32[256]" = torch.ops.aten.mul.Tensor(sum_138, 0.0006377551020408163)
    mul_993: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_994: "f32[256]" = torch.ops.aten.mul.Tensor(mul_992, mul_993);  mul_992 = mul_993 = None
    unsqueeze_215: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_994, 0);  mul_994 = None
    mul_995: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_58);  primals_58 = None
    unsqueeze_216: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_995, 0);  mul_995 = None
    mul_996: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_285, unsqueeze_215);  sub_285 = unsqueeze_215 = None
    sub_287: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_646, mul_996);  view_646 = mul_996 = None
    sub_288: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(sub_287, unsqueeze_214);  sub_287 = unsqueeze_214 = None
    mul_997: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_288, unsqueeze_216);  sub_288 = unsqueeze_216 = None
    mul_998: "f32[256]" = torch.ops.aten.mul.Tensor(sum_138, squeeze_43);  sum_138 = squeeze_43 = None
    view_647: "f32[8, 196, 256]" = torch.ops.aten.view.default(mul_997, [8, 196, 256]);  mul_997 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_648: "f32[1568, 256]" = torch.ops.aten.view.default(view_647, [1568, 256]);  view_647 = None
    permute_401: "f32[256, 1568]" = torch.ops.aten.permute.default(view_648, [1, 0])
    mm_156: "f32[256, 128]" = torch.ops.aten.mm.default(permute_401, view_65);  permute_401 = view_65 = None
    permute_402: "f32[128, 256]" = torch.ops.aten.permute.default(mm_156, [1, 0]);  mm_156 = None
    mm_157: "f32[1568, 128]" = torch.ops.aten.mm.default(view_648, permute_403);  view_648 = permute_403 = None
    view_649: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_157, [8, 196, 128]);  mm_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_438: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_436, view_649);  add_436 = view_649 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_404: "f32[256, 128]" = torch.ops.aten.permute.default(permute_402, [1, 0]);  permute_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_650: "f32[1568, 128]" = torch.ops.aten.view.default(add_438, [1568, 128])
    sum_139: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_650, [0])
    sub_289: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_63, unsqueeze_217);  view_63 = unsqueeze_217 = None
    mul_999: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(view_650, sub_289)
    sum_140: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_999, [0]);  mul_999 = None
    mul_1000: "f32[128]" = torch.ops.aten.mul.Tensor(sum_139, 0.0006377551020408163)
    unsqueeze_218: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1000, 0);  mul_1000 = None
    mul_1001: "f32[128]" = torch.ops.aten.mul.Tensor(sum_140, 0.0006377551020408163)
    mul_1002: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_1003: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1001, mul_1002);  mul_1001 = mul_1002 = None
    unsqueeze_219: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1003, 0);  mul_1003 = None
    mul_1004: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_55);  primals_55 = None
    unsqueeze_220: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1004, 0);  mul_1004 = None
    mul_1005: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_289, unsqueeze_219);  sub_289 = unsqueeze_219 = None
    sub_291: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_650, mul_1005);  view_650 = mul_1005 = None
    sub_292: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(sub_291, unsqueeze_218);  sub_291 = unsqueeze_218 = None
    mul_1006: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_292, unsqueeze_220);  sub_292 = unsqueeze_220 = None
    mul_1007: "f32[128]" = torch.ops.aten.mul.Tensor(sum_140, squeeze_40);  sum_140 = squeeze_40 = None
    view_651: "f32[8, 196, 128]" = torch.ops.aten.view.default(mul_1006, [8, 196, 128]);  mul_1006 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_652: "f32[1568, 128]" = torch.ops.aten.view.default(view_651, [1568, 128]);  view_651 = None
    permute_405: "f32[128, 1568]" = torch.ops.aten.permute.default(view_652, [1, 0])
    mm_158: "f32[128, 128]" = torch.ops.aten.mm.default(permute_405, view_61);  permute_405 = view_61 = None
    permute_406: "f32[128, 128]" = torch.ops.aten.permute.default(mm_158, [1, 0]);  mm_158 = None
    mm_159: "f32[1568, 128]" = torch.ops.aten.mm.default(view_652, permute_407);  view_652 = permute_407 = None
    view_653: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_159, [8, 196, 128]);  mm_159 = None
    permute_408: "f32[128, 128]" = torch.ops.aten.permute.default(permute_406, [1, 0]);  permute_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    lt_23: "b8[8, 196, 128]" = torch.ops.aten.lt.Scalar(view_60, -3)
    le_23: "b8[8, 196, 128]" = torch.ops.aten.le.Scalar(view_60, 3)
    div_71: "f32[8, 196, 128]" = torch.ops.aten.div.Tensor(view_60, 3);  view_60 = None
    add_439: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(div_71, 0.5);  div_71 = None
    mul_1008: "f32[8, 196, 128]" = torch.ops.aten.mul.Tensor(view_653, add_439);  add_439 = None
    where_46: "f32[8, 196, 128]" = torch.ops.aten.where.self(le_23, mul_1008, view_653);  le_23 = mul_1008 = view_653 = None
    where_47: "f32[8, 196, 128]" = torch.ops.aten.where.self(lt_23, full_default, where_46);  lt_23 = where_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    view_654: "f32[8, 196, 4, 32]" = torch.ops.aten.view.default(where_47, [8, 196, 4, 32]);  where_47 = None
    permute_409: "f32[8, 4, 196, 32]" = torch.ops.aten.permute.default(view_654, [0, 2, 1, 3]);  view_654 = None
    clone_96: "f32[8, 4, 196, 32]" = torch.ops.aten.clone.default(permute_409, memory_format = torch.contiguous_format);  permute_409 = None
    view_655: "f32[32, 196, 32]" = torch.ops.aten.view.default(clone_96, [32, 196, 32]);  clone_96 = None
    bmm_72: "f32[32, 196, 32]" = torch.ops.aten.bmm.default(permute_410, view_655);  permute_410 = None
    bmm_73: "f32[32, 196, 196]" = torch.ops.aten.bmm.default(view_655, permute_411);  view_655 = permute_411 = None
    view_656: "f32[8, 4, 196, 32]" = torch.ops.aten.view.default(bmm_72, [8, 4, 196, 32]);  bmm_72 = None
    view_657: "f32[8, 4, 196, 196]" = torch.ops.aten.view.default(bmm_73, [8, 4, 196, 196]);  bmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    mul_1009: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(view_657, alias_25);  view_657 = None
    sum_141: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_1009, [-1], True)
    mul_1010: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(alias_25, sum_141);  alias_25 = sum_141 = None
    sub_293: "f32[8, 4, 196, 196]" = torch.ops.aten.sub.Tensor(mul_1009, mul_1010);  mul_1009 = mul_1010 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    sum_142: "f32[1, 4, 196, 196]" = torch.ops.aten.sum.dim_IntList(sub_293, [0], True)
    view_658: "f32[4, 196, 196]" = torch.ops.aten.view.default(sum_142, [4, 196, 196]);  sum_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    index_put_11: "f32[4, 196]" = torch.ops.aten.index_put.default(full_default_48, [None, primals_211], view_658, True);  primals_211 = view_658 = None
    slice_scatter_17: "f32[4, 196]" = torch.ops.aten.slice_scatter.default(full_default_48, index_put_11, 0, 0, 9223372036854775807);  index_put_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    mul_1011: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(sub_293, 0.25);  sub_293 = None
    view_659: "f32[32, 196, 196]" = torch.ops.aten.view.default(mul_1011, [32, 196, 196]);  mul_1011 = None
    bmm_74: "f32[32, 16, 196]" = torch.ops.aten.bmm.default(permute_412, view_659);  permute_412 = None
    bmm_75: "f32[32, 196, 16]" = torch.ops.aten.bmm.default(view_659, permute_413);  view_659 = permute_413 = None
    view_660: "f32[8, 4, 16, 196]" = torch.ops.aten.view.default(bmm_74, [8, 4, 16, 196]);  bmm_74 = None
    view_661: "f32[8, 4, 196, 16]" = torch.ops.aten.view.default(bmm_75, [8, 4, 196, 16]);  bmm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_414: "f32[8, 196, 4, 32]" = torch.ops.aten.permute.default(view_656, [0, 2, 1, 3]);  view_656 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_415: "f32[8, 196, 4, 16]" = torch.ops.aten.permute.default(view_660, [0, 3, 1, 2]);  view_660 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_416: "f32[8, 196, 4, 16]" = torch.ops.aten.permute.default(view_661, [0, 2, 1, 3]);  view_661 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    cat_11: "f32[8, 196, 4, 64]" = torch.ops.aten.cat.default([permute_416, permute_415, permute_414], 3);  permute_416 = permute_415 = permute_414 = None
    view_662: "f32[8, 196, 256]" = torch.ops.aten.view.default(cat_11, [8, 196, 256]);  cat_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_663: "f32[1568, 256]" = torch.ops.aten.view.default(view_662, [1568, 256]);  view_662 = None
    sum_143: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_663, [0])
    sub_294: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_51, unsqueeze_221);  view_51 = unsqueeze_221 = None
    mul_1012: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(view_663, sub_294)
    sum_144: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1012, [0]);  mul_1012 = None
    mul_1013: "f32[256]" = torch.ops.aten.mul.Tensor(sum_143, 0.0006377551020408163)
    unsqueeze_222: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1013, 0);  mul_1013 = None
    mul_1014: "f32[256]" = torch.ops.aten.mul.Tensor(sum_144, 0.0006377551020408163)
    mul_1015: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_1016: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1014, mul_1015);  mul_1014 = mul_1015 = None
    unsqueeze_223: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1016, 0);  mul_1016 = None
    mul_1017: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_52);  primals_52 = None
    unsqueeze_224: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1017, 0);  mul_1017 = None
    mul_1018: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_294, unsqueeze_223);  sub_294 = unsqueeze_223 = None
    sub_296: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_663, mul_1018);  view_663 = mul_1018 = None
    sub_297: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(sub_296, unsqueeze_222);  sub_296 = unsqueeze_222 = None
    mul_1019: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_297, unsqueeze_224);  sub_297 = unsqueeze_224 = None
    mul_1020: "f32[256]" = torch.ops.aten.mul.Tensor(sum_144, squeeze_37);  sum_144 = squeeze_37 = None
    view_664: "f32[8, 196, 256]" = torch.ops.aten.view.default(mul_1019, [8, 196, 256]);  mul_1019 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_665: "f32[1568, 256]" = torch.ops.aten.view.default(view_664, [1568, 256]);  view_664 = None
    permute_417: "f32[256, 1568]" = torch.ops.aten.permute.default(view_665, [1, 0])
    mm_160: "f32[256, 128]" = torch.ops.aten.mm.default(permute_417, view_49);  permute_417 = view_49 = None
    permute_418: "f32[128, 256]" = torch.ops.aten.permute.default(mm_160, [1, 0]);  mm_160 = None
    mm_161: "f32[1568, 128]" = torch.ops.aten.mm.default(view_665, permute_419);  view_665 = permute_419 = None
    view_666: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_161, [8, 196, 128]);  mm_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_440: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_438, view_666);  add_438 = view_666 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_420: "f32[256, 128]" = torch.ops.aten.permute.default(permute_418, [1, 0]);  permute_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_667: "f32[1568, 128]" = torch.ops.aten.view.default(add_440, [1568, 128])
    sum_145: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_667, [0])
    sub_298: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_47, unsqueeze_225);  view_47 = unsqueeze_225 = None
    mul_1021: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(view_667, sub_298)
    sum_146: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1021, [0]);  mul_1021 = None
    mul_1022: "f32[128]" = torch.ops.aten.mul.Tensor(sum_145, 0.0006377551020408163)
    unsqueeze_226: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1022, 0);  mul_1022 = None
    mul_1023: "f32[128]" = torch.ops.aten.mul.Tensor(sum_146, 0.0006377551020408163)
    mul_1024: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_1025: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1023, mul_1024);  mul_1023 = mul_1024 = None
    unsqueeze_227: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1025, 0);  mul_1025 = None
    mul_1026: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_49);  primals_49 = None
    unsqueeze_228: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1026, 0);  mul_1026 = None
    mul_1027: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_298, unsqueeze_227);  sub_298 = unsqueeze_227 = None
    sub_300: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_667, mul_1027);  view_667 = mul_1027 = None
    sub_301: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(sub_300, unsqueeze_226);  sub_300 = unsqueeze_226 = None
    mul_1028: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_301, unsqueeze_228);  sub_301 = unsqueeze_228 = None
    mul_1029: "f32[128]" = torch.ops.aten.mul.Tensor(sum_146, squeeze_34);  sum_146 = squeeze_34 = None
    view_668: "f32[8, 196, 128]" = torch.ops.aten.view.default(mul_1028, [8, 196, 128]);  mul_1028 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_669: "f32[1568, 128]" = torch.ops.aten.view.default(view_668, [1568, 128]);  view_668 = None
    permute_421: "f32[128, 1568]" = torch.ops.aten.permute.default(view_669, [1, 0])
    mm_162: "f32[128, 256]" = torch.ops.aten.mm.default(permute_421, view_45);  permute_421 = view_45 = None
    permute_422: "f32[256, 128]" = torch.ops.aten.permute.default(mm_162, [1, 0]);  mm_162 = None
    mm_163: "f32[1568, 256]" = torch.ops.aten.mm.default(view_669, permute_423);  view_669 = permute_423 = None
    view_670: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_163, [8, 196, 256]);  mm_163 = None
    permute_424: "f32[128, 256]" = torch.ops.aten.permute.default(permute_422, [1, 0]);  permute_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    lt_24: "b8[8, 196, 256]" = torch.ops.aten.lt.Scalar(view_44, -3)
    le_24: "b8[8, 196, 256]" = torch.ops.aten.le.Scalar(view_44, 3)
    div_72: "f32[8, 196, 256]" = torch.ops.aten.div.Tensor(view_44, 3);  view_44 = None
    add_441: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(div_72, 0.5);  div_72 = None
    mul_1030: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_670, add_441);  add_441 = None
    where_48: "f32[8, 196, 256]" = torch.ops.aten.where.self(le_24, mul_1030, view_670);  le_24 = mul_1030 = view_670 = None
    where_49: "f32[8, 196, 256]" = torch.ops.aten.where.self(lt_24, full_default, where_48);  lt_24 = where_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_671: "f32[1568, 256]" = torch.ops.aten.view.default(where_49, [1568, 256]);  where_49 = None
    sum_147: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_671, [0])
    sub_302: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_43, unsqueeze_229);  view_43 = unsqueeze_229 = None
    mul_1031: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(view_671, sub_302)
    sum_148: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1031, [0]);  mul_1031 = None
    mul_1032: "f32[256]" = torch.ops.aten.mul.Tensor(sum_147, 0.0006377551020408163)
    unsqueeze_230: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1032, 0);  mul_1032 = None
    mul_1033: "f32[256]" = torch.ops.aten.mul.Tensor(sum_148, 0.0006377551020408163)
    mul_1034: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_1035: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1033, mul_1034);  mul_1033 = mul_1034 = None
    unsqueeze_231: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1035, 0);  mul_1035 = None
    mul_1036: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_46);  primals_46 = None
    unsqueeze_232: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1036, 0);  mul_1036 = None
    mul_1037: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_302, unsqueeze_231);  sub_302 = unsqueeze_231 = None
    sub_304: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_671, mul_1037);  view_671 = mul_1037 = None
    sub_305: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(sub_304, unsqueeze_230);  sub_304 = unsqueeze_230 = None
    mul_1038: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_305, unsqueeze_232);  sub_305 = unsqueeze_232 = None
    mul_1039: "f32[256]" = torch.ops.aten.mul.Tensor(sum_148, squeeze_31);  sum_148 = squeeze_31 = None
    view_672: "f32[8, 196, 256]" = torch.ops.aten.view.default(mul_1038, [8, 196, 256]);  mul_1038 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_673: "f32[1568, 256]" = torch.ops.aten.view.default(view_672, [1568, 256]);  view_672 = None
    permute_425: "f32[256, 1568]" = torch.ops.aten.permute.default(view_673, [1, 0])
    mm_164: "f32[256, 128]" = torch.ops.aten.mm.default(permute_425, view_41);  permute_425 = view_41 = None
    permute_426: "f32[128, 256]" = torch.ops.aten.permute.default(mm_164, [1, 0]);  mm_164 = None
    mm_165: "f32[1568, 128]" = torch.ops.aten.mm.default(view_673, permute_427);  view_673 = permute_427 = None
    view_674: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_165, [8, 196, 128]);  mm_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_442: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_440, view_674);  add_440 = view_674 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_428: "f32[256, 128]" = torch.ops.aten.permute.default(permute_426, [1, 0]);  permute_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_675: "f32[1568, 128]" = torch.ops.aten.view.default(add_442, [1568, 128])
    sum_149: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_675, [0])
    sub_306: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_39, unsqueeze_233);  view_39 = unsqueeze_233 = None
    mul_1040: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(view_675, sub_306)
    sum_150: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1040, [0]);  mul_1040 = None
    mul_1041: "f32[128]" = torch.ops.aten.mul.Tensor(sum_149, 0.0006377551020408163)
    unsqueeze_234: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1041, 0);  mul_1041 = None
    mul_1042: "f32[128]" = torch.ops.aten.mul.Tensor(sum_150, 0.0006377551020408163)
    mul_1043: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_1044: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1042, mul_1043);  mul_1042 = mul_1043 = None
    unsqueeze_235: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1044, 0);  mul_1044 = None
    mul_1045: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_43);  primals_43 = None
    unsqueeze_236: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1045, 0);  mul_1045 = None
    mul_1046: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_306, unsqueeze_235);  sub_306 = unsqueeze_235 = None
    sub_308: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_675, mul_1046);  view_675 = mul_1046 = None
    sub_309: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(sub_308, unsqueeze_234);  sub_308 = unsqueeze_234 = None
    mul_1047: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_309, unsqueeze_236);  sub_309 = unsqueeze_236 = None
    mul_1048: "f32[128]" = torch.ops.aten.mul.Tensor(sum_150, squeeze_28);  sum_150 = squeeze_28 = None
    view_676: "f32[8, 196, 128]" = torch.ops.aten.view.default(mul_1047, [8, 196, 128]);  mul_1047 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_677: "f32[1568, 128]" = torch.ops.aten.view.default(view_676, [1568, 128]);  view_676 = None
    permute_429: "f32[128, 1568]" = torch.ops.aten.permute.default(view_677, [1, 0])
    mm_166: "f32[128, 128]" = torch.ops.aten.mm.default(permute_429, view_37);  permute_429 = view_37 = None
    permute_430: "f32[128, 128]" = torch.ops.aten.permute.default(mm_166, [1, 0]);  mm_166 = None
    mm_167: "f32[1568, 128]" = torch.ops.aten.mm.default(view_677, permute_431);  view_677 = permute_431 = None
    view_678: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_167, [8, 196, 128]);  mm_167 = None
    permute_432: "f32[128, 128]" = torch.ops.aten.permute.default(permute_430, [1, 0]);  permute_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    lt_25: "b8[8, 196, 128]" = torch.ops.aten.lt.Scalar(view_36, -3)
    le_25: "b8[8, 196, 128]" = torch.ops.aten.le.Scalar(view_36, 3)
    div_73: "f32[8, 196, 128]" = torch.ops.aten.div.Tensor(view_36, 3);  view_36 = None
    add_443: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(div_73, 0.5);  div_73 = None
    mul_1049: "f32[8, 196, 128]" = torch.ops.aten.mul.Tensor(view_678, add_443);  add_443 = None
    where_50: "f32[8, 196, 128]" = torch.ops.aten.where.self(le_25, mul_1049, view_678);  le_25 = mul_1049 = view_678 = None
    where_51: "f32[8, 196, 128]" = torch.ops.aten.where.self(lt_25, full_default, where_50);  lt_25 = where_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    view_679: "f32[8, 196, 4, 32]" = torch.ops.aten.view.default(where_51, [8, 196, 4, 32]);  where_51 = None
    permute_433: "f32[8, 4, 196, 32]" = torch.ops.aten.permute.default(view_679, [0, 2, 1, 3]);  view_679 = None
    clone_97: "f32[8, 4, 196, 32]" = torch.ops.aten.clone.default(permute_433, memory_format = torch.contiguous_format);  permute_433 = None
    view_680: "f32[32, 196, 32]" = torch.ops.aten.view.default(clone_97, [32, 196, 32]);  clone_97 = None
    bmm_76: "f32[32, 196, 32]" = torch.ops.aten.bmm.default(permute_434, view_680);  permute_434 = None
    bmm_77: "f32[32, 196, 196]" = torch.ops.aten.bmm.default(view_680, permute_435);  view_680 = permute_435 = None
    view_681: "f32[8, 4, 196, 32]" = torch.ops.aten.view.default(bmm_76, [8, 4, 196, 32]);  bmm_76 = None
    view_682: "f32[8, 4, 196, 196]" = torch.ops.aten.view.default(bmm_77, [8, 4, 196, 196]);  bmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    mul_1050: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(view_682, alias_26);  view_682 = None
    sum_151: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_1050, [-1], True)
    mul_1051: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(alias_26, sum_151);  alias_26 = sum_151 = None
    sub_310: "f32[8, 4, 196, 196]" = torch.ops.aten.sub.Tensor(mul_1050, mul_1051);  mul_1050 = mul_1051 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    sum_152: "f32[1, 4, 196, 196]" = torch.ops.aten.sum.dim_IntList(sub_310, [0], True)
    view_683: "f32[4, 196, 196]" = torch.ops.aten.view.default(sum_152, [4, 196, 196]);  sum_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    index_put_12: "f32[4, 196]" = torch.ops.aten.index_put.default(full_default_48, [None, primals_210], view_683, True);  primals_210 = view_683 = None
    slice_scatter_18: "f32[4, 196]" = torch.ops.aten.slice_scatter.default(full_default_48, index_put_12, 0, 0, 9223372036854775807);  index_put_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    mul_1052: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(sub_310, 0.25);  sub_310 = None
    view_684: "f32[32, 196, 196]" = torch.ops.aten.view.default(mul_1052, [32, 196, 196]);  mul_1052 = None
    bmm_78: "f32[32, 16, 196]" = torch.ops.aten.bmm.default(permute_436, view_684);  permute_436 = None
    bmm_79: "f32[32, 196, 16]" = torch.ops.aten.bmm.default(view_684, permute_437);  view_684 = permute_437 = None
    view_685: "f32[8, 4, 16, 196]" = torch.ops.aten.view.default(bmm_78, [8, 4, 16, 196]);  bmm_78 = None
    view_686: "f32[8, 4, 196, 16]" = torch.ops.aten.view.default(bmm_79, [8, 4, 196, 16]);  bmm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_438: "f32[8, 196, 4, 32]" = torch.ops.aten.permute.default(view_681, [0, 2, 1, 3]);  view_681 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_439: "f32[8, 196, 4, 16]" = torch.ops.aten.permute.default(view_685, [0, 3, 1, 2]);  view_685 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_440: "f32[8, 196, 4, 16]" = torch.ops.aten.permute.default(view_686, [0, 2, 1, 3]);  view_686 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    cat_12: "f32[8, 196, 4, 64]" = torch.ops.aten.cat.default([permute_440, permute_439, permute_438], 3);  permute_440 = permute_439 = permute_438 = None
    view_687: "f32[8, 196, 256]" = torch.ops.aten.view.default(cat_12, [8, 196, 256]);  cat_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_688: "f32[1568, 256]" = torch.ops.aten.view.default(view_687, [1568, 256]);  view_687 = None
    sum_153: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_688, [0])
    sub_311: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_27, unsqueeze_237);  view_27 = unsqueeze_237 = None
    mul_1053: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(view_688, sub_311)
    sum_154: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1053, [0]);  mul_1053 = None
    mul_1054: "f32[256]" = torch.ops.aten.mul.Tensor(sum_153, 0.0006377551020408163)
    unsqueeze_238: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1054, 0);  mul_1054 = None
    mul_1055: "f32[256]" = torch.ops.aten.mul.Tensor(sum_154, 0.0006377551020408163)
    mul_1056: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_1057: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1055, mul_1056);  mul_1055 = mul_1056 = None
    unsqueeze_239: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1057, 0);  mul_1057 = None
    mul_1058: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_40);  primals_40 = None
    unsqueeze_240: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1058, 0);  mul_1058 = None
    mul_1059: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_311, unsqueeze_239);  sub_311 = unsqueeze_239 = None
    sub_313: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_688, mul_1059);  view_688 = mul_1059 = None
    sub_314: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(sub_313, unsqueeze_238);  sub_313 = unsqueeze_238 = None
    mul_1060: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_314, unsqueeze_240);  sub_314 = unsqueeze_240 = None
    mul_1061: "f32[256]" = torch.ops.aten.mul.Tensor(sum_154, squeeze_25);  sum_154 = squeeze_25 = None
    view_689: "f32[8, 196, 256]" = torch.ops.aten.view.default(mul_1060, [8, 196, 256]);  mul_1060 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_690: "f32[1568, 256]" = torch.ops.aten.view.default(view_689, [1568, 256]);  view_689 = None
    permute_441: "f32[256, 1568]" = torch.ops.aten.permute.default(view_690, [1, 0])
    mm_168: "f32[256, 128]" = torch.ops.aten.mm.default(permute_441, view_25);  permute_441 = view_25 = None
    permute_442: "f32[128, 256]" = torch.ops.aten.permute.default(mm_168, [1, 0]);  mm_168 = None
    mm_169: "f32[1568, 128]" = torch.ops.aten.mm.default(view_690, permute_443);  view_690 = permute_443 = None
    view_691: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_169, [8, 196, 128]);  mm_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_444: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_442, view_691);  add_442 = view_691 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_444: "f32[256, 128]" = torch.ops.aten.permute.default(permute_442, [1, 0]);  permute_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_692: "f32[1568, 128]" = torch.ops.aten.view.default(add_444, [1568, 128])
    sum_155: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_692, [0])
    sub_315: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_23, unsqueeze_241);  view_23 = unsqueeze_241 = None
    mul_1062: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(view_692, sub_315)
    sum_156: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1062, [0]);  mul_1062 = None
    mul_1063: "f32[128]" = torch.ops.aten.mul.Tensor(sum_155, 0.0006377551020408163)
    unsqueeze_242: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1063, 0);  mul_1063 = None
    mul_1064: "f32[128]" = torch.ops.aten.mul.Tensor(sum_156, 0.0006377551020408163)
    mul_1065: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_1066: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1064, mul_1065);  mul_1064 = mul_1065 = None
    unsqueeze_243: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1066, 0);  mul_1066 = None
    mul_1067: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_37);  primals_37 = None
    unsqueeze_244: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1067, 0);  mul_1067 = None
    mul_1068: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_315, unsqueeze_243);  sub_315 = unsqueeze_243 = None
    sub_317: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_692, mul_1068);  view_692 = mul_1068 = None
    sub_318: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(sub_317, unsqueeze_242);  sub_317 = unsqueeze_242 = None
    mul_1069: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_318, unsqueeze_244);  sub_318 = unsqueeze_244 = None
    mul_1070: "f32[128]" = torch.ops.aten.mul.Tensor(sum_156, squeeze_22);  sum_156 = squeeze_22 = None
    view_693: "f32[8, 196, 128]" = torch.ops.aten.view.default(mul_1069, [8, 196, 128]);  mul_1069 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_694: "f32[1568, 128]" = torch.ops.aten.view.default(view_693, [1568, 128]);  view_693 = None
    permute_445: "f32[128, 1568]" = torch.ops.aten.permute.default(view_694, [1, 0])
    mm_170: "f32[128, 256]" = torch.ops.aten.mm.default(permute_445, view_21);  permute_445 = view_21 = None
    permute_446: "f32[256, 128]" = torch.ops.aten.permute.default(mm_170, [1, 0]);  mm_170 = None
    mm_171: "f32[1568, 256]" = torch.ops.aten.mm.default(view_694, permute_447);  view_694 = permute_447 = None
    view_695: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_171, [8, 196, 256]);  mm_171 = None
    permute_448: "f32[128, 256]" = torch.ops.aten.permute.default(permute_446, [1, 0]);  permute_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    lt_26: "b8[8, 196, 256]" = torch.ops.aten.lt.Scalar(view_20, -3)
    le_26: "b8[8, 196, 256]" = torch.ops.aten.le.Scalar(view_20, 3)
    div_74: "f32[8, 196, 256]" = torch.ops.aten.div.Tensor(view_20, 3);  view_20 = None
    add_445: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(div_74, 0.5);  div_74 = None
    mul_1071: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_695, add_445);  add_445 = None
    where_52: "f32[8, 196, 256]" = torch.ops.aten.where.self(le_26, mul_1071, view_695);  le_26 = mul_1071 = view_695 = None
    where_53: "f32[8, 196, 256]" = torch.ops.aten.where.self(lt_26, full_default, where_52);  lt_26 = where_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_696: "f32[1568, 256]" = torch.ops.aten.view.default(where_53, [1568, 256]);  where_53 = None
    sum_157: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_696, [0])
    sub_319: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_19, unsqueeze_245);  view_19 = unsqueeze_245 = None
    mul_1072: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(view_696, sub_319)
    sum_158: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1072, [0]);  mul_1072 = None
    mul_1073: "f32[256]" = torch.ops.aten.mul.Tensor(sum_157, 0.0006377551020408163)
    unsqueeze_246: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1073, 0);  mul_1073 = None
    mul_1074: "f32[256]" = torch.ops.aten.mul.Tensor(sum_158, 0.0006377551020408163)
    mul_1075: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_1076: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1074, mul_1075);  mul_1074 = mul_1075 = None
    unsqueeze_247: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1076, 0);  mul_1076 = None
    mul_1077: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_34);  primals_34 = None
    unsqueeze_248: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1077, 0);  mul_1077 = None
    mul_1078: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_319, unsqueeze_247);  sub_319 = unsqueeze_247 = None
    sub_321: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_696, mul_1078);  view_696 = mul_1078 = None
    sub_322: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(sub_321, unsqueeze_246);  sub_321 = unsqueeze_246 = None
    mul_1079: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_322, unsqueeze_248);  sub_322 = unsqueeze_248 = None
    mul_1080: "f32[256]" = torch.ops.aten.mul.Tensor(sum_158, squeeze_19);  sum_158 = squeeze_19 = None
    view_697: "f32[8, 196, 256]" = torch.ops.aten.view.default(mul_1079, [8, 196, 256]);  mul_1079 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_698: "f32[1568, 256]" = torch.ops.aten.view.default(view_697, [1568, 256]);  view_697 = None
    permute_449: "f32[256, 1568]" = torch.ops.aten.permute.default(view_698, [1, 0])
    mm_172: "f32[256, 128]" = torch.ops.aten.mm.default(permute_449, view_17);  permute_449 = view_17 = None
    permute_450: "f32[128, 256]" = torch.ops.aten.permute.default(mm_172, [1, 0]);  mm_172 = None
    mm_173: "f32[1568, 128]" = torch.ops.aten.mm.default(view_698, permute_451);  view_698 = permute_451 = None
    view_699: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_173, [8, 196, 128]);  mm_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_446: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_444, view_699);  add_444 = view_699 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_452: "f32[256, 128]" = torch.ops.aten.permute.default(permute_450, [1, 0]);  permute_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_700: "f32[1568, 128]" = torch.ops.aten.view.default(add_446, [1568, 128])
    sum_159: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_700, [0])
    sub_323: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_15, unsqueeze_249);  view_15 = unsqueeze_249 = None
    mul_1081: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(view_700, sub_323)
    sum_160: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1081, [0]);  mul_1081 = None
    mul_1082: "f32[128]" = torch.ops.aten.mul.Tensor(sum_159, 0.0006377551020408163)
    unsqueeze_250: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1082, 0);  mul_1082 = None
    mul_1083: "f32[128]" = torch.ops.aten.mul.Tensor(sum_160, 0.0006377551020408163)
    mul_1084: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_1085: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1083, mul_1084);  mul_1083 = mul_1084 = None
    unsqueeze_251: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1085, 0);  mul_1085 = None
    mul_1086: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_31);  primals_31 = None
    unsqueeze_252: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1086, 0);  mul_1086 = None
    mul_1087: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_323, unsqueeze_251);  sub_323 = unsqueeze_251 = None
    sub_325: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_700, mul_1087);  view_700 = mul_1087 = None
    sub_326: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(sub_325, unsqueeze_250);  sub_325 = unsqueeze_250 = None
    mul_1088: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_326, unsqueeze_252);  sub_326 = unsqueeze_252 = None
    mul_1089: "f32[128]" = torch.ops.aten.mul.Tensor(sum_160, squeeze_16);  sum_160 = squeeze_16 = None
    view_701: "f32[8, 196, 128]" = torch.ops.aten.view.default(mul_1088, [8, 196, 128]);  mul_1088 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_702: "f32[1568, 128]" = torch.ops.aten.view.default(view_701, [1568, 128]);  view_701 = None
    permute_453: "f32[128, 1568]" = torch.ops.aten.permute.default(view_702, [1, 0])
    mm_174: "f32[128, 128]" = torch.ops.aten.mm.default(permute_453, view_13);  permute_453 = view_13 = None
    permute_454: "f32[128, 128]" = torch.ops.aten.permute.default(mm_174, [1, 0]);  mm_174 = None
    mm_175: "f32[1568, 128]" = torch.ops.aten.mm.default(view_702, permute_455);  view_702 = permute_455 = None
    view_703: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_175, [8, 196, 128]);  mm_175 = None
    permute_456: "f32[128, 128]" = torch.ops.aten.permute.default(permute_454, [1, 0]);  permute_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    lt_27: "b8[8, 196, 128]" = torch.ops.aten.lt.Scalar(view_12, -3)
    le_27: "b8[8, 196, 128]" = torch.ops.aten.le.Scalar(view_12, 3)
    div_75: "f32[8, 196, 128]" = torch.ops.aten.div.Tensor(view_12, 3);  view_12 = None
    add_447: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(div_75, 0.5);  div_75 = None
    mul_1090: "f32[8, 196, 128]" = torch.ops.aten.mul.Tensor(view_703, add_447);  add_447 = None
    where_54: "f32[8, 196, 128]" = torch.ops.aten.where.self(le_27, mul_1090, view_703);  le_27 = mul_1090 = view_703 = None
    where_55: "f32[8, 196, 128]" = torch.ops.aten.where.self(lt_27, full_default, where_54);  lt_27 = where_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    view_704: "f32[8, 196, 4, 32]" = torch.ops.aten.view.default(where_55, [8, 196, 4, 32]);  where_55 = None
    permute_457: "f32[8, 4, 196, 32]" = torch.ops.aten.permute.default(view_704, [0, 2, 1, 3]);  view_704 = None
    clone_98: "f32[8, 4, 196, 32]" = torch.ops.aten.clone.default(permute_457, memory_format = torch.contiguous_format);  permute_457 = None
    view_705: "f32[32, 196, 32]" = torch.ops.aten.view.default(clone_98, [32, 196, 32]);  clone_98 = None
    bmm_80: "f32[32, 196, 32]" = torch.ops.aten.bmm.default(permute_458, view_705);  permute_458 = None
    bmm_81: "f32[32, 196, 196]" = torch.ops.aten.bmm.default(view_705, permute_459);  view_705 = permute_459 = None
    view_706: "f32[8, 4, 196, 32]" = torch.ops.aten.view.default(bmm_80, [8, 4, 196, 32]);  bmm_80 = None
    view_707: "f32[8, 4, 196, 196]" = torch.ops.aten.view.default(bmm_81, [8, 4, 196, 196]);  bmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    mul_1091: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(view_707, alias_27);  view_707 = None
    sum_161: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_1091, [-1], True)
    mul_1092: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(alias_27, sum_161);  alias_27 = sum_161 = None
    sub_327: "f32[8, 4, 196, 196]" = torch.ops.aten.sub.Tensor(mul_1091, mul_1092);  mul_1091 = mul_1092 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    sum_162: "f32[1, 4, 196, 196]" = torch.ops.aten.sum.dim_IntList(sub_327, [0], True)
    view_708: "f32[4, 196, 196]" = torch.ops.aten.view.default(sum_162, [4, 196, 196]);  sum_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    index_put_13: "f32[4, 196]" = torch.ops.aten.index_put.default(full_default_48, [None, primals_209], view_708, True);  primals_209 = view_708 = None
    slice_scatter_19: "f32[4, 196]" = torch.ops.aten.slice_scatter.default(full_default_48, index_put_13, 0, 0, 9223372036854775807);  full_default_48 = index_put_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    mul_1093: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(sub_327, 0.25);  sub_327 = None
    view_709: "f32[32, 196, 196]" = torch.ops.aten.view.default(mul_1093, [32, 196, 196]);  mul_1093 = None
    bmm_82: "f32[32, 16, 196]" = torch.ops.aten.bmm.default(permute_460, view_709);  permute_460 = None
    bmm_83: "f32[32, 196, 16]" = torch.ops.aten.bmm.default(view_709, permute_461);  view_709 = permute_461 = None
    view_710: "f32[8, 4, 16, 196]" = torch.ops.aten.view.default(bmm_82, [8, 4, 16, 196]);  bmm_82 = None
    view_711: "f32[8, 4, 196, 16]" = torch.ops.aten.view.default(bmm_83, [8, 4, 196, 16]);  bmm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_462: "f32[8, 196, 4, 32]" = torch.ops.aten.permute.default(view_706, [0, 2, 1, 3]);  view_706 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_463: "f32[8, 196, 4, 16]" = torch.ops.aten.permute.default(view_710, [0, 3, 1, 2]);  view_710 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_464: "f32[8, 196, 4, 16]" = torch.ops.aten.permute.default(view_711, [0, 2, 1, 3]);  view_711 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    cat_13: "f32[8, 196, 4, 64]" = torch.ops.aten.cat.default([permute_464, permute_463, permute_462], 3);  permute_464 = permute_463 = permute_462 = None
    view_712: "f32[8, 196, 256]" = torch.ops.aten.view.default(cat_13, [8, 196, 256]);  cat_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_713: "f32[1568, 256]" = torch.ops.aten.view.default(view_712, [1568, 256]);  view_712 = None
    sum_163: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_713, [0])
    sub_328: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_3, unsqueeze_253);  view_3 = unsqueeze_253 = None
    mul_1094: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(view_713, sub_328)
    sum_164: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1094, [0]);  mul_1094 = None
    mul_1095: "f32[256]" = torch.ops.aten.mul.Tensor(sum_163, 0.0006377551020408163)
    unsqueeze_254: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1095, 0);  mul_1095 = None
    mul_1096: "f32[256]" = torch.ops.aten.mul.Tensor(sum_164, 0.0006377551020408163)
    mul_1097: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_1098: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1096, mul_1097);  mul_1096 = mul_1097 = None
    unsqueeze_255: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1098, 0);  mul_1098 = None
    mul_1099: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_28);  primals_28 = None
    unsqueeze_256: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1099, 0);  mul_1099 = None
    mul_1100: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_328, unsqueeze_255);  sub_328 = unsqueeze_255 = None
    sub_330: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_713, mul_1100);  view_713 = mul_1100 = None
    sub_331: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(sub_330, unsqueeze_254);  sub_330 = unsqueeze_254 = None
    mul_1101: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_331, unsqueeze_256);  sub_331 = unsqueeze_256 = None
    mul_1102: "f32[256]" = torch.ops.aten.mul.Tensor(sum_164, squeeze_13);  sum_164 = squeeze_13 = None
    view_714: "f32[8, 196, 256]" = torch.ops.aten.view.default(mul_1101, [8, 196, 256]);  mul_1101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_715: "f32[1568, 256]" = torch.ops.aten.view.default(view_714, [1568, 256]);  view_714 = None
    permute_465: "f32[256, 1568]" = torch.ops.aten.permute.default(view_715, [1, 0])
    mm_176: "f32[256, 128]" = torch.ops.aten.mm.default(permute_465, view_1);  permute_465 = view_1 = None
    permute_466: "f32[128, 256]" = torch.ops.aten.permute.default(mm_176, [1, 0]);  mm_176 = None
    mm_177: "f32[1568, 128]" = torch.ops.aten.mm.default(view_715, permute_467);  view_715 = permute_467 = None
    view_716: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_177, [8, 196, 128]);  mm_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_448: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_446, view_716);  add_446 = view_716 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_468: "f32[256, 128]" = torch.ops.aten.permute.default(permute_466, [1, 0]);  permute_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:639, code: x = x.flatten(2).transpose(1, 2)
    permute_469: "f32[8, 128, 196]" = torch.ops.aten.permute.default(add_448, [0, 2, 1]);  add_448 = None
    view_717: "f32[8, 128, 14, 14]" = torch.ops.aten.view.default(permute_469, [8, 128, 14, 14]);  permute_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:65, code: return self.bn(self.linear(x))
    sum_165: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_717, [0, 2, 3])
    sub_332: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_259);  convolution_3 = unsqueeze_259 = None
    mul_1103: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(view_717, sub_332)
    sum_166: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1103, [0, 2, 3]);  mul_1103 = None
    mul_1104: "f32[128]" = torch.ops.aten.mul.Tensor(sum_165, 0.0006377551020408163)
    unsqueeze_260: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1104, 0);  mul_1104 = None
    unsqueeze_261: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 2);  unsqueeze_260 = None
    unsqueeze_262: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 3);  unsqueeze_261 = None
    mul_1105: "f32[128]" = torch.ops.aten.mul.Tensor(sum_166, 0.0006377551020408163)
    mul_1106: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_1107: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1105, mul_1106);  mul_1105 = mul_1106 = None
    unsqueeze_263: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1107, 0);  mul_1107 = None
    unsqueeze_264: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 2);  unsqueeze_263 = None
    unsqueeze_265: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, 3);  unsqueeze_264 = None
    mul_1108: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_25);  primals_25 = None
    unsqueeze_266: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1108, 0);  mul_1108 = None
    unsqueeze_267: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 2);  unsqueeze_266 = None
    unsqueeze_268: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 3);  unsqueeze_267 = None
    mul_1109: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_332, unsqueeze_265);  sub_332 = unsqueeze_265 = None
    sub_334: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(view_717, mul_1109);  view_717 = mul_1109 = None
    sub_335: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_334, unsqueeze_262);  sub_334 = unsqueeze_262 = None
    mul_1110: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_335, unsqueeze_268);  sub_335 = unsqueeze_268 = None
    mul_1111: "f32[128]" = torch.ops.aten.mul.Tensor(sum_166, squeeze_10);  sum_166 = squeeze_10 = None
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_1110, div_2, primals_24, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1110 = div_2 = primals_24 = None
    getitem_168: "f32[8, 64, 28, 28]" = convolution_backward[0]
    getitem_169: "f32[128, 64, 3, 3]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:637, code: x = self.stem(x)
    lt_28: "b8[8, 64, 28, 28]" = torch.ops.aten.lt.Scalar(add_16, -3)
    le_28: "b8[8, 64, 28, 28]" = torch.ops.aten.le.Scalar(add_16, 3)
    div_76: "f32[8, 64, 28, 28]" = torch.ops.aten.div.Tensor(add_16, 3);  add_16 = None
    add_449: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(div_76, 0.5);  div_76 = None
    mul_1112: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_168, add_449);  add_449 = None
    where_56: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(le_28, mul_1112, getitem_168);  le_28 = mul_1112 = getitem_168 = None
    where_57: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(lt_28, full_default, where_56);  lt_28 = where_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:65, code: return self.bn(self.linear(x))
    sum_167: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_57, [0, 2, 3])
    sub_336: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_271);  convolution_2 = unsqueeze_271 = None
    mul_1113: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(where_57, sub_336)
    sum_168: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1113, [0, 2, 3]);  mul_1113 = None
    mul_1114: "f32[64]" = torch.ops.aten.mul.Tensor(sum_167, 0.00015943877551020407)
    unsqueeze_272: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1114, 0);  mul_1114 = None
    unsqueeze_273: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 2);  unsqueeze_272 = None
    unsqueeze_274: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 3);  unsqueeze_273 = None
    mul_1115: "f32[64]" = torch.ops.aten.mul.Tensor(sum_168, 0.00015943877551020407)
    mul_1116: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_1117: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1115, mul_1116);  mul_1115 = mul_1116 = None
    unsqueeze_275: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1117, 0);  mul_1117 = None
    unsqueeze_276: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 2);  unsqueeze_275 = None
    unsqueeze_277: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 3);  unsqueeze_276 = None
    mul_1118: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_22);  primals_22 = None
    unsqueeze_278: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1118, 0);  mul_1118 = None
    unsqueeze_279: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 2);  unsqueeze_278 = None
    unsqueeze_280: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 3);  unsqueeze_279 = None
    mul_1119: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_336, unsqueeze_277);  sub_336 = unsqueeze_277 = None
    sub_338: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(where_57, mul_1119);  where_57 = mul_1119 = None
    sub_339: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(sub_338, unsqueeze_274);  sub_338 = unsqueeze_274 = None
    mul_1120: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_339, unsqueeze_280);  sub_339 = unsqueeze_280 = None
    mul_1121: "f32[64]" = torch.ops.aten.mul.Tensor(sum_168, squeeze_7);  sum_168 = squeeze_7 = None
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_1120, div_1, primals_21, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1120 = div_1 = primals_21 = None
    getitem_171: "f32[8, 32, 56, 56]" = convolution_backward_1[0]
    getitem_172: "f32[64, 32, 3, 3]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:637, code: x = self.stem(x)
    lt_29: "b8[8, 32, 56, 56]" = torch.ops.aten.lt.Scalar(add_10, -3)
    le_29: "b8[8, 32, 56, 56]" = torch.ops.aten.le.Scalar(add_10, 3)
    div_77: "f32[8, 32, 56, 56]" = torch.ops.aten.div.Tensor(add_10, 3);  add_10 = None
    add_450: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(div_77, 0.5);  div_77 = None
    mul_1122: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_171, add_450);  add_450 = None
    where_58: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(le_29, mul_1122, getitem_171);  le_29 = mul_1122 = getitem_171 = None
    where_59: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(lt_29, full_default, where_58);  lt_29 = where_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:65, code: return self.bn(self.linear(x))
    sum_169: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_59, [0, 2, 3])
    sub_340: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_283);  convolution_1 = unsqueeze_283 = None
    mul_1123: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(where_59, sub_340)
    sum_170: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1123, [0, 2, 3]);  mul_1123 = None
    mul_1124: "f32[32]" = torch.ops.aten.mul.Tensor(sum_169, 3.985969387755102e-05)
    unsqueeze_284: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1124, 0);  mul_1124 = None
    unsqueeze_285: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 2);  unsqueeze_284 = None
    unsqueeze_286: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 3);  unsqueeze_285 = None
    mul_1125: "f32[32]" = torch.ops.aten.mul.Tensor(sum_170, 3.985969387755102e-05)
    mul_1126: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_1127: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1125, mul_1126);  mul_1125 = mul_1126 = None
    unsqueeze_287: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1127, 0);  mul_1127 = None
    unsqueeze_288: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 2);  unsqueeze_287 = None
    unsqueeze_289: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 3);  unsqueeze_288 = None
    mul_1128: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_19);  primals_19 = None
    unsqueeze_290: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1128, 0);  mul_1128 = None
    unsqueeze_291: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 2);  unsqueeze_290 = None
    unsqueeze_292: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 3);  unsqueeze_291 = None
    mul_1129: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_340, unsqueeze_289);  sub_340 = unsqueeze_289 = None
    sub_342: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(where_59, mul_1129);  where_59 = mul_1129 = None
    sub_343: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(sub_342, unsqueeze_286);  sub_342 = unsqueeze_286 = None
    mul_1130: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_343, unsqueeze_292);  sub_343 = unsqueeze_292 = None
    mul_1131: "f32[32]" = torch.ops.aten.mul.Tensor(sum_170, squeeze_4);  sum_170 = squeeze_4 = None
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_1130, div, primals_18, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1130 = div = primals_18 = None
    getitem_174: "f32[8, 16, 112, 112]" = convolution_backward_2[0]
    getitem_175: "f32[32, 16, 3, 3]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:637, code: x = self.stem(x)
    lt_30: "b8[8, 16, 112, 112]" = torch.ops.aten.lt.Scalar(add_4, -3)
    le_30: "b8[8, 16, 112, 112]" = torch.ops.aten.le.Scalar(add_4, 3)
    div_78: "f32[8, 16, 112, 112]" = torch.ops.aten.div.Tensor(add_4, 3);  add_4 = None
    add_451: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(div_78, 0.5);  div_78 = None
    mul_1132: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_174, add_451);  add_451 = None
    where_60: "f32[8, 16, 112, 112]" = torch.ops.aten.where.self(le_30, mul_1132, getitem_174);  le_30 = mul_1132 = getitem_174 = None
    where_61: "f32[8, 16, 112, 112]" = torch.ops.aten.where.self(lt_30, full_default, where_60);  lt_30 = full_default = where_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:65, code: return self.bn(self.linear(x))
    sum_171: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_61, [0, 2, 3])
    sub_344: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_295);  convolution = unsqueeze_295 = None
    mul_1133: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(where_61, sub_344)
    sum_172: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1133, [0, 2, 3]);  mul_1133 = None
    mul_1134: "f32[16]" = torch.ops.aten.mul.Tensor(sum_171, 9.964923469387754e-06)
    unsqueeze_296: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1134, 0);  mul_1134 = None
    unsqueeze_297: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 2);  unsqueeze_296 = None
    unsqueeze_298: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 3);  unsqueeze_297 = None
    mul_1135: "f32[16]" = torch.ops.aten.mul.Tensor(sum_172, 9.964923469387754e-06)
    mul_1136: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_1137: "f32[16]" = torch.ops.aten.mul.Tensor(mul_1135, mul_1136);  mul_1135 = mul_1136 = None
    unsqueeze_299: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1137, 0);  mul_1137 = None
    unsqueeze_300: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 2);  unsqueeze_299 = None
    unsqueeze_301: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 3);  unsqueeze_300 = None
    mul_1138: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_16);  primals_16 = None
    unsqueeze_302: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1138, 0);  mul_1138 = None
    unsqueeze_303: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 2);  unsqueeze_302 = None
    unsqueeze_304: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 3);  unsqueeze_303 = None
    mul_1139: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_344, unsqueeze_301);  sub_344 = unsqueeze_301 = None
    sub_346: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(where_61, mul_1139);  where_61 = mul_1139 = None
    sub_347: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_346, unsqueeze_298);  sub_346 = unsqueeze_298 = None
    mul_1140: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_347, unsqueeze_304);  sub_347 = unsqueeze_304 = None
    mul_1141: "f32[16]" = torch.ops.aten.mul.Tensor(sum_172, squeeze_1);  sum_172 = squeeze_1 = None
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_1140, primals_415, primals_15, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_1140 = primals_415 = primals_15 = None
    getitem_178: "f32[16, 3, 3, 3]" = convolution_backward_3[1];  convolution_backward_3 = None
    return [slice_scatter_19, slice_scatter_18, slice_scatter_17, slice_scatter_16, slice_scatter_12, slice_scatter_11, slice_scatter_10, slice_scatter_9, slice_scatter_8, slice_scatter_4, slice_scatter_3, slice_scatter_2, slice_scatter_1, slice_scatter, getitem_178, mul_1141, sum_171, getitem_175, mul_1131, sum_169, getitem_172, mul_1121, sum_167, getitem_169, mul_1111, sum_165, permute_468, mul_1102, sum_163, permute_456, mul_1089, sum_159, permute_452, mul_1080, sum_157, permute_448, mul_1070, sum_155, permute_444, mul_1061, sum_153, permute_432, mul_1048, sum_149, permute_428, mul_1039, sum_147, permute_424, mul_1029, sum_145, permute_420, mul_1020, sum_143, permute_408, mul_1007, sum_139, permute_404, mul_998, sum_137, permute_400, mul_988, sum_135, permute_396, mul_979, sum_133, permute_384, mul_966, sum_129, permute_380, mul_957, sum_127, permute_376, mul_947, sum_125, permute_372, mul_938, sum_123, permute_366, mul_929, sum_121, permute_356, mul_916, sum_117, permute_352, mul_907, sum_115, permute_348, mul_897, sum_113, permute_344, mul_888, sum_111, permute_332, mul_875, sum_107, permute_328, mul_866, sum_105, permute_324, mul_856, sum_103, permute_320, mul_847, sum_101, permute_308, mul_834, sum_97, permute_304, mul_825, sum_95, permute_300, mul_815, sum_93, permute_296, mul_806, sum_91, permute_284, mul_793, sum_87, permute_280, mul_784, sum_85, permute_276, mul_774, sum_83, permute_272, mul_765, sum_81, permute_260, mul_752, sum_77, permute_256, mul_743, sum_75, permute_252, mul_733, sum_73, permute_248, mul_724, sum_71, permute_242, mul_715, sum_69, permute_232, mul_702, sum_65, permute_228, mul_693, sum_63, permute_224, mul_683, sum_61, permute_220, mul_674, sum_59, permute_208, mul_661, sum_55, permute_204, mul_652, sum_53, permute_200, mul_642, sum_51, permute_196, mul_633, sum_49, permute_184, mul_620, sum_45, permute_180, mul_611, sum_43, permute_176, mul_601, sum_41, permute_172, mul_592, sum_39, permute_160, mul_579, sum_35, permute_156, mul_570, sum_33, permute_152, mul_560, sum_31, permute_148, mul_551, sum_29, permute_136, mul_538, sum_25, permute_132, mul_529, sum_23, permute_128, mul_519, sum_21, mul_510, sum_19, permute_124, view_351, mul_501, sum_16, permute_120, view_351, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    