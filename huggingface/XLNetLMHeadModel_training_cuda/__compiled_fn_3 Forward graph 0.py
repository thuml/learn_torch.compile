from __future__ import annotations



def forward(self, primals_1: "f32[1024, 16, 64]", primals_2: "f32[1024, 16, 64]", primals_3: "f32[1024, 16, 64]", primals_4: "f32[1024, 16, 64]", primals_5: "f32[16, 64]", primals_6: "f32[16, 64]", primals_7: "f32[1024, 16, 64]", primals_8: "f32[1024, 16, 64]", primals_9: "f32[1024, 16, 64]", primals_10: "f32[1024, 16, 64]", primals_11: "f32[1024, 16, 64]", primals_12: "f32[16, 64]", primals_13: "f32[16, 64]", primals_14: "f32[1024, 16, 64]", primals_15: "f32[1024, 16, 64]", primals_16: "f32[1024, 16, 64]", primals_17: "f32[1024, 16, 64]", primals_18: "f32[1024, 16, 64]", primals_19: "f32[16, 64]", primals_20: "f32[16, 64]", primals_21: "f32[1024, 16, 64]", primals_22: "f32[1024, 16, 64]", primals_23: "f32[1024, 16, 64]", primals_24: "f32[1024, 16, 64]", primals_25: "f32[1024, 16, 64]", primals_26: "f32[16, 64]", primals_27: "f32[16, 64]", primals_28: "f32[1024, 16, 64]", primals_29: "f32[1024, 16, 64]", primals_30: "f32[1024, 16, 64]", primals_31: "f32[1024, 16, 64]", primals_32: "f32[1024, 16, 64]", primals_33: "f32[16, 64]", primals_34: "f32[16, 64]", primals_35: "f32[1024, 16, 64]", primals_36: "f32[1024, 16, 64]", primals_37: "f32[1024, 16, 64]", primals_38: "f32[1024, 16, 64]", primals_39: "f32[1024, 16, 64]", primals_40: "f32[16, 64]", primals_41: "f32[16, 64]", primals_42: "f32[1024, 16, 64]", primals_43: "f32[1024, 16, 64]", primals_44: "f32[1024, 16, 64]", primals_45: "f32[1024, 16, 64]", primals_46: "f32[1024, 16, 64]", primals_47: "f32[16, 64]", primals_48: "f32[16, 64]", primals_49: "f32[1024, 16, 64]", primals_50: "f32[1024, 16, 64]", primals_51: "f32[1024, 16, 64]", primals_52: "f32[1024, 16, 64]", primals_53: "f32[1024, 16, 64]", primals_54: "f32[16, 64]", primals_55: "f32[16, 64]", primals_56: "f32[1024, 16, 64]", primals_57: "f32[1024, 16, 64]", primals_58: "f32[1024, 16, 64]", primals_59: "f32[1024, 16, 64]", primals_60: "f32[1024, 16, 64]", primals_61: "f32[16, 64]", primals_62: "f32[16, 64]", primals_63: "f32[1024, 16, 64]", primals_64: "f32[1024, 16, 64]", primals_65: "f32[1024, 16, 64]", primals_66: "f32[1024, 16, 64]", primals_67: "f32[1024, 16, 64]", primals_68: "f32[16, 64]", primals_69: "f32[16, 64]", primals_70: "f32[1024, 16, 64]", primals_71: "f32[1024, 16, 64]", primals_72: "f32[1024, 16, 64]", primals_73: "f32[1024, 16, 64]", primals_74: "f32[1024, 16, 64]", primals_75: "f32[16, 64]", primals_76: "f32[16, 64]", primals_77: "f32[1024, 16, 64]", primals_78: "f32[1024, 16, 64]", primals_79: "f32[1024, 16, 64]", primals_80: "f32[1024, 16, 64]", primals_81: "f32[1024, 16, 64]", primals_82: "f32[16, 64]", primals_83: "f32[16, 64]", primals_84: "f32[1024, 16, 64]", primals_85: "f32[1024, 16, 64]", primals_86: "f32[1024, 16, 64]", primals_87: "f32[1024, 16, 64]", primals_88: "f32[1024, 16, 64]", primals_89: "f32[16, 64]", primals_90: "f32[16, 64]", primals_91: "f32[1024, 16, 64]", primals_92: "f32[1024, 16, 64]", primals_93: "f32[1024, 16, 64]", primals_94: "f32[1024, 16, 64]", primals_95: "f32[1024, 16, 64]", primals_96: "f32[16, 64]", primals_97: "f32[16, 64]", primals_98: "f32[1024, 16, 64]", primals_99: "f32[1024, 16, 64]", primals_100: "f32[1024, 16, 64]", primals_101: "f32[1024, 16, 64]", primals_102: "f32[1024, 16, 64]", primals_103: "f32[16, 64]", primals_104: "f32[16, 64]", primals_105: "f32[1024, 16, 64]", primals_106: "f32[1024, 16, 64]", primals_107: "f32[1024, 16, 64]", primals_108: "f32[1024, 16, 64]", primals_109: "f32[1024, 16, 64]", primals_110: "f32[16, 64]", primals_111: "f32[16, 64]", primals_112: "f32[1024, 16, 64]", primals_113: "f32[1024, 16, 64]", primals_114: "f32[1024, 16, 64]", primals_115: "f32[1024, 16, 64]", primals_116: "f32[1024, 16, 64]", primals_117: "f32[16, 64]", primals_118: "f32[16, 64]", primals_119: "f32[1024, 16, 64]", primals_120: "f32[1024, 16, 64]", primals_121: "f32[1024, 16, 64]", primals_122: "f32[1024, 16, 64]", primals_123: "f32[1024, 16, 64]", primals_124: "f32[16, 64]", primals_125: "f32[16, 64]", primals_126: "f32[1024, 16, 64]", primals_127: "f32[1024, 16, 64]", primals_128: "f32[1024, 16, 64]", primals_129: "f32[1024, 16, 64]", primals_130: "f32[1024, 16, 64]", primals_131: "f32[16, 64]", primals_132: "f32[16, 64]", primals_133: "f32[1024, 16, 64]", primals_134: "f32[1024, 16, 64]", primals_135: "f32[1024, 16, 64]", primals_136: "f32[1024, 16, 64]", primals_137: "f32[1024, 16, 64]", primals_138: "f32[16, 64]", primals_139: "f32[16, 64]", primals_140: "f32[1024, 16, 64]", primals_141: "f32[1024, 16, 64]", primals_142: "f32[1024, 16, 64]", primals_143: "f32[1024, 16, 64]", primals_144: "f32[1024, 16, 64]", primals_145: "f32[16, 64]", primals_146: "f32[16, 64]", primals_147: "f32[1024, 16, 64]", primals_148: "f32[1024, 16, 64]", primals_149: "f32[1024, 16, 64]", primals_150: "f32[1024, 16, 64]", primals_151: "f32[1024, 16, 64]", primals_152: "f32[16, 64]", primals_153: "f32[16, 64]", primals_154: "f32[1024, 16, 64]", primals_155: "f32[1024, 16, 64]", primals_156: "f32[1024, 16, 64]", primals_157: "f32[1024, 16, 64]", primals_158: "f32[1024, 16, 64]", primals_159: "f32[16, 64]", primals_160: "f32[16, 64]", primals_161: "f32[1024, 16, 64]", primals_162: "f32[1024, 16, 64]", primals_163: "f32[1024, 16, 64]", primals_164: "f32[1024, 16, 64]", primals_165: "f32[1024, 16, 64]", primals_166: "f32[16, 64]", primals_167: "f32[16, 64]", primals_168: "f32[1024, 16, 64]", primals_169: "f32[32000, 1024]", primals_170: "f32[1024]", primals_171: "f32[1024]", primals_172: "f32[4096, 1024]", primals_173: "f32[4096]", primals_174: "f32[1024, 4096]", primals_175: "f32[1024]", primals_176: "f32[1024]", primals_177: "f32[1024]", primals_178: "f32[1024]", primals_179: "f32[1024]", primals_180: "f32[4096, 1024]", primals_181: "f32[4096]", primals_182: "f32[1024, 4096]", primals_183: "f32[1024]", primals_184: "f32[1024]", primals_185: "f32[1024]", primals_186: "f32[1024]", primals_187: "f32[1024]", primals_188: "f32[4096, 1024]", primals_189: "f32[4096]", primals_190: "f32[1024, 4096]", primals_191: "f32[1024]", primals_192: "f32[1024]", primals_193: "f32[1024]", primals_194: "f32[1024]", primals_195: "f32[1024]", primals_196: "f32[4096, 1024]", primals_197: "f32[4096]", primals_198: "f32[1024, 4096]", primals_199: "f32[1024]", primals_200: "f32[1024]", primals_201: "f32[1024]", primals_202: "f32[1024]", primals_203: "f32[1024]", primals_204: "f32[4096, 1024]", primals_205: "f32[4096]", primals_206: "f32[1024, 4096]", primals_207: "f32[1024]", primals_208: "f32[1024]", primals_209: "f32[1024]", primals_210: "f32[1024]", primals_211: "f32[1024]", primals_212: "f32[4096, 1024]", primals_213: "f32[4096]", primals_214: "f32[1024, 4096]", primals_215: "f32[1024]", primals_216: "f32[1024]", primals_217: "f32[1024]", primals_218: "f32[1024]", primals_219: "f32[1024]", primals_220: "f32[4096, 1024]", primals_221: "f32[4096]", primals_222: "f32[1024, 4096]", primals_223: "f32[1024]", primals_224: "f32[1024]", primals_225: "f32[1024]", primals_226: "f32[1024]", primals_227: "f32[1024]", primals_228: "f32[4096, 1024]", primals_229: "f32[4096]", primals_230: "f32[1024, 4096]", primals_231: "f32[1024]", primals_232: "f32[1024]", primals_233: "f32[1024]", primals_234: "f32[1024]", primals_235: "f32[1024]", primals_236: "f32[4096, 1024]", primals_237: "f32[4096]", primals_238: "f32[1024, 4096]", primals_239: "f32[1024]", primals_240: "f32[1024]", primals_241: "f32[1024]", primals_242: "f32[1024]", primals_243: "f32[1024]", primals_244: "f32[4096, 1024]", primals_245: "f32[4096]", primals_246: "f32[1024, 4096]", primals_247: "f32[1024]", primals_248: "f32[1024]", primals_249: "f32[1024]", primals_250: "f32[1024]", primals_251: "f32[1024]", primals_252: "f32[4096, 1024]", primals_253: "f32[4096]", primals_254: "f32[1024, 4096]", primals_255: "f32[1024]", primals_256: "f32[1024]", primals_257: "f32[1024]", primals_258: "f32[1024]", primals_259: "f32[1024]", primals_260: "f32[4096, 1024]", primals_261: "f32[4096]", primals_262: "f32[1024, 4096]", primals_263: "f32[1024]", primals_264: "f32[1024]", primals_265: "f32[1024]", primals_266: "f32[1024]", primals_267: "f32[1024]", primals_268: "f32[4096, 1024]", primals_269: "f32[4096]", primals_270: "f32[1024, 4096]", primals_271: "f32[1024]", primals_272: "f32[1024]", primals_273: "f32[1024]", primals_274: "f32[1024]", primals_275: "f32[1024]", primals_276: "f32[4096, 1024]", primals_277: "f32[4096]", primals_278: "f32[1024, 4096]", primals_279: "f32[1024]", primals_280: "f32[1024]", primals_281: "f32[1024]", primals_282: "f32[1024]", primals_283: "f32[1024]", primals_284: "f32[4096, 1024]", primals_285: "f32[4096]", primals_286: "f32[1024, 4096]", primals_287: "f32[1024]", primals_288: "f32[1024]", primals_289: "f32[1024]", primals_290: "f32[1024]", primals_291: "f32[1024]", primals_292: "f32[4096, 1024]", primals_293: "f32[4096]", primals_294: "f32[1024, 4096]", primals_295: "f32[1024]", primals_296: "f32[1024]", primals_297: "f32[1024]", primals_298: "f32[1024]", primals_299: "f32[1024]", primals_300: "f32[4096, 1024]", primals_301: "f32[4096]", primals_302: "f32[1024, 4096]", primals_303: "f32[1024]", primals_304: "f32[1024]", primals_305: "f32[1024]", primals_306: "f32[1024]", primals_307: "f32[1024]", primals_308: "f32[4096, 1024]", primals_309: "f32[4096]", primals_310: "f32[1024, 4096]", primals_311: "f32[1024]", primals_312: "f32[1024]", primals_313: "f32[1024]", primals_314: "f32[1024]", primals_315: "f32[1024]", primals_316: "f32[4096, 1024]", primals_317: "f32[4096]", primals_318: "f32[1024, 4096]", primals_319: "f32[1024]", primals_320: "f32[1024]", primals_321: "f32[1024]", primals_322: "f32[1024]", primals_323: "f32[1024]", primals_324: "f32[4096, 1024]", primals_325: "f32[4096]", primals_326: "f32[1024, 4096]", primals_327: "f32[1024]", primals_328: "f32[1024]", primals_329: "f32[1024]", primals_330: "f32[1024]", primals_331: "f32[1024]", primals_332: "f32[4096, 1024]", primals_333: "f32[4096]", primals_334: "f32[1024, 4096]", primals_335: "f32[1024]", primals_336: "f32[1024]", primals_337: "f32[1024]", primals_338: "f32[1024]", primals_339: "f32[1024]", primals_340: "f32[4096, 1024]", primals_341: "f32[4096]", primals_342: "f32[1024, 4096]", primals_343: "f32[1024]", primals_344: "f32[1024]", primals_345: "f32[1024]", primals_346: "f32[1024]", primals_347: "f32[1024]", primals_348: "f32[4096, 1024]", primals_349: "f32[4096]", primals_350: "f32[1024, 4096]", primals_351: "f32[1024]", primals_352: "f32[1024]", primals_353: "f32[1024]", primals_354: "f32[1024]", primals_355: "f32[1024]", primals_356: "f32[4096, 1024]", primals_357: "f32[4096]", primals_358: "f32[1024, 4096]", primals_359: "f32[1024]", primals_360: "f32[1024]", primals_361: "f32[1024]", primals_362: "f32[32000, 1024]", primals_363: "f32[32000]", primals_364: "i64[1, 512]", primals_365: "i64[1, 512]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1107, code: input_ids = input_ids.transpose(0, 1).contiguous()
    permute: "i64[512, 1]" = torch.ops.aten.permute.default(primals_364, [1, 0]);  primals_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1176, code: word_emb_k = self.word_embedding(input_ids)
    embedding: "f32[512, 1, 1024]" = torch.ops.aten.embedding.default(primals_169, permute);  primals_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1177, code: output_h = self.dropout(word_emb_k)
    native_dropout = torch.ops.aten.native_dropout.default(embedding, 0.1, True);  embedding = None
    getitem: "f32[512, 1, 1024]" = native_dropout[0]
    getitem_1: "b8[512, 1, 1024]" = native_dropout[1];  native_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1023, code: freq_seq = torch.arange(0, self.d_model, 2.0, dtype=torch.float)
    iota: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    convert_element_type: "f64[512]" = torch.ops.prims.convert_element_type.default(iota, torch.float64);  iota = None
    mul: "f64[512]" = torch.ops.aten.mul.Tensor(convert_element_type, 2.0);  convert_element_type = None
    add: "f64[512]" = torch.ops.aten.add.Tensor(mul, 0);  mul = None
    convert_element_type_1: "f32[512]" = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1024, code: inv_freq = 1 / torch.pow(10000, (freq_seq / self.d_model))
    div: "f32[512]" = torch.ops.aten.div.Tensor(convert_element_type_1, 1024);  convert_element_type_1 = None
    pow_1: "f32[512]" = torch.ops.aten.pow.Scalar(10000, div);  div = None
    reciprocal: "f32[512]" = torch.ops.aten.reciprocal.default(pow_1);  pow_1 = None
    mul_1: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1052, code: fwd_pos_seq = torch.arange(beg, end, -1.0)
    iota_1: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    convert_element_type_2: "f64[1024]" = torch.ops.prims.convert_element_type.default(iota_1, torch.float64);  iota_1 = None
    mul_2: "f64[1024]" = torch.ops.aten.mul.Tensor(convert_element_type_2, -1.0);  convert_element_type_2 = None
    add_1: "f64[1024]" = torch.ops.aten.add.Tensor(mul_2, 512);  mul_2 = None
    convert_element_type_3: "f32[1024]" = torch.ops.prims.convert_element_type.default(add_1, torch.float32);  add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1012, code: sinusoid_inp = torch.einsum("i,d->id", pos_seq, inv_freq)
    unsqueeze: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1);  convert_element_type_3 = None
    permute_1: "f32[1024, 1]" = torch.ops.aten.permute.default(unsqueeze, [0, 1]);  unsqueeze = None
    unsqueeze_1: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_1, 1);  mul_1 = None
    permute_2: "f32[1, 512]" = torch.ops.aten.permute.default(unsqueeze_1, [1, 0]);  unsqueeze_1 = None
    mul_3: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(permute_1, permute_2);  permute_1 = permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1013, code: pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
    sin: "f32[1024, 512]" = torch.ops.aten.sin.default(mul_3)
    cos: "f32[1024, 512]" = torch.ops.aten.cos.default(mul_3);  mul_3 = None
    cat: "f32[1024, 1024]" = torch.ops.aten.cat.default([sin, cos], 1);  sin = cos = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1014, code: pos_emb = pos_emb[:, None, :]
    slice_1: "f32[1024, 1024]" = torch.ops.aten.slice.Tensor(cat, 0, 0, 9223372036854775807);  cat = None
    unsqueeze_2: "f32[1024, 1, 1024]" = torch.ops.aten.unsqueeze.default(slice_1, 1);  slice_1 = None
    slice_2: "f32[1024, 1, 1024]" = torch.ops.aten.slice.Tensor(unsqueeze_2, 2, 0, 9223372036854775807);  unsqueeze_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1017, code: pos_emb = pos_emb.expand(-1, bsz, -1)
    expand: "f32[1024, 1, 1024]" = torch.ops.aten.expand.default(slice_2, [-1, 1, -1]);  slice_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1204, code: pos_emb = pos_emb.to(output_h.device)
    device_put: "f32[1024, 1, 1024]" = torch.ops.prims.device_put.default(expand, device(type='cuda', index=0));  expand = None
    convert_element_type_4: "f32[1024, 1, 1024]" = torch.ops.prims.convert_element_type.default(device_put, torch.float32);  device_put = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1205, code: pos_emb = self.dropout(pos_emb)
    native_dropout_1 = torch.ops.aten.native_dropout.default(convert_element_type_4, 0.1, True);  convert_element_type_4 = None
    getitem_2: "f32[1024, 1, 1024]" = native_dropout_1[0];  native_dropout_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_3: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(getitem, 3)
    unsqueeze_4: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 4);  unsqueeze_3 = None
    permute_3: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_4, [0, 1, 3, 4, 2]);  unsqueeze_4 = None
    unsqueeze_5: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_1, 3);  primals_1 = None
    unsqueeze_6: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 4);  unsqueeze_5 = None
    permute_4: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_6, [3, 4, 1, 2, 0]);  unsqueeze_6 = None
    permute_5: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_3, [0, 4, 1, 2, 3]);  permute_3 = None
    view: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_5, [1, 512, 1024]);  permute_5 = None
    permute_6: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_4, [4, 1, 2, 3, 0]);  permute_4 = None
    view_1: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_6, [1, 1024, 1024]);  permute_6 = None
    bmm: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view, view_1)
    view_2: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm, [512, 1, 1, 16, 64]);  bmm = None
    permute_7: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_2, [0, 2, 3, 4, 1]);  view_2 = None
    view_3: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_7, [512, 1, 16, 64]);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_9: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_2, 3);  primals_2 = None
    unsqueeze_10: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 4);  unsqueeze_9 = None
    permute_9: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_10, [3, 4, 1, 2, 0]);  unsqueeze_10 = None
    permute_11: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_9, [4, 1, 2, 3, 0]);  permute_9 = None
    view_5: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_11, [1, 1024, 1024]);  permute_11 = None
    bmm_1: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view, view_5)
    view_6: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_1, [512, 1, 1, 16, 64]);  bmm_1 = None
    permute_12: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_6, [0, 2, 3, 4, 1]);  view_6 = None
    view_7: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_12, [512, 1, 16, 64]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_13: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_3, 3);  primals_3 = None
    unsqueeze_14: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 4);  unsqueeze_13 = None
    permute_14: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_14, [3, 4, 1, 2, 0]);  unsqueeze_14 = None
    permute_16: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_14, [4, 1, 2, 3, 0]);  permute_14 = None
    view_9: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_16, [1, 1024, 1024]);  permute_16 = None
    bmm_2: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view, view_9)
    view_10: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_2, [512, 1, 1, 16, 64]);  bmm_2 = None
    permute_17: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_10, [0, 2, 3, 4, 1]);  view_10 = None
    view_11: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_17, [512, 1, 16, 64]);  permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_15: "f32[1024, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(getitem_2, 3);  getitem_2 = None
    unsqueeze_16: "f32[1024, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_15, 4);  unsqueeze_15 = None
    permute_18: "f32[1024, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_16, [0, 1, 3, 4, 2]);  unsqueeze_16 = None
    unsqueeze_17: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_4, 3);  primals_4 = None
    unsqueeze_18: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_17, 4);  unsqueeze_17 = None
    permute_19: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_18, [3, 4, 1, 2, 0]);  unsqueeze_18 = None
    permute_20: "f32[1024, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_18, [0, 4, 1, 2, 3]);  permute_18 = None
    view_12: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_20, [1, 1024, 1024]);  permute_20 = None
    permute_21: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_19, [4, 1, 2, 3, 0]);  permute_19 = None
    view_13: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_21, [1, 1024, 1024]);  permute_21 = None
    bmm_3: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_12, view_13);  view_13 = None
    view_14: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_3, [1024, 1, 1, 16, 64]);  bmm_3 = None
    permute_22: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_14, [0, 2, 3, 4, 1]);  view_14 = None
    view_15: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_22, [1024, 1, 16, 64]);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_2: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_3, primals_5);  primals_5 = None
    unsqueeze_19: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_2, 4);  add_2 = None
    permute_23: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_19, [1, 2, 0, 4, 3]);  unsqueeze_19 = None
    unsqueeze_20: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_7, 4);  view_7 = None
    permute_24: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_20, [1, 2, 4, 0, 3]);  unsqueeze_20 = None
    permute_25: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_23, [1, 2, 4, 0, 3]);  permute_23 = None
    view_16: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_25, [16, 512, 64]);  permute_25 = None
    permute_26: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_24, [1, 4, 0, 3, 2]);  permute_24 = None
    view_17: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_26, [16, 64, 512]);  permute_26 = None
    bmm_4: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_16, view_17)
    view_18: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_4, [16, 512, 1, 1, 512]);  bmm_4 = None
    permute_27: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_18, [3, 0, 1, 4, 2]);  view_18 = None
    view_19: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_27, [1, 16, 512, 512]);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_3: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_3, primals_6);  view_3 = primals_6 = None
    unsqueeze_21: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_3, 4);  add_3 = None
    permute_28: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_21, [1, 2, 0, 4, 3]);  unsqueeze_21 = None
    unsqueeze_22: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_15, 4);  view_15 = None
    permute_29: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_22, [1, 2, 4, 0, 3]);  unsqueeze_22 = None
    permute_30: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_28, [1, 2, 4, 0, 3]);  permute_28 = None
    view_20: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_30, [16, 512, 64]);  permute_30 = None
    permute_31: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_29, [1, 4, 0, 3, 2]);  permute_29 = None
    view_21: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_31, [16, 64, 1024]);  permute_31 = None
    bmm_5: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_20, view_21)
    view_22: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_5, [16, 512, 1, 1, 1024]);  bmm_5 = None
    permute_32: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_22, [3, 0, 1, 4, 2]);  view_22 = None
    view_23: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_32, [1, 16, 512, 1024]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_24: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_23, [1, 16, 1024, 512]);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_3: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_24, 0, 0, 9223372036854775807);  view_24 = None
    slice_4: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_3, 1, 0, 9223372036854775807);  slice_3 = None
    slice_5: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_4, 2, 1, 9223372036854775807);  slice_4 = None
    slice_6: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_5, 3, 0, 9223372036854775807);  slice_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_25: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_6, [1, 16, 512, 1023]);  slice_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    iota_2: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    slice_7: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_25, 0, 0, 9223372036854775807);  view_25 = None
    slice_8: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_7, 1, 0, 9223372036854775807);  slice_7 = None
    slice_9: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_8, 2, 0, 9223372036854775807);  slice_8 = None
    index: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_9, [None, None, None, iota_2]);  slice_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_4: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_19, index);  view_19 = index = None
    add_5: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_4, 0);  add_4 = None
    mul_4: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_5, 0.125);  add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_4, [3], True)
    sub: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_4, amax);  mul_4 = amax = None
    exp: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub);  sub = None
    sum_1: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [3], True)
    div_1: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    native_dropout_2 = torch.ops.aten.native_dropout.default(div_1, 0.1, True);  div_1 = None
    getitem_4: "f32[1, 16, 512, 512]" = native_dropout_2[0]
    getitem_5: "b8[1, 16, 512, 512]" = native_dropout_2[1];  native_dropout_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_23: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(getitem_4, 4);  getitem_4 = None
    permute_33: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_23, [2, 0, 1, 4, 3]);  unsqueeze_23 = None
    unsqueeze_24: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_11, 4);  view_11 = None
    permute_34: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_24, [4, 1, 2, 3, 0]);  unsqueeze_24 = None
    permute_35: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_33, [2, 0, 4, 1, 3]);  permute_33 = None
    view_26: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_35, [16, 512, 512]);  permute_35 = None
    permute_36: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_34, [2, 4, 1, 3, 0]);  permute_34 = None
    view_27: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_36, [16, 512, 64]);  permute_36 = None
    bmm_6: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_26, view_27)
    view_28: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_6, [16, 512, 1, 1, 64]);  bmm_6 = None
    permute_37: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_28, [1, 3, 0, 4, 2]);  view_28 = None
    view_29: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_37, [512, 1, 16, 64]);  permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_25: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_29, 4);  view_29 = None
    permute_38: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_25, [0, 1, 4, 3, 2]);  unsqueeze_25 = None
    unsqueeze_26: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_7, 3);  primals_7 = None
    unsqueeze_27: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, 4);  unsqueeze_26 = None
    permute_39: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_27, [3, 4, 0, 2, 1]);  unsqueeze_27 = None
    permute_40: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_38, [0, 3, 4, 1, 2]);  permute_38 = None
    clone: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
    view_30: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone, [1, 512, 1024]);  clone = None
    permute_41: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_39, [3, 4, 1, 2, 0]);  permute_39 = None
    clone_1: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None
    view_31: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_1, [1, 1024, 1024]);  clone_1 = None
    bmm_7: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_30, view_31)
    view_32: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_7, [512, 1, 1, 1, 1024]);  bmm_7 = None
    permute_42: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_32, [0, 3, 4, 1, 2]);  view_32 = None
    view_33: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_42, [512, 1, 1024]);  permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    native_dropout_3 = torch.ops.aten.native_dropout.default(view_33, 0.1, True);  view_33 = None
    getitem_6: "f32[512, 1, 1024]" = native_dropout_3[0]
    getitem_7: "b8[512, 1, 1024]" = native_dropout_3[1];  native_dropout_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_6: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_6, getitem);  getitem_6 = getitem = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean = torch.ops.aten.var_mean.correction(add_6, [2], correction = 0, keepdim = True)
    getitem_8: "f32[512, 1, 1]" = var_mean[0]
    getitem_9: "f32[512, 1, 1]" = var_mean[1];  var_mean = None
    add_7: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
    rsqrt: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    sub_1: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_6, getitem_9);  add_6 = getitem_9 = None
    mul_5: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
    mul_6: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_5, primals_170)
    add_8: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_6, primals_171);  mul_6 = primals_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_34: "f32[512, 1024]" = torch.ops.aten.view.default(add_8, [512, 1024])
    permute_43: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_172, [1, 0]);  primals_172 = None
    addmm: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_173, view_34, permute_43);  primals_173 = None
    view_35: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm, [512, 1, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_7: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_35, 0.5)
    mul_8: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_35, 0.7071067811865476);  view_35 = None
    erf: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
    add_9: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_9: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_7, add_9);  mul_7 = add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    native_dropout_4 = torch.ops.aten.native_dropout.default(mul_9, 0.1, True);  mul_9 = None
    getitem_10: "f32[512, 1, 4096]" = native_dropout_4[0]
    getitem_11: "b8[512, 1, 4096]" = native_dropout_4[1];  native_dropout_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_36: "f32[512, 4096]" = torch.ops.aten.view.default(getitem_10, [512, 4096]);  getitem_10 = None
    permute_44: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_174, [1, 0]);  primals_174 = None
    addmm_1: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_175, view_36, permute_44);  primals_175 = None
    view_37: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_1, [512, 1, 1024]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    native_dropout_5 = torch.ops.aten.native_dropout.default(view_37, 0.1, True);  view_37 = None
    getitem_12: "f32[512, 1, 1024]" = native_dropout_5[0]
    getitem_13: "b8[512, 1, 1024]" = native_dropout_5[1];  native_dropout_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_10: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_12, add_8);  getitem_12 = add_8 = None
    var_mean_1 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
    getitem_14: "f32[512, 1, 1]" = var_mean_1[0]
    getitem_15: "f32[512, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_11: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
    rsqrt_1: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_2: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_10, getitem_15);  add_10 = getitem_15 = None
    mul_10: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
    mul_11: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_10, primals_176)
    add_12: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_11, primals_177);  mul_11 = primals_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_28: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_12, 3)
    unsqueeze_29: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, 4);  unsqueeze_28 = None
    permute_45: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_29, [0, 1, 3, 4, 2]);  unsqueeze_29 = None
    unsqueeze_30: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_8, 3);  primals_8 = None
    unsqueeze_31: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, 4);  unsqueeze_30 = None
    permute_46: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_31, [3, 4, 1, 2, 0]);  unsqueeze_31 = None
    permute_47: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_45, [0, 4, 1, 2, 3]);  permute_45 = None
    view_38: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_47, [1, 512, 1024]);  permute_47 = None
    permute_48: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_46, [4, 1, 2, 3, 0]);  permute_46 = None
    view_39: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_48, [1, 1024, 1024]);  permute_48 = None
    bmm_8: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_38, view_39)
    view_40: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_8, [512, 1, 1, 16, 64]);  bmm_8 = None
    permute_49: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_40, [0, 2, 3, 4, 1]);  view_40 = None
    view_41: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_49, [512, 1, 16, 64]);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_34: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_9, 3);  primals_9 = None
    unsqueeze_35: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, 4);  unsqueeze_34 = None
    permute_51: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_35, [3, 4, 1, 2, 0]);  unsqueeze_35 = None
    permute_53: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_51, [4, 1, 2, 3, 0]);  permute_51 = None
    view_43: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_53, [1, 1024, 1024]);  permute_53 = None
    bmm_9: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_38, view_43)
    view_44: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_9, [512, 1, 1, 16, 64]);  bmm_9 = None
    permute_54: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_44, [0, 2, 3, 4, 1]);  view_44 = None
    view_45: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_54, [512, 1, 16, 64]);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_38: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_10, 3);  primals_10 = None
    unsqueeze_39: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, 4);  unsqueeze_38 = None
    permute_56: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_39, [3, 4, 1, 2, 0]);  unsqueeze_39 = None
    permute_58: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_56, [4, 1, 2, 3, 0]);  permute_56 = None
    view_47: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_58, [1, 1024, 1024]);  permute_58 = None
    bmm_10: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_38, view_47)
    view_48: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_10, [512, 1, 1, 16, 64]);  bmm_10 = None
    permute_59: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_48, [0, 2, 3, 4, 1]);  view_48 = None
    view_49: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_59, [512, 1, 16, 64]);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_42: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_11, 3);  primals_11 = None
    unsqueeze_43: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, 4);  unsqueeze_42 = None
    permute_61: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_43, [3, 4, 1, 2, 0]);  unsqueeze_43 = None
    permute_63: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_61, [4, 1, 2, 3, 0]);  permute_61 = None
    view_51: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_63, [1, 1024, 1024]);  permute_63 = None
    bmm_11: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_12, view_51);  view_51 = None
    view_52: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_11, [1024, 1, 1, 16, 64]);  bmm_11 = None
    permute_64: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_52, [0, 2, 3, 4, 1]);  view_52 = None
    view_53: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_64, [1024, 1, 16, 64]);  permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_13: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_41, primals_12);  primals_12 = None
    unsqueeze_44: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_13, 4);  add_13 = None
    permute_65: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_44, [1, 2, 0, 4, 3]);  unsqueeze_44 = None
    unsqueeze_45: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_45, 4);  view_45 = None
    permute_66: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_45, [1, 2, 4, 0, 3]);  unsqueeze_45 = None
    permute_67: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_65, [1, 2, 4, 0, 3]);  permute_65 = None
    view_54: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_67, [16, 512, 64]);  permute_67 = None
    permute_68: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_66, [1, 4, 0, 3, 2]);  permute_66 = None
    view_55: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_68, [16, 64, 512]);  permute_68 = None
    bmm_12: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_54, view_55)
    view_56: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_12, [16, 512, 1, 1, 512]);  bmm_12 = None
    permute_69: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_56, [3, 0, 1, 4, 2]);  view_56 = None
    view_57: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_69, [1, 16, 512, 512]);  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_14: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_41, primals_13);  view_41 = primals_13 = None
    unsqueeze_46: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_14, 4);  add_14 = None
    permute_70: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_46, [1, 2, 0, 4, 3]);  unsqueeze_46 = None
    unsqueeze_47: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_53, 4);  view_53 = None
    permute_71: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_47, [1, 2, 4, 0, 3]);  unsqueeze_47 = None
    permute_72: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_70, [1, 2, 4, 0, 3]);  permute_70 = None
    view_58: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_72, [16, 512, 64]);  permute_72 = None
    permute_73: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_71, [1, 4, 0, 3, 2]);  permute_71 = None
    view_59: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_73, [16, 64, 1024]);  permute_73 = None
    bmm_13: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_58, view_59)
    view_60: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_13, [16, 512, 1, 1, 1024]);  bmm_13 = None
    permute_74: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_60, [3, 0, 1, 4, 2]);  view_60 = None
    view_61: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_74, [1, 16, 512, 1024]);  permute_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_62: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_61, [1, 16, 1024, 512]);  view_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_10: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_62, 0, 0, 9223372036854775807);  view_62 = None
    slice_11: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_10, 1, 0, 9223372036854775807);  slice_10 = None
    slice_12: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_11, 2, 1, 9223372036854775807);  slice_11 = None
    slice_13: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_12, 3, 0, 9223372036854775807);  slice_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_63: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_13, [1, 16, 512, 1023]);  slice_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    slice_14: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_63, 0, 0, 9223372036854775807);  view_63 = None
    slice_15: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_14, 1, 0, 9223372036854775807);  slice_14 = None
    slice_16: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_15, 2, 0, 9223372036854775807);  slice_15 = None
    index_1: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_16, [None, None, None, iota_2]);  slice_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_15: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_57, index_1);  view_57 = index_1 = None
    add_16: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_15, 0);  add_15 = None
    mul_12: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_16, 0.125);  add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_1: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_12, [3], True)
    sub_3: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_12, amax_1);  mul_12 = amax_1 = None
    exp_1: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
    sum_2: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [3], True)
    div_2: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_1: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    native_dropout_6 = torch.ops.aten.native_dropout.default(div_2, 0.1, True);  div_2 = None
    getitem_16: "f32[1, 16, 512, 512]" = native_dropout_6[0]
    getitem_17: "b8[1, 16, 512, 512]" = native_dropout_6[1];  native_dropout_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_48: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(getitem_16, 4);  getitem_16 = None
    permute_75: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_48, [2, 0, 1, 4, 3]);  unsqueeze_48 = None
    unsqueeze_49: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_49, 4);  view_49 = None
    permute_76: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_49, [4, 1, 2, 3, 0]);  unsqueeze_49 = None
    permute_77: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_75, [2, 0, 4, 1, 3]);  permute_75 = None
    view_64: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_77, [16, 512, 512]);  permute_77 = None
    permute_78: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_76, [2, 4, 1, 3, 0]);  permute_76 = None
    view_65: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_78, [16, 512, 64]);  permute_78 = None
    bmm_14: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_64, view_65)
    view_66: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_14, [16, 512, 1, 1, 64]);  bmm_14 = None
    permute_79: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_66, [1, 3, 0, 4, 2]);  view_66 = None
    view_67: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_79, [512, 1, 16, 64]);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_50: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_67, 4);  view_67 = None
    permute_80: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_50, [0, 1, 4, 3, 2]);  unsqueeze_50 = None
    unsqueeze_51: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_14, 3);  primals_14 = None
    unsqueeze_52: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_51, 4);  unsqueeze_51 = None
    permute_81: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_52, [3, 4, 0, 2, 1]);  unsqueeze_52 = None
    permute_82: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_80, [0, 3, 4, 1, 2]);  permute_80 = None
    clone_2: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
    view_68: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_2, [1, 512, 1024]);  clone_2 = None
    permute_83: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_81, [3, 4, 1, 2, 0]);  permute_81 = None
    clone_3: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_83, memory_format = torch.contiguous_format);  permute_83 = None
    view_69: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_3, [1, 1024, 1024]);  clone_3 = None
    bmm_15: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_68, view_69)
    view_70: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_15, [512, 1, 1, 1, 1024]);  bmm_15 = None
    permute_84: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_70, [0, 3, 4, 1, 2]);  view_70 = None
    view_71: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_84, [512, 1, 1024]);  permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    native_dropout_7 = torch.ops.aten.native_dropout.default(view_71, 0.1, True);  view_71 = None
    getitem_18: "f32[512, 1, 1024]" = native_dropout_7[0]
    getitem_19: "b8[512, 1, 1024]" = native_dropout_7[1];  native_dropout_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_17: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_18, add_12);  getitem_18 = add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
    getitem_20: "f32[512, 1, 1]" = var_mean_2[0]
    getitem_21: "f32[512, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_18: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
    rsqrt_2: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_4: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_17, getitem_21);  add_17 = getitem_21 = None
    mul_13: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = None
    mul_14: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_13, primals_178)
    add_19: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_14, primals_179);  mul_14 = primals_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_72: "f32[512, 1024]" = torch.ops.aten.view.default(add_19, [512, 1024])
    permute_85: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_180, [1, 0]);  primals_180 = None
    addmm_2: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_181, view_72, permute_85);  primals_181 = None
    view_73: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_2, [512, 1, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_15: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_73, 0.5)
    mul_16: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_73, 0.7071067811865476);  view_73 = None
    erf_1: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_16);  mul_16 = None
    add_20: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_17: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_15, add_20);  mul_15 = add_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    native_dropout_8 = torch.ops.aten.native_dropout.default(mul_17, 0.1, True);  mul_17 = None
    getitem_22: "f32[512, 1, 4096]" = native_dropout_8[0]
    getitem_23: "b8[512, 1, 4096]" = native_dropout_8[1];  native_dropout_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_74: "f32[512, 4096]" = torch.ops.aten.view.default(getitem_22, [512, 4096]);  getitem_22 = None
    permute_86: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_182, [1, 0]);  primals_182 = None
    addmm_3: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_183, view_74, permute_86);  primals_183 = None
    view_75: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_3, [512, 1, 1024]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    native_dropout_9 = torch.ops.aten.native_dropout.default(view_75, 0.1, True);  view_75 = None
    getitem_24: "f32[512, 1, 1024]" = native_dropout_9[0]
    getitem_25: "b8[512, 1, 1024]" = native_dropout_9[1];  native_dropout_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_21: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_24, add_19);  getitem_24 = add_19 = None
    var_mean_3 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
    getitem_26: "f32[512, 1, 1]" = var_mean_3[0]
    getitem_27: "f32[512, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_22: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
    rsqrt_3: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
    sub_5: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_21, getitem_27);  add_21 = getitem_27 = None
    mul_18: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
    mul_19: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_18, primals_184)
    add_23: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_19, primals_185);  mul_19 = primals_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_53: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_23, 3)
    unsqueeze_54: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_53, 4);  unsqueeze_53 = None
    permute_87: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_54, [0, 1, 3, 4, 2]);  unsqueeze_54 = None
    unsqueeze_55: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_15, 3);  primals_15 = None
    unsqueeze_56: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_55, 4);  unsqueeze_55 = None
    permute_88: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_56, [3, 4, 1, 2, 0]);  unsqueeze_56 = None
    permute_89: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_87, [0, 4, 1, 2, 3]);  permute_87 = None
    view_76: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_89, [1, 512, 1024]);  permute_89 = None
    permute_90: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_88, [4, 1, 2, 3, 0]);  permute_88 = None
    view_77: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_90, [1, 1024, 1024]);  permute_90 = None
    bmm_16: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_76, view_77)
    view_78: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_16, [512, 1, 1, 16, 64]);  bmm_16 = None
    permute_91: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_78, [0, 2, 3, 4, 1]);  view_78 = None
    view_79: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_91, [512, 1, 16, 64]);  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_59: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_16, 3);  primals_16 = None
    unsqueeze_60: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_59, 4);  unsqueeze_59 = None
    permute_93: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_60, [3, 4, 1, 2, 0]);  unsqueeze_60 = None
    permute_95: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_93, [4, 1, 2, 3, 0]);  permute_93 = None
    view_81: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_95, [1, 1024, 1024]);  permute_95 = None
    bmm_17: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_76, view_81)
    view_82: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_17, [512, 1, 1, 16, 64]);  bmm_17 = None
    permute_96: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_82, [0, 2, 3, 4, 1]);  view_82 = None
    view_83: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_96, [512, 1, 16, 64]);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_63: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_17, 3);  primals_17 = None
    unsqueeze_64: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_63, 4);  unsqueeze_63 = None
    permute_98: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_64, [3, 4, 1, 2, 0]);  unsqueeze_64 = None
    permute_100: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_98, [4, 1, 2, 3, 0]);  permute_98 = None
    view_85: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_100, [1, 1024, 1024]);  permute_100 = None
    bmm_18: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_76, view_85)
    view_86: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_18, [512, 1, 1, 16, 64]);  bmm_18 = None
    permute_101: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_86, [0, 2, 3, 4, 1]);  view_86 = None
    view_87: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_101, [512, 1, 16, 64]);  permute_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_67: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_18, 3);  primals_18 = None
    unsqueeze_68: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_67, 4);  unsqueeze_67 = None
    permute_103: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_68, [3, 4, 1, 2, 0]);  unsqueeze_68 = None
    permute_105: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_103, [4, 1, 2, 3, 0]);  permute_103 = None
    view_89: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_105, [1, 1024, 1024]);  permute_105 = None
    bmm_19: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_12, view_89);  view_89 = None
    view_90: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_19, [1024, 1, 1, 16, 64]);  bmm_19 = None
    permute_106: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_90, [0, 2, 3, 4, 1]);  view_90 = None
    view_91: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_106, [1024, 1, 16, 64]);  permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_24: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_79, primals_19);  primals_19 = None
    unsqueeze_69: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_24, 4);  add_24 = None
    permute_107: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_69, [1, 2, 0, 4, 3]);  unsqueeze_69 = None
    unsqueeze_70: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_83, 4);  view_83 = None
    permute_108: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_70, [1, 2, 4, 0, 3]);  unsqueeze_70 = None
    permute_109: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_107, [1, 2, 4, 0, 3]);  permute_107 = None
    view_92: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_109, [16, 512, 64]);  permute_109 = None
    permute_110: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_108, [1, 4, 0, 3, 2]);  permute_108 = None
    view_93: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_110, [16, 64, 512]);  permute_110 = None
    bmm_20: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_92, view_93)
    view_94: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_20, [16, 512, 1, 1, 512]);  bmm_20 = None
    permute_111: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_94, [3, 0, 1, 4, 2]);  view_94 = None
    view_95: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_111, [1, 16, 512, 512]);  permute_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_25: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_79, primals_20);  view_79 = primals_20 = None
    unsqueeze_71: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_25, 4);  add_25 = None
    permute_112: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_71, [1, 2, 0, 4, 3]);  unsqueeze_71 = None
    unsqueeze_72: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_91, 4);  view_91 = None
    permute_113: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_72, [1, 2, 4, 0, 3]);  unsqueeze_72 = None
    permute_114: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_112, [1, 2, 4, 0, 3]);  permute_112 = None
    view_96: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_114, [16, 512, 64]);  permute_114 = None
    permute_115: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_113, [1, 4, 0, 3, 2]);  permute_113 = None
    view_97: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_115, [16, 64, 1024]);  permute_115 = None
    bmm_21: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_96, view_97)
    view_98: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_21, [16, 512, 1, 1, 1024]);  bmm_21 = None
    permute_116: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_98, [3, 0, 1, 4, 2]);  view_98 = None
    view_99: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_116, [1, 16, 512, 1024]);  permute_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_100: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_99, [1, 16, 1024, 512]);  view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_17: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_100, 0, 0, 9223372036854775807);  view_100 = None
    slice_18: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_17, 1, 0, 9223372036854775807);  slice_17 = None
    slice_19: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_18, 2, 1, 9223372036854775807);  slice_18 = None
    slice_20: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_19, 3, 0, 9223372036854775807);  slice_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_101: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_20, [1, 16, 512, 1023]);  slice_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    slice_21: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_101, 0, 0, 9223372036854775807);  view_101 = None
    slice_22: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_21, 1, 0, 9223372036854775807);  slice_21 = None
    slice_23: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_22, 2, 0, 9223372036854775807);  slice_22 = None
    index_2: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_23, [None, None, None, iota_2]);  slice_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_26: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_95, index_2);  view_95 = index_2 = None
    add_27: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_26, 0);  add_26 = None
    mul_20: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_27, 0.125);  add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_2: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_20, [3], True)
    sub_6: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_20, amax_2);  mul_20 = amax_2 = None
    exp_2: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
    sum_3: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [3], True)
    div_3: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_2: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    native_dropout_10 = torch.ops.aten.native_dropout.default(div_3, 0.1, True);  div_3 = None
    getitem_28: "f32[1, 16, 512, 512]" = native_dropout_10[0]
    getitem_29: "b8[1, 16, 512, 512]" = native_dropout_10[1];  native_dropout_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_73: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(getitem_28, 4);  getitem_28 = None
    permute_117: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_73, [2, 0, 1, 4, 3]);  unsqueeze_73 = None
    unsqueeze_74: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_87, 4);  view_87 = None
    permute_118: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_74, [4, 1, 2, 3, 0]);  unsqueeze_74 = None
    permute_119: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_117, [2, 0, 4, 1, 3]);  permute_117 = None
    view_102: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_119, [16, 512, 512]);  permute_119 = None
    permute_120: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_118, [2, 4, 1, 3, 0]);  permute_118 = None
    view_103: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_120, [16, 512, 64]);  permute_120 = None
    bmm_22: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_102, view_103)
    view_104: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_22, [16, 512, 1, 1, 64]);  bmm_22 = None
    permute_121: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_104, [1, 3, 0, 4, 2]);  view_104 = None
    view_105: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_121, [512, 1, 16, 64]);  permute_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_75: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_105, 4);  view_105 = None
    permute_122: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_75, [0, 1, 4, 3, 2]);  unsqueeze_75 = None
    unsqueeze_76: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_21, 3);  primals_21 = None
    unsqueeze_77: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, 4);  unsqueeze_76 = None
    permute_123: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_77, [3, 4, 0, 2, 1]);  unsqueeze_77 = None
    permute_124: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_122, [0, 3, 4, 1, 2]);  permute_122 = None
    clone_4: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_124, memory_format = torch.contiguous_format);  permute_124 = None
    view_106: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_4, [1, 512, 1024]);  clone_4 = None
    permute_125: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_123, [3, 4, 1, 2, 0]);  permute_123 = None
    clone_5: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
    view_107: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_5, [1, 1024, 1024]);  clone_5 = None
    bmm_23: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_106, view_107)
    view_108: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_23, [512, 1, 1, 1, 1024]);  bmm_23 = None
    permute_126: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_108, [0, 3, 4, 1, 2]);  view_108 = None
    view_109: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_126, [512, 1, 1024]);  permute_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    native_dropout_11 = torch.ops.aten.native_dropout.default(view_109, 0.1, True);  view_109 = None
    getitem_30: "f32[512, 1, 1024]" = native_dropout_11[0]
    getitem_31: "b8[512, 1, 1024]" = native_dropout_11[1];  native_dropout_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_28: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_30, add_23);  getitem_30 = add_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
    getitem_32: "f32[512, 1, 1]" = var_mean_4[0]
    getitem_33: "f32[512, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_29: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
    rsqrt_4: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
    sub_7: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_28, getitem_33);  add_28 = getitem_33 = None
    mul_21: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_4);  sub_7 = None
    mul_22: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_21, primals_186)
    add_30: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_22, primals_187);  mul_22 = primals_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_110: "f32[512, 1024]" = torch.ops.aten.view.default(add_30, [512, 1024])
    permute_127: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
    addmm_4: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_189, view_110, permute_127);  primals_189 = None
    view_111: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_4, [512, 1, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_23: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_111, 0.5)
    mul_24: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_111, 0.7071067811865476);  view_111 = None
    erf_2: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_24);  mul_24 = None
    add_31: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_25: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_23, add_31);  mul_23 = add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    native_dropout_12 = torch.ops.aten.native_dropout.default(mul_25, 0.1, True);  mul_25 = None
    getitem_34: "f32[512, 1, 4096]" = native_dropout_12[0]
    getitem_35: "b8[512, 1, 4096]" = native_dropout_12[1];  native_dropout_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_112: "f32[512, 4096]" = torch.ops.aten.view.default(getitem_34, [512, 4096]);  getitem_34 = None
    permute_128: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_190, [1, 0]);  primals_190 = None
    addmm_5: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_191, view_112, permute_128);  primals_191 = None
    view_113: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_5, [512, 1, 1024]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    native_dropout_13 = torch.ops.aten.native_dropout.default(view_113, 0.1, True);  view_113 = None
    getitem_36: "f32[512, 1, 1024]" = native_dropout_13[0]
    getitem_37: "b8[512, 1, 1024]" = native_dropout_13[1];  native_dropout_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_32: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_36, add_30);  getitem_36 = add_30 = None
    var_mean_5 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
    getitem_38: "f32[512, 1, 1]" = var_mean_5[0]
    getitem_39: "f32[512, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_33: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
    rsqrt_5: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
    sub_8: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_32, getitem_39);  add_32 = getitem_39 = None
    mul_26: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = None
    mul_27: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_26, primals_192)
    add_34: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_27, primals_193);  mul_27 = primals_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_78: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_34, 3)
    unsqueeze_79: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, 4);  unsqueeze_78 = None
    permute_129: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_79, [0, 1, 3, 4, 2]);  unsqueeze_79 = None
    unsqueeze_80: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_22, 3);  primals_22 = None
    unsqueeze_81: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, 4);  unsqueeze_80 = None
    permute_130: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_81, [3, 4, 1, 2, 0]);  unsqueeze_81 = None
    permute_131: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_129, [0, 4, 1, 2, 3]);  permute_129 = None
    view_114: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_131, [1, 512, 1024]);  permute_131 = None
    permute_132: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_130, [4, 1, 2, 3, 0]);  permute_130 = None
    view_115: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_132, [1, 1024, 1024]);  permute_132 = None
    bmm_24: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_114, view_115)
    view_116: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_24, [512, 1, 1, 16, 64]);  bmm_24 = None
    permute_133: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_116, [0, 2, 3, 4, 1]);  view_116 = None
    view_117: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_133, [512, 1, 16, 64]);  permute_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_84: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_23, 3);  primals_23 = None
    unsqueeze_85: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, 4);  unsqueeze_84 = None
    permute_135: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_85, [3, 4, 1, 2, 0]);  unsqueeze_85 = None
    permute_137: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_135, [4, 1, 2, 3, 0]);  permute_135 = None
    view_119: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_137, [1, 1024, 1024]);  permute_137 = None
    bmm_25: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_114, view_119)
    view_120: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_25, [512, 1, 1, 16, 64]);  bmm_25 = None
    permute_138: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_120, [0, 2, 3, 4, 1]);  view_120 = None
    view_121: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_138, [512, 1, 16, 64]);  permute_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_88: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_24, 3);  primals_24 = None
    unsqueeze_89: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, 4);  unsqueeze_88 = None
    permute_140: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_89, [3, 4, 1, 2, 0]);  unsqueeze_89 = None
    permute_142: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_140, [4, 1, 2, 3, 0]);  permute_140 = None
    view_123: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_142, [1, 1024, 1024]);  permute_142 = None
    bmm_26: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_114, view_123)
    view_124: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_26, [512, 1, 1, 16, 64]);  bmm_26 = None
    permute_143: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_124, [0, 2, 3, 4, 1]);  view_124 = None
    view_125: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_143, [512, 1, 16, 64]);  permute_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_92: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_25, 3);  primals_25 = None
    unsqueeze_93: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, 4);  unsqueeze_92 = None
    permute_145: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_93, [3, 4, 1, 2, 0]);  unsqueeze_93 = None
    permute_147: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_145, [4, 1, 2, 3, 0]);  permute_145 = None
    view_127: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_147, [1, 1024, 1024]);  permute_147 = None
    bmm_27: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_12, view_127);  view_127 = None
    view_128: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_27, [1024, 1, 1, 16, 64]);  bmm_27 = None
    permute_148: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_128, [0, 2, 3, 4, 1]);  view_128 = None
    view_129: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_148, [1024, 1, 16, 64]);  permute_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_35: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_117, primals_26);  primals_26 = None
    unsqueeze_94: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_35, 4);  add_35 = None
    permute_149: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_94, [1, 2, 0, 4, 3]);  unsqueeze_94 = None
    unsqueeze_95: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_121, 4);  view_121 = None
    permute_150: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_95, [1, 2, 4, 0, 3]);  unsqueeze_95 = None
    permute_151: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_149, [1, 2, 4, 0, 3]);  permute_149 = None
    view_130: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_151, [16, 512, 64]);  permute_151 = None
    permute_152: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_150, [1, 4, 0, 3, 2]);  permute_150 = None
    view_131: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_152, [16, 64, 512]);  permute_152 = None
    bmm_28: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_130, view_131)
    view_132: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_28, [16, 512, 1, 1, 512]);  bmm_28 = None
    permute_153: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_132, [3, 0, 1, 4, 2]);  view_132 = None
    view_133: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_153, [1, 16, 512, 512]);  permute_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_36: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_117, primals_27);  view_117 = primals_27 = None
    unsqueeze_96: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_36, 4);  add_36 = None
    permute_154: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_96, [1, 2, 0, 4, 3]);  unsqueeze_96 = None
    unsqueeze_97: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_129, 4);  view_129 = None
    permute_155: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_97, [1, 2, 4, 0, 3]);  unsqueeze_97 = None
    permute_156: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_154, [1, 2, 4, 0, 3]);  permute_154 = None
    view_134: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_156, [16, 512, 64]);  permute_156 = None
    permute_157: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_155, [1, 4, 0, 3, 2]);  permute_155 = None
    view_135: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_157, [16, 64, 1024]);  permute_157 = None
    bmm_29: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_134, view_135)
    view_136: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_29, [16, 512, 1, 1, 1024]);  bmm_29 = None
    permute_158: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_136, [3, 0, 1, 4, 2]);  view_136 = None
    view_137: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_158, [1, 16, 512, 1024]);  permute_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_138: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_137, [1, 16, 1024, 512]);  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_24: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_138, 0, 0, 9223372036854775807);  view_138 = None
    slice_25: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_24, 1, 0, 9223372036854775807);  slice_24 = None
    slice_26: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_25, 2, 1, 9223372036854775807);  slice_25 = None
    slice_27: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_26, 3, 0, 9223372036854775807);  slice_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_139: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_27, [1, 16, 512, 1023]);  slice_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    slice_28: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_139, 0, 0, 9223372036854775807);  view_139 = None
    slice_29: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_28, 1, 0, 9223372036854775807);  slice_28 = None
    slice_30: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_29, 2, 0, 9223372036854775807);  slice_29 = None
    index_3: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_30, [None, None, None, iota_2]);  slice_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_37: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_133, index_3);  view_133 = index_3 = None
    add_38: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_37, 0);  add_37 = None
    mul_28: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_38, 0.125);  add_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_3: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_28, [3], True)
    sub_9: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_28, amax_3);  mul_28 = amax_3 = None
    exp_3: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
    sum_4: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [3], True)
    div_4: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_3: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    native_dropout_14 = torch.ops.aten.native_dropout.default(div_4, 0.1, True);  div_4 = None
    getitem_40: "f32[1, 16, 512, 512]" = native_dropout_14[0]
    getitem_41: "b8[1, 16, 512, 512]" = native_dropout_14[1];  native_dropout_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_98: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(getitem_40, 4);  getitem_40 = None
    permute_159: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_98, [2, 0, 1, 4, 3]);  unsqueeze_98 = None
    unsqueeze_99: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_125, 4);  view_125 = None
    permute_160: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_99, [4, 1, 2, 3, 0]);  unsqueeze_99 = None
    permute_161: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_159, [2, 0, 4, 1, 3]);  permute_159 = None
    view_140: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_161, [16, 512, 512]);  permute_161 = None
    permute_162: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_160, [2, 4, 1, 3, 0]);  permute_160 = None
    view_141: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_162, [16, 512, 64]);  permute_162 = None
    bmm_30: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_140, view_141)
    view_142: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_30, [16, 512, 1, 1, 64]);  bmm_30 = None
    permute_163: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_142, [1, 3, 0, 4, 2]);  view_142 = None
    view_143: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_163, [512, 1, 16, 64]);  permute_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_100: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_143, 4);  view_143 = None
    permute_164: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_100, [0, 1, 4, 3, 2]);  unsqueeze_100 = None
    unsqueeze_101: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_28, 3);  primals_28 = None
    unsqueeze_102: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_101, 4);  unsqueeze_101 = None
    permute_165: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_102, [3, 4, 0, 2, 1]);  unsqueeze_102 = None
    permute_166: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_164, [0, 3, 4, 1, 2]);  permute_164 = None
    clone_6: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_166, memory_format = torch.contiguous_format);  permute_166 = None
    view_144: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_6, [1, 512, 1024]);  clone_6 = None
    permute_167: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_165, [3, 4, 1, 2, 0]);  permute_165 = None
    clone_7: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_167, memory_format = torch.contiguous_format);  permute_167 = None
    view_145: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_7, [1, 1024, 1024]);  clone_7 = None
    bmm_31: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_144, view_145)
    view_146: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_31, [512, 1, 1, 1, 1024]);  bmm_31 = None
    permute_168: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_146, [0, 3, 4, 1, 2]);  view_146 = None
    view_147: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_168, [512, 1, 1024]);  permute_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    native_dropout_15 = torch.ops.aten.native_dropout.default(view_147, 0.1, True);  view_147 = None
    getitem_42: "f32[512, 1, 1024]" = native_dropout_15[0]
    getitem_43: "b8[512, 1, 1024]" = native_dropout_15[1];  native_dropout_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_39: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_42, add_34);  getitem_42 = add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
    getitem_44: "f32[512, 1, 1]" = var_mean_6[0]
    getitem_45: "f32[512, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_40: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
    rsqrt_6: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
    sub_10: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_39, getitem_45);  add_39 = getitem_45 = None
    mul_29: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_6);  sub_10 = None
    mul_30: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_29, primals_194)
    add_41: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_30, primals_195);  mul_30 = primals_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_148: "f32[512, 1024]" = torch.ops.aten.view.default(add_41, [512, 1024])
    permute_169: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_196, [1, 0]);  primals_196 = None
    addmm_6: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_197, view_148, permute_169);  primals_197 = None
    view_149: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_6, [512, 1, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_31: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_149, 0.5)
    mul_32: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_149, 0.7071067811865476);  view_149 = None
    erf_3: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_32);  mul_32 = None
    add_42: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_33: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_31, add_42);  mul_31 = add_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    native_dropout_16 = torch.ops.aten.native_dropout.default(mul_33, 0.1, True);  mul_33 = None
    getitem_46: "f32[512, 1, 4096]" = native_dropout_16[0]
    getitem_47: "b8[512, 1, 4096]" = native_dropout_16[1];  native_dropout_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_150: "f32[512, 4096]" = torch.ops.aten.view.default(getitem_46, [512, 4096]);  getitem_46 = None
    permute_170: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_198, [1, 0]);  primals_198 = None
    addmm_7: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_199, view_150, permute_170);  primals_199 = None
    view_151: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_7, [512, 1, 1024]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    native_dropout_17 = torch.ops.aten.native_dropout.default(view_151, 0.1, True);  view_151 = None
    getitem_48: "f32[512, 1, 1024]" = native_dropout_17[0]
    getitem_49: "b8[512, 1, 1024]" = native_dropout_17[1];  native_dropout_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_43: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_48, add_41);  getitem_48 = add_41 = None
    var_mean_7 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
    getitem_50: "f32[512, 1, 1]" = var_mean_7[0]
    getitem_51: "f32[512, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_44: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
    rsqrt_7: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
    sub_11: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_43, getitem_51);  add_43 = getitem_51 = None
    mul_34: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = None
    mul_35: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_34, primals_200)
    add_45: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_35, primals_201);  mul_35 = primals_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_103: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_45, 3)
    unsqueeze_104: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_103, 4);  unsqueeze_103 = None
    permute_171: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_104, [0, 1, 3, 4, 2]);  unsqueeze_104 = None
    unsqueeze_105: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_29, 3);  primals_29 = None
    unsqueeze_106: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_105, 4);  unsqueeze_105 = None
    permute_172: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_106, [3, 4, 1, 2, 0]);  unsqueeze_106 = None
    permute_173: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_171, [0, 4, 1, 2, 3]);  permute_171 = None
    view_152: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_173, [1, 512, 1024]);  permute_173 = None
    permute_174: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_172, [4, 1, 2, 3, 0]);  permute_172 = None
    view_153: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_174, [1, 1024, 1024]);  permute_174 = None
    bmm_32: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_152, view_153)
    view_154: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_32, [512, 1, 1, 16, 64]);  bmm_32 = None
    permute_175: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_154, [0, 2, 3, 4, 1]);  view_154 = None
    view_155: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_175, [512, 1, 16, 64]);  permute_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_109: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_30, 3);  primals_30 = None
    unsqueeze_110: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_109, 4);  unsqueeze_109 = None
    permute_177: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_110, [3, 4, 1, 2, 0]);  unsqueeze_110 = None
    permute_179: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_177, [4, 1, 2, 3, 0]);  permute_177 = None
    view_157: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_179, [1, 1024, 1024]);  permute_179 = None
    bmm_33: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_152, view_157)
    view_158: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_33, [512, 1, 1, 16, 64]);  bmm_33 = None
    permute_180: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_158, [0, 2, 3, 4, 1]);  view_158 = None
    view_159: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_180, [512, 1, 16, 64]);  permute_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_113: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_31, 3);  primals_31 = None
    unsqueeze_114: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_113, 4);  unsqueeze_113 = None
    permute_182: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_114, [3, 4, 1, 2, 0]);  unsqueeze_114 = None
    permute_184: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_182, [4, 1, 2, 3, 0]);  permute_182 = None
    view_161: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_184, [1, 1024, 1024]);  permute_184 = None
    bmm_34: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_152, view_161)
    view_162: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_34, [512, 1, 1, 16, 64]);  bmm_34 = None
    permute_185: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_162, [0, 2, 3, 4, 1]);  view_162 = None
    view_163: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_185, [512, 1, 16, 64]);  permute_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_117: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_32, 3);  primals_32 = None
    unsqueeze_118: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_117, 4);  unsqueeze_117 = None
    permute_187: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_118, [3, 4, 1, 2, 0]);  unsqueeze_118 = None
    permute_189: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_187, [4, 1, 2, 3, 0]);  permute_187 = None
    view_165: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_189, [1, 1024, 1024]);  permute_189 = None
    bmm_35: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_12, view_165);  view_165 = None
    view_166: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_35, [1024, 1, 1, 16, 64]);  bmm_35 = None
    permute_190: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_166, [0, 2, 3, 4, 1]);  view_166 = None
    view_167: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_190, [1024, 1, 16, 64]);  permute_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_46: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_155, primals_33);  primals_33 = None
    unsqueeze_119: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_46, 4);  add_46 = None
    permute_191: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_119, [1, 2, 0, 4, 3]);  unsqueeze_119 = None
    unsqueeze_120: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_159, 4);  view_159 = None
    permute_192: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_120, [1, 2, 4, 0, 3]);  unsqueeze_120 = None
    permute_193: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_191, [1, 2, 4, 0, 3]);  permute_191 = None
    view_168: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_193, [16, 512, 64]);  permute_193 = None
    permute_194: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_192, [1, 4, 0, 3, 2]);  permute_192 = None
    view_169: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_194, [16, 64, 512]);  permute_194 = None
    bmm_36: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_168, view_169)
    view_170: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_36, [16, 512, 1, 1, 512]);  bmm_36 = None
    permute_195: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_170, [3, 0, 1, 4, 2]);  view_170 = None
    view_171: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_195, [1, 16, 512, 512]);  permute_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_47: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_155, primals_34);  view_155 = primals_34 = None
    unsqueeze_121: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_47, 4);  add_47 = None
    permute_196: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_121, [1, 2, 0, 4, 3]);  unsqueeze_121 = None
    unsqueeze_122: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_167, 4);  view_167 = None
    permute_197: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_122, [1, 2, 4, 0, 3]);  unsqueeze_122 = None
    permute_198: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_196, [1, 2, 4, 0, 3]);  permute_196 = None
    view_172: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_198, [16, 512, 64]);  permute_198 = None
    permute_199: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_197, [1, 4, 0, 3, 2]);  permute_197 = None
    view_173: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_199, [16, 64, 1024]);  permute_199 = None
    bmm_37: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_172, view_173)
    view_174: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_37, [16, 512, 1, 1, 1024]);  bmm_37 = None
    permute_200: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_174, [3, 0, 1, 4, 2]);  view_174 = None
    view_175: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_200, [1, 16, 512, 1024]);  permute_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_176: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_175, [1, 16, 1024, 512]);  view_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_31: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_176, 0, 0, 9223372036854775807);  view_176 = None
    slice_32: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_31, 1, 0, 9223372036854775807);  slice_31 = None
    slice_33: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_32, 2, 1, 9223372036854775807);  slice_32 = None
    slice_34: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_33, 3, 0, 9223372036854775807);  slice_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_177: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_34, [1, 16, 512, 1023]);  slice_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    slice_35: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_177, 0, 0, 9223372036854775807);  view_177 = None
    slice_36: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_35, 1, 0, 9223372036854775807);  slice_35 = None
    slice_37: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_36, 2, 0, 9223372036854775807);  slice_36 = None
    index_4: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_37, [None, None, None, iota_2]);  slice_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_48: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_171, index_4);  view_171 = index_4 = None
    add_49: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_48, 0);  add_48 = None
    mul_36: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_49, 0.125);  add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_4: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_36, [3], True)
    sub_12: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_36, amax_4);  mul_36 = amax_4 = None
    exp_4: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
    sum_5: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [3], True)
    div_5: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    alias_4: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    native_dropout_18 = torch.ops.aten.native_dropout.default(div_5, 0.1, True);  div_5 = None
    getitem_52: "f32[1, 16, 512, 512]" = native_dropout_18[0]
    getitem_53: "b8[1, 16, 512, 512]" = native_dropout_18[1];  native_dropout_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_123: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(getitem_52, 4);  getitem_52 = None
    permute_201: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_123, [2, 0, 1, 4, 3]);  unsqueeze_123 = None
    unsqueeze_124: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_163, 4);  view_163 = None
    permute_202: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_124, [4, 1, 2, 3, 0]);  unsqueeze_124 = None
    permute_203: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_201, [2, 0, 4, 1, 3]);  permute_201 = None
    view_178: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_203, [16, 512, 512]);  permute_203 = None
    permute_204: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_202, [2, 4, 1, 3, 0]);  permute_202 = None
    view_179: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_204, [16, 512, 64]);  permute_204 = None
    bmm_38: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_178, view_179)
    view_180: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_38, [16, 512, 1, 1, 64]);  bmm_38 = None
    permute_205: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_180, [1, 3, 0, 4, 2]);  view_180 = None
    view_181: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_205, [512, 1, 16, 64]);  permute_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_125: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_181, 4);  view_181 = None
    permute_206: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_125, [0, 1, 4, 3, 2]);  unsqueeze_125 = None
    unsqueeze_126: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_35, 3);  primals_35 = None
    unsqueeze_127: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, 4);  unsqueeze_126 = None
    permute_207: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_127, [3, 4, 0, 2, 1]);  unsqueeze_127 = None
    permute_208: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_206, [0, 3, 4, 1, 2]);  permute_206 = None
    clone_8: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_208, memory_format = torch.contiguous_format);  permute_208 = None
    view_182: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_8, [1, 512, 1024]);  clone_8 = None
    permute_209: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_207, [3, 4, 1, 2, 0]);  permute_207 = None
    clone_9: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_209, memory_format = torch.contiguous_format);  permute_209 = None
    view_183: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_9, [1, 1024, 1024]);  clone_9 = None
    bmm_39: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_182, view_183)
    view_184: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_39, [512, 1, 1, 1, 1024]);  bmm_39 = None
    permute_210: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_184, [0, 3, 4, 1, 2]);  view_184 = None
    view_185: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_210, [512, 1, 1024]);  permute_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    native_dropout_19 = torch.ops.aten.native_dropout.default(view_185, 0.1, True);  view_185 = None
    getitem_54: "f32[512, 1, 1024]" = native_dropout_19[0]
    getitem_55: "b8[512, 1, 1024]" = native_dropout_19[1];  native_dropout_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_50: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_54, add_45);  getitem_54 = add_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_50, [2], correction = 0, keepdim = True)
    getitem_56: "f32[512, 1, 1]" = var_mean_8[0]
    getitem_57: "f32[512, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_51: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-12);  getitem_56 = None
    rsqrt_8: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
    sub_13: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_50, getitem_57);  add_50 = getitem_57 = None
    mul_37: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_8);  sub_13 = None
    mul_38: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_37, primals_202)
    add_52: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_38, primals_203);  mul_38 = primals_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_186: "f32[512, 1024]" = torch.ops.aten.view.default(add_52, [512, 1024])
    permute_211: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_204, [1, 0]);  primals_204 = None
    addmm_8: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_205, view_186, permute_211);  primals_205 = None
    view_187: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_8, [512, 1, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_39: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_187, 0.5)
    mul_40: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_187, 0.7071067811865476);  view_187 = None
    erf_4: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
    add_53: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_41: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_39, add_53);  mul_39 = add_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    native_dropout_20 = torch.ops.aten.native_dropout.default(mul_41, 0.1, True);  mul_41 = None
    getitem_58: "f32[512, 1, 4096]" = native_dropout_20[0]
    getitem_59: "b8[512, 1, 4096]" = native_dropout_20[1];  native_dropout_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_188: "f32[512, 4096]" = torch.ops.aten.view.default(getitem_58, [512, 4096]);  getitem_58 = None
    permute_212: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_206, [1, 0]);  primals_206 = None
    addmm_9: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_207, view_188, permute_212);  primals_207 = None
    view_189: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_9, [512, 1, 1024]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    native_dropout_21 = torch.ops.aten.native_dropout.default(view_189, 0.1, True);  view_189 = None
    getitem_60: "f32[512, 1, 1024]" = native_dropout_21[0]
    getitem_61: "b8[512, 1, 1024]" = native_dropout_21[1];  native_dropout_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_54: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_60, add_52);  getitem_60 = add_52 = None
    var_mean_9 = torch.ops.aten.var_mean.correction(add_54, [2], correction = 0, keepdim = True)
    getitem_62: "f32[512, 1, 1]" = var_mean_9[0]
    getitem_63: "f32[512, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_55: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-12);  getitem_62 = None
    rsqrt_9: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
    sub_14: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_54, getitem_63);  add_54 = getitem_63 = None
    mul_42: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = None
    mul_43: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_42, primals_208)
    add_56: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_43, primals_209);  mul_43 = primals_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_128: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_56, 3)
    unsqueeze_129: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, 4);  unsqueeze_128 = None
    permute_213: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_129, [0, 1, 3, 4, 2]);  unsqueeze_129 = None
    unsqueeze_130: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_36, 3);  primals_36 = None
    unsqueeze_131: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, 4);  unsqueeze_130 = None
    permute_214: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_131, [3, 4, 1, 2, 0]);  unsqueeze_131 = None
    permute_215: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_213, [0, 4, 1, 2, 3]);  permute_213 = None
    view_190: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_215, [1, 512, 1024]);  permute_215 = None
    permute_216: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_214, [4, 1, 2, 3, 0]);  permute_214 = None
    view_191: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_216, [1, 1024, 1024]);  permute_216 = None
    bmm_40: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_190, view_191)
    view_192: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_40, [512, 1, 1, 16, 64]);  bmm_40 = None
    permute_217: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_192, [0, 2, 3, 4, 1]);  view_192 = None
    view_193: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_217, [512, 1, 16, 64]);  permute_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_134: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_37, 3);  primals_37 = None
    unsqueeze_135: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, 4);  unsqueeze_134 = None
    permute_219: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_135, [3, 4, 1, 2, 0]);  unsqueeze_135 = None
    permute_221: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_219, [4, 1, 2, 3, 0]);  permute_219 = None
    view_195: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_221, [1, 1024, 1024]);  permute_221 = None
    bmm_41: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_190, view_195)
    view_196: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_41, [512, 1, 1, 16, 64]);  bmm_41 = None
    permute_222: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_196, [0, 2, 3, 4, 1]);  view_196 = None
    view_197: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_222, [512, 1, 16, 64]);  permute_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_138: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_38, 3);  primals_38 = None
    unsqueeze_139: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, 4);  unsqueeze_138 = None
    permute_224: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_139, [3, 4, 1, 2, 0]);  unsqueeze_139 = None
    permute_226: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_224, [4, 1, 2, 3, 0]);  permute_224 = None
    view_199: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_226, [1, 1024, 1024]);  permute_226 = None
    bmm_42: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_190, view_199)
    view_200: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_42, [512, 1, 1, 16, 64]);  bmm_42 = None
    permute_227: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_200, [0, 2, 3, 4, 1]);  view_200 = None
    view_201: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_227, [512, 1, 16, 64]);  permute_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_142: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_39, 3);  primals_39 = None
    unsqueeze_143: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, 4);  unsqueeze_142 = None
    permute_229: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_143, [3, 4, 1, 2, 0]);  unsqueeze_143 = None
    permute_231: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_229, [4, 1, 2, 3, 0]);  permute_229 = None
    view_203: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_231, [1, 1024, 1024]);  permute_231 = None
    bmm_43: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_12, view_203);  view_203 = None
    view_204: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_43, [1024, 1, 1, 16, 64]);  bmm_43 = None
    permute_232: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_204, [0, 2, 3, 4, 1]);  view_204 = None
    view_205: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_232, [1024, 1, 16, 64]);  permute_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_57: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_193, primals_40);  primals_40 = None
    unsqueeze_144: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_57, 4);  add_57 = None
    permute_233: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_144, [1, 2, 0, 4, 3]);  unsqueeze_144 = None
    unsqueeze_145: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_197, 4);  view_197 = None
    permute_234: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_145, [1, 2, 4, 0, 3]);  unsqueeze_145 = None
    permute_235: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_233, [1, 2, 4, 0, 3]);  permute_233 = None
    view_206: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_235, [16, 512, 64]);  permute_235 = None
    permute_236: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_234, [1, 4, 0, 3, 2]);  permute_234 = None
    view_207: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_236, [16, 64, 512]);  permute_236 = None
    bmm_44: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_206, view_207)
    view_208: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_44, [16, 512, 1, 1, 512]);  bmm_44 = None
    permute_237: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_208, [3, 0, 1, 4, 2]);  view_208 = None
    view_209: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_237, [1, 16, 512, 512]);  permute_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_58: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_193, primals_41);  view_193 = primals_41 = None
    unsqueeze_146: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_58, 4);  add_58 = None
    permute_238: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_146, [1, 2, 0, 4, 3]);  unsqueeze_146 = None
    unsqueeze_147: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_205, 4);  view_205 = None
    permute_239: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_147, [1, 2, 4, 0, 3]);  unsqueeze_147 = None
    permute_240: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_238, [1, 2, 4, 0, 3]);  permute_238 = None
    view_210: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_240, [16, 512, 64]);  permute_240 = None
    permute_241: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_239, [1, 4, 0, 3, 2]);  permute_239 = None
    view_211: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_241, [16, 64, 1024]);  permute_241 = None
    bmm_45: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_210, view_211)
    view_212: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_45, [16, 512, 1, 1, 1024]);  bmm_45 = None
    permute_242: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_212, [3, 0, 1, 4, 2]);  view_212 = None
    view_213: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_242, [1, 16, 512, 1024]);  permute_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_214: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_213, [1, 16, 1024, 512]);  view_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_38: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_214, 0, 0, 9223372036854775807);  view_214 = None
    slice_39: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_38, 1, 0, 9223372036854775807);  slice_38 = None
    slice_40: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_39, 2, 1, 9223372036854775807);  slice_39 = None
    slice_41: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_40, 3, 0, 9223372036854775807);  slice_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_215: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_41, [1, 16, 512, 1023]);  slice_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    slice_42: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_215, 0, 0, 9223372036854775807);  view_215 = None
    slice_43: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_42, 1, 0, 9223372036854775807);  slice_42 = None
    slice_44: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_43, 2, 0, 9223372036854775807);  slice_43 = None
    index_5: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_44, [None, None, None, iota_2]);  slice_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_59: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_209, index_5);  view_209 = index_5 = None
    add_60: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_59, 0);  add_59 = None
    mul_44: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_60, 0.125);  add_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_5: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_44, [3], True)
    sub_15: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_44, amax_5);  mul_44 = amax_5 = None
    exp_5: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
    sum_6: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [3], True)
    div_6: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    alias_5: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    native_dropout_22 = torch.ops.aten.native_dropout.default(div_6, 0.1, True);  div_6 = None
    getitem_64: "f32[1, 16, 512, 512]" = native_dropout_22[0]
    getitem_65: "b8[1, 16, 512, 512]" = native_dropout_22[1];  native_dropout_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_148: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(getitem_64, 4);  getitem_64 = None
    permute_243: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_148, [2, 0, 1, 4, 3]);  unsqueeze_148 = None
    unsqueeze_149: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_201, 4);  view_201 = None
    permute_244: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_149, [4, 1, 2, 3, 0]);  unsqueeze_149 = None
    permute_245: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_243, [2, 0, 4, 1, 3]);  permute_243 = None
    view_216: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_245, [16, 512, 512]);  permute_245 = None
    permute_246: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_244, [2, 4, 1, 3, 0]);  permute_244 = None
    view_217: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_246, [16, 512, 64]);  permute_246 = None
    bmm_46: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_216, view_217)
    view_218: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_46, [16, 512, 1, 1, 64]);  bmm_46 = None
    permute_247: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_218, [1, 3, 0, 4, 2]);  view_218 = None
    view_219: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_247, [512, 1, 16, 64]);  permute_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_150: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_219, 4);  view_219 = None
    permute_248: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_150, [0, 1, 4, 3, 2]);  unsqueeze_150 = None
    unsqueeze_151: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_42, 3);  primals_42 = None
    unsqueeze_152: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_151, 4);  unsqueeze_151 = None
    permute_249: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_152, [3, 4, 0, 2, 1]);  unsqueeze_152 = None
    permute_250: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_248, [0, 3, 4, 1, 2]);  permute_248 = None
    clone_10: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_250, memory_format = torch.contiguous_format);  permute_250 = None
    view_220: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_10, [1, 512, 1024]);  clone_10 = None
    permute_251: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_249, [3, 4, 1, 2, 0]);  permute_249 = None
    clone_11: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_251, memory_format = torch.contiguous_format);  permute_251 = None
    view_221: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_11, [1, 1024, 1024]);  clone_11 = None
    bmm_47: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_220, view_221)
    view_222: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_47, [512, 1, 1, 1, 1024]);  bmm_47 = None
    permute_252: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_222, [0, 3, 4, 1, 2]);  view_222 = None
    view_223: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_252, [512, 1, 1024]);  permute_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    native_dropout_23 = torch.ops.aten.native_dropout.default(view_223, 0.1, True);  view_223 = None
    getitem_66: "f32[512, 1, 1024]" = native_dropout_23[0]
    getitem_67: "b8[512, 1, 1024]" = native_dropout_23[1];  native_dropout_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_61: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_66, add_56);  getitem_66 = add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_61, [2], correction = 0, keepdim = True)
    getitem_68: "f32[512, 1, 1]" = var_mean_10[0]
    getitem_69: "f32[512, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_62: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-12);  getitem_68 = None
    rsqrt_10: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    sub_16: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_61, getitem_69);  add_61 = getitem_69 = None
    mul_45: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = None
    mul_46: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_45, primals_210)
    add_63: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_46, primals_211);  mul_46 = primals_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_224: "f32[512, 1024]" = torch.ops.aten.view.default(add_63, [512, 1024])
    permute_253: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_212, [1, 0]);  primals_212 = None
    addmm_10: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_213, view_224, permute_253);  primals_213 = None
    view_225: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_10, [512, 1, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_47: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_225, 0.5)
    mul_48: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_225, 0.7071067811865476);  view_225 = None
    erf_5: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
    add_64: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_49: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_47, add_64);  mul_47 = add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    native_dropout_24 = torch.ops.aten.native_dropout.default(mul_49, 0.1, True);  mul_49 = None
    getitem_70: "f32[512, 1, 4096]" = native_dropout_24[0]
    getitem_71: "b8[512, 1, 4096]" = native_dropout_24[1];  native_dropout_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_226: "f32[512, 4096]" = torch.ops.aten.view.default(getitem_70, [512, 4096]);  getitem_70 = None
    permute_254: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_214, [1, 0]);  primals_214 = None
    addmm_11: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_215, view_226, permute_254);  primals_215 = None
    view_227: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_11, [512, 1, 1024]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    native_dropout_25 = torch.ops.aten.native_dropout.default(view_227, 0.1, True);  view_227 = None
    getitem_72: "f32[512, 1, 1024]" = native_dropout_25[0]
    getitem_73: "b8[512, 1, 1024]" = native_dropout_25[1];  native_dropout_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_65: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_72, add_63);  getitem_72 = add_63 = None
    var_mean_11 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
    getitem_74: "f32[512, 1, 1]" = var_mean_11[0]
    getitem_75: "f32[512, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_66: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-12);  getitem_74 = None
    rsqrt_11: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    sub_17: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_65, getitem_75);  add_65 = getitem_75 = None
    mul_50: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = None
    mul_51: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_50, primals_216)
    add_67: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_51, primals_217);  mul_51 = primals_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_153: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_67, 3)
    unsqueeze_154: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_153, 4);  unsqueeze_153 = None
    permute_255: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_154, [0, 1, 3, 4, 2]);  unsqueeze_154 = None
    unsqueeze_155: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_43, 3);  primals_43 = None
    unsqueeze_156: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_155, 4);  unsqueeze_155 = None
    permute_256: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_156, [3, 4, 1, 2, 0]);  unsqueeze_156 = None
    permute_257: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_255, [0, 4, 1, 2, 3]);  permute_255 = None
    view_228: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_257, [1, 512, 1024]);  permute_257 = None
    permute_258: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_256, [4, 1, 2, 3, 0]);  permute_256 = None
    view_229: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_258, [1, 1024, 1024]);  permute_258 = None
    bmm_48: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_228, view_229)
    view_230: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_48, [512, 1, 1, 16, 64]);  bmm_48 = None
    permute_259: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_230, [0, 2, 3, 4, 1]);  view_230 = None
    view_231: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_259, [512, 1, 16, 64]);  permute_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_159: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_44, 3);  primals_44 = None
    unsqueeze_160: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_159, 4);  unsqueeze_159 = None
    permute_261: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_160, [3, 4, 1, 2, 0]);  unsqueeze_160 = None
    permute_263: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_261, [4, 1, 2, 3, 0]);  permute_261 = None
    view_233: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_263, [1, 1024, 1024]);  permute_263 = None
    bmm_49: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_228, view_233)
    view_234: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_49, [512, 1, 1, 16, 64]);  bmm_49 = None
    permute_264: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_234, [0, 2, 3, 4, 1]);  view_234 = None
    view_235: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_264, [512, 1, 16, 64]);  permute_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_163: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_45, 3);  primals_45 = None
    unsqueeze_164: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_163, 4);  unsqueeze_163 = None
    permute_266: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_164, [3, 4, 1, 2, 0]);  unsqueeze_164 = None
    permute_268: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_266, [4, 1, 2, 3, 0]);  permute_266 = None
    view_237: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_268, [1, 1024, 1024]);  permute_268 = None
    bmm_50: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_228, view_237)
    view_238: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_50, [512, 1, 1, 16, 64]);  bmm_50 = None
    permute_269: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_238, [0, 2, 3, 4, 1]);  view_238 = None
    view_239: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_269, [512, 1, 16, 64]);  permute_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_167: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_46, 3);  primals_46 = None
    unsqueeze_168: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 4);  unsqueeze_167 = None
    permute_271: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_168, [3, 4, 1, 2, 0]);  unsqueeze_168 = None
    permute_273: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_271, [4, 1, 2, 3, 0]);  permute_271 = None
    view_241: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_273, [1, 1024, 1024]);  permute_273 = None
    bmm_51: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_12, view_241);  view_241 = None
    view_242: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_51, [1024, 1, 1, 16, 64]);  bmm_51 = None
    permute_274: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_242, [0, 2, 3, 4, 1]);  view_242 = None
    view_243: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_274, [1024, 1, 16, 64]);  permute_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_68: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_231, primals_47);  primals_47 = None
    unsqueeze_169: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_68, 4);  add_68 = None
    permute_275: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_169, [1, 2, 0, 4, 3]);  unsqueeze_169 = None
    unsqueeze_170: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_235, 4);  view_235 = None
    permute_276: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_170, [1, 2, 4, 0, 3]);  unsqueeze_170 = None
    permute_277: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_275, [1, 2, 4, 0, 3]);  permute_275 = None
    view_244: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_277, [16, 512, 64]);  permute_277 = None
    permute_278: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_276, [1, 4, 0, 3, 2]);  permute_276 = None
    view_245: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_278, [16, 64, 512]);  permute_278 = None
    bmm_52: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_244, view_245)
    view_246: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_52, [16, 512, 1, 1, 512]);  bmm_52 = None
    permute_279: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_246, [3, 0, 1, 4, 2]);  view_246 = None
    view_247: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_279, [1, 16, 512, 512]);  permute_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_69: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_231, primals_48);  view_231 = primals_48 = None
    unsqueeze_171: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_69, 4);  add_69 = None
    permute_280: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_171, [1, 2, 0, 4, 3]);  unsqueeze_171 = None
    unsqueeze_172: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_243, 4);  view_243 = None
    permute_281: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_172, [1, 2, 4, 0, 3]);  unsqueeze_172 = None
    permute_282: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_280, [1, 2, 4, 0, 3]);  permute_280 = None
    view_248: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_282, [16, 512, 64]);  permute_282 = None
    permute_283: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_281, [1, 4, 0, 3, 2]);  permute_281 = None
    view_249: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_283, [16, 64, 1024]);  permute_283 = None
    bmm_53: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_248, view_249)
    view_250: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_53, [16, 512, 1, 1, 1024]);  bmm_53 = None
    permute_284: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_250, [3, 0, 1, 4, 2]);  view_250 = None
    view_251: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_284, [1, 16, 512, 1024]);  permute_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_252: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_251, [1, 16, 1024, 512]);  view_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_45: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_252, 0, 0, 9223372036854775807);  view_252 = None
    slice_46: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_45, 1, 0, 9223372036854775807);  slice_45 = None
    slice_47: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_46, 2, 1, 9223372036854775807);  slice_46 = None
    slice_48: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_47, 3, 0, 9223372036854775807);  slice_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_253: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_48, [1, 16, 512, 1023]);  slice_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    slice_49: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_253, 0, 0, 9223372036854775807);  view_253 = None
    slice_50: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_49, 1, 0, 9223372036854775807);  slice_49 = None
    slice_51: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_50, 2, 0, 9223372036854775807);  slice_50 = None
    index_6: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_51, [None, None, None, iota_2]);  slice_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_70: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_247, index_6);  view_247 = index_6 = None
    add_71: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_70, 0);  add_70 = None
    mul_52: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_71, 0.125);  add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_6: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_52, [3], True)
    sub_18: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_52, amax_6);  mul_52 = amax_6 = None
    exp_6: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
    sum_7: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [3], True)
    div_7: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    alias_6: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    native_dropout_26 = torch.ops.aten.native_dropout.default(div_7, 0.1, True);  div_7 = None
    getitem_76: "f32[1, 16, 512, 512]" = native_dropout_26[0]
    getitem_77: "b8[1, 16, 512, 512]" = native_dropout_26[1];  native_dropout_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_173: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(getitem_76, 4);  getitem_76 = None
    permute_285: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_173, [2, 0, 1, 4, 3]);  unsqueeze_173 = None
    unsqueeze_174: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_239, 4);  view_239 = None
    permute_286: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_174, [4, 1, 2, 3, 0]);  unsqueeze_174 = None
    permute_287: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_285, [2, 0, 4, 1, 3]);  permute_285 = None
    view_254: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_287, [16, 512, 512]);  permute_287 = None
    permute_288: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_286, [2, 4, 1, 3, 0]);  permute_286 = None
    view_255: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_288, [16, 512, 64]);  permute_288 = None
    bmm_54: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_254, view_255)
    view_256: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_54, [16, 512, 1, 1, 64]);  bmm_54 = None
    permute_289: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_256, [1, 3, 0, 4, 2]);  view_256 = None
    view_257: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_289, [512, 1, 16, 64]);  permute_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_175: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_257, 4);  view_257 = None
    permute_290: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_175, [0, 1, 4, 3, 2]);  unsqueeze_175 = None
    unsqueeze_176: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_49, 3);  primals_49 = None
    unsqueeze_177: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, 4);  unsqueeze_176 = None
    permute_291: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_177, [3, 4, 0, 2, 1]);  unsqueeze_177 = None
    permute_292: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_290, [0, 3, 4, 1, 2]);  permute_290 = None
    clone_12: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_292, memory_format = torch.contiguous_format);  permute_292 = None
    view_258: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_12, [1, 512, 1024]);  clone_12 = None
    permute_293: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_291, [3, 4, 1, 2, 0]);  permute_291 = None
    clone_13: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
    view_259: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_13, [1, 1024, 1024]);  clone_13 = None
    bmm_55: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_258, view_259)
    view_260: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_55, [512, 1, 1, 1, 1024]);  bmm_55 = None
    permute_294: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_260, [0, 3, 4, 1, 2]);  view_260 = None
    view_261: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_294, [512, 1, 1024]);  permute_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    native_dropout_27 = torch.ops.aten.native_dropout.default(view_261, 0.1, True);  view_261 = None
    getitem_78: "f32[512, 1, 1024]" = native_dropout_27[0]
    getitem_79: "b8[512, 1, 1024]" = native_dropout_27[1];  native_dropout_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_72: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_78, add_67);  getitem_78 = add_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_72, [2], correction = 0, keepdim = True)
    getitem_80: "f32[512, 1, 1]" = var_mean_12[0]
    getitem_81: "f32[512, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_73: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-12);  getitem_80 = None
    rsqrt_12: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    sub_19: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_72, getitem_81);  add_72 = getitem_81 = None
    mul_53: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_12);  sub_19 = None
    mul_54: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_53, primals_218)
    add_74: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_54, primals_219);  mul_54 = primals_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_262: "f32[512, 1024]" = torch.ops.aten.view.default(add_74, [512, 1024])
    permute_295: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_220, [1, 0]);  primals_220 = None
    addmm_12: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_221, view_262, permute_295);  primals_221 = None
    view_263: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_12, [512, 1, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_55: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_263, 0.5)
    mul_56: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_263, 0.7071067811865476);  view_263 = None
    erf_6: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_56);  mul_56 = None
    add_75: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_57: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_55, add_75);  mul_55 = add_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    native_dropout_28 = torch.ops.aten.native_dropout.default(mul_57, 0.1, True);  mul_57 = None
    getitem_82: "f32[512, 1, 4096]" = native_dropout_28[0]
    getitem_83: "b8[512, 1, 4096]" = native_dropout_28[1];  native_dropout_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_264: "f32[512, 4096]" = torch.ops.aten.view.default(getitem_82, [512, 4096]);  getitem_82 = None
    permute_296: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_222, [1, 0]);  primals_222 = None
    addmm_13: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_223, view_264, permute_296);  primals_223 = None
    view_265: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_13, [512, 1, 1024]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    native_dropout_29 = torch.ops.aten.native_dropout.default(view_265, 0.1, True);  view_265 = None
    getitem_84: "f32[512, 1, 1024]" = native_dropout_29[0]
    getitem_85: "b8[512, 1, 1024]" = native_dropout_29[1];  native_dropout_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_76: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_84, add_74);  getitem_84 = add_74 = None
    var_mean_13 = torch.ops.aten.var_mean.correction(add_76, [2], correction = 0, keepdim = True)
    getitem_86: "f32[512, 1, 1]" = var_mean_13[0]
    getitem_87: "f32[512, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_77: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-12);  getitem_86 = None
    rsqrt_13: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
    sub_20: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_76, getitem_87);  add_76 = getitem_87 = None
    mul_58: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = None
    mul_59: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_58, primals_224)
    add_78: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_59, primals_225);  mul_59 = primals_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_178: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_78, 3)
    unsqueeze_179: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, 4);  unsqueeze_178 = None
    permute_297: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_179, [0, 1, 3, 4, 2]);  unsqueeze_179 = None
    unsqueeze_180: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_50, 3);  primals_50 = None
    unsqueeze_181: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, 4);  unsqueeze_180 = None
    permute_298: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_181, [3, 4, 1, 2, 0]);  unsqueeze_181 = None
    permute_299: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_297, [0, 4, 1, 2, 3]);  permute_297 = None
    view_266: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_299, [1, 512, 1024]);  permute_299 = None
    permute_300: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_298, [4, 1, 2, 3, 0]);  permute_298 = None
    view_267: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_300, [1, 1024, 1024]);  permute_300 = None
    bmm_56: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_266, view_267)
    view_268: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_56, [512, 1, 1, 16, 64]);  bmm_56 = None
    permute_301: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_268, [0, 2, 3, 4, 1]);  view_268 = None
    view_269: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_301, [512, 1, 16, 64]);  permute_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_184: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_51, 3);  primals_51 = None
    unsqueeze_185: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, 4);  unsqueeze_184 = None
    permute_303: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_185, [3, 4, 1, 2, 0]);  unsqueeze_185 = None
    permute_305: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_303, [4, 1, 2, 3, 0]);  permute_303 = None
    view_271: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_305, [1, 1024, 1024]);  permute_305 = None
    bmm_57: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_266, view_271)
    view_272: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_57, [512, 1, 1, 16, 64]);  bmm_57 = None
    permute_306: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_272, [0, 2, 3, 4, 1]);  view_272 = None
    view_273: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_306, [512, 1, 16, 64]);  permute_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_188: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_52, 3);  primals_52 = None
    unsqueeze_189: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, 4);  unsqueeze_188 = None
    permute_308: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_189, [3, 4, 1, 2, 0]);  unsqueeze_189 = None
    permute_310: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_308, [4, 1, 2, 3, 0]);  permute_308 = None
    view_275: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_310, [1, 1024, 1024]);  permute_310 = None
    bmm_58: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_266, view_275)
    view_276: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_58, [512, 1, 1, 16, 64]);  bmm_58 = None
    permute_311: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_276, [0, 2, 3, 4, 1]);  view_276 = None
    view_277: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_311, [512, 1, 16, 64]);  permute_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_192: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_53, 3);  primals_53 = None
    unsqueeze_193: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, 4);  unsqueeze_192 = None
    permute_313: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_193, [3, 4, 1, 2, 0]);  unsqueeze_193 = None
    permute_315: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_313, [4, 1, 2, 3, 0]);  permute_313 = None
    view_279: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_315, [1, 1024, 1024]);  permute_315 = None
    bmm_59: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_12, view_279);  view_279 = None
    view_280: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_59, [1024, 1, 1, 16, 64]);  bmm_59 = None
    permute_316: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_280, [0, 2, 3, 4, 1]);  view_280 = None
    view_281: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_316, [1024, 1, 16, 64]);  permute_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_79: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_269, primals_54);  primals_54 = None
    unsqueeze_194: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_79, 4);  add_79 = None
    permute_317: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_194, [1, 2, 0, 4, 3]);  unsqueeze_194 = None
    unsqueeze_195: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_273, 4);  view_273 = None
    permute_318: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_195, [1, 2, 4, 0, 3]);  unsqueeze_195 = None
    permute_319: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_317, [1, 2, 4, 0, 3]);  permute_317 = None
    view_282: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_319, [16, 512, 64]);  permute_319 = None
    permute_320: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_318, [1, 4, 0, 3, 2]);  permute_318 = None
    view_283: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_320, [16, 64, 512]);  permute_320 = None
    bmm_60: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_282, view_283)
    view_284: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_60, [16, 512, 1, 1, 512]);  bmm_60 = None
    permute_321: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_284, [3, 0, 1, 4, 2]);  view_284 = None
    view_285: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_321, [1, 16, 512, 512]);  permute_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_80: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_269, primals_55);  view_269 = primals_55 = None
    unsqueeze_196: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_80, 4);  add_80 = None
    permute_322: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_196, [1, 2, 0, 4, 3]);  unsqueeze_196 = None
    unsqueeze_197: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_281, 4);  view_281 = None
    permute_323: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_197, [1, 2, 4, 0, 3]);  unsqueeze_197 = None
    permute_324: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_322, [1, 2, 4, 0, 3]);  permute_322 = None
    view_286: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_324, [16, 512, 64]);  permute_324 = None
    permute_325: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_323, [1, 4, 0, 3, 2]);  permute_323 = None
    view_287: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_325, [16, 64, 1024]);  permute_325 = None
    bmm_61: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_286, view_287)
    view_288: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_61, [16, 512, 1, 1, 1024]);  bmm_61 = None
    permute_326: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_288, [3, 0, 1, 4, 2]);  view_288 = None
    view_289: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_326, [1, 16, 512, 1024]);  permute_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_290: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_289, [1, 16, 1024, 512]);  view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_52: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_290, 0, 0, 9223372036854775807);  view_290 = None
    slice_53: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_52, 1, 0, 9223372036854775807);  slice_52 = None
    slice_54: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_53, 2, 1, 9223372036854775807);  slice_53 = None
    slice_55: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_54, 3, 0, 9223372036854775807);  slice_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_291: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_55, [1, 16, 512, 1023]);  slice_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    slice_56: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_291, 0, 0, 9223372036854775807);  view_291 = None
    slice_57: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_56, 1, 0, 9223372036854775807);  slice_56 = None
    slice_58: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_57, 2, 0, 9223372036854775807);  slice_57 = None
    index_7: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_58, [None, None, None, iota_2]);  slice_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_81: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_285, index_7);  view_285 = index_7 = None
    add_82: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_81, 0);  add_81 = None
    mul_60: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_82, 0.125);  add_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_7: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_60, [3], True)
    sub_21: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_60, amax_7);  mul_60 = amax_7 = None
    exp_7: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
    sum_8: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [3], True)
    div_8: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    alias_7: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    native_dropout_30 = torch.ops.aten.native_dropout.default(div_8, 0.1, True);  div_8 = None
    getitem_88: "f32[1, 16, 512, 512]" = native_dropout_30[0]
    getitem_89: "b8[1, 16, 512, 512]" = native_dropout_30[1];  native_dropout_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_198: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(getitem_88, 4);  getitem_88 = None
    permute_327: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_198, [2, 0, 1, 4, 3]);  unsqueeze_198 = None
    unsqueeze_199: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_277, 4);  view_277 = None
    permute_328: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_199, [4, 1, 2, 3, 0]);  unsqueeze_199 = None
    permute_329: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_327, [2, 0, 4, 1, 3]);  permute_327 = None
    view_292: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_329, [16, 512, 512]);  permute_329 = None
    permute_330: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_328, [2, 4, 1, 3, 0]);  permute_328 = None
    view_293: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_330, [16, 512, 64]);  permute_330 = None
    bmm_62: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_292, view_293)
    view_294: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_62, [16, 512, 1, 1, 64]);  bmm_62 = None
    permute_331: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_294, [1, 3, 0, 4, 2]);  view_294 = None
    view_295: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_331, [512, 1, 16, 64]);  permute_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_200: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_295, 4);  view_295 = None
    permute_332: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_200, [0, 1, 4, 3, 2]);  unsqueeze_200 = None
    unsqueeze_201: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_56, 3);  primals_56 = None
    unsqueeze_202: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_201, 4);  unsqueeze_201 = None
    permute_333: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_202, [3, 4, 0, 2, 1]);  unsqueeze_202 = None
    permute_334: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_332, [0, 3, 4, 1, 2]);  permute_332 = None
    clone_14: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_334, memory_format = torch.contiguous_format);  permute_334 = None
    view_296: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_14, [1, 512, 1024]);  clone_14 = None
    permute_335: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_333, [3, 4, 1, 2, 0]);  permute_333 = None
    clone_15: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_335, memory_format = torch.contiguous_format);  permute_335 = None
    view_297: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_15, [1, 1024, 1024]);  clone_15 = None
    bmm_63: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_296, view_297)
    view_298: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_63, [512, 1, 1, 1, 1024]);  bmm_63 = None
    permute_336: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_298, [0, 3, 4, 1, 2]);  view_298 = None
    view_299: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_336, [512, 1, 1024]);  permute_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    native_dropout_31 = torch.ops.aten.native_dropout.default(view_299, 0.1, True);  view_299 = None
    getitem_90: "f32[512, 1, 1024]" = native_dropout_31[0]
    getitem_91: "b8[512, 1, 1024]" = native_dropout_31[1];  native_dropout_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_83: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_90, add_78);  getitem_90 = add_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
    getitem_92: "f32[512, 1, 1]" = var_mean_14[0]
    getitem_93: "f32[512, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_84: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-12);  getitem_92 = None
    rsqrt_14: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_22: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_83, getitem_93);  add_83 = getitem_93 = None
    mul_61: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_14);  sub_22 = None
    mul_62: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_61, primals_226)
    add_85: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_62, primals_227);  mul_62 = primals_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_300: "f32[512, 1024]" = torch.ops.aten.view.default(add_85, [512, 1024])
    permute_337: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_228, [1, 0]);  primals_228 = None
    addmm_14: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_229, view_300, permute_337);  primals_229 = None
    view_301: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_14, [512, 1, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_63: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_301, 0.5)
    mul_64: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_301, 0.7071067811865476);  view_301 = None
    erf_7: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_64);  mul_64 = None
    add_86: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_65: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_63, add_86);  mul_63 = add_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    native_dropout_32 = torch.ops.aten.native_dropout.default(mul_65, 0.1, True);  mul_65 = None
    getitem_94: "f32[512, 1, 4096]" = native_dropout_32[0]
    getitem_95: "b8[512, 1, 4096]" = native_dropout_32[1];  native_dropout_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_302: "f32[512, 4096]" = torch.ops.aten.view.default(getitem_94, [512, 4096]);  getitem_94 = None
    permute_338: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_230, [1, 0]);  primals_230 = None
    addmm_15: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_231, view_302, permute_338);  primals_231 = None
    view_303: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_15, [512, 1, 1024]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    native_dropout_33 = torch.ops.aten.native_dropout.default(view_303, 0.1, True);  view_303 = None
    getitem_96: "f32[512, 1, 1024]" = native_dropout_33[0]
    getitem_97: "b8[512, 1, 1024]" = native_dropout_33[1];  native_dropout_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_87: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_96, add_85);  getitem_96 = add_85 = None
    var_mean_15 = torch.ops.aten.var_mean.correction(add_87, [2], correction = 0, keepdim = True)
    getitem_98: "f32[512, 1, 1]" = var_mean_15[0]
    getitem_99: "f32[512, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_88: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-12);  getitem_98 = None
    rsqrt_15: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
    sub_23: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_87, getitem_99);  add_87 = getitem_99 = None
    mul_66: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = None
    mul_67: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_66, primals_232)
    add_89: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_67, primals_233);  mul_67 = primals_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_203: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_89, 3)
    unsqueeze_204: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 4);  unsqueeze_203 = None
    permute_339: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_204, [0, 1, 3, 4, 2]);  unsqueeze_204 = None
    unsqueeze_205: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_57, 3);  primals_57 = None
    unsqueeze_206: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 4);  unsqueeze_205 = None
    permute_340: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_206, [3, 4, 1, 2, 0]);  unsqueeze_206 = None
    permute_341: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_339, [0, 4, 1, 2, 3]);  permute_339 = None
    view_304: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_341, [1, 512, 1024]);  permute_341 = None
    permute_342: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_340, [4, 1, 2, 3, 0]);  permute_340 = None
    view_305: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_342, [1, 1024, 1024]);  permute_342 = None
    bmm_64: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_304, view_305)
    view_306: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_64, [512, 1, 1, 16, 64]);  bmm_64 = None
    permute_343: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_306, [0, 2, 3, 4, 1]);  view_306 = None
    view_307: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_343, [512, 1, 16, 64]);  permute_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_209: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_58, 3);  primals_58 = None
    unsqueeze_210: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 4);  unsqueeze_209 = None
    permute_345: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_210, [3, 4, 1, 2, 0]);  unsqueeze_210 = None
    permute_347: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_345, [4, 1, 2, 3, 0]);  permute_345 = None
    view_309: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_347, [1, 1024, 1024]);  permute_347 = None
    bmm_65: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_304, view_309)
    view_310: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_65, [512, 1, 1, 16, 64]);  bmm_65 = None
    permute_348: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_310, [0, 2, 3, 4, 1]);  view_310 = None
    view_311: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_348, [512, 1, 16, 64]);  permute_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_213: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_59, 3);  primals_59 = None
    unsqueeze_214: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 4);  unsqueeze_213 = None
    permute_350: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_214, [3, 4, 1, 2, 0]);  unsqueeze_214 = None
    permute_352: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_350, [4, 1, 2, 3, 0]);  permute_350 = None
    view_313: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_352, [1, 1024, 1024]);  permute_352 = None
    bmm_66: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_304, view_313)
    view_314: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_66, [512, 1, 1, 16, 64]);  bmm_66 = None
    permute_353: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_314, [0, 2, 3, 4, 1]);  view_314 = None
    view_315: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_353, [512, 1, 16, 64]);  permute_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_217: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_60, 3);  primals_60 = None
    unsqueeze_218: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 4);  unsqueeze_217 = None
    permute_355: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_218, [3, 4, 1, 2, 0]);  unsqueeze_218 = None
    permute_357: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_355, [4, 1, 2, 3, 0]);  permute_355 = None
    view_317: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_357, [1, 1024, 1024]);  permute_357 = None
    bmm_67: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_12, view_317);  view_317 = None
    view_318: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_67, [1024, 1, 1, 16, 64]);  bmm_67 = None
    permute_358: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_318, [0, 2, 3, 4, 1]);  view_318 = None
    view_319: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_358, [1024, 1, 16, 64]);  permute_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_90: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_307, primals_61);  primals_61 = None
    unsqueeze_219: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_90, 4);  add_90 = None
    permute_359: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_219, [1, 2, 0, 4, 3]);  unsqueeze_219 = None
    unsqueeze_220: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_311, 4);  view_311 = None
    permute_360: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_220, [1, 2, 4, 0, 3]);  unsqueeze_220 = None
    permute_361: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_359, [1, 2, 4, 0, 3]);  permute_359 = None
    view_320: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_361, [16, 512, 64]);  permute_361 = None
    permute_362: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_360, [1, 4, 0, 3, 2]);  permute_360 = None
    view_321: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_362, [16, 64, 512]);  permute_362 = None
    bmm_68: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_320, view_321)
    view_322: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_68, [16, 512, 1, 1, 512]);  bmm_68 = None
    permute_363: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_322, [3, 0, 1, 4, 2]);  view_322 = None
    view_323: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_363, [1, 16, 512, 512]);  permute_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_91: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_307, primals_62);  view_307 = primals_62 = None
    unsqueeze_221: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_91, 4);  add_91 = None
    permute_364: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_221, [1, 2, 0, 4, 3]);  unsqueeze_221 = None
    unsqueeze_222: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_319, 4);  view_319 = None
    permute_365: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_222, [1, 2, 4, 0, 3]);  unsqueeze_222 = None
    permute_366: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_364, [1, 2, 4, 0, 3]);  permute_364 = None
    view_324: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_366, [16, 512, 64]);  permute_366 = None
    permute_367: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_365, [1, 4, 0, 3, 2]);  permute_365 = None
    view_325: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_367, [16, 64, 1024]);  permute_367 = None
    bmm_69: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_324, view_325)
    view_326: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_69, [16, 512, 1, 1, 1024]);  bmm_69 = None
    permute_368: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_326, [3, 0, 1, 4, 2]);  view_326 = None
    view_327: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_368, [1, 16, 512, 1024]);  permute_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_328: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_327, [1, 16, 1024, 512]);  view_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_59: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_328, 0, 0, 9223372036854775807);  view_328 = None
    slice_60: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_59, 1, 0, 9223372036854775807);  slice_59 = None
    slice_61: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_60, 2, 1, 9223372036854775807);  slice_60 = None
    slice_62: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_61, 3, 0, 9223372036854775807);  slice_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_329: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_62, [1, 16, 512, 1023]);  slice_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    slice_63: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_329, 0, 0, 9223372036854775807);  view_329 = None
    slice_64: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_63, 1, 0, 9223372036854775807);  slice_63 = None
    slice_65: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_64, 2, 0, 9223372036854775807);  slice_64 = None
    index_8: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_65, [None, None, None, iota_2]);  slice_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_92: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_323, index_8);  view_323 = index_8 = None
    add_93: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_92, 0);  add_92 = None
    mul_68: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_93, 0.125);  add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_8: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_68, [3], True)
    sub_24: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_68, amax_8);  mul_68 = amax_8 = None
    exp_8: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_24);  sub_24 = None
    sum_9: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [3], True)
    div_9: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    alias_8: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    native_dropout_34 = torch.ops.aten.native_dropout.default(div_9, 0.1, True);  div_9 = None
    getitem_100: "f32[1, 16, 512, 512]" = native_dropout_34[0]
    getitem_101: "b8[1, 16, 512, 512]" = native_dropout_34[1];  native_dropout_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_223: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(getitem_100, 4);  getitem_100 = None
    permute_369: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_223, [2, 0, 1, 4, 3]);  unsqueeze_223 = None
    unsqueeze_224: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_315, 4);  view_315 = None
    permute_370: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_224, [4, 1, 2, 3, 0]);  unsqueeze_224 = None
    permute_371: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_369, [2, 0, 4, 1, 3]);  permute_369 = None
    view_330: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_371, [16, 512, 512]);  permute_371 = None
    permute_372: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_370, [2, 4, 1, 3, 0]);  permute_370 = None
    view_331: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_372, [16, 512, 64]);  permute_372 = None
    bmm_70: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_330, view_331)
    view_332: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_70, [16, 512, 1, 1, 64]);  bmm_70 = None
    permute_373: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_332, [1, 3, 0, 4, 2]);  view_332 = None
    view_333: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_373, [512, 1, 16, 64]);  permute_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_225: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_333, 4);  view_333 = None
    permute_374: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_225, [0, 1, 4, 3, 2]);  unsqueeze_225 = None
    unsqueeze_226: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_63, 3);  primals_63 = None
    unsqueeze_227: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 4);  unsqueeze_226 = None
    permute_375: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_227, [3, 4, 0, 2, 1]);  unsqueeze_227 = None
    permute_376: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_374, [0, 3, 4, 1, 2]);  permute_374 = None
    clone_16: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_376, memory_format = torch.contiguous_format);  permute_376 = None
    view_334: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_16, [1, 512, 1024]);  clone_16 = None
    permute_377: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_375, [3, 4, 1, 2, 0]);  permute_375 = None
    clone_17: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_377, memory_format = torch.contiguous_format);  permute_377 = None
    view_335: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_17, [1, 1024, 1024]);  clone_17 = None
    bmm_71: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_334, view_335)
    view_336: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_71, [512, 1, 1, 1, 1024]);  bmm_71 = None
    permute_378: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_336, [0, 3, 4, 1, 2]);  view_336 = None
    view_337: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_378, [512, 1, 1024]);  permute_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    native_dropout_35 = torch.ops.aten.native_dropout.default(view_337, 0.1, True);  view_337 = None
    getitem_102: "f32[512, 1, 1024]" = native_dropout_35[0]
    getitem_103: "b8[512, 1, 1024]" = native_dropout_35[1];  native_dropout_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_94: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_102, add_89);  getitem_102 = add_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_16 = torch.ops.aten.var_mean.correction(add_94, [2], correction = 0, keepdim = True)
    getitem_104: "f32[512, 1, 1]" = var_mean_16[0]
    getitem_105: "f32[512, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_95: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-12);  getitem_104 = None
    rsqrt_16: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
    sub_25: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_94, getitem_105);  add_94 = getitem_105 = None
    mul_69: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_16);  sub_25 = None
    mul_70: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_69, primals_234)
    add_96: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_70, primals_235);  mul_70 = primals_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_338: "f32[512, 1024]" = torch.ops.aten.view.default(add_96, [512, 1024])
    permute_379: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_236, [1, 0]);  primals_236 = None
    addmm_16: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_237, view_338, permute_379);  primals_237 = None
    view_339: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_16, [512, 1, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_71: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_339, 0.5)
    mul_72: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_339, 0.7071067811865476);  view_339 = None
    erf_8: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_72);  mul_72 = None
    add_97: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_73: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_71, add_97);  mul_71 = add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    native_dropout_36 = torch.ops.aten.native_dropout.default(mul_73, 0.1, True);  mul_73 = None
    getitem_106: "f32[512, 1, 4096]" = native_dropout_36[0]
    getitem_107: "b8[512, 1, 4096]" = native_dropout_36[1];  native_dropout_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_340: "f32[512, 4096]" = torch.ops.aten.view.default(getitem_106, [512, 4096]);  getitem_106 = None
    permute_380: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_238, [1, 0]);  primals_238 = None
    addmm_17: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_239, view_340, permute_380);  primals_239 = None
    view_341: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_17, [512, 1, 1024]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    native_dropout_37 = torch.ops.aten.native_dropout.default(view_341, 0.1, True);  view_341 = None
    getitem_108: "f32[512, 1, 1024]" = native_dropout_37[0]
    getitem_109: "b8[512, 1, 1024]" = native_dropout_37[1];  native_dropout_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_98: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_108, add_96);  getitem_108 = add_96 = None
    var_mean_17 = torch.ops.aten.var_mean.correction(add_98, [2], correction = 0, keepdim = True)
    getitem_110: "f32[512, 1, 1]" = var_mean_17[0]
    getitem_111: "f32[512, 1, 1]" = var_mean_17[1];  var_mean_17 = None
    add_99: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-12);  getitem_110 = None
    rsqrt_17: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
    sub_26: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_98, getitem_111);  add_98 = getitem_111 = None
    mul_74: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = None
    mul_75: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_74, primals_240)
    add_100: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_75, primals_241);  mul_75 = primals_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_228: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_100, 3)
    unsqueeze_229: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, 4);  unsqueeze_228 = None
    permute_381: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_229, [0, 1, 3, 4, 2]);  unsqueeze_229 = None
    unsqueeze_230: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_64, 3);  primals_64 = None
    unsqueeze_231: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 4);  unsqueeze_230 = None
    permute_382: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_231, [3, 4, 1, 2, 0]);  unsqueeze_231 = None
    permute_383: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_381, [0, 4, 1, 2, 3]);  permute_381 = None
    view_342: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_383, [1, 512, 1024]);  permute_383 = None
    permute_384: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_382, [4, 1, 2, 3, 0]);  permute_382 = None
    view_343: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_384, [1, 1024, 1024]);  permute_384 = None
    bmm_72: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_342, view_343)
    view_344: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_72, [512, 1, 1, 16, 64]);  bmm_72 = None
    permute_385: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_344, [0, 2, 3, 4, 1]);  view_344 = None
    view_345: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_385, [512, 1, 16, 64]);  permute_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_234: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_65, 3);  primals_65 = None
    unsqueeze_235: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, 4);  unsqueeze_234 = None
    permute_387: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_235, [3, 4, 1, 2, 0]);  unsqueeze_235 = None
    permute_389: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_387, [4, 1, 2, 3, 0]);  permute_387 = None
    view_347: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_389, [1, 1024, 1024]);  permute_389 = None
    bmm_73: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_342, view_347)
    view_348: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_73, [512, 1, 1, 16, 64]);  bmm_73 = None
    permute_390: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_348, [0, 2, 3, 4, 1]);  view_348 = None
    view_349: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_390, [512, 1, 16, 64]);  permute_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_238: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_66, 3);  primals_66 = None
    unsqueeze_239: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 4);  unsqueeze_238 = None
    permute_392: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_239, [3, 4, 1, 2, 0]);  unsqueeze_239 = None
    permute_394: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_392, [4, 1, 2, 3, 0]);  permute_392 = None
    view_351: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_394, [1, 1024, 1024]);  permute_394 = None
    bmm_74: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_342, view_351)
    view_352: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_74, [512, 1, 1, 16, 64]);  bmm_74 = None
    permute_395: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_352, [0, 2, 3, 4, 1]);  view_352 = None
    view_353: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_395, [512, 1, 16, 64]);  permute_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_242: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_67, 3);  primals_67 = None
    unsqueeze_243: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 4);  unsqueeze_242 = None
    permute_397: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_243, [3, 4, 1, 2, 0]);  unsqueeze_243 = None
    permute_399: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_397, [4, 1, 2, 3, 0]);  permute_397 = None
    view_355: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_399, [1, 1024, 1024]);  permute_399 = None
    bmm_75: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_12, view_355);  view_355 = None
    view_356: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_75, [1024, 1, 1, 16, 64]);  bmm_75 = None
    permute_400: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_356, [0, 2, 3, 4, 1]);  view_356 = None
    view_357: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_400, [1024, 1, 16, 64]);  permute_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_101: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_345, primals_68);  primals_68 = None
    unsqueeze_244: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_101, 4);  add_101 = None
    permute_401: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_244, [1, 2, 0, 4, 3]);  unsqueeze_244 = None
    unsqueeze_245: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_349, 4);  view_349 = None
    permute_402: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_245, [1, 2, 4, 0, 3]);  unsqueeze_245 = None
    permute_403: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_401, [1, 2, 4, 0, 3]);  permute_401 = None
    view_358: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_403, [16, 512, 64]);  permute_403 = None
    permute_404: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_402, [1, 4, 0, 3, 2]);  permute_402 = None
    view_359: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_404, [16, 64, 512]);  permute_404 = None
    bmm_76: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_358, view_359)
    view_360: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_76, [16, 512, 1, 1, 512]);  bmm_76 = None
    permute_405: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_360, [3, 0, 1, 4, 2]);  view_360 = None
    view_361: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_405, [1, 16, 512, 512]);  permute_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_102: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_345, primals_69);  view_345 = primals_69 = None
    unsqueeze_246: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_102, 4);  add_102 = None
    permute_406: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_246, [1, 2, 0, 4, 3]);  unsqueeze_246 = None
    unsqueeze_247: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_357, 4);  view_357 = None
    permute_407: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_247, [1, 2, 4, 0, 3]);  unsqueeze_247 = None
    permute_408: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_406, [1, 2, 4, 0, 3]);  permute_406 = None
    view_362: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_408, [16, 512, 64]);  permute_408 = None
    permute_409: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_407, [1, 4, 0, 3, 2]);  permute_407 = None
    view_363: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_409, [16, 64, 1024]);  permute_409 = None
    bmm_77: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_362, view_363)
    view_364: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_77, [16, 512, 1, 1, 1024]);  bmm_77 = None
    permute_410: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_364, [3, 0, 1, 4, 2]);  view_364 = None
    view_365: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_410, [1, 16, 512, 1024]);  permute_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_366: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_365, [1, 16, 1024, 512]);  view_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_66: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_366, 0, 0, 9223372036854775807);  view_366 = None
    slice_67: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_66, 1, 0, 9223372036854775807);  slice_66 = None
    slice_68: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_67, 2, 1, 9223372036854775807);  slice_67 = None
    slice_69: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_68, 3, 0, 9223372036854775807);  slice_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_367: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_69, [1, 16, 512, 1023]);  slice_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    slice_70: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_367, 0, 0, 9223372036854775807);  view_367 = None
    slice_71: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_70, 1, 0, 9223372036854775807);  slice_70 = None
    slice_72: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_71, 2, 0, 9223372036854775807);  slice_71 = None
    index_9: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_72, [None, None, None, iota_2]);  slice_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_103: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_361, index_9);  view_361 = index_9 = None
    add_104: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_103, 0);  add_103 = None
    mul_76: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_104, 0.125);  add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_9: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_76, [3], True)
    sub_27: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_76, amax_9);  mul_76 = amax_9 = None
    exp_9: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_27);  sub_27 = None
    sum_10: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [3], True)
    div_10: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    alias_9: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    native_dropout_38 = torch.ops.aten.native_dropout.default(div_10, 0.1, True);  div_10 = None
    getitem_112: "f32[1, 16, 512, 512]" = native_dropout_38[0]
    getitem_113: "b8[1, 16, 512, 512]" = native_dropout_38[1];  native_dropout_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_248: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(getitem_112, 4);  getitem_112 = None
    permute_411: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_248, [2, 0, 1, 4, 3]);  unsqueeze_248 = None
    unsqueeze_249: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_353, 4);  view_353 = None
    permute_412: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_249, [4, 1, 2, 3, 0]);  unsqueeze_249 = None
    permute_413: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_411, [2, 0, 4, 1, 3]);  permute_411 = None
    view_368: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_413, [16, 512, 512]);  permute_413 = None
    permute_414: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_412, [2, 4, 1, 3, 0]);  permute_412 = None
    view_369: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_414, [16, 512, 64]);  permute_414 = None
    bmm_78: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_368, view_369)
    view_370: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_78, [16, 512, 1, 1, 64]);  bmm_78 = None
    permute_415: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_370, [1, 3, 0, 4, 2]);  view_370 = None
    view_371: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_415, [512, 1, 16, 64]);  permute_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_250: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_371, 4);  view_371 = None
    permute_416: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_250, [0, 1, 4, 3, 2]);  unsqueeze_250 = None
    unsqueeze_251: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_70, 3);  primals_70 = None
    unsqueeze_252: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 4);  unsqueeze_251 = None
    permute_417: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_252, [3, 4, 0, 2, 1]);  unsqueeze_252 = None
    permute_418: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_416, [0, 3, 4, 1, 2]);  permute_416 = None
    clone_18: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_418, memory_format = torch.contiguous_format);  permute_418 = None
    view_372: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_18, [1, 512, 1024]);  clone_18 = None
    permute_419: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_417, [3, 4, 1, 2, 0]);  permute_417 = None
    clone_19: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_419, memory_format = torch.contiguous_format);  permute_419 = None
    view_373: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_19, [1, 1024, 1024]);  clone_19 = None
    bmm_79: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_372, view_373)
    view_374: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_79, [512, 1, 1, 1, 1024]);  bmm_79 = None
    permute_420: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_374, [0, 3, 4, 1, 2]);  view_374 = None
    view_375: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_420, [512, 1, 1024]);  permute_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    native_dropout_39 = torch.ops.aten.native_dropout.default(view_375, 0.1, True);  view_375 = None
    getitem_114: "f32[512, 1, 1024]" = native_dropout_39[0]
    getitem_115: "b8[512, 1, 1024]" = native_dropout_39[1];  native_dropout_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_105: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_114, add_100);  getitem_114 = add_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_18 = torch.ops.aten.var_mean.correction(add_105, [2], correction = 0, keepdim = True)
    getitem_116: "f32[512, 1, 1]" = var_mean_18[0]
    getitem_117: "f32[512, 1, 1]" = var_mean_18[1];  var_mean_18 = None
    add_106: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-12);  getitem_116 = None
    rsqrt_18: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
    sub_28: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_105, getitem_117);  add_105 = getitem_117 = None
    mul_77: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_18);  sub_28 = None
    mul_78: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_77, primals_242)
    add_107: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_78, primals_243);  mul_78 = primals_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_376: "f32[512, 1024]" = torch.ops.aten.view.default(add_107, [512, 1024])
    permute_421: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_244, [1, 0]);  primals_244 = None
    addmm_18: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_245, view_376, permute_421);  primals_245 = None
    view_377: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_18, [512, 1, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_79: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_377, 0.5)
    mul_80: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_377, 0.7071067811865476);  view_377 = None
    erf_9: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_80);  mul_80 = None
    add_108: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_81: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_79, add_108);  mul_79 = add_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    native_dropout_40 = torch.ops.aten.native_dropout.default(mul_81, 0.1, True);  mul_81 = None
    getitem_118: "f32[512, 1, 4096]" = native_dropout_40[0]
    getitem_119: "b8[512, 1, 4096]" = native_dropout_40[1];  native_dropout_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_378: "f32[512, 4096]" = torch.ops.aten.view.default(getitem_118, [512, 4096]);  getitem_118 = None
    permute_422: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_246, [1, 0]);  primals_246 = None
    addmm_19: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_247, view_378, permute_422);  primals_247 = None
    view_379: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_19, [512, 1, 1024]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    native_dropout_41 = torch.ops.aten.native_dropout.default(view_379, 0.1, True);  view_379 = None
    getitem_120: "f32[512, 1, 1024]" = native_dropout_41[0]
    getitem_121: "b8[512, 1, 1024]" = native_dropout_41[1];  native_dropout_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_109: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_120, add_107);  getitem_120 = add_107 = None
    var_mean_19 = torch.ops.aten.var_mean.correction(add_109, [2], correction = 0, keepdim = True)
    getitem_122: "f32[512, 1, 1]" = var_mean_19[0]
    getitem_123: "f32[512, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_110: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-12);  getitem_122 = None
    rsqrt_19: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
    sub_29: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_109, getitem_123);  add_109 = getitem_123 = None
    mul_82: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = None
    mul_83: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_82, primals_248)
    add_111: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_83, primals_249);  mul_83 = primals_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_253: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_111, 3)
    unsqueeze_254: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 4);  unsqueeze_253 = None
    permute_423: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_254, [0, 1, 3, 4, 2]);  unsqueeze_254 = None
    unsqueeze_255: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_71, 3);  primals_71 = None
    unsqueeze_256: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 4);  unsqueeze_255 = None
    permute_424: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_256, [3, 4, 1, 2, 0]);  unsqueeze_256 = None
    permute_425: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_423, [0, 4, 1, 2, 3]);  permute_423 = None
    view_380: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_425, [1, 512, 1024]);  permute_425 = None
    permute_426: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_424, [4, 1, 2, 3, 0]);  permute_424 = None
    view_381: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_426, [1, 1024, 1024]);  permute_426 = None
    bmm_80: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_380, view_381)
    view_382: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_80, [512, 1, 1, 16, 64]);  bmm_80 = None
    permute_427: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_382, [0, 2, 3, 4, 1]);  view_382 = None
    view_383: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_427, [512, 1, 16, 64]);  permute_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_259: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_72, 3);  primals_72 = None
    unsqueeze_260: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 4);  unsqueeze_259 = None
    permute_429: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_260, [3, 4, 1, 2, 0]);  unsqueeze_260 = None
    permute_431: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_429, [4, 1, 2, 3, 0]);  permute_429 = None
    view_385: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_431, [1, 1024, 1024]);  permute_431 = None
    bmm_81: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_380, view_385)
    view_386: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_81, [512, 1, 1, 16, 64]);  bmm_81 = None
    permute_432: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_386, [0, 2, 3, 4, 1]);  view_386 = None
    view_387: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_432, [512, 1, 16, 64]);  permute_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_263: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_73, 3);  primals_73 = None
    unsqueeze_264: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 4);  unsqueeze_263 = None
    permute_434: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_264, [3, 4, 1, 2, 0]);  unsqueeze_264 = None
    permute_436: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_434, [4, 1, 2, 3, 0]);  permute_434 = None
    view_389: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_436, [1, 1024, 1024]);  permute_436 = None
    bmm_82: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_380, view_389)
    view_390: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_82, [512, 1, 1, 16, 64]);  bmm_82 = None
    permute_437: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_390, [0, 2, 3, 4, 1]);  view_390 = None
    view_391: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_437, [512, 1, 16, 64]);  permute_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_267: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_74, 3);  primals_74 = None
    unsqueeze_268: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 4);  unsqueeze_267 = None
    permute_439: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_268, [3, 4, 1, 2, 0]);  unsqueeze_268 = None
    permute_441: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_439, [4, 1, 2, 3, 0]);  permute_439 = None
    view_393: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_441, [1, 1024, 1024]);  permute_441 = None
    bmm_83: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_12, view_393);  view_393 = None
    view_394: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_83, [1024, 1, 1, 16, 64]);  bmm_83 = None
    permute_442: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_394, [0, 2, 3, 4, 1]);  view_394 = None
    view_395: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_442, [1024, 1, 16, 64]);  permute_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_112: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_383, primals_75);  primals_75 = None
    unsqueeze_269: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_112, 4);  add_112 = None
    permute_443: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_269, [1, 2, 0, 4, 3]);  unsqueeze_269 = None
    unsqueeze_270: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_387, 4);  view_387 = None
    permute_444: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_270, [1, 2, 4, 0, 3]);  unsqueeze_270 = None
    permute_445: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_443, [1, 2, 4, 0, 3]);  permute_443 = None
    view_396: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_445, [16, 512, 64]);  permute_445 = None
    permute_446: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_444, [1, 4, 0, 3, 2]);  permute_444 = None
    view_397: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_446, [16, 64, 512]);  permute_446 = None
    bmm_84: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_396, view_397)
    view_398: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_84, [16, 512, 1, 1, 512]);  bmm_84 = None
    permute_447: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_398, [3, 0, 1, 4, 2]);  view_398 = None
    view_399: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_447, [1, 16, 512, 512]);  permute_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_113: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_383, primals_76);  view_383 = primals_76 = None
    unsqueeze_271: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_113, 4);  add_113 = None
    permute_448: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_271, [1, 2, 0, 4, 3]);  unsqueeze_271 = None
    unsqueeze_272: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_395, 4);  view_395 = None
    permute_449: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_272, [1, 2, 4, 0, 3]);  unsqueeze_272 = None
    permute_450: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_448, [1, 2, 4, 0, 3]);  permute_448 = None
    view_400: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_450, [16, 512, 64]);  permute_450 = None
    permute_451: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_449, [1, 4, 0, 3, 2]);  permute_449 = None
    view_401: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_451, [16, 64, 1024]);  permute_451 = None
    bmm_85: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_400, view_401)
    view_402: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_85, [16, 512, 1, 1, 1024]);  bmm_85 = None
    permute_452: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_402, [3, 0, 1, 4, 2]);  view_402 = None
    view_403: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_452, [1, 16, 512, 1024]);  permute_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_404: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_403, [1, 16, 1024, 512]);  view_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_73: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_404, 0, 0, 9223372036854775807);  view_404 = None
    slice_74: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_73, 1, 0, 9223372036854775807);  slice_73 = None
    slice_75: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_74, 2, 1, 9223372036854775807);  slice_74 = None
    slice_76: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_75, 3, 0, 9223372036854775807);  slice_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_405: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_76, [1, 16, 512, 1023]);  slice_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    slice_77: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_405, 0, 0, 9223372036854775807);  view_405 = None
    slice_78: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_77, 1, 0, 9223372036854775807);  slice_77 = None
    slice_79: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_78, 2, 0, 9223372036854775807);  slice_78 = None
    index_10: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_79, [None, None, None, iota_2]);  slice_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_114: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_399, index_10);  view_399 = index_10 = None
    add_115: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_114, 0);  add_114 = None
    mul_84: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_115, 0.125);  add_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_10: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_84, [3], True)
    sub_30: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_84, amax_10);  mul_84 = amax_10 = None
    exp_10: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_30);  sub_30 = None
    sum_11: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [3], True)
    div_11: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    alias_10: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    native_dropout_42 = torch.ops.aten.native_dropout.default(div_11, 0.1, True);  div_11 = None
    getitem_124: "f32[1, 16, 512, 512]" = native_dropout_42[0]
    getitem_125: "b8[1, 16, 512, 512]" = native_dropout_42[1];  native_dropout_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_273: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(getitem_124, 4);  getitem_124 = None
    permute_453: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_273, [2, 0, 1, 4, 3]);  unsqueeze_273 = None
    unsqueeze_274: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_391, 4);  view_391 = None
    permute_454: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_274, [4, 1, 2, 3, 0]);  unsqueeze_274 = None
    permute_455: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_453, [2, 0, 4, 1, 3]);  permute_453 = None
    view_406: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_455, [16, 512, 512]);  permute_455 = None
    permute_456: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_454, [2, 4, 1, 3, 0]);  permute_454 = None
    view_407: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_456, [16, 512, 64]);  permute_456 = None
    bmm_86: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_406, view_407)
    view_408: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_86, [16, 512, 1, 1, 64]);  bmm_86 = None
    permute_457: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_408, [1, 3, 0, 4, 2]);  view_408 = None
    view_409: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_457, [512, 1, 16, 64]);  permute_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_275: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_409, 4);  view_409 = None
    permute_458: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_275, [0, 1, 4, 3, 2]);  unsqueeze_275 = None
    unsqueeze_276: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_77, 3);  primals_77 = None
    unsqueeze_277: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 4);  unsqueeze_276 = None
    permute_459: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_277, [3, 4, 0, 2, 1]);  unsqueeze_277 = None
    permute_460: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_458, [0, 3, 4, 1, 2]);  permute_458 = None
    clone_20: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_460, memory_format = torch.contiguous_format);  permute_460 = None
    view_410: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_20, [1, 512, 1024]);  clone_20 = None
    permute_461: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_459, [3, 4, 1, 2, 0]);  permute_459 = None
    clone_21: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_461, memory_format = torch.contiguous_format);  permute_461 = None
    view_411: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_21, [1, 1024, 1024]);  clone_21 = None
    bmm_87: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_410, view_411)
    view_412: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_87, [512, 1, 1, 1, 1024]);  bmm_87 = None
    permute_462: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_412, [0, 3, 4, 1, 2]);  view_412 = None
    view_413: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_462, [512, 1, 1024]);  permute_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    native_dropout_43 = torch.ops.aten.native_dropout.default(view_413, 0.1, True);  view_413 = None
    getitem_126: "f32[512, 1, 1024]" = native_dropout_43[0]
    getitem_127: "b8[512, 1, 1024]" = native_dropout_43[1];  native_dropout_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_116: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_126, add_111);  getitem_126 = add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_116, [2], correction = 0, keepdim = True)
    getitem_128: "f32[512, 1, 1]" = var_mean_20[0]
    getitem_129: "f32[512, 1, 1]" = var_mean_20[1];  var_mean_20 = None
    add_117: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-12);  getitem_128 = None
    rsqrt_20: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
    sub_31: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_116, getitem_129);  add_116 = getitem_129 = None
    mul_85: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_20);  sub_31 = None
    mul_86: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_85, primals_250)
    add_118: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_86, primals_251);  mul_86 = primals_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_414: "f32[512, 1024]" = torch.ops.aten.view.default(add_118, [512, 1024])
    permute_463: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_252, [1, 0]);  primals_252 = None
    addmm_20: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_253, view_414, permute_463);  primals_253 = None
    view_415: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_20, [512, 1, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_87: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_415, 0.5)
    mul_88: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_415, 0.7071067811865476);  view_415 = None
    erf_10: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
    add_119: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_89: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_87, add_119);  mul_87 = add_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    native_dropout_44 = torch.ops.aten.native_dropout.default(mul_89, 0.1, True);  mul_89 = None
    getitem_130: "f32[512, 1, 4096]" = native_dropout_44[0]
    getitem_131: "b8[512, 1, 4096]" = native_dropout_44[1];  native_dropout_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_416: "f32[512, 4096]" = torch.ops.aten.view.default(getitem_130, [512, 4096]);  getitem_130 = None
    permute_464: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_254, [1, 0]);  primals_254 = None
    addmm_21: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_255, view_416, permute_464);  primals_255 = None
    view_417: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_21, [512, 1, 1024]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    native_dropout_45 = torch.ops.aten.native_dropout.default(view_417, 0.1, True);  view_417 = None
    getitem_132: "f32[512, 1, 1024]" = native_dropout_45[0]
    getitem_133: "b8[512, 1, 1024]" = native_dropout_45[1];  native_dropout_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_120: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_132, add_118);  getitem_132 = add_118 = None
    var_mean_21 = torch.ops.aten.var_mean.correction(add_120, [2], correction = 0, keepdim = True)
    getitem_134: "f32[512, 1, 1]" = var_mean_21[0]
    getitem_135: "f32[512, 1, 1]" = var_mean_21[1];  var_mean_21 = None
    add_121: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_134, 1e-12);  getitem_134 = None
    rsqrt_21: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_121);  add_121 = None
    sub_32: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_120, getitem_135);  add_120 = getitem_135 = None
    mul_90: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = None
    mul_91: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_90, primals_256)
    add_122: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_91, primals_257);  mul_91 = primals_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_278: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_122, 3)
    unsqueeze_279: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 4);  unsqueeze_278 = None
    permute_465: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_279, [0, 1, 3, 4, 2]);  unsqueeze_279 = None
    unsqueeze_280: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_78, 3);  primals_78 = None
    unsqueeze_281: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 4);  unsqueeze_280 = None
    permute_466: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_281, [3, 4, 1, 2, 0]);  unsqueeze_281 = None
    permute_467: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_465, [0, 4, 1, 2, 3]);  permute_465 = None
    view_418: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_467, [1, 512, 1024]);  permute_467 = None
    permute_468: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_466, [4, 1, 2, 3, 0]);  permute_466 = None
    view_419: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_468, [1, 1024, 1024]);  permute_468 = None
    bmm_88: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_418, view_419)
    view_420: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_88, [512, 1, 1, 16, 64]);  bmm_88 = None
    permute_469: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_420, [0, 2, 3, 4, 1]);  view_420 = None
    view_421: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_469, [512, 1, 16, 64]);  permute_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_284: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_79, 3);  primals_79 = None
    unsqueeze_285: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 4);  unsqueeze_284 = None
    permute_471: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_285, [3, 4, 1, 2, 0]);  unsqueeze_285 = None
    permute_473: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_471, [4, 1, 2, 3, 0]);  permute_471 = None
    view_423: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_473, [1, 1024, 1024]);  permute_473 = None
    bmm_89: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_418, view_423)
    view_424: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_89, [512, 1, 1, 16, 64]);  bmm_89 = None
    permute_474: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_424, [0, 2, 3, 4, 1]);  view_424 = None
    view_425: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_474, [512, 1, 16, 64]);  permute_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_288: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_80, 3);  primals_80 = None
    unsqueeze_289: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 4);  unsqueeze_288 = None
    permute_476: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_289, [3, 4, 1, 2, 0]);  unsqueeze_289 = None
    permute_478: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_476, [4, 1, 2, 3, 0]);  permute_476 = None
    view_427: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_478, [1, 1024, 1024]);  permute_478 = None
    bmm_90: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_418, view_427)
    view_428: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_90, [512, 1, 1, 16, 64]);  bmm_90 = None
    permute_479: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_428, [0, 2, 3, 4, 1]);  view_428 = None
    view_429: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_479, [512, 1, 16, 64]);  permute_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_292: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_81, 3);  primals_81 = None
    unsqueeze_293: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 4);  unsqueeze_292 = None
    permute_481: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_293, [3, 4, 1, 2, 0]);  unsqueeze_293 = None
    permute_483: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_481, [4, 1, 2, 3, 0]);  permute_481 = None
    view_431: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_483, [1, 1024, 1024]);  permute_483 = None
    bmm_91: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_12, view_431);  view_431 = None
    view_432: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_91, [1024, 1, 1, 16, 64]);  bmm_91 = None
    permute_484: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_432, [0, 2, 3, 4, 1]);  view_432 = None
    view_433: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_484, [1024, 1, 16, 64]);  permute_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_123: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_421, primals_82);  primals_82 = None
    unsqueeze_294: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_123, 4);  add_123 = None
    permute_485: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_294, [1, 2, 0, 4, 3]);  unsqueeze_294 = None
    unsqueeze_295: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_425, 4);  view_425 = None
    permute_486: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_295, [1, 2, 4, 0, 3]);  unsqueeze_295 = None
    permute_487: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_485, [1, 2, 4, 0, 3]);  permute_485 = None
    view_434: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_487, [16, 512, 64]);  permute_487 = None
    permute_488: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_486, [1, 4, 0, 3, 2]);  permute_486 = None
    view_435: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_488, [16, 64, 512]);  permute_488 = None
    bmm_92: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_434, view_435)
    view_436: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_92, [16, 512, 1, 1, 512]);  bmm_92 = None
    permute_489: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_436, [3, 0, 1, 4, 2]);  view_436 = None
    view_437: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_489, [1, 16, 512, 512]);  permute_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_124: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_421, primals_83);  view_421 = primals_83 = None
    unsqueeze_296: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_124, 4);  add_124 = None
    permute_490: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_296, [1, 2, 0, 4, 3]);  unsqueeze_296 = None
    unsqueeze_297: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_433, 4);  view_433 = None
    permute_491: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_297, [1, 2, 4, 0, 3]);  unsqueeze_297 = None
    permute_492: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_490, [1, 2, 4, 0, 3]);  permute_490 = None
    view_438: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_492, [16, 512, 64]);  permute_492 = None
    permute_493: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_491, [1, 4, 0, 3, 2]);  permute_491 = None
    view_439: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_493, [16, 64, 1024]);  permute_493 = None
    bmm_93: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_438, view_439)
    view_440: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_93, [16, 512, 1, 1, 1024]);  bmm_93 = None
    permute_494: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_440, [3, 0, 1, 4, 2]);  view_440 = None
    view_441: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_494, [1, 16, 512, 1024]);  permute_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_442: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_441, [1, 16, 1024, 512]);  view_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_80: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_442, 0, 0, 9223372036854775807);  view_442 = None
    slice_81: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_80, 1, 0, 9223372036854775807);  slice_80 = None
    slice_82: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_81, 2, 1, 9223372036854775807);  slice_81 = None
    slice_83: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_82, 3, 0, 9223372036854775807);  slice_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_443: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_83, [1, 16, 512, 1023]);  slice_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    slice_84: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_443, 0, 0, 9223372036854775807);  view_443 = None
    slice_85: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_84, 1, 0, 9223372036854775807);  slice_84 = None
    slice_86: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_85, 2, 0, 9223372036854775807);  slice_85 = None
    index_11: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_86, [None, None, None, iota_2]);  slice_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_125: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_437, index_11);  view_437 = index_11 = None
    add_126: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_125, 0);  add_125 = None
    mul_92: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_126, 0.125);  add_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_11: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_92, [3], True)
    sub_33: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_92, amax_11);  mul_92 = amax_11 = None
    exp_11: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_33);  sub_33 = None
    sum_12: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [3], True)
    div_12: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    alias_11: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    native_dropout_46 = torch.ops.aten.native_dropout.default(div_12, 0.1, True);  div_12 = None
    getitem_136: "f32[1, 16, 512, 512]" = native_dropout_46[0]
    getitem_137: "b8[1, 16, 512, 512]" = native_dropout_46[1];  native_dropout_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_298: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(getitem_136, 4);  getitem_136 = None
    permute_495: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_298, [2, 0, 1, 4, 3]);  unsqueeze_298 = None
    unsqueeze_299: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_429, 4);  view_429 = None
    permute_496: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_299, [4, 1, 2, 3, 0]);  unsqueeze_299 = None
    permute_497: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_495, [2, 0, 4, 1, 3]);  permute_495 = None
    view_444: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_497, [16, 512, 512]);  permute_497 = None
    permute_498: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_496, [2, 4, 1, 3, 0]);  permute_496 = None
    view_445: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_498, [16, 512, 64]);  permute_498 = None
    bmm_94: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_444, view_445)
    view_446: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_94, [16, 512, 1, 1, 64]);  bmm_94 = None
    permute_499: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_446, [1, 3, 0, 4, 2]);  view_446 = None
    view_447: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_499, [512, 1, 16, 64]);  permute_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_300: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_447, 4);  view_447 = None
    permute_500: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_300, [0, 1, 4, 3, 2]);  unsqueeze_300 = None
    unsqueeze_301: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_84, 3);  primals_84 = None
    unsqueeze_302: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 4);  unsqueeze_301 = None
    permute_501: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_302, [3, 4, 0, 2, 1]);  unsqueeze_302 = None
    permute_502: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_500, [0, 3, 4, 1, 2]);  permute_500 = None
    clone_22: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_502, memory_format = torch.contiguous_format);  permute_502 = None
    view_448: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_22, [1, 512, 1024]);  clone_22 = None
    permute_503: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_501, [3, 4, 1, 2, 0]);  permute_501 = None
    clone_23: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_503, memory_format = torch.contiguous_format);  permute_503 = None
    view_449: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_23, [1, 1024, 1024]);  clone_23 = None
    bmm_95: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_448, view_449)
    view_450: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_95, [512, 1, 1, 1, 1024]);  bmm_95 = None
    permute_504: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_450, [0, 3, 4, 1, 2]);  view_450 = None
    view_451: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_504, [512, 1, 1024]);  permute_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    native_dropout_47 = torch.ops.aten.native_dropout.default(view_451, 0.1, True);  view_451 = None
    getitem_138: "f32[512, 1, 1024]" = native_dropout_47[0]
    getitem_139: "b8[512, 1, 1024]" = native_dropout_47[1];  native_dropout_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_127: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_138, add_122);  getitem_138 = add_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_127, [2], correction = 0, keepdim = True)
    getitem_140: "f32[512, 1, 1]" = var_mean_22[0]
    getitem_141: "f32[512, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_128: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_140, 1e-12);  getitem_140 = None
    rsqrt_22: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
    sub_34: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_127, getitem_141);  add_127 = getitem_141 = None
    mul_93: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = None
    mul_94: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_93, primals_258)
    add_129: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_94, primals_259);  mul_94 = primals_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_452: "f32[512, 1024]" = torch.ops.aten.view.default(add_129, [512, 1024])
    permute_505: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_260, [1, 0]);  primals_260 = None
    addmm_22: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_261, view_452, permute_505);  primals_261 = None
    view_453: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_22, [512, 1, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_95: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_453, 0.5)
    mul_96: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_453, 0.7071067811865476);  view_453 = None
    erf_11: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_96);  mul_96 = None
    add_130: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_97: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_95, add_130);  mul_95 = add_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    native_dropout_48 = torch.ops.aten.native_dropout.default(mul_97, 0.1, True);  mul_97 = None
    getitem_142: "f32[512, 1, 4096]" = native_dropout_48[0]
    getitem_143: "b8[512, 1, 4096]" = native_dropout_48[1];  native_dropout_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_454: "f32[512, 4096]" = torch.ops.aten.view.default(getitem_142, [512, 4096]);  getitem_142 = None
    permute_506: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_262, [1, 0]);  primals_262 = None
    addmm_23: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_263, view_454, permute_506);  primals_263 = None
    view_455: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_23, [512, 1, 1024]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    native_dropout_49 = torch.ops.aten.native_dropout.default(view_455, 0.1, True);  view_455 = None
    getitem_144: "f32[512, 1, 1024]" = native_dropout_49[0]
    getitem_145: "b8[512, 1, 1024]" = native_dropout_49[1];  native_dropout_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_131: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_144, add_129);  getitem_144 = add_129 = None
    var_mean_23 = torch.ops.aten.var_mean.correction(add_131, [2], correction = 0, keepdim = True)
    getitem_146: "f32[512, 1, 1]" = var_mean_23[0]
    getitem_147: "f32[512, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_132: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_146, 1e-12);  getitem_146 = None
    rsqrt_23: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
    sub_35: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_131, getitem_147);  add_131 = getitem_147 = None
    mul_98: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = None
    mul_99: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_98, primals_264)
    add_133: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_99, primals_265);  mul_99 = primals_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_303: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_133, 3)
    unsqueeze_304: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 4);  unsqueeze_303 = None
    permute_507: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_304, [0, 1, 3, 4, 2]);  unsqueeze_304 = None
    unsqueeze_305: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_85, 3);  primals_85 = None
    unsqueeze_306: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 4);  unsqueeze_305 = None
    permute_508: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_306, [3, 4, 1, 2, 0]);  unsqueeze_306 = None
    permute_509: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_507, [0, 4, 1, 2, 3]);  permute_507 = None
    view_456: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_509, [1, 512, 1024]);  permute_509 = None
    permute_510: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_508, [4, 1, 2, 3, 0]);  permute_508 = None
    view_457: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_510, [1, 1024, 1024]);  permute_510 = None
    bmm_96: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_456, view_457)
    view_458: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_96, [512, 1, 1, 16, 64]);  bmm_96 = None
    permute_511: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_458, [0, 2, 3, 4, 1]);  view_458 = None
    view_459: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_511, [512, 1, 16, 64]);  permute_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_309: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_86, 3);  primals_86 = None
    unsqueeze_310: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 4);  unsqueeze_309 = None
    permute_513: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_310, [3, 4, 1, 2, 0]);  unsqueeze_310 = None
    permute_515: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_513, [4, 1, 2, 3, 0]);  permute_513 = None
    view_461: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_515, [1, 1024, 1024]);  permute_515 = None
    bmm_97: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_456, view_461)
    view_462: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_97, [512, 1, 1, 16, 64]);  bmm_97 = None
    permute_516: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_462, [0, 2, 3, 4, 1]);  view_462 = None
    view_463: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_516, [512, 1, 16, 64]);  permute_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_313: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_87, 3);  primals_87 = None
    unsqueeze_314: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 4);  unsqueeze_313 = None
    permute_518: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_314, [3, 4, 1, 2, 0]);  unsqueeze_314 = None
    permute_520: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_518, [4, 1, 2, 3, 0]);  permute_518 = None
    view_465: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_520, [1, 1024, 1024]);  permute_520 = None
    bmm_98: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_456, view_465)
    view_466: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_98, [512, 1, 1, 16, 64]);  bmm_98 = None
    permute_521: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_466, [0, 2, 3, 4, 1]);  view_466 = None
    view_467: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_521, [512, 1, 16, 64]);  permute_521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_317: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_88, 3);  primals_88 = None
    unsqueeze_318: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 4);  unsqueeze_317 = None
    permute_523: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_318, [3, 4, 1, 2, 0]);  unsqueeze_318 = None
    permute_525: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_523, [4, 1, 2, 3, 0]);  permute_523 = None
    view_469: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_525, [1, 1024, 1024]);  permute_525 = None
    bmm_99: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_12, view_469);  view_469 = None
    view_470: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_99, [1024, 1, 1, 16, 64]);  bmm_99 = None
    permute_526: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_470, [0, 2, 3, 4, 1]);  view_470 = None
    view_471: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_526, [1024, 1, 16, 64]);  permute_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_134: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_459, primals_89);  primals_89 = None
    unsqueeze_319: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_134, 4);  add_134 = None
    permute_527: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_319, [1, 2, 0, 4, 3]);  unsqueeze_319 = None
    unsqueeze_320: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_463, 4);  view_463 = None
    permute_528: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_320, [1, 2, 4, 0, 3]);  unsqueeze_320 = None
    permute_529: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_527, [1, 2, 4, 0, 3]);  permute_527 = None
    view_472: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_529, [16, 512, 64]);  permute_529 = None
    permute_530: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_528, [1, 4, 0, 3, 2]);  permute_528 = None
    view_473: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_530, [16, 64, 512]);  permute_530 = None
    bmm_100: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_472, view_473)
    view_474: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_100, [16, 512, 1, 1, 512]);  bmm_100 = None
    permute_531: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_474, [3, 0, 1, 4, 2]);  view_474 = None
    view_475: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_531, [1, 16, 512, 512]);  permute_531 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_135: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_459, primals_90);  view_459 = primals_90 = None
    unsqueeze_321: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_135, 4);  add_135 = None
    permute_532: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_321, [1, 2, 0, 4, 3]);  unsqueeze_321 = None
    unsqueeze_322: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_471, 4);  view_471 = None
    permute_533: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_322, [1, 2, 4, 0, 3]);  unsqueeze_322 = None
    permute_534: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_532, [1, 2, 4, 0, 3]);  permute_532 = None
    view_476: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_534, [16, 512, 64]);  permute_534 = None
    permute_535: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_533, [1, 4, 0, 3, 2]);  permute_533 = None
    view_477: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_535, [16, 64, 1024]);  permute_535 = None
    bmm_101: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_476, view_477)
    view_478: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_101, [16, 512, 1, 1, 1024]);  bmm_101 = None
    permute_536: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_478, [3, 0, 1, 4, 2]);  view_478 = None
    view_479: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_536, [1, 16, 512, 1024]);  permute_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_480: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_479, [1, 16, 1024, 512]);  view_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_87: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_480, 0, 0, 9223372036854775807);  view_480 = None
    slice_88: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_87, 1, 0, 9223372036854775807);  slice_87 = None
    slice_89: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_88, 2, 1, 9223372036854775807);  slice_88 = None
    slice_90: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_89, 3, 0, 9223372036854775807);  slice_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_481: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_90, [1, 16, 512, 1023]);  slice_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    slice_91: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_481, 0, 0, 9223372036854775807);  view_481 = None
    slice_92: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_91, 1, 0, 9223372036854775807);  slice_91 = None
    slice_93: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_92, 2, 0, 9223372036854775807);  slice_92 = None
    index_12: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_93, [None, None, None, iota_2]);  slice_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_136: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_475, index_12);  view_475 = index_12 = None
    add_137: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_136, 0);  add_136 = None
    mul_100: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_137, 0.125);  add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_12: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_100, [3], True)
    sub_36: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_100, amax_12);  mul_100 = amax_12 = None
    exp_12: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_36);  sub_36 = None
    sum_13: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [3], True)
    div_13: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    alias_12: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    native_dropout_50 = torch.ops.aten.native_dropout.default(div_13, 0.1, True);  div_13 = None
    getitem_148: "f32[1, 16, 512, 512]" = native_dropout_50[0]
    getitem_149: "b8[1, 16, 512, 512]" = native_dropout_50[1];  native_dropout_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_323: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(getitem_148, 4);  getitem_148 = None
    permute_537: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_323, [2, 0, 1, 4, 3]);  unsqueeze_323 = None
    unsqueeze_324: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_467, 4);  view_467 = None
    permute_538: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_324, [4, 1, 2, 3, 0]);  unsqueeze_324 = None
    permute_539: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_537, [2, 0, 4, 1, 3]);  permute_537 = None
    view_482: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_539, [16, 512, 512]);  permute_539 = None
    permute_540: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_538, [2, 4, 1, 3, 0]);  permute_538 = None
    view_483: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_540, [16, 512, 64]);  permute_540 = None
    bmm_102: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_482, view_483)
    view_484: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_102, [16, 512, 1, 1, 64]);  bmm_102 = None
    permute_541: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_484, [1, 3, 0, 4, 2]);  view_484 = None
    view_485: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_541, [512, 1, 16, 64]);  permute_541 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_325: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_485, 4);  view_485 = None
    permute_542: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_325, [0, 1, 4, 3, 2]);  unsqueeze_325 = None
    unsqueeze_326: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_91, 3);  primals_91 = None
    unsqueeze_327: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 4);  unsqueeze_326 = None
    permute_543: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_327, [3, 4, 0, 2, 1]);  unsqueeze_327 = None
    permute_544: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_542, [0, 3, 4, 1, 2]);  permute_542 = None
    clone_24: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_544, memory_format = torch.contiguous_format);  permute_544 = None
    view_486: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_24, [1, 512, 1024]);  clone_24 = None
    permute_545: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_543, [3, 4, 1, 2, 0]);  permute_543 = None
    clone_25: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_545, memory_format = torch.contiguous_format);  permute_545 = None
    view_487: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_25, [1, 1024, 1024]);  clone_25 = None
    bmm_103: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_486, view_487)
    view_488: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_103, [512, 1, 1, 1, 1024]);  bmm_103 = None
    permute_546: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_488, [0, 3, 4, 1, 2]);  view_488 = None
    view_489: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_546, [512, 1, 1024]);  permute_546 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    native_dropout_51 = torch.ops.aten.native_dropout.default(view_489, 0.1, True);  view_489 = None
    getitem_150: "f32[512, 1, 1024]" = native_dropout_51[0]
    getitem_151: "b8[512, 1, 1024]" = native_dropout_51[1];  native_dropout_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_138: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_150, add_133);  getitem_150 = add_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_138, [2], correction = 0, keepdim = True)
    getitem_152: "f32[512, 1, 1]" = var_mean_24[0]
    getitem_153: "f32[512, 1, 1]" = var_mean_24[1];  var_mean_24 = None
    add_139: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_152, 1e-12);  getitem_152 = None
    rsqrt_24: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
    sub_37: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_138, getitem_153);  add_138 = getitem_153 = None
    mul_101: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_24);  sub_37 = None
    mul_102: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_101, primals_266)
    add_140: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_102, primals_267);  mul_102 = primals_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_490: "f32[512, 1024]" = torch.ops.aten.view.default(add_140, [512, 1024])
    permute_547: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_268, [1, 0]);  primals_268 = None
    addmm_24: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_269, view_490, permute_547);  primals_269 = None
    view_491: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_24, [512, 1, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_103: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_491, 0.5)
    mul_104: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_491, 0.7071067811865476);  view_491 = None
    erf_12: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_104);  mul_104 = None
    add_141: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_105: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_103, add_141);  mul_103 = add_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    native_dropout_52 = torch.ops.aten.native_dropout.default(mul_105, 0.1, True);  mul_105 = None
    getitem_154: "f32[512, 1, 4096]" = native_dropout_52[0]
    getitem_155: "b8[512, 1, 4096]" = native_dropout_52[1];  native_dropout_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_492: "f32[512, 4096]" = torch.ops.aten.view.default(getitem_154, [512, 4096]);  getitem_154 = None
    permute_548: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_270, [1, 0]);  primals_270 = None
    addmm_25: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_271, view_492, permute_548);  primals_271 = None
    view_493: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_25, [512, 1, 1024]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    native_dropout_53 = torch.ops.aten.native_dropout.default(view_493, 0.1, True);  view_493 = None
    getitem_156: "f32[512, 1, 1024]" = native_dropout_53[0]
    getitem_157: "b8[512, 1, 1024]" = native_dropout_53[1];  native_dropout_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_142: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_156, add_140);  getitem_156 = add_140 = None
    var_mean_25 = torch.ops.aten.var_mean.correction(add_142, [2], correction = 0, keepdim = True)
    getitem_158: "f32[512, 1, 1]" = var_mean_25[0]
    getitem_159: "f32[512, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_143: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-12);  getitem_158 = None
    rsqrt_25: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
    sub_38: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_142, getitem_159);  add_142 = getitem_159 = None
    mul_106: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_25);  sub_38 = None
    mul_107: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_106, primals_272)
    add_144: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_107, primals_273);  mul_107 = primals_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_328: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_144, 3)
    unsqueeze_329: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 4);  unsqueeze_328 = None
    permute_549: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_329, [0, 1, 3, 4, 2]);  unsqueeze_329 = None
    unsqueeze_330: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_92, 3);  primals_92 = None
    unsqueeze_331: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, 4);  unsqueeze_330 = None
    permute_550: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_331, [3, 4, 1, 2, 0]);  unsqueeze_331 = None
    permute_551: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_549, [0, 4, 1, 2, 3]);  permute_549 = None
    view_494: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_551, [1, 512, 1024]);  permute_551 = None
    permute_552: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_550, [4, 1, 2, 3, 0]);  permute_550 = None
    view_495: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_552, [1, 1024, 1024]);  permute_552 = None
    bmm_104: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_494, view_495)
    view_496: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_104, [512, 1, 1, 16, 64]);  bmm_104 = None
    permute_553: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_496, [0, 2, 3, 4, 1]);  view_496 = None
    view_497: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_553, [512, 1, 16, 64]);  permute_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_334: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_93, 3);  primals_93 = None
    unsqueeze_335: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 4);  unsqueeze_334 = None
    permute_555: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_335, [3, 4, 1, 2, 0]);  unsqueeze_335 = None
    permute_557: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_555, [4, 1, 2, 3, 0]);  permute_555 = None
    view_499: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_557, [1, 1024, 1024]);  permute_557 = None
    bmm_105: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_494, view_499)
    view_500: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_105, [512, 1, 1, 16, 64]);  bmm_105 = None
    permute_558: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_500, [0, 2, 3, 4, 1]);  view_500 = None
    view_501: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_558, [512, 1, 16, 64]);  permute_558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_338: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_94, 3);  primals_94 = None
    unsqueeze_339: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 4);  unsqueeze_338 = None
    permute_560: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_339, [3, 4, 1, 2, 0]);  unsqueeze_339 = None
    permute_562: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_560, [4, 1, 2, 3, 0]);  permute_560 = None
    view_503: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_562, [1, 1024, 1024]);  permute_562 = None
    bmm_106: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_494, view_503)
    view_504: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_106, [512, 1, 1, 16, 64]);  bmm_106 = None
    permute_563: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_504, [0, 2, 3, 4, 1]);  view_504 = None
    view_505: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_563, [512, 1, 16, 64]);  permute_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_342: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_95, 3);  primals_95 = None
    unsqueeze_343: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 4);  unsqueeze_342 = None
    permute_565: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_343, [3, 4, 1, 2, 0]);  unsqueeze_343 = None
    permute_567: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_565, [4, 1, 2, 3, 0]);  permute_565 = None
    view_507: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_567, [1, 1024, 1024]);  permute_567 = None
    bmm_107: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_12, view_507);  view_507 = None
    view_508: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_107, [1024, 1, 1, 16, 64]);  bmm_107 = None
    permute_568: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_508, [0, 2, 3, 4, 1]);  view_508 = None
    view_509: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_568, [1024, 1, 16, 64]);  permute_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_145: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_497, primals_96);  primals_96 = None
    unsqueeze_344: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_145, 4);  add_145 = None
    permute_569: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_344, [1, 2, 0, 4, 3]);  unsqueeze_344 = None
    unsqueeze_345: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_501, 4);  view_501 = None
    permute_570: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_345, [1, 2, 4, 0, 3]);  unsqueeze_345 = None
    permute_571: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_569, [1, 2, 4, 0, 3]);  permute_569 = None
    view_510: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_571, [16, 512, 64]);  permute_571 = None
    permute_572: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_570, [1, 4, 0, 3, 2]);  permute_570 = None
    view_511: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_572, [16, 64, 512]);  permute_572 = None
    bmm_108: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_510, view_511)
    view_512: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_108, [16, 512, 1, 1, 512]);  bmm_108 = None
    permute_573: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_512, [3, 0, 1, 4, 2]);  view_512 = None
    view_513: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_573, [1, 16, 512, 512]);  permute_573 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_146: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_497, primals_97);  view_497 = primals_97 = None
    unsqueeze_346: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_146, 4);  add_146 = None
    permute_574: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_346, [1, 2, 0, 4, 3]);  unsqueeze_346 = None
    unsqueeze_347: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_509, 4);  view_509 = None
    permute_575: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_347, [1, 2, 4, 0, 3]);  unsqueeze_347 = None
    permute_576: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_574, [1, 2, 4, 0, 3]);  permute_574 = None
    view_514: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_576, [16, 512, 64]);  permute_576 = None
    permute_577: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_575, [1, 4, 0, 3, 2]);  permute_575 = None
    view_515: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_577, [16, 64, 1024]);  permute_577 = None
    bmm_109: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_514, view_515)
    view_516: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_109, [16, 512, 1, 1, 1024]);  bmm_109 = None
    permute_578: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_516, [3, 0, 1, 4, 2]);  view_516 = None
    view_517: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_578, [1, 16, 512, 1024]);  permute_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_518: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_517, [1, 16, 1024, 512]);  view_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_94: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_518, 0, 0, 9223372036854775807);  view_518 = None
    slice_95: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_94, 1, 0, 9223372036854775807);  slice_94 = None
    slice_96: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_95, 2, 1, 9223372036854775807);  slice_95 = None
    slice_97: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_96, 3, 0, 9223372036854775807);  slice_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_519: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_97, [1, 16, 512, 1023]);  slice_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    slice_98: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_519, 0, 0, 9223372036854775807);  view_519 = None
    slice_99: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_98, 1, 0, 9223372036854775807);  slice_98 = None
    slice_100: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_99, 2, 0, 9223372036854775807);  slice_99 = None
    index_13: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_100, [None, None, None, iota_2]);  slice_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_147: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_513, index_13);  view_513 = index_13 = None
    add_148: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_147, 0);  add_147 = None
    mul_108: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_148, 0.125);  add_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_13: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_108, [3], True)
    sub_39: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_108, amax_13);  mul_108 = amax_13 = None
    exp_13: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_39);  sub_39 = None
    sum_14: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [3], True)
    div_14: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
    alias_13: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    native_dropout_54 = torch.ops.aten.native_dropout.default(div_14, 0.1, True);  div_14 = None
    getitem_160: "f32[1, 16, 512, 512]" = native_dropout_54[0]
    getitem_161: "b8[1, 16, 512, 512]" = native_dropout_54[1];  native_dropout_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_348: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(getitem_160, 4);  getitem_160 = None
    permute_579: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_348, [2, 0, 1, 4, 3]);  unsqueeze_348 = None
    unsqueeze_349: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_505, 4);  view_505 = None
    permute_580: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_349, [4, 1, 2, 3, 0]);  unsqueeze_349 = None
    permute_581: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_579, [2, 0, 4, 1, 3]);  permute_579 = None
    view_520: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_581, [16, 512, 512]);  permute_581 = None
    permute_582: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_580, [2, 4, 1, 3, 0]);  permute_580 = None
    view_521: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_582, [16, 512, 64]);  permute_582 = None
    bmm_110: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_520, view_521)
    view_522: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_110, [16, 512, 1, 1, 64]);  bmm_110 = None
    permute_583: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_522, [1, 3, 0, 4, 2]);  view_522 = None
    view_523: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_583, [512, 1, 16, 64]);  permute_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_350: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_523, 4);  view_523 = None
    permute_584: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_350, [0, 1, 4, 3, 2]);  unsqueeze_350 = None
    unsqueeze_351: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_98, 3);  primals_98 = None
    unsqueeze_352: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 4);  unsqueeze_351 = None
    permute_585: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_352, [3, 4, 0, 2, 1]);  unsqueeze_352 = None
    permute_586: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_584, [0, 3, 4, 1, 2]);  permute_584 = None
    clone_26: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_586, memory_format = torch.contiguous_format);  permute_586 = None
    view_524: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_26, [1, 512, 1024]);  clone_26 = None
    permute_587: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_585, [3, 4, 1, 2, 0]);  permute_585 = None
    clone_27: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_587, memory_format = torch.contiguous_format);  permute_587 = None
    view_525: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_27, [1, 1024, 1024]);  clone_27 = None
    bmm_111: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_524, view_525)
    view_526: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_111, [512, 1, 1, 1, 1024]);  bmm_111 = None
    permute_588: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_526, [0, 3, 4, 1, 2]);  view_526 = None
    view_527: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_588, [512, 1, 1024]);  permute_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    native_dropout_55 = torch.ops.aten.native_dropout.default(view_527, 0.1, True);  view_527 = None
    getitem_162: "f32[512, 1, 1024]" = native_dropout_55[0]
    getitem_163: "b8[512, 1, 1024]" = native_dropout_55[1];  native_dropout_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_149: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_162, add_144);  getitem_162 = add_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_26 = torch.ops.aten.var_mean.correction(add_149, [2], correction = 0, keepdim = True)
    getitem_164: "f32[512, 1, 1]" = var_mean_26[0]
    getitem_165: "f32[512, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_150: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_164, 1e-12);  getitem_164 = None
    rsqrt_26: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
    sub_40: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_149, getitem_165);  add_149 = getitem_165 = None
    mul_109: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_26);  sub_40 = None
    mul_110: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_109, primals_274)
    add_151: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_110, primals_275);  mul_110 = primals_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_528: "f32[512, 1024]" = torch.ops.aten.view.default(add_151, [512, 1024])
    permute_589: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_276, [1, 0]);  primals_276 = None
    addmm_26: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_277, view_528, permute_589);  primals_277 = None
    view_529: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_26, [512, 1, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_111: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_529, 0.5)
    mul_112: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_529, 0.7071067811865476);  view_529 = None
    erf_13: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_112);  mul_112 = None
    add_152: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_113: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_111, add_152);  mul_111 = add_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    native_dropout_56 = torch.ops.aten.native_dropout.default(mul_113, 0.1, True);  mul_113 = None
    getitem_166: "f32[512, 1, 4096]" = native_dropout_56[0]
    getitem_167: "b8[512, 1, 4096]" = native_dropout_56[1];  native_dropout_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_530: "f32[512, 4096]" = torch.ops.aten.view.default(getitem_166, [512, 4096]);  getitem_166 = None
    permute_590: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_278, [1, 0]);  primals_278 = None
    addmm_27: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_279, view_530, permute_590);  primals_279 = None
    view_531: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_27, [512, 1, 1024]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    native_dropout_57 = torch.ops.aten.native_dropout.default(view_531, 0.1, True);  view_531 = None
    getitem_168: "f32[512, 1, 1024]" = native_dropout_57[0]
    getitem_169: "b8[512, 1, 1024]" = native_dropout_57[1];  native_dropout_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_153: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_168, add_151);  getitem_168 = add_151 = None
    var_mean_27 = torch.ops.aten.var_mean.correction(add_153, [2], correction = 0, keepdim = True)
    getitem_170: "f32[512, 1, 1]" = var_mean_27[0]
    getitem_171: "f32[512, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_154: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_170, 1e-12);  getitem_170 = None
    rsqrt_27: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
    sub_41: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_153, getitem_171);  add_153 = getitem_171 = None
    mul_114: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_27);  sub_41 = None
    mul_115: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_114, primals_280)
    add_155: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_115, primals_281);  mul_115 = primals_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_353: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_155, 3)
    unsqueeze_354: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 4);  unsqueeze_353 = None
    permute_591: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_354, [0, 1, 3, 4, 2]);  unsqueeze_354 = None
    unsqueeze_355: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_99, 3);  primals_99 = None
    unsqueeze_356: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 4);  unsqueeze_355 = None
    permute_592: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_356, [3, 4, 1, 2, 0]);  unsqueeze_356 = None
    permute_593: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_591, [0, 4, 1, 2, 3]);  permute_591 = None
    view_532: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_593, [1, 512, 1024]);  permute_593 = None
    permute_594: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_592, [4, 1, 2, 3, 0]);  permute_592 = None
    view_533: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_594, [1, 1024, 1024]);  permute_594 = None
    bmm_112: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_532, view_533)
    view_534: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_112, [512, 1, 1, 16, 64]);  bmm_112 = None
    permute_595: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_534, [0, 2, 3, 4, 1]);  view_534 = None
    view_535: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_595, [512, 1, 16, 64]);  permute_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_359: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_100, 3);  primals_100 = None
    unsqueeze_360: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 4);  unsqueeze_359 = None
    permute_597: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_360, [3, 4, 1, 2, 0]);  unsqueeze_360 = None
    permute_599: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_597, [4, 1, 2, 3, 0]);  permute_597 = None
    view_537: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_599, [1, 1024, 1024]);  permute_599 = None
    bmm_113: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_532, view_537)
    view_538: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_113, [512, 1, 1, 16, 64]);  bmm_113 = None
    permute_600: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_538, [0, 2, 3, 4, 1]);  view_538 = None
    view_539: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_600, [512, 1, 16, 64]);  permute_600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_363: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_101, 3);  primals_101 = None
    unsqueeze_364: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 4);  unsqueeze_363 = None
    permute_602: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_364, [3, 4, 1, 2, 0]);  unsqueeze_364 = None
    permute_604: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_602, [4, 1, 2, 3, 0]);  permute_602 = None
    view_541: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_604, [1, 1024, 1024]);  permute_604 = None
    bmm_114: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_532, view_541)
    view_542: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_114, [512, 1, 1, 16, 64]);  bmm_114 = None
    permute_605: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_542, [0, 2, 3, 4, 1]);  view_542 = None
    view_543: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_605, [512, 1, 16, 64]);  permute_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_367: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_102, 3);  primals_102 = None
    unsqueeze_368: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 4);  unsqueeze_367 = None
    permute_607: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_368, [3, 4, 1, 2, 0]);  unsqueeze_368 = None
    permute_609: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_607, [4, 1, 2, 3, 0]);  permute_607 = None
    view_545: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_609, [1, 1024, 1024]);  permute_609 = None
    bmm_115: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_12, view_545);  view_545 = None
    view_546: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_115, [1024, 1, 1, 16, 64]);  bmm_115 = None
    permute_610: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_546, [0, 2, 3, 4, 1]);  view_546 = None
    view_547: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_610, [1024, 1, 16, 64]);  permute_610 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_156: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_535, primals_103);  primals_103 = None
    unsqueeze_369: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_156, 4);  add_156 = None
    permute_611: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_369, [1, 2, 0, 4, 3]);  unsqueeze_369 = None
    unsqueeze_370: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_539, 4);  view_539 = None
    permute_612: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_370, [1, 2, 4, 0, 3]);  unsqueeze_370 = None
    permute_613: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_611, [1, 2, 4, 0, 3]);  permute_611 = None
    view_548: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_613, [16, 512, 64]);  permute_613 = None
    permute_614: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_612, [1, 4, 0, 3, 2]);  permute_612 = None
    view_549: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_614, [16, 64, 512]);  permute_614 = None
    bmm_116: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_548, view_549)
    view_550: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_116, [16, 512, 1, 1, 512]);  bmm_116 = None
    permute_615: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_550, [3, 0, 1, 4, 2]);  view_550 = None
    view_551: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_615, [1, 16, 512, 512]);  permute_615 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_157: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_535, primals_104);  view_535 = primals_104 = None
    unsqueeze_371: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_157, 4);  add_157 = None
    permute_616: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_371, [1, 2, 0, 4, 3]);  unsqueeze_371 = None
    unsqueeze_372: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_547, 4);  view_547 = None
    permute_617: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_372, [1, 2, 4, 0, 3]);  unsqueeze_372 = None
    permute_618: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_616, [1, 2, 4, 0, 3]);  permute_616 = None
    view_552: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_618, [16, 512, 64]);  permute_618 = None
    permute_619: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_617, [1, 4, 0, 3, 2]);  permute_617 = None
    view_553: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_619, [16, 64, 1024]);  permute_619 = None
    bmm_117: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_552, view_553)
    view_554: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_117, [16, 512, 1, 1, 1024]);  bmm_117 = None
    permute_620: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_554, [3, 0, 1, 4, 2]);  view_554 = None
    view_555: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_620, [1, 16, 512, 1024]);  permute_620 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_556: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_555, [1, 16, 1024, 512]);  view_555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_101: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_556, 0, 0, 9223372036854775807);  view_556 = None
    slice_102: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_101, 1, 0, 9223372036854775807);  slice_101 = None
    slice_103: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_102, 2, 1, 9223372036854775807);  slice_102 = None
    slice_104: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_103, 3, 0, 9223372036854775807);  slice_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_557: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_104, [1, 16, 512, 1023]);  slice_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    slice_105: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_557, 0, 0, 9223372036854775807);  view_557 = None
    slice_106: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_105, 1, 0, 9223372036854775807);  slice_105 = None
    slice_107: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_106, 2, 0, 9223372036854775807);  slice_106 = None
    index_14: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_107, [None, None, None, iota_2]);  slice_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_158: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_551, index_14);  view_551 = index_14 = None
    add_159: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_158, 0);  add_158 = None
    mul_116: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_159, 0.125);  add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_14: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_116, [3], True)
    sub_42: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_116, amax_14);  mul_116 = amax_14 = None
    exp_14: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_42);  sub_42 = None
    sum_15: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [3], True)
    div_15: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
    alias_14: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    native_dropout_58 = torch.ops.aten.native_dropout.default(div_15, 0.1, True);  div_15 = None
    getitem_172: "f32[1, 16, 512, 512]" = native_dropout_58[0]
    getitem_173: "b8[1, 16, 512, 512]" = native_dropout_58[1];  native_dropout_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_373: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(getitem_172, 4);  getitem_172 = None
    permute_621: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_373, [2, 0, 1, 4, 3]);  unsqueeze_373 = None
    unsqueeze_374: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_543, 4);  view_543 = None
    permute_622: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_374, [4, 1, 2, 3, 0]);  unsqueeze_374 = None
    permute_623: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_621, [2, 0, 4, 1, 3]);  permute_621 = None
    view_558: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_623, [16, 512, 512]);  permute_623 = None
    permute_624: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_622, [2, 4, 1, 3, 0]);  permute_622 = None
    view_559: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_624, [16, 512, 64]);  permute_624 = None
    bmm_118: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_558, view_559)
    view_560: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_118, [16, 512, 1, 1, 64]);  bmm_118 = None
    permute_625: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_560, [1, 3, 0, 4, 2]);  view_560 = None
    view_561: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_625, [512, 1, 16, 64]);  permute_625 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_375: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_561, 4);  view_561 = None
    permute_626: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_375, [0, 1, 4, 3, 2]);  unsqueeze_375 = None
    unsqueeze_376: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_105, 3);  primals_105 = None
    unsqueeze_377: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 4);  unsqueeze_376 = None
    permute_627: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_377, [3, 4, 0, 2, 1]);  unsqueeze_377 = None
    permute_628: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_626, [0, 3, 4, 1, 2]);  permute_626 = None
    clone_28: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_628, memory_format = torch.contiguous_format);  permute_628 = None
    view_562: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_28, [1, 512, 1024]);  clone_28 = None
    permute_629: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_627, [3, 4, 1, 2, 0]);  permute_627 = None
    clone_29: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_629, memory_format = torch.contiguous_format);  permute_629 = None
    view_563: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_29, [1, 1024, 1024]);  clone_29 = None
    bmm_119: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_562, view_563)
    view_564: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_119, [512, 1, 1, 1, 1024]);  bmm_119 = None
    permute_630: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_564, [0, 3, 4, 1, 2]);  view_564 = None
    view_565: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_630, [512, 1, 1024]);  permute_630 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    native_dropout_59 = torch.ops.aten.native_dropout.default(view_565, 0.1, True);  view_565 = None
    getitem_174: "f32[512, 1, 1024]" = native_dropout_59[0]
    getitem_175: "b8[512, 1, 1024]" = native_dropout_59[1];  native_dropout_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_160: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_174, add_155);  getitem_174 = add_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_28 = torch.ops.aten.var_mean.correction(add_160, [2], correction = 0, keepdim = True)
    getitem_176: "f32[512, 1, 1]" = var_mean_28[0]
    getitem_177: "f32[512, 1, 1]" = var_mean_28[1];  var_mean_28 = None
    add_161: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-12);  getitem_176 = None
    rsqrt_28: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    sub_43: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_160, getitem_177);  add_160 = getitem_177 = None
    mul_117: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_28);  sub_43 = None
    mul_118: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_117, primals_282)
    add_162: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_118, primals_283);  mul_118 = primals_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_566: "f32[512, 1024]" = torch.ops.aten.view.default(add_162, [512, 1024])
    permute_631: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_284, [1, 0]);  primals_284 = None
    addmm_28: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_285, view_566, permute_631);  primals_285 = None
    view_567: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_28, [512, 1, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_119: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_567, 0.5)
    mul_120: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_567, 0.7071067811865476);  view_567 = None
    erf_14: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_120);  mul_120 = None
    add_163: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_121: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_119, add_163);  mul_119 = add_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    native_dropout_60 = torch.ops.aten.native_dropout.default(mul_121, 0.1, True);  mul_121 = None
    getitem_178: "f32[512, 1, 4096]" = native_dropout_60[0]
    getitem_179: "b8[512, 1, 4096]" = native_dropout_60[1];  native_dropout_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_568: "f32[512, 4096]" = torch.ops.aten.view.default(getitem_178, [512, 4096]);  getitem_178 = None
    permute_632: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_286, [1, 0]);  primals_286 = None
    addmm_29: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_287, view_568, permute_632);  primals_287 = None
    view_569: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_29, [512, 1, 1024]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    native_dropout_61 = torch.ops.aten.native_dropout.default(view_569, 0.1, True);  view_569 = None
    getitem_180: "f32[512, 1, 1024]" = native_dropout_61[0]
    getitem_181: "b8[512, 1, 1024]" = native_dropout_61[1];  native_dropout_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_164: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_180, add_162);  getitem_180 = add_162 = None
    var_mean_29 = torch.ops.aten.var_mean.correction(add_164, [2], correction = 0, keepdim = True)
    getitem_182: "f32[512, 1, 1]" = var_mean_29[0]
    getitem_183: "f32[512, 1, 1]" = var_mean_29[1];  var_mean_29 = None
    add_165: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_182, 1e-12);  getitem_182 = None
    rsqrt_29: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
    sub_44: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_164, getitem_183);  add_164 = getitem_183 = None
    mul_122: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_29);  sub_44 = None
    mul_123: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_122, primals_288)
    add_166: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_123, primals_289);  mul_123 = primals_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_378: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_166, 3)
    unsqueeze_379: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, 4);  unsqueeze_378 = None
    permute_633: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_379, [0, 1, 3, 4, 2]);  unsqueeze_379 = None
    unsqueeze_380: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_106, 3);  primals_106 = None
    unsqueeze_381: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 4);  unsqueeze_380 = None
    permute_634: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_381, [3, 4, 1, 2, 0]);  unsqueeze_381 = None
    permute_635: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_633, [0, 4, 1, 2, 3]);  permute_633 = None
    view_570: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_635, [1, 512, 1024]);  permute_635 = None
    permute_636: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_634, [4, 1, 2, 3, 0]);  permute_634 = None
    view_571: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_636, [1, 1024, 1024]);  permute_636 = None
    bmm_120: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_570, view_571)
    view_572: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_120, [512, 1, 1, 16, 64]);  bmm_120 = None
    permute_637: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_572, [0, 2, 3, 4, 1]);  view_572 = None
    view_573: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_637, [512, 1, 16, 64]);  permute_637 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_384: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_107, 3);  primals_107 = None
    unsqueeze_385: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, 4);  unsqueeze_384 = None
    permute_639: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_385, [3, 4, 1, 2, 0]);  unsqueeze_385 = None
    permute_641: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_639, [4, 1, 2, 3, 0]);  permute_639 = None
    view_575: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_641, [1, 1024, 1024]);  permute_641 = None
    bmm_121: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_570, view_575)
    view_576: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_121, [512, 1, 1, 16, 64]);  bmm_121 = None
    permute_642: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_576, [0, 2, 3, 4, 1]);  view_576 = None
    view_577: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_642, [512, 1, 16, 64]);  permute_642 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_388: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_108, 3);  primals_108 = None
    unsqueeze_389: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 4);  unsqueeze_388 = None
    permute_644: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_389, [3, 4, 1, 2, 0]);  unsqueeze_389 = None
    permute_646: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_644, [4, 1, 2, 3, 0]);  permute_644 = None
    view_579: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_646, [1, 1024, 1024]);  permute_646 = None
    bmm_122: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_570, view_579)
    view_580: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_122, [512, 1, 1, 16, 64]);  bmm_122 = None
    permute_647: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_580, [0, 2, 3, 4, 1]);  view_580 = None
    view_581: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_647, [512, 1, 16, 64]);  permute_647 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_392: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_109, 3);  primals_109 = None
    unsqueeze_393: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 4);  unsqueeze_392 = None
    permute_649: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_393, [3, 4, 1, 2, 0]);  unsqueeze_393 = None
    permute_651: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_649, [4, 1, 2, 3, 0]);  permute_649 = None
    view_583: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_651, [1, 1024, 1024]);  permute_651 = None
    bmm_123: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_12, view_583);  view_583 = None
    view_584: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_123, [1024, 1, 1, 16, 64]);  bmm_123 = None
    permute_652: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_584, [0, 2, 3, 4, 1]);  view_584 = None
    view_585: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_652, [1024, 1, 16, 64]);  permute_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_167: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_573, primals_110);  primals_110 = None
    unsqueeze_394: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_167, 4);  add_167 = None
    permute_653: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_394, [1, 2, 0, 4, 3]);  unsqueeze_394 = None
    unsqueeze_395: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_577, 4);  view_577 = None
    permute_654: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_395, [1, 2, 4, 0, 3]);  unsqueeze_395 = None
    permute_655: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_653, [1, 2, 4, 0, 3]);  permute_653 = None
    view_586: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_655, [16, 512, 64]);  permute_655 = None
    permute_656: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_654, [1, 4, 0, 3, 2]);  permute_654 = None
    view_587: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_656, [16, 64, 512]);  permute_656 = None
    bmm_124: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_586, view_587)
    view_588: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_124, [16, 512, 1, 1, 512]);  bmm_124 = None
    permute_657: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_588, [3, 0, 1, 4, 2]);  view_588 = None
    view_589: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_657, [1, 16, 512, 512]);  permute_657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_168: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_573, primals_111);  view_573 = primals_111 = None
    unsqueeze_396: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_168, 4);  add_168 = None
    permute_658: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_396, [1, 2, 0, 4, 3]);  unsqueeze_396 = None
    unsqueeze_397: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_585, 4);  view_585 = None
    permute_659: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_397, [1, 2, 4, 0, 3]);  unsqueeze_397 = None
    permute_660: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_658, [1, 2, 4, 0, 3]);  permute_658 = None
    view_590: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_660, [16, 512, 64]);  permute_660 = None
    permute_661: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_659, [1, 4, 0, 3, 2]);  permute_659 = None
    view_591: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_661, [16, 64, 1024]);  permute_661 = None
    bmm_125: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_590, view_591)
    view_592: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_125, [16, 512, 1, 1, 1024]);  bmm_125 = None
    permute_662: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_592, [3, 0, 1, 4, 2]);  view_592 = None
    view_593: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_662, [1, 16, 512, 1024]);  permute_662 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_594: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_593, [1, 16, 1024, 512]);  view_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_108: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_594, 0, 0, 9223372036854775807);  view_594 = None
    slice_109: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_108, 1, 0, 9223372036854775807);  slice_108 = None
    slice_110: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_109, 2, 1, 9223372036854775807);  slice_109 = None
    slice_111: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_110, 3, 0, 9223372036854775807);  slice_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_595: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_111, [1, 16, 512, 1023]);  slice_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    slice_112: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_595, 0, 0, 9223372036854775807);  view_595 = None
    slice_113: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_112, 1, 0, 9223372036854775807);  slice_112 = None
    slice_114: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_113, 2, 0, 9223372036854775807);  slice_113 = None
    index_15: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_114, [None, None, None, iota_2]);  slice_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_169: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_589, index_15);  view_589 = index_15 = None
    add_170: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_169, 0);  add_169 = None
    mul_124: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_170, 0.125);  add_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_15: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_124, [3], True)
    sub_45: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_124, amax_15);  mul_124 = amax_15 = None
    exp_15: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_45);  sub_45 = None
    sum_16: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [3], True)
    div_16: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
    alias_15: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    native_dropout_62 = torch.ops.aten.native_dropout.default(div_16, 0.1, True);  div_16 = None
    getitem_184: "f32[1, 16, 512, 512]" = native_dropout_62[0]
    getitem_185: "b8[1, 16, 512, 512]" = native_dropout_62[1];  native_dropout_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_398: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(getitem_184, 4);  getitem_184 = None
    permute_663: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_398, [2, 0, 1, 4, 3]);  unsqueeze_398 = None
    unsqueeze_399: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_581, 4);  view_581 = None
    permute_664: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_399, [4, 1, 2, 3, 0]);  unsqueeze_399 = None
    permute_665: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_663, [2, 0, 4, 1, 3]);  permute_663 = None
    view_596: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_665, [16, 512, 512]);  permute_665 = None
    permute_666: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_664, [2, 4, 1, 3, 0]);  permute_664 = None
    view_597: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_666, [16, 512, 64]);  permute_666 = None
    bmm_126: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_596, view_597)
    view_598: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_126, [16, 512, 1, 1, 64]);  bmm_126 = None
    permute_667: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_598, [1, 3, 0, 4, 2]);  view_598 = None
    view_599: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_667, [512, 1, 16, 64]);  permute_667 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_400: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_599, 4);  view_599 = None
    permute_668: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_400, [0, 1, 4, 3, 2]);  unsqueeze_400 = None
    unsqueeze_401: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_112, 3);  primals_112 = None
    unsqueeze_402: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 4);  unsqueeze_401 = None
    permute_669: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_402, [3, 4, 0, 2, 1]);  unsqueeze_402 = None
    permute_670: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_668, [0, 3, 4, 1, 2]);  permute_668 = None
    clone_30: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_670, memory_format = torch.contiguous_format);  permute_670 = None
    view_600: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_30, [1, 512, 1024]);  clone_30 = None
    permute_671: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_669, [3, 4, 1, 2, 0]);  permute_669 = None
    clone_31: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_671, memory_format = torch.contiguous_format);  permute_671 = None
    view_601: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_31, [1, 1024, 1024]);  clone_31 = None
    bmm_127: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_600, view_601)
    view_602: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_127, [512, 1, 1, 1, 1024]);  bmm_127 = None
    permute_672: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_602, [0, 3, 4, 1, 2]);  view_602 = None
    view_603: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_672, [512, 1, 1024]);  permute_672 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    native_dropout_63 = torch.ops.aten.native_dropout.default(view_603, 0.1, True);  view_603 = None
    getitem_186: "f32[512, 1, 1024]" = native_dropout_63[0]
    getitem_187: "b8[512, 1, 1024]" = native_dropout_63[1];  native_dropout_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_171: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_186, add_166);  getitem_186 = add_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_30 = torch.ops.aten.var_mean.correction(add_171, [2], correction = 0, keepdim = True)
    getitem_188: "f32[512, 1, 1]" = var_mean_30[0]
    getitem_189: "f32[512, 1, 1]" = var_mean_30[1];  var_mean_30 = None
    add_172: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_188, 1e-12);  getitem_188 = None
    rsqrt_30: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
    sub_46: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_171, getitem_189);  add_171 = getitem_189 = None
    mul_125: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_30);  sub_46 = None
    mul_126: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_125, primals_290)
    add_173: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_126, primals_291);  mul_126 = primals_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_604: "f32[512, 1024]" = torch.ops.aten.view.default(add_173, [512, 1024])
    permute_673: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_292, [1, 0]);  primals_292 = None
    addmm_30: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_293, view_604, permute_673);  primals_293 = None
    view_605: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_30, [512, 1, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_127: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_605, 0.5)
    mul_128: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_605, 0.7071067811865476);  view_605 = None
    erf_15: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_128);  mul_128 = None
    add_174: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_129: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_127, add_174);  mul_127 = add_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    native_dropout_64 = torch.ops.aten.native_dropout.default(mul_129, 0.1, True);  mul_129 = None
    getitem_190: "f32[512, 1, 4096]" = native_dropout_64[0]
    getitem_191: "b8[512, 1, 4096]" = native_dropout_64[1];  native_dropout_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_606: "f32[512, 4096]" = torch.ops.aten.view.default(getitem_190, [512, 4096]);  getitem_190 = None
    permute_674: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_294, [1, 0]);  primals_294 = None
    addmm_31: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_295, view_606, permute_674);  primals_295 = None
    view_607: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_31, [512, 1, 1024]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    native_dropout_65 = torch.ops.aten.native_dropout.default(view_607, 0.1, True);  view_607 = None
    getitem_192: "f32[512, 1, 1024]" = native_dropout_65[0]
    getitem_193: "b8[512, 1, 1024]" = native_dropout_65[1];  native_dropout_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_175: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_192, add_173);  getitem_192 = add_173 = None
    var_mean_31 = torch.ops.aten.var_mean.correction(add_175, [2], correction = 0, keepdim = True)
    getitem_194: "f32[512, 1, 1]" = var_mean_31[0]
    getitem_195: "f32[512, 1, 1]" = var_mean_31[1];  var_mean_31 = None
    add_176: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_194, 1e-12);  getitem_194 = None
    rsqrt_31: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_176);  add_176 = None
    sub_47: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_175, getitem_195);  add_175 = getitem_195 = None
    mul_130: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_31);  sub_47 = None
    mul_131: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_130, primals_296)
    add_177: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_131, primals_297);  mul_131 = primals_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_403: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_177, 3)
    unsqueeze_404: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 4);  unsqueeze_403 = None
    permute_675: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_404, [0, 1, 3, 4, 2]);  unsqueeze_404 = None
    unsqueeze_405: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_113, 3);  primals_113 = None
    unsqueeze_406: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 4);  unsqueeze_405 = None
    permute_676: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_406, [3, 4, 1, 2, 0]);  unsqueeze_406 = None
    permute_677: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_675, [0, 4, 1, 2, 3]);  permute_675 = None
    view_608: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_677, [1, 512, 1024]);  permute_677 = None
    permute_678: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_676, [4, 1, 2, 3, 0]);  permute_676 = None
    view_609: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_678, [1, 1024, 1024]);  permute_678 = None
    bmm_128: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_608, view_609)
    view_610: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_128, [512, 1, 1, 16, 64]);  bmm_128 = None
    permute_679: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_610, [0, 2, 3, 4, 1]);  view_610 = None
    view_611: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_679, [512, 1, 16, 64]);  permute_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_409: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_114, 3);  primals_114 = None
    unsqueeze_410: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 4);  unsqueeze_409 = None
    permute_681: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_410, [3, 4, 1, 2, 0]);  unsqueeze_410 = None
    permute_683: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_681, [4, 1, 2, 3, 0]);  permute_681 = None
    view_613: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_683, [1, 1024, 1024]);  permute_683 = None
    bmm_129: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_608, view_613)
    view_614: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_129, [512, 1, 1, 16, 64]);  bmm_129 = None
    permute_684: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_614, [0, 2, 3, 4, 1]);  view_614 = None
    view_615: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_684, [512, 1, 16, 64]);  permute_684 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_413: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_115, 3);  primals_115 = None
    unsqueeze_414: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 4);  unsqueeze_413 = None
    permute_686: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_414, [3, 4, 1, 2, 0]);  unsqueeze_414 = None
    permute_688: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_686, [4, 1, 2, 3, 0]);  permute_686 = None
    view_617: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_688, [1, 1024, 1024]);  permute_688 = None
    bmm_130: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_608, view_617)
    view_618: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_130, [512, 1, 1, 16, 64]);  bmm_130 = None
    permute_689: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_618, [0, 2, 3, 4, 1]);  view_618 = None
    view_619: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_689, [512, 1, 16, 64]);  permute_689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_417: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_116, 3);  primals_116 = None
    unsqueeze_418: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 4);  unsqueeze_417 = None
    permute_691: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_418, [3, 4, 1, 2, 0]);  unsqueeze_418 = None
    permute_693: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_691, [4, 1, 2, 3, 0]);  permute_691 = None
    view_621: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_693, [1, 1024, 1024]);  permute_693 = None
    bmm_131: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_12, view_621);  view_621 = None
    view_622: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_131, [1024, 1, 1, 16, 64]);  bmm_131 = None
    permute_694: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_622, [0, 2, 3, 4, 1]);  view_622 = None
    view_623: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_694, [1024, 1, 16, 64]);  permute_694 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_178: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_611, primals_117);  primals_117 = None
    unsqueeze_419: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_178, 4);  add_178 = None
    permute_695: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_419, [1, 2, 0, 4, 3]);  unsqueeze_419 = None
    unsqueeze_420: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_615, 4);  view_615 = None
    permute_696: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_420, [1, 2, 4, 0, 3]);  unsqueeze_420 = None
    permute_697: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_695, [1, 2, 4, 0, 3]);  permute_695 = None
    view_624: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_697, [16, 512, 64]);  permute_697 = None
    permute_698: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_696, [1, 4, 0, 3, 2]);  permute_696 = None
    view_625: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_698, [16, 64, 512]);  permute_698 = None
    bmm_132: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_624, view_625)
    view_626: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_132, [16, 512, 1, 1, 512]);  bmm_132 = None
    permute_699: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_626, [3, 0, 1, 4, 2]);  view_626 = None
    view_627: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_699, [1, 16, 512, 512]);  permute_699 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_179: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_611, primals_118);  view_611 = primals_118 = None
    unsqueeze_421: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_179, 4);  add_179 = None
    permute_700: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_421, [1, 2, 0, 4, 3]);  unsqueeze_421 = None
    unsqueeze_422: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_623, 4);  view_623 = None
    permute_701: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_422, [1, 2, 4, 0, 3]);  unsqueeze_422 = None
    permute_702: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_700, [1, 2, 4, 0, 3]);  permute_700 = None
    view_628: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_702, [16, 512, 64]);  permute_702 = None
    permute_703: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_701, [1, 4, 0, 3, 2]);  permute_701 = None
    view_629: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_703, [16, 64, 1024]);  permute_703 = None
    bmm_133: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_628, view_629)
    view_630: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_133, [16, 512, 1, 1, 1024]);  bmm_133 = None
    permute_704: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_630, [3, 0, 1, 4, 2]);  view_630 = None
    view_631: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_704, [1, 16, 512, 1024]);  permute_704 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_632: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_631, [1, 16, 1024, 512]);  view_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_115: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_632, 0, 0, 9223372036854775807);  view_632 = None
    slice_116: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_115, 1, 0, 9223372036854775807);  slice_115 = None
    slice_117: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_116, 2, 1, 9223372036854775807);  slice_116 = None
    slice_118: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_117, 3, 0, 9223372036854775807);  slice_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_633: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_118, [1, 16, 512, 1023]);  slice_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    slice_119: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_633, 0, 0, 9223372036854775807);  view_633 = None
    slice_120: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_119, 1, 0, 9223372036854775807);  slice_119 = None
    slice_121: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_120, 2, 0, 9223372036854775807);  slice_120 = None
    index_16: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_121, [None, None, None, iota_2]);  slice_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_180: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_627, index_16);  view_627 = index_16 = None
    add_181: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_180, 0);  add_180 = None
    mul_132: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_181, 0.125);  add_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_16: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_132, [3], True)
    sub_48: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_132, amax_16);  mul_132 = amax_16 = None
    exp_16: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_48);  sub_48 = None
    sum_17: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [3], True)
    div_17: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
    alias_16: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    native_dropout_66 = torch.ops.aten.native_dropout.default(div_17, 0.1, True);  div_17 = None
    getitem_196: "f32[1, 16, 512, 512]" = native_dropout_66[0]
    getitem_197: "b8[1, 16, 512, 512]" = native_dropout_66[1];  native_dropout_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_423: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(getitem_196, 4);  getitem_196 = None
    permute_705: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_423, [2, 0, 1, 4, 3]);  unsqueeze_423 = None
    unsqueeze_424: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_619, 4);  view_619 = None
    permute_706: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_424, [4, 1, 2, 3, 0]);  unsqueeze_424 = None
    permute_707: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_705, [2, 0, 4, 1, 3]);  permute_705 = None
    view_634: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_707, [16, 512, 512]);  permute_707 = None
    permute_708: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_706, [2, 4, 1, 3, 0]);  permute_706 = None
    view_635: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_708, [16, 512, 64]);  permute_708 = None
    bmm_134: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_634, view_635)
    view_636: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_134, [16, 512, 1, 1, 64]);  bmm_134 = None
    permute_709: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_636, [1, 3, 0, 4, 2]);  view_636 = None
    view_637: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_709, [512, 1, 16, 64]);  permute_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_425: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_637, 4);  view_637 = None
    permute_710: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_425, [0, 1, 4, 3, 2]);  unsqueeze_425 = None
    unsqueeze_426: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_119, 3);  primals_119 = None
    unsqueeze_427: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 4);  unsqueeze_426 = None
    permute_711: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_427, [3, 4, 0, 2, 1]);  unsqueeze_427 = None
    permute_712: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_710, [0, 3, 4, 1, 2]);  permute_710 = None
    clone_32: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_712, memory_format = torch.contiguous_format);  permute_712 = None
    view_638: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_32, [1, 512, 1024]);  clone_32 = None
    permute_713: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_711, [3, 4, 1, 2, 0]);  permute_711 = None
    clone_33: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_713, memory_format = torch.contiguous_format);  permute_713 = None
    view_639: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_33, [1, 1024, 1024]);  clone_33 = None
    bmm_135: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_638, view_639)
    view_640: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_135, [512, 1, 1, 1, 1024]);  bmm_135 = None
    permute_714: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_640, [0, 3, 4, 1, 2]);  view_640 = None
    view_641: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_714, [512, 1, 1024]);  permute_714 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    native_dropout_67 = torch.ops.aten.native_dropout.default(view_641, 0.1, True);  view_641 = None
    getitem_198: "f32[512, 1, 1024]" = native_dropout_67[0]
    getitem_199: "b8[512, 1, 1024]" = native_dropout_67[1];  native_dropout_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_182: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_198, add_177);  getitem_198 = add_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_32 = torch.ops.aten.var_mean.correction(add_182, [2], correction = 0, keepdim = True)
    getitem_200: "f32[512, 1, 1]" = var_mean_32[0]
    getitem_201: "f32[512, 1, 1]" = var_mean_32[1];  var_mean_32 = None
    add_183: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_200, 1e-12);  getitem_200 = None
    rsqrt_32: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_183);  add_183 = None
    sub_49: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_182, getitem_201);  add_182 = getitem_201 = None
    mul_133: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_32);  sub_49 = None
    mul_134: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_133, primals_298)
    add_184: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_134, primals_299);  mul_134 = primals_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_642: "f32[512, 1024]" = torch.ops.aten.view.default(add_184, [512, 1024])
    permute_715: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_300, [1, 0]);  primals_300 = None
    addmm_32: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_301, view_642, permute_715);  primals_301 = None
    view_643: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_32, [512, 1, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_135: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_643, 0.5)
    mul_136: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_643, 0.7071067811865476);  view_643 = None
    erf_16: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_136);  mul_136 = None
    add_185: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_137: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_135, add_185);  mul_135 = add_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    native_dropout_68 = torch.ops.aten.native_dropout.default(mul_137, 0.1, True);  mul_137 = None
    getitem_202: "f32[512, 1, 4096]" = native_dropout_68[0]
    getitem_203: "b8[512, 1, 4096]" = native_dropout_68[1];  native_dropout_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_644: "f32[512, 4096]" = torch.ops.aten.view.default(getitem_202, [512, 4096]);  getitem_202 = None
    permute_716: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_302, [1, 0]);  primals_302 = None
    addmm_33: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_303, view_644, permute_716);  primals_303 = None
    view_645: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_33, [512, 1, 1024]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    native_dropout_69 = torch.ops.aten.native_dropout.default(view_645, 0.1, True);  view_645 = None
    getitem_204: "f32[512, 1, 1024]" = native_dropout_69[0]
    getitem_205: "b8[512, 1, 1024]" = native_dropout_69[1];  native_dropout_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_186: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_204, add_184);  getitem_204 = add_184 = None
    var_mean_33 = torch.ops.aten.var_mean.correction(add_186, [2], correction = 0, keepdim = True)
    getitem_206: "f32[512, 1, 1]" = var_mean_33[0]
    getitem_207: "f32[512, 1, 1]" = var_mean_33[1];  var_mean_33 = None
    add_187: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_206, 1e-12);  getitem_206 = None
    rsqrt_33: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_187);  add_187 = None
    sub_50: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_186, getitem_207);  add_186 = getitem_207 = None
    mul_138: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_33);  sub_50 = None
    mul_139: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_138, primals_304)
    add_188: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_139, primals_305);  mul_139 = primals_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_428: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_188, 3)
    unsqueeze_429: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 4);  unsqueeze_428 = None
    permute_717: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_429, [0, 1, 3, 4, 2]);  unsqueeze_429 = None
    unsqueeze_430: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_120, 3);  primals_120 = None
    unsqueeze_431: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 4);  unsqueeze_430 = None
    permute_718: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_431, [3, 4, 1, 2, 0]);  unsqueeze_431 = None
    permute_719: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_717, [0, 4, 1, 2, 3]);  permute_717 = None
    view_646: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_719, [1, 512, 1024]);  permute_719 = None
    permute_720: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_718, [4, 1, 2, 3, 0]);  permute_718 = None
    view_647: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_720, [1, 1024, 1024]);  permute_720 = None
    bmm_136: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_646, view_647)
    view_648: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_136, [512, 1, 1, 16, 64]);  bmm_136 = None
    permute_721: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_648, [0, 2, 3, 4, 1]);  view_648 = None
    view_649: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_721, [512, 1, 16, 64]);  permute_721 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_434: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_121, 3);  primals_121 = None
    unsqueeze_435: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 4);  unsqueeze_434 = None
    permute_723: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_435, [3, 4, 1, 2, 0]);  unsqueeze_435 = None
    permute_725: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_723, [4, 1, 2, 3, 0]);  permute_723 = None
    view_651: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_725, [1, 1024, 1024]);  permute_725 = None
    bmm_137: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_646, view_651)
    view_652: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_137, [512, 1, 1, 16, 64]);  bmm_137 = None
    permute_726: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_652, [0, 2, 3, 4, 1]);  view_652 = None
    view_653: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_726, [512, 1, 16, 64]);  permute_726 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_438: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_122, 3);  primals_122 = None
    unsqueeze_439: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 4);  unsqueeze_438 = None
    permute_728: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_439, [3, 4, 1, 2, 0]);  unsqueeze_439 = None
    permute_730: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_728, [4, 1, 2, 3, 0]);  permute_728 = None
    view_655: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_730, [1, 1024, 1024]);  permute_730 = None
    bmm_138: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_646, view_655)
    view_656: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_138, [512, 1, 1, 16, 64]);  bmm_138 = None
    permute_731: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_656, [0, 2, 3, 4, 1]);  view_656 = None
    view_657: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_731, [512, 1, 16, 64]);  permute_731 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_442: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_123, 3);  primals_123 = None
    unsqueeze_443: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 4);  unsqueeze_442 = None
    permute_733: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_443, [3, 4, 1, 2, 0]);  unsqueeze_443 = None
    permute_735: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_733, [4, 1, 2, 3, 0]);  permute_733 = None
    view_659: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_735, [1, 1024, 1024]);  permute_735 = None
    bmm_139: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_12, view_659);  view_659 = None
    view_660: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_139, [1024, 1, 1, 16, 64]);  bmm_139 = None
    permute_736: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_660, [0, 2, 3, 4, 1]);  view_660 = None
    view_661: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_736, [1024, 1, 16, 64]);  permute_736 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_189: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_649, primals_124);  primals_124 = None
    unsqueeze_444: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_189, 4);  add_189 = None
    permute_737: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_444, [1, 2, 0, 4, 3]);  unsqueeze_444 = None
    unsqueeze_445: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_653, 4);  view_653 = None
    permute_738: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_445, [1, 2, 4, 0, 3]);  unsqueeze_445 = None
    permute_739: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_737, [1, 2, 4, 0, 3]);  permute_737 = None
    view_662: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_739, [16, 512, 64]);  permute_739 = None
    permute_740: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_738, [1, 4, 0, 3, 2]);  permute_738 = None
    view_663: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_740, [16, 64, 512]);  permute_740 = None
    bmm_140: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_662, view_663)
    view_664: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_140, [16, 512, 1, 1, 512]);  bmm_140 = None
    permute_741: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_664, [3, 0, 1, 4, 2]);  view_664 = None
    view_665: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_741, [1, 16, 512, 512]);  permute_741 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_190: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_649, primals_125);  view_649 = primals_125 = None
    unsqueeze_446: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_190, 4);  add_190 = None
    permute_742: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_446, [1, 2, 0, 4, 3]);  unsqueeze_446 = None
    unsqueeze_447: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_661, 4);  view_661 = None
    permute_743: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_447, [1, 2, 4, 0, 3]);  unsqueeze_447 = None
    permute_744: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_742, [1, 2, 4, 0, 3]);  permute_742 = None
    view_666: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_744, [16, 512, 64]);  permute_744 = None
    permute_745: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_743, [1, 4, 0, 3, 2]);  permute_743 = None
    view_667: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_745, [16, 64, 1024]);  permute_745 = None
    bmm_141: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_666, view_667)
    view_668: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_141, [16, 512, 1, 1, 1024]);  bmm_141 = None
    permute_746: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_668, [3, 0, 1, 4, 2]);  view_668 = None
    view_669: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_746, [1, 16, 512, 1024]);  permute_746 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_670: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_669, [1, 16, 1024, 512]);  view_669 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_122: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_670, 0, 0, 9223372036854775807);  view_670 = None
    slice_123: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_122, 1, 0, 9223372036854775807);  slice_122 = None
    slice_124: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_123, 2, 1, 9223372036854775807);  slice_123 = None
    slice_125: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_124, 3, 0, 9223372036854775807);  slice_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_671: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_125, [1, 16, 512, 1023]);  slice_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    slice_126: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_671, 0, 0, 9223372036854775807);  view_671 = None
    slice_127: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_126, 1, 0, 9223372036854775807);  slice_126 = None
    slice_128: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_127, 2, 0, 9223372036854775807);  slice_127 = None
    index_17: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_128, [None, None, None, iota_2]);  slice_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_191: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_665, index_17);  view_665 = index_17 = None
    add_192: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_191, 0);  add_191 = None
    mul_140: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_192, 0.125);  add_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_17: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_140, [3], True)
    sub_51: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_140, amax_17);  mul_140 = amax_17 = None
    exp_17: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_51);  sub_51 = None
    sum_18: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [3], True)
    div_18: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
    alias_17: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    native_dropout_70 = torch.ops.aten.native_dropout.default(div_18, 0.1, True);  div_18 = None
    getitem_208: "f32[1, 16, 512, 512]" = native_dropout_70[0]
    getitem_209: "b8[1, 16, 512, 512]" = native_dropout_70[1];  native_dropout_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_448: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(getitem_208, 4);  getitem_208 = None
    permute_747: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_448, [2, 0, 1, 4, 3]);  unsqueeze_448 = None
    unsqueeze_449: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_657, 4);  view_657 = None
    permute_748: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_449, [4, 1, 2, 3, 0]);  unsqueeze_449 = None
    permute_749: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_747, [2, 0, 4, 1, 3]);  permute_747 = None
    view_672: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_749, [16, 512, 512]);  permute_749 = None
    permute_750: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_748, [2, 4, 1, 3, 0]);  permute_748 = None
    view_673: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_750, [16, 512, 64]);  permute_750 = None
    bmm_142: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_672, view_673)
    view_674: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_142, [16, 512, 1, 1, 64]);  bmm_142 = None
    permute_751: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_674, [1, 3, 0, 4, 2]);  view_674 = None
    view_675: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_751, [512, 1, 16, 64]);  permute_751 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_450: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_675, 4);  view_675 = None
    permute_752: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_450, [0, 1, 4, 3, 2]);  unsqueeze_450 = None
    unsqueeze_451: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_126, 3);  primals_126 = None
    unsqueeze_452: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 4);  unsqueeze_451 = None
    permute_753: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_452, [3, 4, 0, 2, 1]);  unsqueeze_452 = None
    permute_754: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_752, [0, 3, 4, 1, 2]);  permute_752 = None
    clone_34: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_754, memory_format = torch.contiguous_format);  permute_754 = None
    view_676: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_34, [1, 512, 1024]);  clone_34 = None
    permute_755: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_753, [3, 4, 1, 2, 0]);  permute_753 = None
    clone_35: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_755, memory_format = torch.contiguous_format);  permute_755 = None
    view_677: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_35, [1, 1024, 1024]);  clone_35 = None
    bmm_143: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_676, view_677)
    view_678: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_143, [512, 1, 1, 1, 1024]);  bmm_143 = None
    permute_756: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_678, [0, 3, 4, 1, 2]);  view_678 = None
    view_679: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_756, [512, 1, 1024]);  permute_756 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    native_dropout_71 = torch.ops.aten.native_dropout.default(view_679, 0.1, True);  view_679 = None
    getitem_210: "f32[512, 1, 1024]" = native_dropout_71[0]
    getitem_211: "b8[512, 1, 1024]" = native_dropout_71[1];  native_dropout_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_193: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_210, add_188);  getitem_210 = add_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_34 = torch.ops.aten.var_mean.correction(add_193, [2], correction = 0, keepdim = True)
    getitem_212: "f32[512, 1, 1]" = var_mean_34[0]
    getitem_213: "f32[512, 1, 1]" = var_mean_34[1];  var_mean_34 = None
    add_194: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_212, 1e-12);  getitem_212 = None
    rsqrt_34: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_194);  add_194 = None
    sub_52: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_193, getitem_213);  add_193 = getitem_213 = None
    mul_141: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_34);  sub_52 = None
    mul_142: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_141, primals_306)
    add_195: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_142, primals_307);  mul_142 = primals_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_680: "f32[512, 1024]" = torch.ops.aten.view.default(add_195, [512, 1024])
    permute_757: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_308, [1, 0]);  primals_308 = None
    addmm_34: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_309, view_680, permute_757);  primals_309 = None
    view_681: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_34, [512, 1, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_143: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_681, 0.5)
    mul_144: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_681, 0.7071067811865476);  view_681 = None
    erf_17: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_144);  mul_144 = None
    add_196: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_145: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_143, add_196);  mul_143 = add_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    native_dropout_72 = torch.ops.aten.native_dropout.default(mul_145, 0.1, True);  mul_145 = None
    getitem_214: "f32[512, 1, 4096]" = native_dropout_72[0]
    getitem_215: "b8[512, 1, 4096]" = native_dropout_72[1];  native_dropout_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_682: "f32[512, 4096]" = torch.ops.aten.view.default(getitem_214, [512, 4096]);  getitem_214 = None
    permute_758: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_310, [1, 0]);  primals_310 = None
    addmm_35: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_311, view_682, permute_758);  primals_311 = None
    view_683: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_35, [512, 1, 1024]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    native_dropout_73 = torch.ops.aten.native_dropout.default(view_683, 0.1, True);  view_683 = None
    getitem_216: "f32[512, 1, 1024]" = native_dropout_73[0]
    getitem_217: "b8[512, 1, 1024]" = native_dropout_73[1];  native_dropout_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_197: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_216, add_195);  getitem_216 = add_195 = None
    var_mean_35 = torch.ops.aten.var_mean.correction(add_197, [2], correction = 0, keepdim = True)
    getitem_218: "f32[512, 1, 1]" = var_mean_35[0]
    getitem_219: "f32[512, 1, 1]" = var_mean_35[1];  var_mean_35 = None
    add_198: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_218, 1e-12);  getitem_218 = None
    rsqrt_35: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_198);  add_198 = None
    sub_53: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_197, getitem_219);  add_197 = getitem_219 = None
    mul_146: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_35);  sub_53 = None
    mul_147: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_146, primals_312)
    add_199: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_147, primals_313);  mul_147 = primals_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_453: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_199, 3)
    unsqueeze_454: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 4);  unsqueeze_453 = None
    permute_759: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_454, [0, 1, 3, 4, 2]);  unsqueeze_454 = None
    unsqueeze_455: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_127, 3);  primals_127 = None
    unsqueeze_456: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 4);  unsqueeze_455 = None
    permute_760: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_456, [3, 4, 1, 2, 0]);  unsqueeze_456 = None
    permute_761: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_759, [0, 4, 1, 2, 3]);  permute_759 = None
    view_684: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_761, [1, 512, 1024]);  permute_761 = None
    permute_762: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_760, [4, 1, 2, 3, 0]);  permute_760 = None
    view_685: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_762, [1, 1024, 1024]);  permute_762 = None
    bmm_144: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_684, view_685)
    view_686: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_144, [512, 1, 1, 16, 64]);  bmm_144 = None
    permute_763: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_686, [0, 2, 3, 4, 1]);  view_686 = None
    view_687: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_763, [512, 1, 16, 64]);  permute_763 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_459: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_128, 3);  primals_128 = None
    unsqueeze_460: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 4);  unsqueeze_459 = None
    permute_765: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_460, [3, 4, 1, 2, 0]);  unsqueeze_460 = None
    permute_767: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_765, [4, 1, 2, 3, 0]);  permute_765 = None
    view_689: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_767, [1, 1024, 1024]);  permute_767 = None
    bmm_145: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_684, view_689)
    view_690: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_145, [512, 1, 1, 16, 64]);  bmm_145 = None
    permute_768: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_690, [0, 2, 3, 4, 1]);  view_690 = None
    view_691: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_768, [512, 1, 16, 64]);  permute_768 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_463: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_129, 3);  primals_129 = None
    unsqueeze_464: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 4);  unsqueeze_463 = None
    permute_770: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_464, [3, 4, 1, 2, 0]);  unsqueeze_464 = None
    permute_772: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_770, [4, 1, 2, 3, 0]);  permute_770 = None
    view_693: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_772, [1, 1024, 1024]);  permute_772 = None
    bmm_146: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_684, view_693)
    view_694: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_146, [512, 1, 1, 16, 64]);  bmm_146 = None
    permute_773: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_694, [0, 2, 3, 4, 1]);  view_694 = None
    view_695: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_773, [512, 1, 16, 64]);  permute_773 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_467: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_130, 3);  primals_130 = None
    unsqueeze_468: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 4);  unsqueeze_467 = None
    permute_775: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_468, [3, 4, 1, 2, 0]);  unsqueeze_468 = None
    permute_777: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_775, [4, 1, 2, 3, 0]);  permute_775 = None
    view_697: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_777, [1, 1024, 1024]);  permute_777 = None
    bmm_147: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_12, view_697);  view_697 = None
    view_698: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_147, [1024, 1, 1, 16, 64]);  bmm_147 = None
    permute_778: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_698, [0, 2, 3, 4, 1]);  view_698 = None
    view_699: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_778, [1024, 1, 16, 64]);  permute_778 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_200: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_687, primals_131);  primals_131 = None
    unsqueeze_469: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_200, 4);  add_200 = None
    permute_779: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_469, [1, 2, 0, 4, 3]);  unsqueeze_469 = None
    unsqueeze_470: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_691, 4);  view_691 = None
    permute_780: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_470, [1, 2, 4, 0, 3]);  unsqueeze_470 = None
    permute_781: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_779, [1, 2, 4, 0, 3]);  permute_779 = None
    view_700: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_781, [16, 512, 64]);  permute_781 = None
    permute_782: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_780, [1, 4, 0, 3, 2]);  permute_780 = None
    view_701: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_782, [16, 64, 512]);  permute_782 = None
    bmm_148: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_700, view_701)
    view_702: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_148, [16, 512, 1, 1, 512]);  bmm_148 = None
    permute_783: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_702, [3, 0, 1, 4, 2]);  view_702 = None
    view_703: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_783, [1, 16, 512, 512]);  permute_783 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_201: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_687, primals_132);  view_687 = primals_132 = None
    unsqueeze_471: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_201, 4);  add_201 = None
    permute_784: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_471, [1, 2, 0, 4, 3]);  unsqueeze_471 = None
    unsqueeze_472: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_699, 4);  view_699 = None
    permute_785: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_472, [1, 2, 4, 0, 3]);  unsqueeze_472 = None
    permute_786: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_784, [1, 2, 4, 0, 3]);  permute_784 = None
    view_704: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_786, [16, 512, 64]);  permute_786 = None
    permute_787: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_785, [1, 4, 0, 3, 2]);  permute_785 = None
    view_705: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_787, [16, 64, 1024]);  permute_787 = None
    bmm_149: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_704, view_705)
    view_706: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_149, [16, 512, 1, 1, 1024]);  bmm_149 = None
    permute_788: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_706, [3, 0, 1, 4, 2]);  view_706 = None
    view_707: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_788, [1, 16, 512, 1024]);  permute_788 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_708: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_707, [1, 16, 1024, 512]);  view_707 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_129: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_708, 0, 0, 9223372036854775807);  view_708 = None
    slice_130: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_129, 1, 0, 9223372036854775807);  slice_129 = None
    slice_131: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_130, 2, 1, 9223372036854775807);  slice_130 = None
    slice_132: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_131, 3, 0, 9223372036854775807);  slice_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_709: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_132, [1, 16, 512, 1023]);  slice_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    slice_133: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_709, 0, 0, 9223372036854775807);  view_709 = None
    slice_134: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_133, 1, 0, 9223372036854775807);  slice_133 = None
    slice_135: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_134, 2, 0, 9223372036854775807);  slice_134 = None
    index_18: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_135, [None, None, None, iota_2]);  slice_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_202: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_703, index_18);  view_703 = index_18 = None
    add_203: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_202, 0);  add_202 = None
    mul_148: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_203, 0.125);  add_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_18: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_148, [3], True)
    sub_54: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_148, amax_18);  mul_148 = amax_18 = None
    exp_18: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_54);  sub_54 = None
    sum_19: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [3], True)
    div_19: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
    alias_18: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    native_dropout_74 = torch.ops.aten.native_dropout.default(div_19, 0.1, True);  div_19 = None
    getitem_220: "f32[1, 16, 512, 512]" = native_dropout_74[0]
    getitem_221: "b8[1, 16, 512, 512]" = native_dropout_74[1];  native_dropout_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_473: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(getitem_220, 4);  getitem_220 = None
    permute_789: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_473, [2, 0, 1, 4, 3]);  unsqueeze_473 = None
    unsqueeze_474: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_695, 4);  view_695 = None
    permute_790: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_474, [4, 1, 2, 3, 0]);  unsqueeze_474 = None
    permute_791: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_789, [2, 0, 4, 1, 3]);  permute_789 = None
    view_710: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_791, [16, 512, 512]);  permute_791 = None
    permute_792: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_790, [2, 4, 1, 3, 0]);  permute_790 = None
    view_711: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_792, [16, 512, 64]);  permute_792 = None
    bmm_150: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_710, view_711)
    view_712: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_150, [16, 512, 1, 1, 64]);  bmm_150 = None
    permute_793: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_712, [1, 3, 0, 4, 2]);  view_712 = None
    view_713: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_793, [512, 1, 16, 64]);  permute_793 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_475: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_713, 4);  view_713 = None
    permute_794: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_475, [0, 1, 4, 3, 2]);  unsqueeze_475 = None
    unsqueeze_476: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_133, 3);  primals_133 = None
    unsqueeze_477: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 4);  unsqueeze_476 = None
    permute_795: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_477, [3, 4, 0, 2, 1]);  unsqueeze_477 = None
    permute_796: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_794, [0, 3, 4, 1, 2]);  permute_794 = None
    clone_36: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_796, memory_format = torch.contiguous_format);  permute_796 = None
    view_714: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_36, [1, 512, 1024]);  clone_36 = None
    permute_797: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_795, [3, 4, 1, 2, 0]);  permute_795 = None
    clone_37: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_797, memory_format = torch.contiguous_format);  permute_797 = None
    view_715: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_37, [1, 1024, 1024]);  clone_37 = None
    bmm_151: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_714, view_715)
    view_716: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_151, [512, 1, 1, 1, 1024]);  bmm_151 = None
    permute_798: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_716, [0, 3, 4, 1, 2]);  view_716 = None
    view_717: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_798, [512, 1, 1024]);  permute_798 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    native_dropout_75 = torch.ops.aten.native_dropout.default(view_717, 0.1, True);  view_717 = None
    getitem_222: "f32[512, 1, 1024]" = native_dropout_75[0]
    getitem_223: "b8[512, 1, 1024]" = native_dropout_75[1];  native_dropout_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_204: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_222, add_199);  getitem_222 = add_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_36 = torch.ops.aten.var_mean.correction(add_204, [2], correction = 0, keepdim = True)
    getitem_224: "f32[512, 1, 1]" = var_mean_36[0]
    getitem_225: "f32[512, 1, 1]" = var_mean_36[1];  var_mean_36 = None
    add_205: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_224, 1e-12);  getitem_224 = None
    rsqrt_36: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_205);  add_205 = None
    sub_55: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_204, getitem_225);  add_204 = getitem_225 = None
    mul_149: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_36);  sub_55 = None
    mul_150: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_149, primals_314)
    add_206: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_150, primals_315);  mul_150 = primals_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_718: "f32[512, 1024]" = torch.ops.aten.view.default(add_206, [512, 1024])
    permute_799: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_316, [1, 0]);  primals_316 = None
    addmm_36: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_317, view_718, permute_799);  primals_317 = None
    view_719: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_36, [512, 1, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_151: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_719, 0.5)
    mul_152: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_719, 0.7071067811865476);  view_719 = None
    erf_18: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_152);  mul_152 = None
    add_207: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_153: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_151, add_207);  mul_151 = add_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    native_dropout_76 = torch.ops.aten.native_dropout.default(mul_153, 0.1, True);  mul_153 = None
    getitem_226: "f32[512, 1, 4096]" = native_dropout_76[0]
    getitem_227: "b8[512, 1, 4096]" = native_dropout_76[1];  native_dropout_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_720: "f32[512, 4096]" = torch.ops.aten.view.default(getitem_226, [512, 4096]);  getitem_226 = None
    permute_800: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_318, [1, 0]);  primals_318 = None
    addmm_37: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_319, view_720, permute_800);  primals_319 = None
    view_721: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_37, [512, 1, 1024]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    native_dropout_77 = torch.ops.aten.native_dropout.default(view_721, 0.1, True);  view_721 = None
    getitem_228: "f32[512, 1, 1024]" = native_dropout_77[0]
    getitem_229: "b8[512, 1, 1024]" = native_dropout_77[1];  native_dropout_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_208: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_228, add_206);  getitem_228 = add_206 = None
    var_mean_37 = torch.ops.aten.var_mean.correction(add_208, [2], correction = 0, keepdim = True)
    getitem_230: "f32[512, 1, 1]" = var_mean_37[0]
    getitem_231: "f32[512, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    add_209: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_230, 1e-12);  getitem_230 = None
    rsqrt_37: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_209);  add_209 = None
    sub_56: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_208, getitem_231);  add_208 = getitem_231 = None
    mul_154: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_37);  sub_56 = None
    mul_155: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_154, primals_320)
    add_210: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_155, primals_321);  mul_155 = primals_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_478: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_210, 3)
    unsqueeze_479: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 4);  unsqueeze_478 = None
    permute_801: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_479, [0, 1, 3, 4, 2]);  unsqueeze_479 = None
    unsqueeze_480: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_134, 3);  primals_134 = None
    unsqueeze_481: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, 4);  unsqueeze_480 = None
    permute_802: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_481, [3, 4, 1, 2, 0]);  unsqueeze_481 = None
    permute_803: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_801, [0, 4, 1, 2, 3]);  permute_801 = None
    view_722: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_803, [1, 512, 1024]);  permute_803 = None
    permute_804: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_802, [4, 1, 2, 3, 0]);  permute_802 = None
    view_723: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_804, [1, 1024, 1024]);  permute_804 = None
    bmm_152: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_722, view_723)
    view_724: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_152, [512, 1, 1, 16, 64]);  bmm_152 = None
    permute_805: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_724, [0, 2, 3, 4, 1]);  view_724 = None
    view_725: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_805, [512, 1, 16, 64]);  permute_805 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_484: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_135, 3);  primals_135 = None
    unsqueeze_485: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 4);  unsqueeze_484 = None
    permute_807: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_485, [3, 4, 1, 2, 0]);  unsqueeze_485 = None
    permute_809: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_807, [4, 1, 2, 3, 0]);  permute_807 = None
    view_727: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_809, [1, 1024, 1024]);  permute_809 = None
    bmm_153: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_722, view_727)
    view_728: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_153, [512, 1, 1, 16, 64]);  bmm_153 = None
    permute_810: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_728, [0, 2, 3, 4, 1]);  view_728 = None
    view_729: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_810, [512, 1, 16, 64]);  permute_810 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_488: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_136, 3);  primals_136 = None
    unsqueeze_489: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 4);  unsqueeze_488 = None
    permute_812: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_489, [3, 4, 1, 2, 0]);  unsqueeze_489 = None
    permute_814: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_812, [4, 1, 2, 3, 0]);  permute_812 = None
    view_731: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_814, [1, 1024, 1024]);  permute_814 = None
    bmm_154: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_722, view_731)
    view_732: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_154, [512, 1, 1, 16, 64]);  bmm_154 = None
    permute_815: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_732, [0, 2, 3, 4, 1]);  view_732 = None
    view_733: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_815, [512, 1, 16, 64]);  permute_815 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_492: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_137, 3);  primals_137 = None
    unsqueeze_493: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, 4);  unsqueeze_492 = None
    permute_817: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_493, [3, 4, 1, 2, 0]);  unsqueeze_493 = None
    permute_819: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_817, [4, 1, 2, 3, 0]);  permute_817 = None
    view_735: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_819, [1, 1024, 1024]);  permute_819 = None
    bmm_155: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_12, view_735);  view_735 = None
    view_736: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_155, [1024, 1, 1, 16, 64]);  bmm_155 = None
    permute_820: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_736, [0, 2, 3, 4, 1]);  view_736 = None
    view_737: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_820, [1024, 1, 16, 64]);  permute_820 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_211: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_725, primals_138);  primals_138 = None
    unsqueeze_494: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_211, 4);  add_211 = None
    permute_821: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_494, [1, 2, 0, 4, 3]);  unsqueeze_494 = None
    unsqueeze_495: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_729, 4);  view_729 = None
    permute_822: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_495, [1, 2, 4, 0, 3]);  unsqueeze_495 = None
    permute_823: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_821, [1, 2, 4, 0, 3]);  permute_821 = None
    view_738: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_823, [16, 512, 64]);  permute_823 = None
    permute_824: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_822, [1, 4, 0, 3, 2]);  permute_822 = None
    view_739: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_824, [16, 64, 512]);  permute_824 = None
    bmm_156: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_738, view_739)
    view_740: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_156, [16, 512, 1, 1, 512]);  bmm_156 = None
    permute_825: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_740, [3, 0, 1, 4, 2]);  view_740 = None
    view_741: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_825, [1, 16, 512, 512]);  permute_825 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_212: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_725, primals_139);  view_725 = primals_139 = None
    unsqueeze_496: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_212, 4);  add_212 = None
    permute_826: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_496, [1, 2, 0, 4, 3]);  unsqueeze_496 = None
    unsqueeze_497: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_737, 4);  view_737 = None
    permute_827: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_497, [1, 2, 4, 0, 3]);  unsqueeze_497 = None
    permute_828: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_826, [1, 2, 4, 0, 3]);  permute_826 = None
    view_742: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_828, [16, 512, 64]);  permute_828 = None
    permute_829: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_827, [1, 4, 0, 3, 2]);  permute_827 = None
    view_743: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_829, [16, 64, 1024]);  permute_829 = None
    bmm_157: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_742, view_743)
    view_744: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_157, [16, 512, 1, 1, 1024]);  bmm_157 = None
    permute_830: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_744, [3, 0, 1, 4, 2]);  view_744 = None
    view_745: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_830, [1, 16, 512, 1024]);  permute_830 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_746: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_745, [1, 16, 1024, 512]);  view_745 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_136: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_746, 0, 0, 9223372036854775807);  view_746 = None
    slice_137: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_136, 1, 0, 9223372036854775807);  slice_136 = None
    slice_138: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_137, 2, 1, 9223372036854775807);  slice_137 = None
    slice_139: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_138, 3, 0, 9223372036854775807);  slice_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_747: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_139, [1, 16, 512, 1023]);  slice_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    slice_140: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_747, 0, 0, 9223372036854775807);  view_747 = None
    slice_141: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_140, 1, 0, 9223372036854775807);  slice_140 = None
    slice_142: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_141, 2, 0, 9223372036854775807);  slice_141 = None
    index_19: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_142, [None, None, None, iota_2]);  slice_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_213: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_741, index_19);  view_741 = index_19 = None
    add_214: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_213, 0);  add_213 = None
    mul_156: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_214, 0.125);  add_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_19: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_156, [3], True)
    sub_57: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_156, amax_19);  mul_156 = amax_19 = None
    exp_19: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_57);  sub_57 = None
    sum_20: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [3], True)
    div_20: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
    alias_19: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    native_dropout_78 = torch.ops.aten.native_dropout.default(div_20, 0.1, True);  div_20 = None
    getitem_232: "f32[1, 16, 512, 512]" = native_dropout_78[0]
    getitem_233: "b8[1, 16, 512, 512]" = native_dropout_78[1];  native_dropout_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_498: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(getitem_232, 4);  getitem_232 = None
    permute_831: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_498, [2, 0, 1, 4, 3]);  unsqueeze_498 = None
    unsqueeze_499: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_733, 4);  view_733 = None
    permute_832: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_499, [4, 1, 2, 3, 0]);  unsqueeze_499 = None
    permute_833: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_831, [2, 0, 4, 1, 3]);  permute_831 = None
    view_748: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_833, [16, 512, 512]);  permute_833 = None
    permute_834: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_832, [2, 4, 1, 3, 0]);  permute_832 = None
    view_749: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_834, [16, 512, 64]);  permute_834 = None
    bmm_158: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_748, view_749)
    view_750: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_158, [16, 512, 1, 1, 64]);  bmm_158 = None
    permute_835: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_750, [1, 3, 0, 4, 2]);  view_750 = None
    view_751: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_835, [512, 1, 16, 64]);  permute_835 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_500: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_751, 4);  view_751 = None
    permute_836: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_500, [0, 1, 4, 3, 2]);  unsqueeze_500 = None
    unsqueeze_501: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_140, 3);  primals_140 = None
    unsqueeze_502: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 4);  unsqueeze_501 = None
    permute_837: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_502, [3, 4, 0, 2, 1]);  unsqueeze_502 = None
    permute_838: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_836, [0, 3, 4, 1, 2]);  permute_836 = None
    clone_38: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_838, memory_format = torch.contiguous_format);  permute_838 = None
    view_752: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_38, [1, 512, 1024]);  clone_38 = None
    permute_839: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_837, [3, 4, 1, 2, 0]);  permute_837 = None
    clone_39: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_839, memory_format = torch.contiguous_format);  permute_839 = None
    view_753: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_39, [1, 1024, 1024]);  clone_39 = None
    bmm_159: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_752, view_753)
    view_754: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_159, [512, 1, 1, 1, 1024]);  bmm_159 = None
    permute_840: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_754, [0, 3, 4, 1, 2]);  view_754 = None
    view_755: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_840, [512, 1, 1024]);  permute_840 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    native_dropout_79 = torch.ops.aten.native_dropout.default(view_755, 0.1, True);  view_755 = None
    getitem_234: "f32[512, 1, 1024]" = native_dropout_79[0]
    getitem_235: "b8[512, 1, 1024]" = native_dropout_79[1];  native_dropout_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_215: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_234, add_210);  getitem_234 = add_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_38 = torch.ops.aten.var_mean.correction(add_215, [2], correction = 0, keepdim = True)
    getitem_236: "f32[512, 1, 1]" = var_mean_38[0]
    getitem_237: "f32[512, 1, 1]" = var_mean_38[1];  var_mean_38 = None
    add_216: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_236, 1e-12);  getitem_236 = None
    rsqrt_38: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_216);  add_216 = None
    sub_58: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_215, getitem_237);  add_215 = getitem_237 = None
    mul_157: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_38);  sub_58 = None
    mul_158: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_157, primals_322)
    add_217: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_158, primals_323);  mul_158 = primals_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_756: "f32[512, 1024]" = torch.ops.aten.view.default(add_217, [512, 1024])
    permute_841: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_324, [1, 0]);  primals_324 = None
    addmm_38: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_325, view_756, permute_841);  primals_325 = None
    view_757: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_38, [512, 1, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_159: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_757, 0.5)
    mul_160: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_757, 0.7071067811865476);  view_757 = None
    erf_19: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_160);  mul_160 = None
    add_218: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_161: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_159, add_218);  mul_159 = add_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    native_dropout_80 = torch.ops.aten.native_dropout.default(mul_161, 0.1, True);  mul_161 = None
    getitem_238: "f32[512, 1, 4096]" = native_dropout_80[0]
    getitem_239: "b8[512, 1, 4096]" = native_dropout_80[1];  native_dropout_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_758: "f32[512, 4096]" = torch.ops.aten.view.default(getitem_238, [512, 4096]);  getitem_238 = None
    permute_842: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_326, [1, 0]);  primals_326 = None
    addmm_39: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_327, view_758, permute_842);  primals_327 = None
    view_759: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_39, [512, 1, 1024]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    native_dropout_81 = torch.ops.aten.native_dropout.default(view_759, 0.1, True);  view_759 = None
    getitem_240: "f32[512, 1, 1024]" = native_dropout_81[0]
    getitem_241: "b8[512, 1, 1024]" = native_dropout_81[1];  native_dropout_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_219: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_240, add_217);  getitem_240 = add_217 = None
    var_mean_39 = torch.ops.aten.var_mean.correction(add_219, [2], correction = 0, keepdim = True)
    getitem_242: "f32[512, 1, 1]" = var_mean_39[0]
    getitem_243: "f32[512, 1, 1]" = var_mean_39[1];  var_mean_39 = None
    add_220: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_242, 1e-12);  getitem_242 = None
    rsqrt_39: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_220);  add_220 = None
    sub_59: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_219, getitem_243);  add_219 = getitem_243 = None
    mul_162: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_39);  sub_59 = None
    mul_163: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_162, primals_328)
    add_221: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_163, primals_329);  mul_163 = primals_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_503: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_221, 3)
    unsqueeze_504: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 4);  unsqueeze_503 = None
    permute_843: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_504, [0, 1, 3, 4, 2]);  unsqueeze_504 = None
    unsqueeze_505: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_141, 3);  primals_141 = None
    unsqueeze_506: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 4);  unsqueeze_505 = None
    permute_844: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_506, [3, 4, 1, 2, 0]);  unsqueeze_506 = None
    permute_845: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_843, [0, 4, 1, 2, 3]);  permute_843 = None
    view_760: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_845, [1, 512, 1024]);  permute_845 = None
    permute_846: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_844, [4, 1, 2, 3, 0]);  permute_844 = None
    view_761: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_846, [1, 1024, 1024]);  permute_846 = None
    bmm_160: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_760, view_761)
    view_762: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_160, [512, 1, 1, 16, 64]);  bmm_160 = None
    permute_847: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_762, [0, 2, 3, 4, 1]);  view_762 = None
    view_763: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_847, [512, 1, 16, 64]);  permute_847 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_509: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_142, 3);  primals_142 = None
    unsqueeze_510: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 4);  unsqueeze_509 = None
    permute_849: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_510, [3, 4, 1, 2, 0]);  unsqueeze_510 = None
    permute_851: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_849, [4, 1, 2, 3, 0]);  permute_849 = None
    view_765: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_851, [1, 1024, 1024]);  permute_851 = None
    bmm_161: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_760, view_765)
    view_766: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_161, [512, 1, 1, 16, 64]);  bmm_161 = None
    permute_852: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_766, [0, 2, 3, 4, 1]);  view_766 = None
    view_767: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_852, [512, 1, 16, 64]);  permute_852 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_513: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_143, 3);  primals_143 = None
    unsqueeze_514: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_513, 4);  unsqueeze_513 = None
    permute_854: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_514, [3, 4, 1, 2, 0]);  unsqueeze_514 = None
    permute_856: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_854, [4, 1, 2, 3, 0]);  permute_854 = None
    view_769: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_856, [1, 1024, 1024]);  permute_856 = None
    bmm_162: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_760, view_769)
    view_770: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_162, [512, 1, 1, 16, 64]);  bmm_162 = None
    permute_857: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_770, [0, 2, 3, 4, 1]);  view_770 = None
    view_771: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_857, [512, 1, 16, 64]);  permute_857 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_517: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_144, 3);  primals_144 = None
    unsqueeze_518: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 4);  unsqueeze_517 = None
    permute_859: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_518, [3, 4, 1, 2, 0]);  unsqueeze_518 = None
    permute_861: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_859, [4, 1, 2, 3, 0]);  permute_859 = None
    view_773: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_861, [1, 1024, 1024]);  permute_861 = None
    bmm_163: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_12, view_773);  view_773 = None
    view_774: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_163, [1024, 1, 1, 16, 64]);  bmm_163 = None
    permute_862: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_774, [0, 2, 3, 4, 1]);  view_774 = None
    view_775: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_862, [1024, 1, 16, 64]);  permute_862 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_222: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_763, primals_145);  primals_145 = None
    unsqueeze_519: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_222, 4);  add_222 = None
    permute_863: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_519, [1, 2, 0, 4, 3]);  unsqueeze_519 = None
    unsqueeze_520: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_767, 4);  view_767 = None
    permute_864: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_520, [1, 2, 4, 0, 3]);  unsqueeze_520 = None
    permute_865: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_863, [1, 2, 4, 0, 3]);  permute_863 = None
    view_776: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_865, [16, 512, 64]);  permute_865 = None
    permute_866: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_864, [1, 4, 0, 3, 2]);  permute_864 = None
    view_777: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_866, [16, 64, 512]);  permute_866 = None
    bmm_164: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_776, view_777)
    view_778: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_164, [16, 512, 1, 1, 512]);  bmm_164 = None
    permute_867: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_778, [3, 0, 1, 4, 2]);  view_778 = None
    view_779: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_867, [1, 16, 512, 512]);  permute_867 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_223: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_763, primals_146);  view_763 = primals_146 = None
    unsqueeze_521: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_223, 4);  add_223 = None
    permute_868: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_521, [1, 2, 0, 4, 3]);  unsqueeze_521 = None
    unsqueeze_522: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_775, 4);  view_775 = None
    permute_869: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_522, [1, 2, 4, 0, 3]);  unsqueeze_522 = None
    permute_870: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_868, [1, 2, 4, 0, 3]);  permute_868 = None
    view_780: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_870, [16, 512, 64]);  permute_870 = None
    permute_871: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_869, [1, 4, 0, 3, 2]);  permute_869 = None
    view_781: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_871, [16, 64, 1024]);  permute_871 = None
    bmm_165: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_780, view_781)
    view_782: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_165, [16, 512, 1, 1, 1024]);  bmm_165 = None
    permute_872: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_782, [3, 0, 1, 4, 2]);  view_782 = None
    view_783: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_872, [1, 16, 512, 1024]);  permute_872 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_784: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_783, [1, 16, 1024, 512]);  view_783 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_143: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_784, 0, 0, 9223372036854775807);  view_784 = None
    slice_144: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_143, 1, 0, 9223372036854775807);  slice_143 = None
    slice_145: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_144, 2, 1, 9223372036854775807);  slice_144 = None
    slice_146: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_145, 3, 0, 9223372036854775807);  slice_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_785: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_146, [1, 16, 512, 1023]);  slice_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    slice_147: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_785, 0, 0, 9223372036854775807);  view_785 = None
    slice_148: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_147, 1, 0, 9223372036854775807);  slice_147 = None
    slice_149: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_148, 2, 0, 9223372036854775807);  slice_148 = None
    index_20: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_149, [None, None, None, iota_2]);  slice_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_224: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_779, index_20);  view_779 = index_20 = None
    add_225: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_224, 0);  add_224 = None
    mul_164: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_225, 0.125);  add_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_20: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_164, [3], True)
    sub_60: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_164, amax_20);  mul_164 = amax_20 = None
    exp_20: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_60);  sub_60 = None
    sum_21: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [3], True)
    div_21: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
    alias_20: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    native_dropout_82 = torch.ops.aten.native_dropout.default(div_21, 0.1, True);  div_21 = None
    getitem_244: "f32[1, 16, 512, 512]" = native_dropout_82[0]
    getitem_245: "b8[1, 16, 512, 512]" = native_dropout_82[1];  native_dropout_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_523: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(getitem_244, 4);  getitem_244 = None
    permute_873: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_523, [2, 0, 1, 4, 3]);  unsqueeze_523 = None
    unsqueeze_524: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_771, 4);  view_771 = None
    permute_874: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_524, [4, 1, 2, 3, 0]);  unsqueeze_524 = None
    permute_875: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_873, [2, 0, 4, 1, 3]);  permute_873 = None
    view_786: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_875, [16, 512, 512]);  permute_875 = None
    permute_876: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_874, [2, 4, 1, 3, 0]);  permute_874 = None
    view_787: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_876, [16, 512, 64]);  permute_876 = None
    bmm_166: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_786, view_787)
    view_788: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_166, [16, 512, 1, 1, 64]);  bmm_166 = None
    permute_877: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_788, [1, 3, 0, 4, 2]);  view_788 = None
    view_789: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_877, [512, 1, 16, 64]);  permute_877 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_525: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_789, 4);  view_789 = None
    permute_878: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_525, [0, 1, 4, 3, 2]);  unsqueeze_525 = None
    unsqueeze_526: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_147, 3);  primals_147 = None
    unsqueeze_527: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 4);  unsqueeze_526 = None
    permute_879: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_527, [3, 4, 0, 2, 1]);  unsqueeze_527 = None
    permute_880: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_878, [0, 3, 4, 1, 2]);  permute_878 = None
    clone_40: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_880, memory_format = torch.contiguous_format);  permute_880 = None
    view_790: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_40, [1, 512, 1024]);  clone_40 = None
    permute_881: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_879, [3, 4, 1, 2, 0]);  permute_879 = None
    clone_41: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_881, memory_format = torch.contiguous_format);  permute_881 = None
    view_791: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_41, [1, 1024, 1024]);  clone_41 = None
    bmm_167: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_790, view_791)
    view_792: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_167, [512, 1, 1, 1, 1024]);  bmm_167 = None
    permute_882: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_792, [0, 3, 4, 1, 2]);  view_792 = None
    view_793: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_882, [512, 1, 1024]);  permute_882 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    native_dropout_83 = torch.ops.aten.native_dropout.default(view_793, 0.1, True);  view_793 = None
    getitem_246: "f32[512, 1, 1024]" = native_dropout_83[0]
    getitem_247: "b8[512, 1, 1024]" = native_dropout_83[1];  native_dropout_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_226: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_246, add_221);  getitem_246 = add_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_40 = torch.ops.aten.var_mean.correction(add_226, [2], correction = 0, keepdim = True)
    getitem_248: "f32[512, 1, 1]" = var_mean_40[0]
    getitem_249: "f32[512, 1, 1]" = var_mean_40[1];  var_mean_40 = None
    add_227: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_248, 1e-12);  getitem_248 = None
    rsqrt_40: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_227);  add_227 = None
    sub_61: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_226, getitem_249);  add_226 = getitem_249 = None
    mul_165: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_40);  sub_61 = None
    mul_166: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_165, primals_330)
    add_228: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_166, primals_331);  mul_166 = primals_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_794: "f32[512, 1024]" = torch.ops.aten.view.default(add_228, [512, 1024])
    permute_883: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_332, [1, 0]);  primals_332 = None
    addmm_40: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_333, view_794, permute_883);  primals_333 = None
    view_795: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_40, [512, 1, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_167: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_795, 0.5)
    mul_168: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_795, 0.7071067811865476);  view_795 = None
    erf_20: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_168);  mul_168 = None
    add_229: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_169: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_167, add_229);  mul_167 = add_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    native_dropout_84 = torch.ops.aten.native_dropout.default(mul_169, 0.1, True);  mul_169 = None
    getitem_250: "f32[512, 1, 4096]" = native_dropout_84[0]
    getitem_251: "b8[512, 1, 4096]" = native_dropout_84[1];  native_dropout_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_796: "f32[512, 4096]" = torch.ops.aten.view.default(getitem_250, [512, 4096]);  getitem_250 = None
    permute_884: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_334, [1, 0]);  primals_334 = None
    addmm_41: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_335, view_796, permute_884);  primals_335 = None
    view_797: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_41, [512, 1, 1024]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    native_dropout_85 = torch.ops.aten.native_dropout.default(view_797, 0.1, True);  view_797 = None
    getitem_252: "f32[512, 1, 1024]" = native_dropout_85[0]
    getitem_253: "b8[512, 1, 1024]" = native_dropout_85[1];  native_dropout_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_230: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_252, add_228);  getitem_252 = add_228 = None
    var_mean_41 = torch.ops.aten.var_mean.correction(add_230, [2], correction = 0, keepdim = True)
    getitem_254: "f32[512, 1, 1]" = var_mean_41[0]
    getitem_255: "f32[512, 1, 1]" = var_mean_41[1];  var_mean_41 = None
    add_231: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_254, 1e-12);  getitem_254 = None
    rsqrt_41: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_231);  add_231 = None
    sub_62: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_230, getitem_255);  add_230 = getitem_255 = None
    mul_170: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_41);  sub_62 = None
    mul_171: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_170, primals_336)
    add_232: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_171, primals_337);  mul_171 = primals_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_528: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_232, 3)
    unsqueeze_529: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, 4);  unsqueeze_528 = None
    permute_885: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_529, [0, 1, 3, 4, 2]);  unsqueeze_529 = None
    unsqueeze_530: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_148, 3);  primals_148 = None
    unsqueeze_531: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 4);  unsqueeze_530 = None
    permute_886: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_531, [3, 4, 1, 2, 0]);  unsqueeze_531 = None
    permute_887: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_885, [0, 4, 1, 2, 3]);  permute_885 = None
    view_798: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_887, [1, 512, 1024]);  permute_887 = None
    permute_888: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_886, [4, 1, 2, 3, 0]);  permute_886 = None
    view_799: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_888, [1, 1024, 1024]);  permute_888 = None
    bmm_168: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_798, view_799)
    view_800: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_168, [512, 1, 1, 16, 64]);  bmm_168 = None
    permute_889: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_800, [0, 2, 3, 4, 1]);  view_800 = None
    view_801: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_889, [512, 1, 16, 64]);  permute_889 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_534: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_149, 3);  primals_149 = None
    unsqueeze_535: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, 4);  unsqueeze_534 = None
    permute_891: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_535, [3, 4, 1, 2, 0]);  unsqueeze_535 = None
    permute_893: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_891, [4, 1, 2, 3, 0]);  permute_891 = None
    view_803: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_893, [1, 1024, 1024]);  permute_893 = None
    bmm_169: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_798, view_803)
    view_804: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_169, [512, 1, 1, 16, 64]);  bmm_169 = None
    permute_894: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_804, [0, 2, 3, 4, 1]);  view_804 = None
    view_805: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_894, [512, 1, 16, 64]);  permute_894 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_538: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_150, 3);  primals_150 = None
    unsqueeze_539: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 4);  unsqueeze_538 = None
    permute_896: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_539, [3, 4, 1, 2, 0]);  unsqueeze_539 = None
    permute_898: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_896, [4, 1, 2, 3, 0]);  permute_896 = None
    view_807: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_898, [1, 1024, 1024]);  permute_898 = None
    bmm_170: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_798, view_807)
    view_808: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_170, [512, 1, 1, 16, 64]);  bmm_170 = None
    permute_899: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_808, [0, 2, 3, 4, 1]);  view_808 = None
    view_809: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_899, [512, 1, 16, 64]);  permute_899 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_542: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_151, 3);  primals_151 = None
    unsqueeze_543: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 4);  unsqueeze_542 = None
    permute_901: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_543, [3, 4, 1, 2, 0]);  unsqueeze_543 = None
    permute_903: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_901, [4, 1, 2, 3, 0]);  permute_901 = None
    view_811: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_903, [1, 1024, 1024]);  permute_903 = None
    bmm_171: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_12, view_811);  view_811 = None
    view_812: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_171, [1024, 1, 1, 16, 64]);  bmm_171 = None
    permute_904: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_812, [0, 2, 3, 4, 1]);  view_812 = None
    view_813: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_904, [1024, 1, 16, 64]);  permute_904 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_233: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_801, primals_152);  primals_152 = None
    unsqueeze_544: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_233, 4);  add_233 = None
    permute_905: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_544, [1, 2, 0, 4, 3]);  unsqueeze_544 = None
    unsqueeze_545: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_805, 4);  view_805 = None
    permute_906: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_545, [1, 2, 4, 0, 3]);  unsqueeze_545 = None
    permute_907: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_905, [1, 2, 4, 0, 3]);  permute_905 = None
    view_814: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_907, [16, 512, 64]);  permute_907 = None
    permute_908: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_906, [1, 4, 0, 3, 2]);  permute_906 = None
    view_815: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_908, [16, 64, 512]);  permute_908 = None
    bmm_172: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_814, view_815)
    view_816: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_172, [16, 512, 1, 1, 512]);  bmm_172 = None
    permute_909: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_816, [3, 0, 1, 4, 2]);  view_816 = None
    view_817: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_909, [1, 16, 512, 512]);  permute_909 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_234: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_801, primals_153);  view_801 = primals_153 = None
    unsqueeze_546: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_234, 4);  add_234 = None
    permute_910: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_546, [1, 2, 0, 4, 3]);  unsqueeze_546 = None
    unsqueeze_547: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_813, 4);  view_813 = None
    permute_911: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_547, [1, 2, 4, 0, 3]);  unsqueeze_547 = None
    permute_912: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_910, [1, 2, 4, 0, 3]);  permute_910 = None
    view_818: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_912, [16, 512, 64]);  permute_912 = None
    permute_913: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_911, [1, 4, 0, 3, 2]);  permute_911 = None
    view_819: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_913, [16, 64, 1024]);  permute_913 = None
    bmm_173: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_818, view_819)
    view_820: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_173, [16, 512, 1, 1, 1024]);  bmm_173 = None
    permute_914: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_820, [3, 0, 1, 4, 2]);  view_820 = None
    view_821: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_914, [1, 16, 512, 1024]);  permute_914 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_822: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_821, [1, 16, 1024, 512]);  view_821 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_150: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_822, 0, 0, 9223372036854775807);  view_822 = None
    slice_151: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_150, 1, 0, 9223372036854775807);  slice_150 = None
    slice_152: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_151, 2, 1, 9223372036854775807);  slice_151 = None
    slice_153: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_152, 3, 0, 9223372036854775807);  slice_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_823: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_153, [1, 16, 512, 1023]);  slice_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    slice_154: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_823, 0, 0, 9223372036854775807);  view_823 = None
    slice_155: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_154, 1, 0, 9223372036854775807);  slice_154 = None
    slice_156: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_155, 2, 0, 9223372036854775807);  slice_155 = None
    index_21: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_156, [None, None, None, iota_2]);  slice_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_235: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_817, index_21);  view_817 = index_21 = None
    add_236: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_235, 0);  add_235 = None
    mul_172: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_236, 0.125);  add_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_21: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_172, [3], True)
    sub_63: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_172, amax_21);  mul_172 = amax_21 = None
    exp_21: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_63);  sub_63 = None
    sum_22: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [3], True)
    div_22: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
    alias_21: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    native_dropout_86 = torch.ops.aten.native_dropout.default(div_22, 0.1, True);  div_22 = None
    getitem_256: "f32[1, 16, 512, 512]" = native_dropout_86[0]
    getitem_257: "b8[1, 16, 512, 512]" = native_dropout_86[1];  native_dropout_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_548: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(getitem_256, 4);  getitem_256 = None
    permute_915: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_548, [2, 0, 1, 4, 3]);  unsqueeze_548 = None
    unsqueeze_549: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_809, 4);  view_809 = None
    permute_916: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_549, [4, 1, 2, 3, 0]);  unsqueeze_549 = None
    permute_917: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_915, [2, 0, 4, 1, 3]);  permute_915 = None
    view_824: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_917, [16, 512, 512]);  permute_917 = None
    permute_918: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_916, [2, 4, 1, 3, 0]);  permute_916 = None
    view_825: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_918, [16, 512, 64]);  permute_918 = None
    bmm_174: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_824, view_825)
    view_826: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_174, [16, 512, 1, 1, 64]);  bmm_174 = None
    permute_919: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_826, [1, 3, 0, 4, 2]);  view_826 = None
    view_827: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_919, [512, 1, 16, 64]);  permute_919 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_550: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_827, 4);  view_827 = None
    permute_920: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_550, [0, 1, 4, 3, 2]);  unsqueeze_550 = None
    unsqueeze_551: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_154, 3);  primals_154 = None
    unsqueeze_552: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 4);  unsqueeze_551 = None
    permute_921: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_552, [3, 4, 0, 2, 1]);  unsqueeze_552 = None
    permute_922: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_920, [0, 3, 4, 1, 2]);  permute_920 = None
    clone_42: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_922, memory_format = torch.contiguous_format);  permute_922 = None
    view_828: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_42, [1, 512, 1024]);  clone_42 = None
    permute_923: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_921, [3, 4, 1, 2, 0]);  permute_921 = None
    clone_43: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_923, memory_format = torch.contiguous_format);  permute_923 = None
    view_829: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_43, [1, 1024, 1024]);  clone_43 = None
    bmm_175: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_828, view_829)
    view_830: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_175, [512, 1, 1, 1, 1024]);  bmm_175 = None
    permute_924: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_830, [0, 3, 4, 1, 2]);  view_830 = None
    view_831: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_924, [512, 1, 1024]);  permute_924 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    native_dropout_87 = torch.ops.aten.native_dropout.default(view_831, 0.1, True);  view_831 = None
    getitem_258: "f32[512, 1, 1024]" = native_dropout_87[0]
    getitem_259: "b8[512, 1, 1024]" = native_dropout_87[1];  native_dropout_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_237: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_258, add_232);  getitem_258 = add_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_42 = torch.ops.aten.var_mean.correction(add_237, [2], correction = 0, keepdim = True)
    getitem_260: "f32[512, 1, 1]" = var_mean_42[0]
    getitem_261: "f32[512, 1, 1]" = var_mean_42[1];  var_mean_42 = None
    add_238: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_260, 1e-12);  getitem_260 = None
    rsqrt_42: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_238);  add_238 = None
    sub_64: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_237, getitem_261);  add_237 = getitem_261 = None
    mul_173: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_42);  sub_64 = None
    mul_174: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_173, primals_338)
    add_239: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_174, primals_339);  mul_174 = primals_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_832: "f32[512, 1024]" = torch.ops.aten.view.default(add_239, [512, 1024])
    permute_925: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_340, [1, 0]);  primals_340 = None
    addmm_42: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_341, view_832, permute_925);  primals_341 = None
    view_833: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_42, [512, 1, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_175: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_833, 0.5)
    mul_176: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_833, 0.7071067811865476);  view_833 = None
    erf_21: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_176);  mul_176 = None
    add_240: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_177: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_175, add_240);  mul_175 = add_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    native_dropout_88 = torch.ops.aten.native_dropout.default(mul_177, 0.1, True);  mul_177 = None
    getitem_262: "f32[512, 1, 4096]" = native_dropout_88[0]
    getitem_263: "b8[512, 1, 4096]" = native_dropout_88[1];  native_dropout_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_834: "f32[512, 4096]" = torch.ops.aten.view.default(getitem_262, [512, 4096]);  getitem_262 = None
    permute_926: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_342, [1, 0]);  primals_342 = None
    addmm_43: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_343, view_834, permute_926);  primals_343 = None
    view_835: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_43, [512, 1, 1024]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    native_dropout_89 = torch.ops.aten.native_dropout.default(view_835, 0.1, True);  view_835 = None
    getitem_264: "f32[512, 1, 1024]" = native_dropout_89[0]
    getitem_265: "b8[512, 1, 1024]" = native_dropout_89[1];  native_dropout_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_241: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_264, add_239);  getitem_264 = add_239 = None
    var_mean_43 = torch.ops.aten.var_mean.correction(add_241, [2], correction = 0, keepdim = True)
    getitem_266: "f32[512, 1, 1]" = var_mean_43[0]
    getitem_267: "f32[512, 1, 1]" = var_mean_43[1];  var_mean_43 = None
    add_242: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_266, 1e-12);  getitem_266 = None
    rsqrt_43: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_242);  add_242 = None
    sub_65: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_241, getitem_267);  add_241 = getitem_267 = None
    mul_178: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_43);  sub_65 = None
    mul_179: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_178, primals_344)
    add_243: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_179, primals_345);  mul_179 = primals_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_553: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_243, 3)
    unsqueeze_554: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 4);  unsqueeze_553 = None
    permute_927: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_554, [0, 1, 3, 4, 2]);  unsqueeze_554 = None
    unsqueeze_555: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_155, 3);  primals_155 = None
    unsqueeze_556: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_555, 4);  unsqueeze_555 = None
    permute_928: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_556, [3, 4, 1, 2, 0]);  unsqueeze_556 = None
    permute_929: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_927, [0, 4, 1, 2, 3]);  permute_927 = None
    view_836: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_929, [1, 512, 1024]);  permute_929 = None
    permute_930: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_928, [4, 1, 2, 3, 0]);  permute_928 = None
    view_837: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_930, [1, 1024, 1024]);  permute_930 = None
    bmm_176: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_836, view_837)
    view_838: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_176, [512, 1, 1, 16, 64]);  bmm_176 = None
    permute_931: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_838, [0, 2, 3, 4, 1]);  view_838 = None
    view_839: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_931, [512, 1, 16, 64]);  permute_931 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_559: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_156, 3);  primals_156 = None
    unsqueeze_560: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 4);  unsqueeze_559 = None
    permute_933: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_560, [3, 4, 1, 2, 0]);  unsqueeze_560 = None
    permute_935: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_933, [4, 1, 2, 3, 0]);  permute_933 = None
    view_841: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_935, [1, 1024, 1024]);  permute_935 = None
    bmm_177: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_836, view_841)
    view_842: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_177, [512, 1, 1, 16, 64]);  bmm_177 = None
    permute_936: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_842, [0, 2, 3, 4, 1]);  view_842 = None
    view_843: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_936, [512, 1, 16, 64]);  permute_936 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_563: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_157, 3);  primals_157 = None
    unsqueeze_564: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 4);  unsqueeze_563 = None
    permute_938: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_564, [3, 4, 1, 2, 0]);  unsqueeze_564 = None
    permute_940: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_938, [4, 1, 2, 3, 0]);  permute_938 = None
    view_845: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_940, [1, 1024, 1024]);  permute_940 = None
    bmm_178: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_836, view_845)
    view_846: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_178, [512, 1, 1, 16, 64]);  bmm_178 = None
    permute_941: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_846, [0, 2, 3, 4, 1]);  view_846 = None
    view_847: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_941, [512, 1, 16, 64]);  permute_941 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_567: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_158, 3);  primals_158 = None
    unsqueeze_568: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_567, 4);  unsqueeze_567 = None
    permute_943: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_568, [3, 4, 1, 2, 0]);  unsqueeze_568 = None
    permute_945: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_943, [4, 1, 2, 3, 0]);  permute_943 = None
    view_849: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_945, [1, 1024, 1024]);  permute_945 = None
    bmm_179: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_12, view_849);  view_849 = None
    view_850: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_179, [1024, 1, 1, 16, 64]);  bmm_179 = None
    permute_946: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_850, [0, 2, 3, 4, 1]);  view_850 = None
    view_851: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_946, [1024, 1, 16, 64]);  permute_946 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_244: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_839, primals_159);  primals_159 = None
    unsqueeze_569: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_244, 4);  add_244 = None
    permute_947: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_569, [1, 2, 0, 4, 3]);  unsqueeze_569 = None
    unsqueeze_570: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_843, 4);  view_843 = None
    permute_948: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_570, [1, 2, 4, 0, 3]);  unsqueeze_570 = None
    permute_949: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_947, [1, 2, 4, 0, 3]);  permute_947 = None
    view_852: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_949, [16, 512, 64]);  permute_949 = None
    permute_950: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_948, [1, 4, 0, 3, 2]);  permute_948 = None
    view_853: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_950, [16, 64, 512]);  permute_950 = None
    bmm_180: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_852, view_853)
    view_854: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_180, [16, 512, 1, 1, 512]);  bmm_180 = None
    permute_951: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_854, [3, 0, 1, 4, 2]);  view_854 = None
    view_855: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_951, [1, 16, 512, 512]);  permute_951 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_245: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_839, primals_160);  view_839 = primals_160 = None
    unsqueeze_571: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_245, 4);  add_245 = None
    permute_952: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_571, [1, 2, 0, 4, 3]);  unsqueeze_571 = None
    unsqueeze_572: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_851, 4);  view_851 = None
    permute_953: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_572, [1, 2, 4, 0, 3]);  unsqueeze_572 = None
    permute_954: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_952, [1, 2, 4, 0, 3]);  permute_952 = None
    view_856: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_954, [16, 512, 64]);  permute_954 = None
    permute_955: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_953, [1, 4, 0, 3, 2]);  permute_953 = None
    view_857: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_955, [16, 64, 1024]);  permute_955 = None
    bmm_181: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_856, view_857)
    view_858: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_181, [16, 512, 1, 1, 1024]);  bmm_181 = None
    permute_956: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_858, [3, 0, 1, 4, 2]);  view_858 = None
    view_859: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_956, [1, 16, 512, 1024]);  permute_956 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_860: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_859, [1, 16, 1024, 512]);  view_859 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_157: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_860, 0, 0, 9223372036854775807);  view_860 = None
    slice_158: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_157, 1, 0, 9223372036854775807);  slice_157 = None
    slice_159: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_158, 2, 1, 9223372036854775807);  slice_158 = None
    slice_160: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_159, 3, 0, 9223372036854775807);  slice_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_861: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_160, [1, 16, 512, 1023]);  slice_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    slice_161: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_861, 0, 0, 9223372036854775807);  view_861 = None
    slice_162: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_161, 1, 0, 9223372036854775807);  slice_161 = None
    slice_163: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_162, 2, 0, 9223372036854775807);  slice_162 = None
    index_22: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_163, [None, None, None, iota_2]);  slice_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_246: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_855, index_22);  view_855 = index_22 = None
    add_247: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_246, 0);  add_246 = None
    mul_180: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_247, 0.125);  add_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_22: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_180, [3], True)
    sub_66: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_180, amax_22);  mul_180 = amax_22 = None
    exp_22: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_66);  sub_66 = None
    sum_23: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [3], True)
    div_23: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
    alias_22: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    native_dropout_90 = torch.ops.aten.native_dropout.default(div_23, 0.1, True);  div_23 = None
    getitem_268: "f32[1, 16, 512, 512]" = native_dropout_90[0]
    getitem_269: "b8[1, 16, 512, 512]" = native_dropout_90[1];  native_dropout_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_573: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(getitem_268, 4);  getitem_268 = None
    permute_957: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_573, [2, 0, 1, 4, 3]);  unsqueeze_573 = None
    unsqueeze_574: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_847, 4);  view_847 = None
    permute_958: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_574, [4, 1, 2, 3, 0]);  unsqueeze_574 = None
    permute_959: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_957, [2, 0, 4, 1, 3]);  permute_957 = None
    view_862: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_959, [16, 512, 512]);  permute_959 = None
    permute_960: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_958, [2, 4, 1, 3, 0]);  permute_958 = None
    view_863: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_960, [16, 512, 64]);  permute_960 = None
    bmm_182: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_862, view_863)
    view_864: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_182, [16, 512, 1, 1, 64]);  bmm_182 = None
    permute_961: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_864, [1, 3, 0, 4, 2]);  view_864 = None
    view_865: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_961, [512, 1, 16, 64]);  permute_961 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_575: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_865, 4);  view_865 = None
    permute_962: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_575, [0, 1, 4, 3, 2]);  unsqueeze_575 = None
    unsqueeze_576: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_161, 3);  primals_161 = None
    unsqueeze_577: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, 4);  unsqueeze_576 = None
    permute_963: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_577, [3, 4, 0, 2, 1]);  unsqueeze_577 = None
    permute_964: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_962, [0, 3, 4, 1, 2]);  permute_962 = None
    clone_44: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_964, memory_format = torch.contiguous_format);  permute_964 = None
    view_866: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_44, [1, 512, 1024]);  clone_44 = None
    permute_965: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_963, [3, 4, 1, 2, 0]);  permute_963 = None
    clone_45: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_965, memory_format = torch.contiguous_format);  permute_965 = None
    view_867: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_45, [1, 1024, 1024]);  clone_45 = None
    bmm_183: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_866, view_867)
    view_868: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_183, [512, 1, 1, 1, 1024]);  bmm_183 = None
    permute_966: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_868, [0, 3, 4, 1, 2]);  view_868 = None
    view_869: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_966, [512, 1, 1024]);  permute_966 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    native_dropout_91 = torch.ops.aten.native_dropout.default(view_869, 0.1, True);  view_869 = None
    getitem_270: "f32[512, 1, 1024]" = native_dropout_91[0]
    getitem_271: "b8[512, 1, 1024]" = native_dropout_91[1];  native_dropout_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_248: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_270, add_243);  getitem_270 = add_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_44 = torch.ops.aten.var_mean.correction(add_248, [2], correction = 0, keepdim = True)
    getitem_272: "f32[512, 1, 1]" = var_mean_44[0]
    getitem_273: "f32[512, 1, 1]" = var_mean_44[1];  var_mean_44 = None
    add_249: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_272, 1e-12);  getitem_272 = None
    rsqrt_44: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_249);  add_249 = None
    sub_67: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_248, getitem_273);  add_248 = getitem_273 = None
    mul_181: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_44);  sub_67 = None
    mul_182: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_181, primals_346)
    add_250: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_182, primals_347);  mul_182 = primals_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_870: "f32[512, 1024]" = torch.ops.aten.view.default(add_250, [512, 1024])
    permute_967: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_348, [1, 0]);  primals_348 = None
    addmm_44: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_349, view_870, permute_967);  primals_349 = None
    view_871: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_44, [512, 1, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_183: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_871, 0.5)
    mul_184: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_871, 0.7071067811865476);  view_871 = None
    erf_22: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_184);  mul_184 = None
    add_251: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_185: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_183, add_251);  mul_183 = add_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    native_dropout_92 = torch.ops.aten.native_dropout.default(mul_185, 0.1, True);  mul_185 = None
    getitem_274: "f32[512, 1, 4096]" = native_dropout_92[0]
    getitem_275: "b8[512, 1, 4096]" = native_dropout_92[1];  native_dropout_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_872: "f32[512, 4096]" = torch.ops.aten.view.default(getitem_274, [512, 4096]);  getitem_274 = None
    permute_968: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_350, [1, 0]);  primals_350 = None
    addmm_45: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_351, view_872, permute_968);  primals_351 = None
    view_873: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_45, [512, 1, 1024]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    native_dropout_93 = torch.ops.aten.native_dropout.default(view_873, 0.1, True);  view_873 = None
    getitem_276: "f32[512, 1, 1024]" = native_dropout_93[0]
    getitem_277: "b8[512, 1, 1024]" = native_dropout_93[1];  native_dropout_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_252: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_276, add_250);  getitem_276 = add_250 = None
    var_mean_45 = torch.ops.aten.var_mean.correction(add_252, [2], correction = 0, keepdim = True)
    getitem_278: "f32[512, 1, 1]" = var_mean_45[0]
    getitem_279: "f32[512, 1, 1]" = var_mean_45[1];  var_mean_45 = None
    add_253: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_278, 1e-12);  getitem_278 = None
    rsqrt_45: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_253);  add_253 = None
    sub_68: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_252, getitem_279);  add_252 = getitem_279 = None
    mul_186: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_45);  sub_68 = None
    mul_187: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_186, primals_352)
    add_254: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_187, primals_353);  mul_187 = primals_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_578: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_254, 3)
    unsqueeze_579: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 4);  unsqueeze_578 = None
    permute_969: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_579, [0, 1, 3, 4, 2]);  unsqueeze_579 = None
    unsqueeze_580: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_162, 3);  primals_162 = None
    unsqueeze_581: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, 4);  unsqueeze_580 = None
    permute_970: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_581, [3, 4, 1, 2, 0]);  unsqueeze_581 = None
    permute_971: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_969, [0, 4, 1, 2, 3]);  permute_969 = None
    view_874: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_971, [1, 512, 1024]);  permute_971 = None
    permute_972: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_970, [4, 1, 2, 3, 0]);  permute_970 = None
    view_875: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_972, [1, 1024, 1024]);  permute_972 = None
    bmm_184: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_874, view_875)
    view_876: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_184, [512, 1, 1, 16, 64]);  bmm_184 = None
    permute_973: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_876, [0, 2, 3, 4, 1]);  view_876 = None
    view_877: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_973, [512, 1, 16, 64]);  permute_973 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_584: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_163, 3);  primals_163 = None
    unsqueeze_585: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 4);  unsqueeze_584 = None
    permute_975: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_585, [3, 4, 1, 2, 0]);  unsqueeze_585 = None
    permute_977: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_975, [4, 1, 2, 3, 0]);  permute_975 = None
    view_879: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_977, [1, 1024, 1024]);  permute_977 = None
    bmm_185: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_874, view_879)
    view_880: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_185, [512, 1, 1, 16, 64]);  bmm_185 = None
    permute_978: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_880, [0, 2, 3, 4, 1]);  view_880 = None
    view_881: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_978, [512, 1, 16, 64]);  permute_978 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_588: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_164, 3);  primals_164 = None
    unsqueeze_589: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, 4);  unsqueeze_588 = None
    permute_980: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_589, [3, 4, 1, 2, 0]);  unsqueeze_589 = None
    permute_982: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_980, [4, 1, 2, 3, 0]);  permute_980 = None
    view_883: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_982, [1, 1024, 1024]);  permute_982 = None
    bmm_186: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_874, view_883)
    view_884: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_186, [512, 1, 1, 16, 64]);  bmm_186 = None
    permute_983: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_884, [0, 2, 3, 4, 1]);  view_884 = None
    view_885: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_983, [512, 1, 16, 64]);  permute_983 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_592: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_165, 3);  primals_165 = None
    unsqueeze_593: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 4);  unsqueeze_592 = None
    permute_985: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_593, [3, 4, 1, 2, 0]);  unsqueeze_593 = None
    permute_987: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_985, [4, 1, 2, 3, 0]);  permute_985 = None
    view_887: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_987, [1, 1024, 1024]);  permute_987 = None
    bmm_187: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_12, view_887);  view_887 = None
    view_888: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_187, [1024, 1, 1, 16, 64]);  bmm_187 = None
    permute_988: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_888, [0, 2, 3, 4, 1]);  view_888 = None
    view_889: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_988, [1024, 1, 16, 64]);  permute_988 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_255: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_877, primals_166);  primals_166 = None
    unsqueeze_594: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_255, 4);  add_255 = None
    permute_989: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_594, [1, 2, 0, 4, 3]);  unsqueeze_594 = None
    unsqueeze_595: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_881, 4);  view_881 = None
    permute_990: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_595, [1, 2, 4, 0, 3]);  unsqueeze_595 = None
    permute_991: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_989, [1, 2, 4, 0, 3]);  permute_989 = None
    view_890: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_991, [16, 512, 64]);  permute_991 = None
    permute_992: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_990, [1, 4, 0, 3, 2]);  permute_990 = None
    view_891: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_992, [16, 64, 512]);  permute_992 = None
    bmm_188: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_890, view_891)
    view_892: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_188, [16, 512, 1, 1, 512]);  bmm_188 = None
    permute_993: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_892, [3, 0, 1, 4, 2]);  view_892 = None
    view_893: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_993, [1, 16, 512, 512]);  permute_993 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_256: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_877, primals_167);  view_877 = primals_167 = None
    unsqueeze_596: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_256, 4);  add_256 = None
    permute_994: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_596, [1, 2, 0, 4, 3]);  unsqueeze_596 = None
    unsqueeze_597: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_889, 4);  view_889 = None
    permute_995: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_597, [1, 2, 4, 0, 3]);  unsqueeze_597 = None
    permute_996: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_994, [1, 2, 4, 0, 3]);  permute_994 = None
    view_894: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_996, [16, 512, 64]);  permute_996 = None
    permute_997: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_995, [1, 4, 0, 3, 2]);  permute_995 = None
    view_895: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_997, [16, 64, 1024]);  permute_997 = None
    bmm_189: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_894, view_895)
    view_896: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_189, [16, 512, 1, 1, 1024]);  bmm_189 = None
    permute_998: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_896, [3, 0, 1, 4, 2]);  view_896 = None
    view_897: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_998, [1, 16, 512, 1024]);  permute_998 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_898: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_897, [1, 16, 1024, 512]);  view_897 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_164: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_898, 0, 0, 9223372036854775807);  view_898 = None
    slice_165: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_164, 1, 0, 9223372036854775807);  slice_164 = None
    slice_166: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_165, 2, 1, 9223372036854775807);  slice_165 = None
    slice_167: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_166, 3, 0, 9223372036854775807);  slice_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_899: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_167, [1, 16, 512, 1023]);  slice_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    slice_168: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_899, 0, 0, 9223372036854775807);  view_899 = None
    slice_169: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_168, 1, 0, 9223372036854775807);  slice_168 = None
    slice_170: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_169, 2, 0, 9223372036854775807);  slice_169 = None
    index_23: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_170, [None, None, None, iota_2]);  slice_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_257: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_893, index_23);  view_893 = index_23 = None
    add_258: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_257, 0);  add_257 = None
    mul_188: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_258, 0.125);  add_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_23: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_188, [3], True)
    sub_69: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_188, amax_23);  mul_188 = amax_23 = None
    exp_23: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_69);  sub_69 = None
    sum_24: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [3], True)
    div_24: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
    alias_23: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    native_dropout_94 = torch.ops.aten.native_dropout.default(div_24, 0.1, True);  div_24 = None
    getitem_280: "f32[1, 16, 512, 512]" = native_dropout_94[0]
    getitem_281: "b8[1, 16, 512, 512]" = native_dropout_94[1];  native_dropout_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_598: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(getitem_280, 4);  getitem_280 = None
    permute_999: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_598, [2, 0, 1, 4, 3]);  unsqueeze_598 = None
    unsqueeze_599: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_885, 4);  view_885 = None
    permute_1000: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_599, [4, 1, 2, 3, 0]);  unsqueeze_599 = None
    permute_1001: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_999, [2, 0, 4, 1, 3]);  permute_999 = None
    view_900: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_1001, [16, 512, 512]);  permute_1001 = None
    permute_1002: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_1000, [2, 4, 1, 3, 0]);  permute_1000 = None
    view_901: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_1002, [16, 512, 64]);  permute_1002 = None
    bmm_190: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_900, view_901)
    view_902: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_190, [16, 512, 1, 1, 64]);  bmm_190 = None
    permute_1003: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_902, [1, 3, 0, 4, 2]);  view_902 = None
    view_903: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_1003, [512, 1, 16, 64]);  permute_1003 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_600: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_903, 4);  view_903 = None
    permute_1004: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_600, [0, 1, 4, 3, 2]);  unsqueeze_600 = None
    unsqueeze_601: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_168, 3);  primals_168 = None
    unsqueeze_602: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 4);  unsqueeze_601 = None
    permute_1005: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_602, [3, 4, 0, 2, 1]);  unsqueeze_602 = None
    permute_1006: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_1004, [0, 3, 4, 1, 2]);  permute_1004 = None
    clone_46: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_1006, memory_format = torch.contiguous_format);  permute_1006 = None
    view_904: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_46, [1, 512, 1024]);  clone_46 = None
    permute_1007: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_1005, [3, 4, 1, 2, 0]);  permute_1005 = None
    clone_47: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_1007, memory_format = torch.contiguous_format);  permute_1007 = None
    view_905: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_47, [1, 1024, 1024]);  clone_47 = None
    bmm_191: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_904, view_905)
    view_906: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_191, [512, 1, 1, 1, 1024]);  bmm_191 = None
    permute_1008: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_906, [0, 3, 4, 1, 2]);  view_906 = None
    view_907: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_1008, [512, 1, 1024]);  permute_1008 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    native_dropout_95 = torch.ops.aten.native_dropout.default(view_907, 0.1, True);  view_907 = None
    getitem_282: "f32[512, 1, 1024]" = native_dropout_95[0]
    getitem_283: "b8[512, 1, 1024]" = native_dropout_95[1];  native_dropout_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_259: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_282, add_254);  getitem_282 = add_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_46 = torch.ops.aten.var_mean.correction(add_259, [2], correction = 0, keepdim = True)
    getitem_284: "f32[512, 1, 1]" = var_mean_46[0]
    getitem_285: "f32[512, 1, 1]" = var_mean_46[1];  var_mean_46 = None
    add_260: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_284, 1e-12);  getitem_284 = None
    rsqrt_46: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_260);  add_260 = None
    sub_70: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_259, getitem_285);  add_259 = getitem_285 = None
    mul_189: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_46);  sub_70 = None
    mul_190: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_189, primals_354)
    add_261: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_190, primals_355);  mul_190 = primals_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_908: "f32[512, 1024]" = torch.ops.aten.view.default(add_261, [512, 1024])
    permute_1009: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_356, [1, 0]);  primals_356 = None
    addmm_46: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_357, view_908, permute_1009);  primals_357 = None
    view_909: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_46, [512, 1, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_191: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_909, 0.5)
    mul_192: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_909, 0.7071067811865476);  view_909 = None
    erf_23: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_192);  mul_192 = None
    add_262: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_193: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_191, add_262);  mul_191 = add_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    native_dropout_96 = torch.ops.aten.native_dropout.default(mul_193, 0.1, True);  mul_193 = None
    getitem_286: "f32[512, 1, 4096]" = native_dropout_96[0]
    getitem_287: "b8[512, 1, 4096]" = native_dropout_96[1];  native_dropout_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_910: "f32[512, 4096]" = torch.ops.aten.view.default(getitem_286, [512, 4096]);  getitem_286 = None
    permute_1010: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_358, [1, 0]);  primals_358 = None
    addmm_47: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_359, view_910, permute_1010);  primals_359 = None
    view_911: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_47, [512, 1, 1024]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    native_dropout_97 = torch.ops.aten.native_dropout.default(view_911, 0.1, True);  view_911 = None
    getitem_288: "f32[512, 1, 1024]" = native_dropout_97[0]
    getitem_289: "b8[512, 1, 1024]" = native_dropout_97[1];  native_dropout_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_263: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(getitem_288, add_261);  getitem_288 = add_261 = None
    var_mean_47 = torch.ops.aten.var_mean.correction(add_263, [2], correction = 0, keepdim = True)
    getitem_290: "f32[512, 1, 1]" = var_mean_47[0]
    getitem_291: "f32[512, 1, 1]" = var_mean_47[1];  var_mean_47 = None
    add_264: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_290, 1e-12);  getitem_290 = None
    rsqrt_47: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_264);  add_264 = None
    sub_71: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_263, getitem_291);  add_263 = getitem_291 = None
    mul_194: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_47);  sub_71 = None
    mul_195: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_194, primals_360)
    add_265: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_195, primals_361);  mul_195 = primals_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1257, code: output = self.dropout(output_g if output_g is not None else output_h)
    native_dropout_98 = torch.ops.aten.native_dropout.default(add_265, 0.1, True);  add_265 = None
    getitem_292: "f32[512, 1, 1024]" = native_dropout_98[0]
    getitem_293: "b8[512, 1, 1024]" = native_dropout_98[1];  native_dropout_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1260, code: output = output.permute(1, 0, 2).contiguous()
    permute_1011: "f32[1, 512, 1024]" = torch.ops.aten.permute.default(getitem_292, [1, 0, 2]);  getitem_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1463, code: logits = self.lm_loss(transformer_outputs[0])
    view_912: "f32[512, 1024]" = torch.ops.aten.view.default(permute_1011, [512, 1024]);  permute_1011 = None
    permute_1012: "f32[1024, 32000]" = torch.ops.aten.permute.default(primals_362, [1, 0]);  primals_362 = None
    addmm_48: "f32[512, 32000]" = torch.ops.aten.addmm.default(primals_363, view_912, permute_1012);  primals_363 = None
    view_913: "f32[1, 512, 32000]" = torch.ops.aten.view.default(addmm_48, [1, 512, 32000]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1469, code: loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    view_914: "f32[512, 32000]" = torch.ops.aten.view.default(view_913, [-1, 32000])
    view_915: "i64[512]" = torch.ops.aten.view.default(primals_365, [-1])
    amax_24: "f32[512, 1]" = torch.ops.aten.amax.default(view_914, [1], True)
    sub_72: "f32[512, 32000]" = torch.ops.aten.sub.Tensor(view_914, amax_24);  view_914 = amax_24 = None
    exp_24: "f32[512, 32000]" = torch.ops.aten.exp.default(sub_72)
    sum_25: "f32[512, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
    log: "f32[512, 1]" = torch.ops.aten.log.default(sum_25);  sum_25 = None
    sub_73: "f32[512, 32000]" = torch.ops.aten.sub.Tensor(sub_72, log);  sub_72 = log = None
    ne: "b8[512]" = torch.ops.aten.ne.Scalar(view_915, -100)
    full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where: "i64[512]" = torch.ops.aten.where.self(ne, view_915, full_default);  view_915 = full_default = None
    unsqueeze_603: "i64[512, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
    gather: "f32[512, 1]" = torch.ops.aten.gather.default(sub_73, 1, unsqueeze_603);  unsqueeze_603 = None
    squeeze: "f32[512]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[512]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where_1: "f32[512]" = torch.ops.aten.where.self(ne, neg, full_default_1);  neg = full_default_1 = None
    sum_26: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
    convert_element_type_5: "f32[]" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
    sum_27: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
    div_25: "f32[]" = torch.ops.aten.div.Tensor(sum_27, convert_element_type_5);  sum_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1463, code: logits = self.lm_loss(transformer_outputs[0])
    permute_1013: "f32[32000, 1024]" = torch.ops.aten.permute.default(permute_1012, [1, 0]);  permute_1012 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    div_27: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_47, 1024);  rsqrt_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    permute_1018: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_1010, [1, 0]);  permute_1010 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    permute_1022: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_1009, [1, 0]);  permute_1009 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    div_28: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_46, 1024);  rsqrt_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    permute_1027: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_904, [0, 2, 1]);  view_904 = None
    permute_1028: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_905, [0, 2, 1]);  view_905 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    permute_1034: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_900, [0, 2, 1]);  view_900 = None
    permute_1035: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_901, [0, 2, 1]);  view_901 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    alias_26: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_23);  alias_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    permute_1041: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_894, [0, 2, 1]);  view_894 = None
    permute_1042: "f32[16, 1024, 64]" = torch.ops.aten.permute.default(view_895, [0, 2, 1]);  view_895 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    permute_1048: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_890, [0, 2, 1]);  view_890 = None
    permute_1049: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_891, [0, 2, 1]);  view_891 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    permute_1055: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    permute_1059: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_874, [0, 2, 1]);  view_874 = None
    permute_1060: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_883, [0, 2, 1]);  view_883 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    permute_1067: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_879, [0, 2, 1]);  view_879 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    permute_1074: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_875, [0, 2, 1]);  view_875 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    div_29: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_45, 1024);  rsqrt_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    permute_1079: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_968, [1, 0]);  permute_968 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    permute_1083: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_967, [1, 0]);  permute_967 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    div_30: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 1024);  rsqrt_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    permute_1088: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_866, [0, 2, 1]);  view_866 = None
    permute_1089: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_867, [0, 2, 1]);  view_867 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    permute_1095: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_862, [0, 2, 1]);  view_862 = None
    permute_1096: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_863, [0, 2, 1]);  view_863 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    alias_27: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_22);  alias_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    permute_1102: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_856, [0, 2, 1]);  view_856 = None
    permute_1103: "f32[16, 1024, 64]" = torch.ops.aten.permute.default(view_857, [0, 2, 1]);  view_857 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    permute_1109: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_852, [0, 2, 1]);  view_852 = None
    permute_1110: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_853, [0, 2, 1]);  view_853 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    permute_1120: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_836, [0, 2, 1]);  view_836 = None
    permute_1121: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_845, [0, 2, 1]);  view_845 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    permute_1128: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_841, [0, 2, 1]);  view_841 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    permute_1135: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_837, [0, 2, 1]);  view_837 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    div_31: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 1024);  rsqrt_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    permute_1140: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_926, [1, 0]);  permute_926 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    permute_1144: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_925, [1, 0]);  permute_925 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    div_32: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_42, 1024);  rsqrt_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    permute_1149: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_828, [0, 2, 1]);  view_828 = None
    permute_1150: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_829, [0, 2, 1]);  view_829 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    permute_1156: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_824, [0, 2, 1]);  view_824 = None
    permute_1157: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_825, [0, 2, 1]);  view_825 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    alias_28: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_21);  alias_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    permute_1163: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_818, [0, 2, 1]);  view_818 = None
    permute_1164: "f32[16, 1024, 64]" = torch.ops.aten.permute.default(view_819, [0, 2, 1]);  view_819 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    permute_1170: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_814, [0, 2, 1]);  view_814 = None
    permute_1171: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_815, [0, 2, 1]);  view_815 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    permute_1181: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_798, [0, 2, 1]);  view_798 = None
    permute_1182: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_807, [0, 2, 1]);  view_807 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    permute_1189: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_803, [0, 2, 1]);  view_803 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    permute_1196: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_799, [0, 2, 1]);  view_799 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    div_33: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_41, 1024);  rsqrt_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    permute_1201: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_884, [1, 0]);  permute_884 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    permute_1205: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_883, [1, 0]);  permute_883 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    div_34: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 1024);  rsqrt_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    permute_1210: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_790, [0, 2, 1]);  view_790 = None
    permute_1211: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_791, [0, 2, 1]);  view_791 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    permute_1217: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_786, [0, 2, 1]);  view_786 = None
    permute_1218: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_787, [0, 2, 1]);  view_787 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    alias_29: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_20);  alias_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    permute_1224: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_780, [0, 2, 1]);  view_780 = None
    permute_1225: "f32[16, 1024, 64]" = torch.ops.aten.permute.default(view_781, [0, 2, 1]);  view_781 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    permute_1231: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_776, [0, 2, 1]);  view_776 = None
    permute_1232: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_777, [0, 2, 1]);  view_777 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    permute_1242: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_760, [0, 2, 1]);  view_760 = None
    permute_1243: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_769, [0, 2, 1]);  view_769 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    permute_1250: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_765, [0, 2, 1]);  view_765 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    permute_1257: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_761, [0, 2, 1]);  view_761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    div_35: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 1024);  rsqrt_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    permute_1262: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_842, [1, 0]);  permute_842 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    permute_1266: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_841, [1, 0]);  permute_841 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    div_36: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 1024);  rsqrt_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    permute_1271: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_752, [0, 2, 1]);  view_752 = None
    permute_1272: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_753, [0, 2, 1]);  view_753 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    permute_1278: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_748, [0, 2, 1]);  view_748 = None
    permute_1279: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_749, [0, 2, 1]);  view_749 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    alias_30: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_19);  alias_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    permute_1285: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_742, [0, 2, 1]);  view_742 = None
    permute_1286: "f32[16, 1024, 64]" = torch.ops.aten.permute.default(view_743, [0, 2, 1]);  view_743 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    permute_1292: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_738, [0, 2, 1]);  view_738 = None
    permute_1293: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_739, [0, 2, 1]);  view_739 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    permute_1303: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_722, [0, 2, 1]);  view_722 = None
    permute_1304: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_731, [0, 2, 1]);  view_731 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    permute_1311: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_727, [0, 2, 1]);  view_727 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    permute_1318: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_723, [0, 2, 1]);  view_723 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    div_37: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 1024);  rsqrt_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    permute_1323: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_800, [1, 0]);  permute_800 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    permute_1327: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_799, [1, 0]);  permute_799 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    div_38: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 1024);  rsqrt_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    permute_1332: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_714, [0, 2, 1]);  view_714 = None
    permute_1333: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_715, [0, 2, 1]);  view_715 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    permute_1339: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_710, [0, 2, 1]);  view_710 = None
    permute_1340: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_711, [0, 2, 1]);  view_711 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    alias_31: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_18);  alias_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    permute_1346: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_704, [0, 2, 1]);  view_704 = None
    permute_1347: "f32[16, 1024, 64]" = torch.ops.aten.permute.default(view_705, [0, 2, 1]);  view_705 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    permute_1353: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_700, [0, 2, 1]);  view_700 = None
    permute_1354: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_701, [0, 2, 1]);  view_701 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    permute_1364: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_684, [0, 2, 1]);  view_684 = None
    permute_1365: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_693, [0, 2, 1]);  view_693 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    permute_1372: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_689, [0, 2, 1]);  view_689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    permute_1379: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_685, [0, 2, 1]);  view_685 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    div_39: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 1024);  rsqrt_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    permute_1384: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_758, [1, 0]);  permute_758 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    permute_1388: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_757, [1, 0]);  permute_757 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    div_40: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 1024);  rsqrt_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    permute_1393: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_676, [0, 2, 1]);  view_676 = None
    permute_1394: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_677, [0, 2, 1]);  view_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    permute_1400: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_672, [0, 2, 1]);  view_672 = None
    permute_1401: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_673, [0, 2, 1]);  view_673 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    alias_32: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    permute_1407: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_666, [0, 2, 1]);  view_666 = None
    permute_1408: "f32[16, 1024, 64]" = torch.ops.aten.permute.default(view_667, [0, 2, 1]);  view_667 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    permute_1414: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_662, [0, 2, 1]);  view_662 = None
    permute_1415: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_663, [0, 2, 1]);  view_663 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    permute_1425: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_646, [0, 2, 1]);  view_646 = None
    permute_1426: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_655, [0, 2, 1]);  view_655 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    permute_1433: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_651, [0, 2, 1]);  view_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    permute_1440: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_647, [0, 2, 1]);  view_647 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    div_41: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 1024);  rsqrt_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    permute_1445: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_716, [1, 0]);  permute_716 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    permute_1449: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_715, [1, 0]);  permute_715 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    div_42: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 1024);  rsqrt_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    permute_1454: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_638, [0, 2, 1]);  view_638 = None
    permute_1455: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_639, [0, 2, 1]);  view_639 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    permute_1461: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_634, [0, 2, 1]);  view_634 = None
    permute_1462: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_635, [0, 2, 1]);  view_635 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    alias_33: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_16);  alias_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    permute_1468: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_628, [0, 2, 1]);  view_628 = None
    permute_1469: "f32[16, 1024, 64]" = torch.ops.aten.permute.default(view_629, [0, 2, 1]);  view_629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    permute_1475: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_624, [0, 2, 1]);  view_624 = None
    permute_1476: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_625, [0, 2, 1]);  view_625 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    permute_1486: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_608, [0, 2, 1]);  view_608 = None
    permute_1487: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_617, [0, 2, 1]);  view_617 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    permute_1494: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_613, [0, 2, 1]);  view_613 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    permute_1501: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_609, [0, 2, 1]);  view_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    div_43: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 1024);  rsqrt_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    permute_1506: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_674, [1, 0]);  permute_674 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    permute_1510: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_673, [1, 0]);  permute_673 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    div_44: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 1024);  rsqrt_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    permute_1515: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_600, [0, 2, 1]);  view_600 = None
    permute_1516: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_601, [0, 2, 1]);  view_601 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    permute_1522: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_596, [0, 2, 1]);  view_596 = None
    permute_1523: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_597, [0, 2, 1]);  view_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    alias_34: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    permute_1529: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_590, [0, 2, 1]);  view_590 = None
    permute_1530: "f32[16, 1024, 64]" = torch.ops.aten.permute.default(view_591, [0, 2, 1]);  view_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    permute_1536: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_586, [0, 2, 1]);  view_586 = None
    permute_1537: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_587, [0, 2, 1]);  view_587 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    permute_1547: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_570, [0, 2, 1]);  view_570 = None
    permute_1548: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_579, [0, 2, 1]);  view_579 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    permute_1555: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_575, [0, 2, 1]);  view_575 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    permute_1562: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_571, [0, 2, 1]);  view_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    div_45: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 1024);  rsqrt_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    permute_1567: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_632, [1, 0]);  permute_632 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    permute_1571: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_631, [1, 0]);  permute_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    div_46: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 1024);  rsqrt_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    permute_1576: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_562, [0, 2, 1]);  view_562 = None
    permute_1577: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_563, [0, 2, 1]);  view_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    permute_1583: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_558, [0, 2, 1]);  view_558 = None
    permute_1584: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_559, [0, 2, 1]);  view_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    alias_35: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    permute_1590: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_552, [0, 2, 1]);  view_552 = None
    permute_1591: "f32[16, 1024, 64]" = torch.ops.aten.permute.default(view_553, [0, 2, 1]);  view_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    permute_1597: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_548, [0, 2, 1]);  view_548 = None
    permute_1598: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_549, [0, 2, 1]);  view_549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    permute_1608: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_532, [0, 2, 1]);  view_532 = None
    permute_1609: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_541, [0, 2, 1]);  view_541 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    permute_1616: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_537, [0, 2, 1]);  view_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    permute_1623: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_533, [0, 2, 1]);  view_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    div_47: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 1024);  rsqrt_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    permute_1628: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_590, [1, 0]);  permute_590 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    permute_1632: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_589, [1, 0]);  permute_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    div_48: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 1024);  rsqrt_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    permute_1637: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_524, [0, 2, 1]);  view_524 = None
    permute_1638: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_525, [0, 2, 1]);  view_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    permute_1644: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_520, [0, 2, 1]);  view_520 = None
    permute_1645: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_521, [0, 2, 1]);  view_521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    alias_36: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    permute_1651: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_514, [0, 2, 1]);  view_514 = None
    permute_1652: "f32[16, 1024, 64]" = torch.ops.aten.permute.default(view_515, [0, 2, 1]);  view_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    permute_1658: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_510, [0, 2, 1]);  view_510 = None
    permute_1659: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_511, [0, 2, 1]);  view_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    permute_1669: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_494, [0, 2, 1]);  view_494 = None
    permute_1670: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_503, [0, 2, 1]);  view_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    permute_1677: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_499, [0, 2, 1]);  view_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    permute_1684: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_495, [0, 2, 1]);  view_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    div_49: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 1024);  rsqrt_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    permute_1689: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_548, [1, 0]);  permute_548 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    permute_1693: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_547, [1, 0]);  permute_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    div_50: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 1024);  rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    permute_1698: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_486, [0, 2, 1]);  view_486 = None
    permute_1699: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_487, [0, 2, 1]);  view_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    permute_1705: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_482, [0, 2, 1]);  view_482 = None
    permute_1706: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_483, [0, 2, 1]);  view_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    alias_37: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    permute_1712: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_476, [0, 2, 1]);  view_476 = None
    permute_1713: "f32[16, 1024, 64]" = torch.ops.aten.permute.default(view_477, [0, 2, 1]);  view_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    permute_1719: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_472, [0, 2, 1]);  view_472 = None
    permute_1720: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_473, [0, 2, 1]);  view_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    permute_1730: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_456, [0, 2, 1]);  view_456 = None
    permute_1731: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_465, [0, 2, 1]);  view_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    permute_1738: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_461, [0, 2, 1]);  view_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    permute_1745: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_457, [0, 2, 1]);  view_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    div_51: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 1024);  rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    permute_1750: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_506, [1, 0]);  permute_506 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    permute_1754: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_505, [1, 0]);  permute_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    div_52: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 1024);  rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    permute_1759: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_448, [0, 2, 1]);  view_448 = None
    permute_1760: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_449, [0, 2, 1]);  view_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    permute_1766: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_444, [0, 2, 1]);  view_444 = None
    permute_1767: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_445, [0, 2, 1]);  view_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    alias_38: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    permute_1773: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_438, [0, 2, 1]);  view_438 = None
    permute_1774: "f32[16, 1024, 64]" = torch.ops.aten.permute.default(view_439, [0, 2, 1]);  view_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    permute_1780: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_434, [0, 2, 1]);  view_434 = None
    permute_1781: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_435, [0, 2, 1]);  view_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    permute_1791: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_418, [0, 2, 1]);  view_418 = None
    permute_1792: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_427, [0, 2, 1]);  view_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    permute_1799: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_423, [0, 2, 1]);  view_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    permute_1806: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_419, [0, 2, 1]);  view_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    div_53: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 1024);  rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    permute_1811: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_464, [1, 0]);  permute_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    permute_1815: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_463, [1, 0]);  permute_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    div_54: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 1024);  rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    permute_1820: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_410, [0, 2, 1]);  view_410 = None
    permute_1821: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_411, [0, 2, 1]);  view_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    permute_1827: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_406, [0, 2, 1]);  view_406 = None
    permute_1828: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_407, [0, 2, 1]);  view_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    alias_39: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    permute_1834: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_400, [0, 2, 1]);  view_400 = None
    permute_1835: "f32[16, 1024, 64]" = torch.ops.aten.permute.default(view_401, [0, 2, 1]);  view_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    permute_1841: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_396, [0, 2, 1]);  view_396 = None
    permute_1842: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_397, [0, 2, 1]);  view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    permute_1852: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_380, [0, 2, 1]);  view_380 = None
    permute_1853: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_389, [0, 2, 1]);  view_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    permute_1860: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_385, [0, 2, 1]);  view_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    permute_1867: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_381, [0, 2, 1]);  view_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    div_55: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 1024);  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    permute_1872: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_422, [1, 0]);  permute_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    permute_1876: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_421, [1, 0]);  permute_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    div_56: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 1024);  rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    permute_1881: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_372, [0, 2, 1]);  view_372 = None
    permute_1882: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_373, [0, 2, 1]);  view_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    permute_1888: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_368, [0, 2, 1]);  view_368 = None
    permute_1889: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_369, [0, 2, 1]);  view_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    alias_40: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    permute_1895: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_362, [0, 2, 1]);  view_362 = None
    permute_1896: "f32[16, 1024, 64]" = torch.ops.aten.permute.default(view_363, [0, 2, 1]);  view_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    permute_1902: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_358, [0, 2, 1]);  view_358 = None
    permute_1903: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_359, [0, 2, 1]);  view_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    permute_1913: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_342, [0, 2, 1]);  view_342 = None
    permute_1914: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_351, [0, 2, 1]);  view_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    permute_1921: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_347, [0, 2, 1]);  view_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    permute_1928: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_343, [0, 2, 1]);  view_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    div_57: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 1024);  rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    permute_1933: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_380, [1, 0]);  permute_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    permute_1937: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_379, [1, 0]);  permute_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    div_58: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 1024);  rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    permute_1942: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_334, [0, 2, 1]);  view_334 = None
    permute_1943: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_335, [0, 2, 1]);  view_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    permute_1949: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_330, [0, 2, 1]);  view_330 = None
    permute_1950: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_331, [0, 2, 1]);  view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    alias_41: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    permute_1956: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_324, [0, 2, 1]);  view_324 = None
    permute_1957: "f32[16, 1024, 64]" = torch.ops.aten.permute.default(view_325, [0, 2, 1]);  view_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    permute_1963: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_320, [0, 2, 1]);  view_320 = None
    permute_1964: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_321, [0, 2, 1]);  view_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    permute_1974: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_304, [0, 2, 1]);  view_304 = None
    permute_1975: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_313, [0, 2, 1]);  view_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    permute_1982: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_309, [0, 2, 1]);  view_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    permute_1989: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_305, [0, 2, 1]);  view_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    div_59: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 1024);  rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    permute_1994: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_338, [1, 0]);  permute_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    permute_1998: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_337, [1, 0]);  permute_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    div_60: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 1024);  rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    permute_2003: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_296, [0, 2, 1]);  view_296 = None
    permute_2004: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_297, [0, 2, 1]);  view_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    permute_2010: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_292, [0, 2, 1]);  view_292 = None
    permute_2011: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_293, [0, 2, 1]);  view_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    alias_42: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    permute_2017: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_286, [0, 2, 1]);  view_286 = None
    permute_2018: "f32[16, 1024, 64]" = torch.ops.aten.permute.default(view_287, [0, 2, 1]);  view_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    permute_2024: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_282, [0, 2, 1]);  view_282 = None
    permute_2025: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_283, [0, 2, 1]);  view_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    permute_2035: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_266, [0, 2, 1]);  view_266 = None
    permute_2036: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_275, [0, 2, 1]);  view_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    permute_2043: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_271, [0, 2, 1]);  view_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    permute_2050: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_267, [0, 2, 1]);  view_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    div_61: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 1024);  rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    permute_2055: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_296, [1, 0]);  permute_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    permute_2059: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_295, [1, 0]);  permute_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    div_62: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 1024);  rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    permute_2064: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_258, [0, 2, 1]);  view_258 = None
    permute_2065: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_259, [0, 2, 1]);  view_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    permute_2071: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
    permute_2072: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_255, [0, 2, 1]);  view_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    alias_43: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    permute_2078: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_248, [0, 2, 1]);  view_248 = None
    permute_2079: "f32[16, 1024, 64]" = torch.ops.aten.permute.default(view_249, [0, 2, 1]);  view_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    permute_2085: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_244, [0, 2, 1]);  view_244 = None
    permute_2086: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_245, [0, 2, 1]);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    permute_2096: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_228, [0, 2, 1]);  view_228 = None
    permute_2097: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_237, [0, 2, 1]);  view_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    permute_2104: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_233, [0, 2, 1]);  view_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    permute_2111: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_229, [0, 2, 1]);  view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    div_63: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 1024);  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    permute_2116: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_254, [1, 0]);  permute_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    permute_2120: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_253, [1, 0]);  permute_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    div_64: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 1024);  rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    permute_2125: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_220, [0, 2, 1]);  view_220 = None
    permute_2126: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_221, [0, 2, 1]);  view_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    permute_2132: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_216, [0, 2, 1]);  view_216 = None
    permute_2133: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_217, [0, 2, 1]);  view_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    alias_44: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    permute_2139: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
    permute_2140: "f32[16, 1024, 64]" = torch.ops.aten.permute.default(view_211, [0, 2, 1]);  view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    permute_2146: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_206, [0, 2, 1]);  view_206 = None
    permute_2147: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_207, [0, 2, 1]);  view_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    permute_2157: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_190, [0, 2, 1]);  view_190 = None
    permute_2158: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_199, [0, 2, 1]);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    permute_2165: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_195, [0, 2, 1]);  view_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    permute_2172: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_191, [0, 2, 1]);  view_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    div_65: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 1024);  rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    permute_2177: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_212, [1, 0]);  permute_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    permute_2181: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    div_66: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 1024);  rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    permute_2186: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_182, [0, 2, 1]);  view_182 = None
    permute_2187: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_183, [0, 2, 1]);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    permute_2193: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_178, [0, 2, 1]);  view_178 = None
    permute_2194: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_179, [0, 2, 1]);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    alias_45: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    permute_2200: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_172, [0, 2, 1]);  view_172 = None
    permute_2201: "f32[16, 1024, 64]" = torch.ops.aten.permute.default(view_173, [0, 2, 1]);  view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    permute_2207: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_168, [0, 2, 1]);  view_168 = None
    permute_2208: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_169, [0, 2, 1]);  view_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    permute_2218: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_152, [0, 2, 1]);  view_152 = None
    permute_2219: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_161, [0, 2, 1]);  view_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    permute_2226: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_157, [0, 2, 1]);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    permute_2233: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_153, [0, 2, 1]);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    div_67: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 1024);  rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    permute_2238: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_170, [1, 0]);  permute_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    permute_2242: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_169, [1, 0]);  permute_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    div_68: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 1024);  rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    permute_2247: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
    permute_2248: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    permute_2254: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_140, [0, 2, 1]);  view_140 = None
    permute_2255: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_141, [0, 2, 1]);  view_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    alias_46: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    permute_2261: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_134, [0, 2, 1]);  view_134 = None
    permute_2262: "f32[16, 1024, 64]" = torch.ops.aten.permute.default(view_135, [0, 2, 1]);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    permute_2268: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_130, [0, 2, 1]);  view_130 = None
    permute_2269: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_131, [0, 2, 1]);  view_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    permute_2279: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_114, [0, 2, 1]);  view_114 = None
    permute_2280: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    permute_2287: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_119, [0, 2, 1]);  view_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    permute_2294: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_115, [0, 2, 1]);  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    div_69: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 1024);  rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    permute_2299: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    permute_2303: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_127, [1, 0]);  permute_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    div_70: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 1024);  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    permute_2308: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_106, [0, 2, 1]);  view_106 = None
    permute_2309: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_107, [0, 2, 1]);  view_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    permute_2315: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_102, [0, 2, 1]);  view_102 = None
    permute_2316: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_103, [0, 2, 1]);  view_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    alias_47: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    permute_2322: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_96, [0, 2, 1]);  view_96 = None
    permute_2323: "f32[16, 1024, 64]" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    permute_2329: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_92, [0, 2, 1]);  view_92 = None
    permute_2330: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_93, [0, 2, 1]);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    permute_2340: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_76, [0, 2, 1]);  view_76 = None
    permute_2341: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_85, [0, 2, 1]);  view_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    permute_2348: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_81, [0, 2, 1]);  view_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    permute_2355: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_77, [0, 2, 1]);  view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    div_71: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 1024);  rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    permute_2360: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    permute_2364: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    div_72: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 1024);  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    permute_2369: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_68, [0, 2, 1]);  view_68 = None
    permute_2370: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_69, [0, 2, 1]);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    permute_2376: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_64, [0, 2, 1]);  view_64 = None
    permute_2377: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_65, [0, 2, 1]);  view_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    alias_48: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    permute_2383: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_58, [0, 2, 1]);  view_58 = None
    permute_2384: "f32[16, 1024, 64]" = torch.ops.aten.permute.default(view_59, [0, 2, 1]);  view_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    permute_2390: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_54, [0, 2, 1]);  view_54 = None
    permute_2391: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_55, [0, 2, 1]);  view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    permute_2401: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_38, [0, 2, 1]);  view_38 = None
    permute_2402: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_47, [0, 2, 1]);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    permute_2409: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_43, [0, 2, 1]);  view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    permute_2416: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_39, [0, 2, 1]);  view_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    div_73: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 1024);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    permute_2421: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    permute_2425: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    div_74: "f32[512, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt, 1024);  rsqrt = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    permute_2430: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view_30, [0, 2, 1]);  view_30 = None
    permute_2431: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    permute_2437: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_26, [0, 2, 1]);  view_26 = None
    permute_2438: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_27, [0, 2, 1]);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    alias_49: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias);  alias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    permute_2444: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_20, [0, 2, 1]);  view_20 = None
    permute_2445: "f32[16, 1024, 64]" = torch.ops.aten.permute.default(view_21, [0, 2, 1]);  view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    permute_2451: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_16, [0, 2, 1]);  view_16 = None
    permute_2452: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_17, [0, 2, 1]);  view_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    permute_2462: "f32[1, 1024, 512]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    permute_2463: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    permute_2470: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_5, [0, 2, 1]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    permute_2477: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(view_1, [0, 2, 1]);  view_1 = None
    return [div_25, view_913, primals_170, primals_176, primals_178, primals_184, primals_186, primals_192, primals_194, primals_200, primals_202, primals_208, primals_210, primals_216, primals_218, primals_224, primals_226, primals_232, primals_234, primals_240, primals_242, primals_248, primals_250, primals_256, primals_258, primals_264, primals_266, primals_272, primals_274, primals_280, primals_282, primals_288, primals_290, primals_296, primals_298, primals_304, primals_306, primals_312, primals_314, primals_320, primals_322, primals_328, primals_330, primals_336, primals_338, primals_344, primals_346, primals_352, primals_354, primals_360, primals_365, permute, getitem_1, iota_2, getitem_5, getitem_7, mul_5, view_34, addmm, getitem_11, view_36, getitem_13, mul_10, getitem_17, getitem_19, mul_13, view_72, addmm_2, getitem_23, view_74, getitem_25, mul_18, getitem_29, getitem_31, mul_21, view_110, addmm_4, getitem_35, view_112, getitem_37, mul_26, getitem_41, getitem_43, mul_29, view_148, addmm_6, getitem_47, view_150, getitem_49, mul_34, getitem_53, getitem_55, mul_37, view_186, addmm_8, getitem_59, view_188, getitem_61, mul_42, getitem_65, getitem_67, mul_45, view_224, addmm_10, getitem_71, view_226, getitem_73, mul_50, getitem_77, getitem_79, mul_53, view_262, addmm_12, getitem_83, view_264, getitem_85, mul_58, getitem_89, getitem_91, mul_61, view_300, addmm_14, getitem_95, view_302, getitem_97, mul_66, getitem_101, getitem_103, mul_69, view_338, addmm_16, getitem_107, view_340, getitem_109, mul_74, getitem_113, getitem_115, mul_77, view_376, addmm_18, getitem_119, view_378, getitem_121, mul_82, getitem_125, getitem_127, mul_85, view_414, addmm_20, getitem_131, view_416, getitem_133, mul_90, getitem_137, getitem_139, mul_93, view_452, addmm_22, getitem_143, view_454, getitem_145, mul_98, getitem_149, getitem_151, mul_101, view_490, addmm_24, getitem_155, view_492, getitem_157, mul_106, getitem_161, getitem_163, mul_109, view_528, addmm_26, getitem_167, view_530, getitem_169, mul_114, getitem_173, getitem_175, mul_117, view_566, addmm_28, getitem_179, view_568, getitem_181, mul_122, getitem_185, getitem_187, mul_125, view_604, addmm_30, getitem_191, view_606, getitem_193, mul_130, getitem_197, getitem_199, mul_133, view_642, addmm_32, getitem_203, view_644, getitem_205, mul_138, getitem_209, getitem_211, mul_141, view_680, addmm_34, getitem_215, view_682, getitem_217, mul_146, getitem_221, getitem_223, mul_149, view_718, addmm_36, getitem_227, view_720, getitem_229, mul_154, getitem_233, getitem_235, mul_157, view_756, addmm_38, getitem_239, view_758, getitem_241, mul_162, getitem_245, getitem_247, mul_165, view_794, addmm_40, getitem_251, view_796, getitem_253, mul_170, getitem_257, getitem_259, mul_173, view_832, addmm_42, getitem_263, view_834, getitem_265, mul_178, getitem_269, getitem_271, mul_181, view_870, addmm_44, getitem_275, view_872, getitem_277, mul_186, getitem_281, getitem_283, mul_189, view_908, addmm_46, getitem_287, view_910, getitem_289, mul_194, getitem_293, view_912, sub_73, convert_element_type_5, permute_1013, div_27, permute_1018, permute_1022, div_28, permute_1027, permute_1028, permute_1034, permute_1035, alias_26, permute_1041, permute_1042, permute_1048, permute_1049, permute_1055, permute_1059, permute_1060, permute_1067, permute_1074, div_29, permute_1079, permute_1083, div_30, permute_1088, permute_1089, permute_1095, permute_1096, alias_27, permute_1102, permute_1103, permute_1109, permute_1110, permute_1120, permute_1121, permute_1128, permute_1135, div_31, permute_1140, permute_1144, div_32, permute_1149, permute_1150, permute_1156, permute_1157, alias_28, permute_1163, permute_1164, permute_1170, permute_1171, permute_1181, permute_1182, permute_1189, permute_1196, div_33, permute_1201, permute_1205, div_34, permute_1210, permute_1211, permute_1217, permute_1218, alias_29, permute_1224, permute_1225, permute_1231, permute_1232, permute_1242, permute_1243, permute_1250, permute_1257, div_35, permute_1262, permute_1266, div_36, permute_1271, permute_1272, permute_1278, permute_1279, alias_30, permute_1285, permute_1286, permute_1292, permute_1293, permute_1303, permute_1304, permute_1311, permute_1318, div_37, permute_1323, permute_1327, div_38, permute_1332, permute_1333, permute_1339, permute_1340, alias_31, permute_1346, permute_1347, permute_1353, permute_1354, permute_1364, permute_1365, permute_1372, permute_1379, div_39, permute_1384, permute_1388, div_40, permute_1393, permute_1394, permute_1400, permute_1401, alias_32, permute_1407, permute_1408, permute_1414, permute_1415, permute_1425, permute_1426, permute_1433, permute_1440, div_41, permute_1445, permute_1449, div_42, permute_1454, permute_1455, permute_1461, permute_1462, alias_33, permute_1468, permute_1469, permute_1475, permute_1476, permute_1486, permute_1487, permute_1494, permute_1501, div_43, permute_1506, permute_1510, div_44, permute_1515, permute_1516, permute_1522, permute_1523, alias_34, permute_1529, permute_1530, permute_1536, permute_1537, permute_1547, permute_1548, permute_1555, permute_1562, div_45, permute_1567, permute_1571, div_46, permute_1576, permute_1577, permute_1583, permute_1584, alias_35, permute_1590, permute_1591, permute_1597, permute_1598, permute_1608, permute_1609, permute_1616, permute_1623, div_47, permute_1628, permute_1632, div_48, permute_1637, permute_1638, permute_1644, permute_1645, alias_36, permute_1651, permute_1652, permute_1658, permute_1659, permute_1669, permute_1670, permute_1677, permute_1684, div_49, permute_1689, permute_1693, div_50, permute_1698, permute_1699, permute_1705, permute_1706, alias_37, permute_1712, permute_1713, permute_1719, permute_1720, permute_1730, permute_1731, permute_1738, permute_1745, div_51, permute_1750, permute_1754, div_52, permute_1759, permute_1760, permute_1766, permute_1767, alias_38, permute_1773, permute_1774, permute_1780, permute_1781, permute_1791, permute_1792, permute_1799, permute_1806, div_53, permute_1811, permute_1815, div_54, permute_1820, permute_1821, permute_1827, permute_1828, alias_39, permute_1834, permute_1835, permute_1841, permute_1842, permute_1852, permute_1853, permute_1860, permute_1867, div_55, permute_1872, permute_1876, div_56, permute_1881, permute_1882, permute_1888, permute_1889, alias_40, permute_1895, permute_1896, permute_1902, permute_1903, permute_1913, permute_1914, permute_1921, permute_1928, div_57, permute_1933, permute_1937, div_58, permute_1942, permute_1943, permute_1949, permute_1950, alias_41, permute_1956, permute_1957, permute_1963, permute_1964, permute_1974, permute_1975, permute_1982, permute_1989, div_59, permute_1994, permute_1998, div_60, permute_2003, permute_2004, permute_2010, permute_2011, alias_42, permute_2017, permute_2018, permute_2024, permute_2025, permute_2035, permute_2036, permute_2043, permute_2050, div_61, permute_2055, permute_2059, div_62, permute_2064, permute_2065, permute_2071, permute_2072, alias_43, permute_2078, permute_2079, permute_2085, permute_2086, permute_2096, permute_2097, permute_2104, permute_2111, div_63, permute_2116, permute_2120, div_64, permute_2125, permute_2126, permute_2132, permute_2133, alias_44, permute_2139, permute_2140, permute_2146, permute_2147, permute_2157, permute_2158, permute_2165, permute_2172, div_65, permute_2177, permute_2181, div_66, permute_2186, permute_2187, permute_2193, permute_2194, alias_45, permute_2200, permute_2201, permute_2207, permute_2208, permute_2218, permute_2219, permute_2226, permute_2233, div_67, permute_2238, permute_2242, div_68, permute_2247, permute_2248, permute_2254, permute_2255, alias_46, permute_2261, permute_2262, permute_2268, permute_2269, permute_2279, permute_2280, permute_2287, permute_2294, div_69, permute_2299, permute_2303, div_70, permute_2308, permute_2309, permute_2315, permute_2316, alias_47, permute_2322, permute_2323, permute_2329, permute_2330, permute_2340, permute_2341, permute_2348, permute_2355, div_71, permute_2360, permute_2364, div_72, permute_2369, permute_2370, permute_2376, permute_2377, alias_48, permute_2383, permute_2384, permute_2390, permute_2391, permute_2401, permute_2402, permute_2409, permute_2416, div_73, permute_2421, permute_2425, div_74, permute_2430, permute_2431, permute_2437, permute_2438, alias_49, permute_2444, permute_2445, permute_2451, permute_2452, permute_2462, permute_2463, permute_2470, permute_2477]
    