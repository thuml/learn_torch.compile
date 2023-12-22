from __future__ import annotations



def forward(self, primals_1: "f32[29056, 1024]", primals_2: "f32[2, 1024]", primals_3: "f32[512, 1024]", primals_4: "f32[1024]", primals_5: "f32[1024]", primals_6: "f32[1024, 1024]", primals_7: "f32[1024]", primals_8: "f32[1024, 1024]", primals_9: "f32[1024]", primals_10: "f32[1024, 1024]", primals_11: "f32[1024]", primals_12: "f32[1024, 1024]", primals_13: "f32[1024]", primals_14: "f32[1024]", primals_15: "f32[1024]", primals_16: "f32[4096, 1024]", primals_17: "f32[4096]", primals_18: "f32[1024, 4096]", primals_19: "f32[1024]", primals_20: "f32[1024]", primals_21: "f32[1024]", primals_22: "f32[1024, 1024]", primals_23: "f32[1024]", primals_24: "f32[1024, 1024]", primals_25: "f32[1024]", primals_26: "f32[1024, 1024]", primals_27: "f32[1024]", primals_28: "f32[1024, 1024]", primals_29: "f32[1024]", primals_30: "f32[1024]", primals_31: "f32[1024]", primals_32: "f32[4096, 1024]", primals_33: "f32[4096]", primals_34: "f32[1024, 4096]", primals_35: "f32[1024]", primals_36: "f32[1024]", primals_37: "f32[1024]", primals_38: "f32[1024, 1024]", primals_39: "f32[1024]", primals_40: "f32[1024, 1024]", primals_41: "f32[1024]", primals_42: "f32[1024, 1024]", primals_43: "f32[1024]", primals_44: "f32[1024, 1024]", primals_45: "f32[1024]", primals_46: "f32[1024]", primals_47: "f32[1024]", primals_48: "f32[4096, 1024]", primals_49: "f32[4096]", primals_50: "f32[1024, 4096]", primals_51: "f32[1024]", primals_52: "f32[1024]", primals_53: "f32[1024]", primals_54: "f32[1024, 1024]", primals_55: "f32[1024]", primals_56: "f32[1024, 1024]", primals_57: "f32[1024]", primals_58: "f32[1024, 1024]", primals_59: "f32[1024]", primals_60: "f32[1024, 1024]", primals_61: "f32[1024]", primals_62: "f32[1024]", primals_63: "f32[1024]", primals_64: "f32[4096, 1024]", primals_65: "f32[4096]", primals_66: "f32[1024, 4096]", primals_67: "f32[1024]", primals_68: "f32[1024]", primals_69: "f32[1024]", primals_70: "f32[1024, 1024]", primals_71: "f32[1024]", primals_72: "f32[1024, 1024]", primals_73: "f32[1024]", primals_74: "f32[1024, 1024]", primals_75: "f32[1024]", primals_76: "f32[1024, 1024]", primals_77: "f32[1024]", primals_78: "f32[1024]", primals_79: "f32[1024]", primals_80: "f32[4096, 1024]", primals_81: "f32[4096]", primals_82: "f32[1024, 4096]", primals_83: "f32[1024]", primals_84: "f32[1024]", primals_85: "f32[1024]", primals_86: "f32[1024, 1024]", primals_87: "f32[1024]", primals_88: "f32[1024, 1024]", primals_89: "f32[1024]", primals_90: "f32[1024, 1024]", primals_91: "f32[1024]", primals_92: "f32[1024, 1024]", primals_93: "f32[1024]", primals_94: "f32[1024]", primals_95: "f32[1024]", primals_96: "f32[4096, 1024]", primals_97: "f32[4096]", primals_98: "f32[1024, 4096]", primals_99: "f32[1024]", primals_100: "f32[1024]", primals_101: "f32[1024]", primals_102: "f32[1024, 1024]", primals_103: "f32[1024]", primals_104: "f32[1024, 1024]", primals_105: "f32[1024]", primals_106: "f32[1024, 1024]", primals_107: "f32[1024]", primals_108: "f32[1024, 1024]", primals_109: "f32[1024]", primals_110: "f32[1024]", primals_111: "f32[1024]", primals_112: "f32[4096, 1024]", primals_113: "f32[4096]", primals_114: "f32[1024, 4096]", primals_115: "f32[1024]", primals_116: "f32[1024]", primals_117: "f32[1024]", primals_118: "f32[1024, 1024]", primals_119: "f32[1024]", primals_120: "f32[1024, 1024]", primals_121: "f32[1024]", primals_122: "f32[1024, 1024]", primals_123: "f32[1024]", primals_124: "f32[1024, 1024]", primals_125: "f32[1024]", primals_126: "f32[1024]", primals_127: "f32[1024]", primals_128: "f32[4096, 1024]", primals_129: "f32[4096]", primals_130: "f32[1024, 4096]", primals_131: "f32[1024]", primals_132: "f32[1024]", primals_133: "f32[1024]", primals_134: "f32[1024, 1024]", primals_135: "f32[1024]", primals_136: "f32[1024, 1024]", primals_137: "f32[1024]", primals_138: "f32[1024, 1024]", primals_139: "f32[1024]", primals_140: "f32[1024, 1024]", primals_141: "f32[1024]", primals_142: "f32[1024]", primals_143: "f32[1024]", primals_144: "f32[4096, 1024]", primals_145: "f32[4096]", primals_146: "f32[1024, 4096]", primals_147: "f32[1024]", primals_148: "f32[1024]", primals_149: "f32[1024]", primals_150: "f32[1024, 1024]", primals_151: "f32[1024]", primals_152: "f32[1024, 1024]", primals_153: "f32[1024]", primals_154: "f32[1024, 1024]", primals_155: "f32[1024]", primals_156: "f32[1024, 1024]", primals_157: "f32[1024]", primals_158: "f32[1024]", primals_159: "f32[1024]", primals_160: "f32[4096, 1024]", primals_161: "f32[4096]", primals_162: "f32[1024, 4096]", primals_163: "f32[1024]", primals_164: "f32[1024]", primals_165: "f32[1024]", primals_166: "f32[1024, 1024]", primals_167: "f32[1024]", primals_168: "f32[1024, 1024]", primals_169: "f32[1024]", primals_170: "f32[1024, 1024]", primals_171: "f32[1024]", primals_172: "f32[1024, 1024]", primals_173: "f32[1024]", primals_174: "f32[1024]", primals_175: "f32[1024]", primals_176: "f32[4096, 1024]", primals_177: "f32[4096]", primals_178: "f32[1024, 4096]", primals_179: "f32[1024]", primals_180: "f32[1024]", primals_181: "f32[1024]", primals_182: "f32[1024, 1024]", primals_183: "f32[1024]", primals_184: "f32[1024, 1024]", primals_185: "f32[1024]", primals_186: "f32[1024, 1024]", primals_187: "f32[1024]", primals_188: "f32[1024, 1024]", primals_189: "f32[1024]", primals_190: "f32[1024]", primals_191: "f32[1024]", primals_192: "f32[4096, 1024]", primals_193: "f32[4096]", primals_194: "f32[1024, 4096]", primals_195: "f32[1024]", primals_196: "f32[1024]", primals_197: "f32[1024]", primals_198: "f32[1024, 1024]", primals_199: "f32[1024]", primals_200: "f32[1024, 1024]", primals_201: "f32[1024]", primals_202: "f32[1024, 1024]", primals_203: "f32[1024]", primals_204: "f32[1024, 1024]", primals_205: "f32[1024]", primals_206: "f32[1024]", primals_207: "f32[1024]", primals_208: "f32[4096, 1024]", primals_209: "f32[4096]", primals_210: "f32[1024, 4096]", primals_211: "f32[1024]", primals_212: "f32[1024]", primals_213: "f32[1024]", primals_214: "f32[1024, 1024]", primals_215: "f32[1024]", primals_216: "f32[1024, 1024]", primals_217: "f32[1024]", primals_218: "f32[1024, 1024]", primals_219: "f32[1024]", primals_220: "f32[1024, 1024]", primals_221: "f32[1024]", primals_222: "f32[1024]", primals_223: "f32[1024]", primals_224: "f32[4096, 1024]", primals_225: "f32[4096]", primals_226: "f32[1024, 4096]", primals_227: "f32[1024]", primals_228: "f32[1024]", primals_229: "f32[1024]", primals_230: "f32[1024, 1024]", primals_231: "f32[1024]", primals_232: "f32[1024, 1024]", primals_233: "f32[1024]", primals_234: "f32[1024, 1024]", primals_235: "f32[1024]", primals_236: "f32[1024, 1024]", primals_237: "f32[1024]", primals_238: "f32[1024]", primals_239: "f32[1024]", primals_240: "f32[4096, 1024]", primals_241: "f32[4096]", primals_242: "f32[1024, 4096]", primals_243: "f32[1024]", primals_244: "f32[1024]", primals_245: "f32[1024]", primals_246: "f32[1024, 1024]", primals_247: "f32[1024]", primals_248: "f32[1024, 1024]", primals_249: "f32[1024]", primals_250: "f32[1024, 1024]", primals_251: "f32[1024]", primals_252: "f32[1024, 1024]", primals_253: "f32[1024]", primals_254: "f32[1024]", primals_255: "f32[1024]", primals_256: "f32[4096, 1024]", primals_257: "f32[4096]", primals_258: "f32[1024, 4096]", primals_259: "f32[1024]", primals_260: "f32[1024]", primals_261: "f32[1024]", primals_262: "f32[1024, 1024]", primals_263: "f32[1024]", primals_264: "f32[1024, 1024]", primals_265: "f32[1024]", primals_266: "f32[1024, 1024]", primals_267: "f32[1024]", primals_268: "f32[1024, 1024]", primals_269: "f32[1024]", primals_270: "f32[1024]", primals_271: "f32[1024]", primals_272: "f32[4096, 1024]", primals_273: "f32[4096]", primals_274: "f32[1024, 4096]", primals_275: "f32[1024]", primals_276: "f32[1024]", primals_277: "f32[1024]", primals_278: "f32[1024, 1024]", primals_279: "f32[1024]", primals_280: "f32[1024, 1024]", primals_281: "f32[1024]", primals_282: "f32[1024, 1024]", primals_283: "f32[1024]", primals_284: "f32[1024, 1024]", primals_285: "f32[1024]", primals_286: "f32[1024]", primals_287: "f32[1024]", primals_288: "f32[4096, 1024]", primals_289: "f32[4096]", primals_290: "f32[1024, 4096]", primals_291: "f32[1024]", primals_292: "f32[1024]", primals_293: "f32[1024]", primals_294: "f32[1024, 1024]", primals_295: "f32[1024]", primals_296: "f32[1024, 1024]", primals_297: "f32[1024]", primals_298: "f32[1024, 1024]", primals_299: "f32[1024]", primals_300: "f32[1024, 1024]", primals_301: "f32[1024]", primals_302: "f32[1024]", primals_303: "f32[1024]", primals_304: "f32[4096, 1024]", primals_305: "f32[4096]", primals_306: "f32[1024, 4096]", primals_307: "f32[1024]", primals_308: "f32[1024]", primals_309: "f32[1024]", primals_310: "f32[1024, 1024]", primals_311: "f32[1024]", primals_312: "f32[1024, 1024]", primals_313: "f32[1024]", primals_314: "f32[1024, 1024]", primals_315: "f32[1024]", primals_316: "f32[1024, 1024]", primals_317: "f32[1024]", primals_318: "f32[1024]", primals_319: "f32[1024]", primals_320: "f32[4096, 1024]", primals_321: "f32[4096]", primals_322: "f32[1024, 4096]", primals_323: "f32[1024]", primals_324: "f32[1024]", primals_325: "f32[1024]", primals_326: "f32[1024, 1024]", primals_327: "f32[1024]", primals_328: "f32[1024, 1024]", primals_329: "f32[1024]", primals_330: "f32[1024, 1024]", primals_331: "f32[1024]", primals_332: "f32[1024, 1024]", primals_333: "f32[1024]", primals_334: "f32[1024]", primals_335: "f32[1024]", primals_336: "f32[4096, 1024]", primals_337: "f32[4096]", primals_338: "f32[1024, 4096]", primals_339: "f32[1024]", primals_340: "f32[1024]", primals_341: "f32[1024]", primals_342: "f32[1024, 1024]", primals_343: "f32[1024]", primals_344: "f32[1024, 1024]", primals_345: "f32[1024]", primals_346: "f32[1024, 1024]", primals_347: "f32[1024]", primals_348: "f32[1024, 1024]", primals_349: "f32[1024]", primals_350: "f32[1024]", primals_351: "f32[1024]", primals_352: "f32[4096, 1024]", primals_353: "f32[4096]", primals_354: "f32[1024, 4096]", primals_355: "f32[1024]", primals_356: "f32[1024]", primals_357: "f32[1024]", primals_358: "f32[1024, 1024]", primals_359: "f32[1024]", primals_360: "f32[1024, 1024]", primals_361: "f32[1024]", primals_362: "f32[1024, 1024]", primals_363: "f32[1024]", primals_364: "f32[1024, 1024]", primals_365: "f32[1024]", primals_366: "f32[1024]", primals_367: "f32[1024]", primals_368: "f32[4096, 1024]", primals_369: "f32[4096]", primals_370: "f32[1024, 4096]", primals_371: "f32[1024]", primals_372: "f32[1024]", primals_373: "f32[1024]", primals_374: "f32[1024, 1024]", primals_375: "f32[1024]", primals_376: "f32[1024, 1024]", primals_377: "f32[1024]", primals_378: "f32[1024, 1024]", primals_379: "f32[1024]", primals_380: "f32[1024, 1024]", primals_381: "f32[1024]", primals_382: "f32[1024]", primals_383: "f32[1024]", primals_384: "f32[4096, 1024]", primals_385: "f32[4096]", primals_386: "f32[1024, 4096]", primals_387: "f32[1024]", primals_388: "f32[1024]", primals_389: "f32[1024]", primals_390: "f32[2, 1024]", primals_391: "f32[2]", primals_392: "i64[1, 512]", primals_393: "i64[1, 512]", primals_394: "i64[1]", primals_395: "i64[1]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:952, code: token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
    full_default: "i64[1, 512]" = torch.ops.aten.full.default([1, 512], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:173, code: position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
    slice_3: "i64[1, 512]" = torch.ops.aten.slice.Tensor(primals_392, 0, 0, 9223372036854775807);  primals_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:179, code: inputs_embeds = self.word_embeddings(input_ids)
    embedding: "f32[1, 512, 1024]" = torch.ops.aten.embedding.default(primals_1, primals_393, 0);  primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:180, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    embedding_1: "f32[1, 512, 1024]" = torch.ops.aten.embedding.default(primals_2, full_default);  primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:182, code: embeddings = inputs_embeds + token_type_embeddings
    add: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:184, code: position_embeddings = self.position_embeddings(position_ids)
    embedding_2: "f32[1, 512, 1024]" = torch.ops.aten.embedding.default(primals_3, slice_3);  primals_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185, code: embeddings += position_embeddings
    add_1: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189, code: embeddings = self.dropout(embeddings)
    native_dropout = torch.ops.aten.native_dropout.default(add_1, 0.1, True);  add_1 = None
    getitem: "f32[1, 512, 1024]" = native_dropout[0]
    getitem_1: "b8[1, 512, 1024]" = native_dropout[1];  native_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(getitem, [2], correction = 0, keepdim = True)
    getitem_2: "f32[1, 512, 1]" = var_mean[0]
    getitem_3: "f32[1, 512, 1]" = var_mean[1];  var_mean = None
    add_2: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
    rsqrt: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
    sub_1: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(getitem, getitem_3);  getitem_3 = None
    mul_1: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
    mul_2: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_1, primals_4)
    add_3: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_2, primals_5);  mul_2 = primals_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view: "f32[512, 1024]" = torch.ops.aten.view.default(add_3, [512, 1024]);  add_3 = None
    permute: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
    addmm: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_7, view, permute);  primals_7 = None
    view_1: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm, [1, 512, 1024]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_1: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
    addmm_1: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_9, view, permute_1);  primals_9 = None
    view_3: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_1, [1, 512, 1024]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_4: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_3, [1, 512, 16, 64]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_2: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_3: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
    addmm_2: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_11, view, permute_3);  primals_11 = None
    view_6: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_2, [1, 512, 1024]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_7: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_6, [1, 512, 16, 64]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_4: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_8: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_1, [1, 512, 16, 64]);  view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_5: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    
    # No stacktrace found for following nodes
    clone_default_92: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    clone_default_93: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    clone_default_94: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    mul_scalar_92: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_92, 0.3535533905932738);  clone_default_92 = None
    permute_default_138: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(clone_default_93, [0, 1, 3, 2]);  clone_default_93 = None
    mul_scalar_93: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_138, 0.3535533905932738);  permute_default_138 = None
    expand_default_92: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_92, [1, 16, 512, 64]);  mul_scalar_92 = None
    view_default_276: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_92, [16, 512, 64]);  expand_default_92 = None
    expand_default_93: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_93, [1, 16, 64, 512]);  mul_scalar_93 = None
    view_default_277: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_default_93, [16, 64, 512]);  expand_default_93 = None
    bmm_default_138: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_276, view_default_277)
    view_default_278: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_138, [1, 16, 512, 512]);  bmm_default_138 = None
    amax_default_23: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(view_default_278, [-1], True)
    sub_tensor_46: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_278, amax_default_23);  view_default_278 = amax_default_23 = None
    exp_default_23: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_46);  sub_tensor_46 = None
    sum_dim_int_list_46: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_23, [-1], True)
    div_tensor_23: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_23, sum_dim_int_list_46);  exp_default_23 = sum_dim_int_list_46 = None
    alias_default_46: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_tensor_23)
    native_dropout_default_23 = torch.ops.aten.native_dropout.default(div_tensor_23, 0.1, True);  div_tensor_23 = None
    getitem_292: "f32[1, 16, 512, 512]" = native_dropout_default_23[0]
    getitem_293: "b8[1, 16, 512, 512]" = native_dropout_default_23[1];  native_dropout_default_23 = None
    expand_default_94: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_292, [1, 16, 512, 512]);  getitem_292 = None
    view_default_279: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_default_94, [16, 512, 512]);  expand_default_94 = None
    expand_default_95: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(clone_default_94, [1, 16, 512, 64]);  clone_default_94 = None
    view_default_280: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_95, [16, 512, 64]);  expand_default_95 = None
    bmm_default_139: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_279, view_default_280)
    view_default_281: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_139, [1, 16, 512, 64]);  bmm_default_139 = None
    permute_default_139: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_default_279, [0, 2, 1]);  view_default_279 = None
    permute_default_140: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_280, [0, 2, 1]);  view_default_280 = None
    alias_default_47: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_default_46);  alias_default_46 = None
    permute_default_141: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_276, [0, 2, 1]);  view_default_276 = None
    permute_default_142: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_default_277, [0, 2, 1]);  view_default_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_7: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_281, [0, 2, 1, 3]);  view_default_281 = None
    clone: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_15: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone, [1, 512, 1024]);  clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_16: "f32[512, 1024]" = torch.ops.aten.view.default(view_15, [512, 1024]);  view_15 = None
    permute_8: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
    addmm_3: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_13, view_16, permute_8);  primals_13 = None
    view_17: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_3, [1, 512, 1024]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_2 = torch.ops.aten.native_dropout.default(view_17, 0.1, True);  view_17 = None
    getitem_6: "f32[1, 512, 1024]" = native_dropout_2[0]
    getitem_7: "b8[1, 512, 1024]" = native_dropout_2[1];  native_dropout_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_5: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(getitem, getitem_6);  getitem = getitem_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 512, 1]" = var_mean_1[0]
    getitem_9: "f32[1, 512, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
    rsqrt_1: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_3: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_5, getitem_9);  getitem_9 = None
    mul_3: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = None
    mul_4: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_3, primals_14)
    add_7: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_4, primals_15);  mul_4 = primals_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_18: "f32[512, 1024]" = torch.ops.aten.view.default(add_7, [512, 1024]);  add_7 = None
    permute_9: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
    addmm_4: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_17, view_18, permute_9);  primals_17 = None
    view_19: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_4, [1, 512, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_5: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_19, 0.5)
    mul_6: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476);  view_19 = None
    erf: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
    add_8: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_7: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_5, add_8);  mul_5 = add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_20: "f32[512, 4096]" = torch.ops.aten.view.default(mul_7, [512, 4096]);  mul_7 = None
    permute_10: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
    addmm_5: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_19, view_20, permute_10);  primals_19 = None
    view_21: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_5, [1, 512, 1024]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_3 = torch.ops.aten.native_dropout.default(view_21, 0.1, True);  view_21 = None
    getitem_10: "f32[1, 512, 1024]" = native_dropout_3[0]
    getitem_11: "b8[1, 512, 1024]" = native_dropout_3[1];  native_dropout_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_9: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_5, getitem_10);  add_5 = getitem_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 512, 1]" = var_mean_2[0]
    getitem_13: "f32[1, 512, 1]" = var_mean_2[1];  var_mean_2 = None
    add_10: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
    rsqrt_2: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
    sub_4: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_9, getitem_13);  getitem_13 = None
    mul_8: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = None
    mul_9: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_8, primals_20)
    add_11: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_9, primals_21);  mul_9 = primals_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_22: "f32[512, 1024]" = torch.ops.aten.view.default(add_11, [512, 1024]);  add_11 = None
    permute_11: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_22, [1, 0]);  primals_22 = None
    addmm_6: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_23, view_22, permute_11);  primals_23 = None
    view_23: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_6, [1, 512, 1024]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_12: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
    addmm_7: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_25, view_22, permute_12);  primals_25 = None
    view_25: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_7, [1, 512, 1024]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_26: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_25, [1, 512, 16, 64]);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_13: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_14: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
    addmm_8: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_27, view_22, permute_14);  primals_27 = None
    view_28: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_8, [1, 512, 1024]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_29: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_28, [1, 512, 16, 64]);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_15: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_30: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_23, [1, 512, 16, 64]);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_16: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    
    # No stacktrace found for following nodes
    clone_default_88: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
    clone_default_89: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
    clone_default_90: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    mul_scalar_88: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_88, 0.3535533905932738);  clone_default_88 = None
    permute_default_132: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(clone_default_89, [0, 1, 3, 2]);  clone_default_89 = None
    mul_scalar_89: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_132, 0.3535533905932738);  permute_default_132 = None
    expand_default_88: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_88, [1, 16, 512, 64]);  mul_scalar_88 = None
    view_default_264: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_88, [16, 512, 64]);  expand_default_88 = None
    expand_default_89: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_89, [1, 16, 64, 512]);  mul_scalar_89 = None
    view_default_265: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_default_89, [16, 64, 512]);  expand_default_89 = None
    bmm_default_132: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_264, view_default_265)
    view_default_266: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_132, [1, 16, 512, 512]);  bmm_default_132 = None
    amax_default_22: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(view_default_266, [-1], True)
    sub_tensor_44: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_266, amax_default_22);  view_default_266 = amax_default_22 = None
    exp_default_22: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_44);  sub_tensor_44 = None
    sum_dim_int_list_44: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_22, [-1], True)
    div_tensor_22: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_22, sum_dim_int_list_44);  exp_default_22 = sum_dim_int_list_44 = None
    alias_default_44: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_tensor_22)
    native_dropout_default_22 = torch.ops.aten.native_dropout.default(div_tensor_22, 0.1, True);  div_tensor_22 = None
    getitem_290: "f32[1, 16, 512, 512]" = native_dropout_default_22[0]
    getitem_291: "b8[1, 16, 512, 512]" = native_dropout_default_22[1];  native_dropout_default_22 = None
    expand_default_90: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_290, [1, 16, 512, 512]);  getitem_290 = None
    view_default_267: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_default_90, [16, 512, 512]);  expand_default_90 = None
    expand_default_91: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(clone_default_90, [1, 16, 512, 64]);  clone_default_90 = None
    view_default_268: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_91, [16, 512, 64]);  expand_default_91 = None
    bmm_default_133: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_267, view_default_268)
    view_default_269: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_133, [1, 16, 512, 64]);  bmm_default_133 = None
    permute_default_133: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_default_267, [0, 2, 1]);  view_default_267 = None
    permute_default_134: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_268, [0, 2, 1]);  view_default_268 = None
    alias_default_45: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_default_44);  alias_default_44 = None
    permute_default_135: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_264, [0, 2, 1]);  view_default_264 = None
    permute_default_136: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_default_265, [0, 2, 1]);  view_default_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_18: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_269, [0, 2, 1, 3]);  view_default_269 = None
    clone_1: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_37: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_1, [1, 512, 1024]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_38: "f32[512, 1024]" = torch.ops.aten.view.default(view_37, [512, 1024]);  view_37 = None
    permute_19: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
    addmm_9: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_29, view_38, permute_19);  primals_29 = None
    view_39: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_9, [1, 512, 1024]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_5 = torch.ops.aten.native_dropout.default(view_39, 0.1, True);  view_39 = None
    getitem_16: "f32[1, 512, 1024]" = native_dropout_5[0]
    getitem_17: "b8[1, 512, 1024]" = native_dropout_5[1];  native_dropout_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_13: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_9, getitem_16);  add_9 = getitem_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_13, [2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 512, 1]" = var_mean_3[0]
    getitem_19: "f32[1, 512, 1]" = var_mean_3[1];  var_mean_3 = None
    add_14: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
    rsqrt_3: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    sub_6: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_13, getitem_19);  getitem_19 = None
    mul_10: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_3);  sub_6 = None
    mul_11: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_10, primals_30)
    add_15: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_11, primals_31);  mul_11 = primals_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_40: "f32[512, 1024]" = torch.ops.aten.view.default(add_15, [512, 1024]);  add_15 = None
    permute_20: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
    addmm_10: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_33, view_40, permute_20);  primals_33 = None
    view_41: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_10, [1, 512, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_12: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_41, 0.5)
    mul_13: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476);  view_41 = None
    erf_1: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
    add_16: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_14: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_12, add_16);  mul_12 = add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_42: "f32[512, 4096]" = torch.ops.aten.view.default(mul_14, [512, 4096]);  mul_14 = None
    permute_21: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
    addmm_11: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_35, view_42, permute_21);  primals_35 = None
    view_43: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_11, [1, 512, 1024]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_6 = torch.ops.aten.native_dropout.default(view_43, 0.1, True);  view_43 = None
    getitem_20: "f32[1, 512, 1024]" = native_dropout_6[0]
    getitem_21: "b8[1, 512, 1024]" = native_dropout_6[1];  native_dropout_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_17: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_13, getitem_20);  add_13 = getitem_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 512, 1]" = var_mean_4[0]
    getitem_23: "f32[1, 512, 1]" = var_mean_4[1];  var_mean_4 = None
    add_18: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
    rsqrt_4: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_7: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_17, getitem_23);  getitem_23 = None
    mul_15: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_4);  sub_7 = None
    mul_16: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_15, primals_36)
    add_19: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_16, primals_37);  mul_16 = primals_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_44: "f32[512, 1024]" = torch.ops.aten.view.default(add_19, [512, 1024]);  add_19 = None
    permute_22: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
    addmm_12: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_39, view_44, permute_22);  primals_39 = None
    view_45: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_12, [1, 512, 1024]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_23: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
    addmm_13: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_41, view_44, permute_23);  primals_41 = None
    view_47: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_13, [1, 512, 1024]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_48: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_47, [1, 512, 16, 64]);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_24: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_25: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
    addmm_14: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_43, view_44, permute_25);  primals_43 = None
    view_50: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_14, [1, 512, 1024]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_51: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_50, [1, 512, 16, 64]);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_26: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_52: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_45, [1, 512, 16, 64]);  view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_27: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
    
    # No stacktrace found for following nodes
    clone_default_84: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
    clone_default_85: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
    clone_default_86: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
    mul_scalar_84: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_84, 0.3535533905932738);  clone_default_84 = None
    permute_default_126: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(clone_default_85, [0, 1, 3, 2]);  clone_default_85 = None
    mul_scalar_85: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_126, 0.3535533905932738);  permute_default_126 = None
    expand_default_84: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_84, [1, 16, 512, 64]);  mul_scalar_84 = None
    view_default_252: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_84, [16, 512, 64]);  expand_default_84 = None
    expand_default_85: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_85, [1, 16, 64, 512]);  mul_scalar_85 = None
    view_default_253: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_default_85, [16, 64, 512]);  expand_default_85 = None
    bmm_default_126: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_252, view_default_253)
    view_default_254: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_126, [1, 16, 512, 512]);  bmm_default_126 = None
    amax_default_21: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(view_default_254, [-1], True)
    sub_tensor_42: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_254, amax_default_21);  view_default_254 = amax_default_21 = None
    exp_default_21: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_42);  sub_tensor_42 = None
    sum_dim_int_list_42: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_21, [-1], True)
    div_tensor_21: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_21, sum_dim_int_list_42);  exp_default_21 = sum_dim_int_list_42 = None
    alias_default_42: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_tensor_21)
    native_dropout_default_21 = torch.ops.aten.native_dropout.default(div_tensor_21, 0.1, True);  div_tensor_21 = None
    getitem_288: "f32[1, 16, 512, 512]" = native_dropout_default_21[0]
    getitem_289: "b8[1, 16, 512, 512]" = native_dropout_default_21[1];  native_dropout_default_21 = None
    expand_default_86: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_288, [1, 16, 512, 512]);  getitem_288 = None
    view_default_255: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_default_86, [16, 512, 512]);  expand_default_86 = None
    expand_default_87: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(clone_default_86, [1, 16, 512, 64]);  clone_default_86 = None
    view_default_256: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_87, [16, 512, 64]);  expand_default_87 = None
    bmm_default_127: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_255, view_default_256)
    view_default_257: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_127, [1, 16, 512, 64]);  bmm_default_127 = None
    permute_default_127: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_default_255, [0, 2, 1]);  view_default_255 = None
    permute_default_128: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_256, [0, 2, 1]);  view_default_256 = None
    alias_default_43: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_default_42);  alias_default_42 = None
    permute_default_129: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_252, [0, 2, 1]);  view_default_252 = None
    permute_default_130: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_default_253, [0, 2, 1]);  view_default_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_29: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_257, [0, 2, 1, 3]);  view_default_257 = None
    clone_2: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_59: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_2, [1, 512, 1024]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_60: "f32[512, 1024]" = torch.ops.aten.view.default(view_59, [512, 1024]);  view_59 = None
    permute_30: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
    addmm_15: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_45, view_60, permute_30);  primals_45 = None
    view_61: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_15, [1, 512, 1024]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_8 = torch.ops.aten.native_dropout.default(view_61, 0.1, True);  view_61 = None
    getitem_26: "f32[1, 512, 1024]" = native_dropout_8[0]
    getitem_27: "b8[1, 512, 1024]" = native_dropout_8[1];  native_dropout_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_21: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_17, getitem_26);  add_17 = getitem_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 512, 1]" = var_mean_5[0]
    getitem_29: "f32[1, 512, 1]" = var_mean_5[1];  var_mean_5 = None
    add_22: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
    rsqrt_5: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
    sub_9: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_21, getitem_29);  getitem_29 = None
    mul_17: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_5);  sub_9 = None
    mul_18: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_17, primals_46)
    add_23: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_18, primals_47);  mul_18 = primals_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_62: "f32[512, 1024]" = torch.ops.aten.view.default(add_23, [512, 1024]);  add_23 = None
    permute_31: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
    addmm_16: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_49, view_62, permute_31);  primals_49 = None
    view_63: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_16, [1, 512, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_19: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_63, 0.5)
    mul_20: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476);  view_63 = None
    erf_2: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
    add_24: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_21: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_19, add_24);  mul_19 = add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_64: "f32[512, 4096]" = torch.ops.aten.view.default(mul_21, [512, 4096]);  mul_21 = None
    permute_32: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
    addmm_17: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_51, view_64, permute_32);  primals_51 = None
    view_65: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_17, [1, 512, 1024]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_9 = torch.ops.aten.native_dropout.default(view_65, 0.1, True);  view_65 = None
    getitem_30: "f32[1, 512, 1024]" = native_dropout_9[0]
    getitem_31: "b8[1, 512, 1024]" = native_dropout_9[1];  native_dropout_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_25: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_21, getitem_30);  add_21 = getitem_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 512, 1]" = var_mean_6[0]
    getitem_33: "f32[1, 512, 1]" = var_mean_6[1];  var_mean_6 = None
    add_26: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
    rsqrt_6: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    sub_10: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_25, getitem_33);  getitem_33 = None
    mul_22: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_6);  sub_10 = None
    mul_23: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_22, primals_52)
    add_27: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_23, primals_53);  mul_23 = primals_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_66: "f32[512, 1024]" = torch.ops.aten.view.default(add_27, [512, 1024]);  add_27 = None
    permute_33: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_54, [1, 0]);  primals_54 = None
    addmm_18: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_55, view_66, permute_33);  primals_55 = None
    view_67: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_18, [1, 512, 1024]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_34: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
    addmm_19: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_57, view_66, permute_34);  primals_57 = None
    view_69: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_19, [1, 512, 1024]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_70: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_69, [1, 512, 16, 64]);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_35: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_36: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
    addmm_20: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_59, view_66, permute_36);  primals_59 = None
    view_72: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_20, [1, 512, 1024]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_73: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_72, [1, 512, 16, 64]);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_37: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_74: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_67, [1, 512, 16, 64]);  view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_38: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
    
    # No stacktrace found for following nodes
    clone_default_80: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    clone_default_81: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
    clone_default_82: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    mul_scalar_80: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_80, 0.3535533905932738);  clone_default_80 = None
    permute_default_120: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(clone_default_81, [0, 1, 3, 2]);  clone_default_81 = None
    mul_scalar_81: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_120, 0.3535533905932738);  permute_default_120 = None
    expand_default_80: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_80, [1, 16, 512, 64]);  mul_scalar_80 = None
    view_default_240: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_80, [16, 512, 64]);  expand_default_80 = None
    expand_default_81: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_81, [1, 16, 64, 512]);  mul_scalar_81 = None
    view_default_241: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_default_81, [16, 64, 512]);  expand_default_81 = None
    bmm_default_120: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_240, view_default_241)
    view_default_242: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_120, [1, 16, 512, 512]);  bmm_default_120 = None
    amax_default_20: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(view_default_242, [-1], True)
    sub_tensor_40: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_242, amax_default_20);  view_default_242 = amax_default_20 = None
    exp_default_20: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_40);  sub_tensor_40 = None
    sum_dim_int_list_40: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_20, [-1], True)
    div_tensor_20: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_20, sum_dim_int_list_40);  exp_default_20 = sum_dim_int_list_40 = None
    alias_default_40: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_tensor_20)
    native_dropout_default_20 = torch.ops.aten.native_dropout.default(div_tensor_20, 0.1, True);  div_tensor_20 = None
    getitem_286: "f32[1, 16, 512, 512]" = native_dropout_default_20[0]
    getitem_287: "b8[1, 16, 512, 512]" = native_dropout_default_20[1];  native_dropout_default_20 = None
    expand_default_82: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_286, [1, 16, 512, 512]);  getitem_286 = None
    view_default_243: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_default_82, [16, 512, 512]);  expand_default_82 = None
    expand_default_83: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(clone_default_82, [1, 16, 512, 64]);  clone_default_82 = None
    view_default_244: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_83, [16, 512, 64]);  expand_default_83 = None
    bmm_default_121: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_243, view_default_244)
    view_default_245: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_121, [1, 16, 512, 64]);  bmm_default_121 = None
    permute_default_121: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_default_243, [0, 2, 1]);  view_default_243 = None
    permute_default_122: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_244, [0, 2, 1]);  view_default_244 = None
    alias_default_41: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_default_40);  alias_default_40 = None
    permute_default_123: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_240, [0, 2, 1]);  view_default_240 = None
    permute_default_124: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_default_241, [0, 2, 1]);  view_default_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_40: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_245, [0, 2, 1, 3]);  view_default_245 = None
    clone_3: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_81: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_3, [1, 512, 1024]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_82: "f32[512, 1024]" = torch.ops.aten.view.default(view_81, [512, 1024]);  view_81 = None
    permute_41: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
    addmm_21: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_61, view_82, permute_41);  primals_61 = None
    view_83: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_21, [1, 512, 1024]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_11 = torch.ops.aten.native_dropout.default(view_83, 0.1, True);  view_83 = None
    getitem_36: "f32[1, 512, 1024]" = native_dropout_11[0]
    getitem_37: "b8[1, 512, 1024]" = native_dropout_11[1];  native_dropout_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_29: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_25, getitem_36);  add_25 = getitem_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 512, 1]" = var_mean_7[0]
    getitem_39: "f32[1, 512, 1]" = var_mean_7[1];  var_mean_7 = None
    add_30: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
    rsqrt_7: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
    sub_12: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_29, getitem_39);  getitem_39 = None
    mul_24: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_7);  sub_12 = None
    mul_25: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_24, primals_62)
    add_31: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_25, primals_63);  mul_25 = primals_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_84: "f32[512, 1024]" = torch.ops.aten.view.default(add_31, [512, 1024]);  add_31 = None
    permute_42: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
    addmm_22: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_65, view_84, permute_42);  primals_65 = None
    view_85: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_22, [1, 512, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_26: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_85, 0.5)
    mul_27: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476);  view_85 = None
    erf_3: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
    add_32: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_28: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_26, add_32);  mul_26 = add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_86: "f32[512, 4096]" = torch.ops.aten.view.default(mul_28, [512, 4096]);  mul_28 = None
    permute_43: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
    addmm_23: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_67, view_86, permute_43);  primals_67 = None
    view_87: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_23, [1, 512, 1024]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_12 = torch.ops.aten.native_dropout.default(view_87, 0.1, True);  view_87 = None
    getitem_40: "f32[1, 512, 1024]" = native_dropout_12[0]
    getitem_41: "b8[1, 512, 1024]" = native_dropout_12[1];  native_dropout_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_33: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_29, getitem_40);  add_29 = getitem_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 512, 1]" = var_mean_8[0]
    getitem_43: "f32[1, 512, 1]" = var_mean_8[1];  var_mean_8 = None
    add_34: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
    rsqrt_8: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
    sub_13: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_33, getitem_43);  getitem_43 = None
    mul_29: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_8);  sub_13 = None
    mul_30: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_29, primals_68)
    add_35: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_30, primals_69);  mul_30 = primals_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_88: "f32[512, 1024]" = torch.ops.aten.view.default(add_35, [512, 1024]);  add_35 = None
    permute_44: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
    addmm_24: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_71, view_88, permute_44);  primals_71 = None
    view_89: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_24, [1, 512, 1024]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_45: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
    addmm_25: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_73, view_88, permute_45);  primals_73 = None
    view_91: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_25, [1, 512, 1024]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_92: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_91, [1, 512, 16, 64]);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_46: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_47: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
    addmm_26: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_75, view_88, permute_47);  primals_75 = None
    view_94: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_26, [1, 512, 1024]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_95: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_94, [1, 512, 16, 64]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_48: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_96: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_89, [1, 512, 16, 64]);  view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_49: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
    
    # No stacktrace found for following nodes
    clone_default_76: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    clone_default_77: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
    clone_default_78: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
    mul_scalar_76: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_76, 0.3535533905932738);  clone_default_76 = None
    permute_default_114: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(clone_default_77, [0, 1, 3, 2]);  clone_default_77 = None
    mul_scalar_77: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_114, 0.3535533905932738);  permute_default_114 = None
    expand_default_76: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_76, [1, 16, 512, 64]);  mul_scalar_76 = None
    view_default_228: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_76, [16, 512, 64]);  expand_default_76 = None
    expand_default_77: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_77, [1, 16, 64, 512]);  mul_scalar_77 = None
    view_default_229: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_default_77, [16, 64, 512]);  expand_default_77 = None
    bmm_default_114: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_228, view_default_229)
    view_default_230: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_114, [1, 16, 512, 512]);  bmm_default_114 = None
    amax_default_19: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(view_default_230, [-1], True)
    sub_tensor_38: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_230, amax_default_19);  view_default_230 = amax_default_19 = None
    exp_default_19: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_38);  sub_tensor_38 = None
    sum_dim_int_list_38: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_19, [-1], True)
    div_tensor_19: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_19, sum_dim_int_list_38);  exp_default_19 = sum_dim_int_list_38 = None
    alias_default_38: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_tensor_19)
    native_dropout_default_19 = torch.ops.aten.native_dropout.default(div_tensor_19, 0.1, True);  div_tensor_19 = None
    getitem_284: "f32[1, 16, 512, 512]" = native_dropout_default_19[0]
    getitem_285: "b8[1, 16, 512, 512]" = native_dropout_default_19[1];  native_dropout_default_19 = None
    expand_default_78: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_284, [1, 16, 512, 512]);  getitem_284 = None
    view_default_231: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_default_78, [16, 512, 512]);  expand_default_78 = None
    expand_default_79: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(clone_default_78, [1, 16, 512, 64]);  clone_default_78 = None
    view_default_232: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_79, [16, 512, 64]);  expand_default_79 = None
    bmm_default_115: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_231, view_default_232)
    view_default_233: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_115, [1, 16, 512, 64]);  bmm_default_115 = None
    permute_default_115: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_default_231, [0, 2, 1]);  view_default_231 = None
    permute_default_116: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_232, [0, 2, 1]);  view_default_232 = None
    alias_default_39: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_default_38);  alias_default_38 = None
    permute_default_117: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_228, [0, 2, 1]);  view_default_228 = None
    permute_default_118: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_default_229, [0, 2, 1]);  view_default_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_51: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_233, [0, 2, 1, 3]);  view_default_233 = None
    clone_4: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_103: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_4, [1, 512, 1024]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_104: "f32[512, 1024]" = torch.ops.aten.view.default(view_103, [512, 1024]);  view_103 = None
    permute_52: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
    addmm_27: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_77, view_104, permute_52);  primals_77 = None
    view_105: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_27, [1, 512, 1024]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_14 = torch.ops.aten.native_dropout.default(view_105, 0.1, True);  view_105 = None
    getitem_46: "f32[1, 512, 1024]" = native_dropout_14[0]
    getitem_47: "b8[1, 512, 1024]" = native_dropout_14[1];  native_dropout_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_37: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_33, getitem_46);  add_33 = getitem_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_9 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 512, 1]" = var_mean_9[0]
    getitem_49: "f32[1, 512, 1]" = var_mean_9[1];  var_mean_9 = None
    add_38: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
    rsqrt_9: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_15: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_37, getitem_49);  getitem_49 = None
    mul_31: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_9);  sub_15 = None
    mul_32: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_31, primals_78)
    add_39: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_32, primals_79);  mul_32 = primals_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_106: "f32[512, 1024]" = torch.ops.aten.view.default(add_39, [512, 1024]);  add_39 = None
    permute_53: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
    addmm_28: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_81, view_106, permute_53);  primals_81 = None
    view_107: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_28, [1, 512, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_33: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_107, 0.5)
    mul_34: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476);  view_107 = None
    erf_4: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_34);  mul_34 = None
    add_40: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_35: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_33, add_40);  mul_33 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_108: "f32[512, 4096]" = torch.ops.aten.view.default(mul_35, [512, 4096]);  mul_35 = None
    permute_54: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
    addmm_29: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_83, view_108, permute_54);  primals_83 = None
    view_109: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_29, [1, 512, 1024]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_15 = torch.ops.aten.native_dropout.default(view_109, 0.1, True);  view_109 = None
    getitem_50: "f32[1, 512, 1024]" = native_dropout_15[0]
    getitem_51: "b8[1, 512, 1024]" = native_dropout_15[1];  native_dropout_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_41: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_37, getitem_50);  add_37 = getitem_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
    getitem_52: "f32[1, 512, 1]" = var_mean_10[0]
    getitem_53: "f32[1, 512, 1]" = var_mean_10[1];  var_mean_10 = None
    add_42: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-12);  getitem_52 = None
    rsqrt_10: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_16: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_41, getitem_53);  getitem_53 = None
    mul_36: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = None
    mul_37: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_36, primals_84)
    add_43: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_37, primals_85);  mul_37 = primals_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_110: "f32[512, 1024]" = torch.ops.aten.view.default(add_43, [512, 1024]);  add_43 = None
    permute_55: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
    addmm_30: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_87, view_110, permute_55);  primals_87 = None
    view_111: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_30, [1, 512, 1024]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_56: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
    addmm_31: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_89, view_110, permute_56);  primals_89 = None
    view_113: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_31, [1, 512, 1024]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_114: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_113, [1, 512, 16, 64]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_57: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_58: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
    addmm_32: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_91, view_110, permute_58);  primals_91 = None
    view_116: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_32, [1, 512, 1024]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_117: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_116, [1, 512, 16, 64]);  view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_59: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_118: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_111, [1, 512, 16, 64]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_60: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
    
    # No stacktrace found for following nodes
    clone_default_72: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
    clone_default_73: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
    clone_default_74: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    mul_scalar_72: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_72, 0.3535533905932738);  clone_default_72 = None
    permute_default_108: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(clone_default_73, [0, 1, 3, 2]);  clone_default_73 = None
    mul_scalar_73: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_108, 0.3535533905932738);  permute_default_108 = None
    expand_default_72: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_72, [1, 16, 512, 64]);  mul_scalar_72 = None
    view_default_216: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_72, [16, 512, 64]);  expand_default_72 = None
    expand_default_73: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_73, [1, 16, 64, 512]);  mul_scalar_73 = None
    view_default_217: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_default_73, [16, 64, 512]);  expand_default_73 = None
    bmm_default_108: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_216, view_default_217)
    view_default_218: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_108, [1, 16, 512, 512]);  bmm_default_108 = None
    amax_default_18: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(view_default_218, [-1], True)
    sub_tensor_36: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_218, amax_default_18);  view_default_218 = amax_default_18 = None
    exp_default_18: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_36);  sub_tensor_36 = None
    sum_dim_int_list_36: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_18, [-1], True)
    div_tensor_18: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_18, sum_dim_int_list_36);  exp_default_18 = sum_dim_int_list_36 = None
    alias_default_36: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_tensor_18)
    native_dropout_default_18 = torch.ops.aten.native_dropout.default(div_tensor_18, 0.1, True);  div_tensor_18 = None
    getitem_282: "f32[1, 16, 512, 512]" = native_dropout_default_18[0]
    getitem_283: "b8[1, 16, 512, 512]" = native_dropout_default_18[1];  native_dropout_default_18 = None
    expand_default_74: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_282, [1, 16, 512, 512]);  getitem_282 = None
    view_default_219: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_default_74, [16, 512, 512]);  expand_default_74 = None
    expand_default_75: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(clone_default_74, [1, 16, 512, 64]);  clone_default_74 = None
    view_default_220: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_75, [16, 512, 64]);  expand_default_75 = None
    bmm_default_109: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_219, view_default_220)
    view_default_221: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_109, [1, 16, 512, 64]);  bmm_default_109 = None
    permute_default_109: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_default_219, [0, 2, 1]);  view_default_219 = None
    permute_default_110: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_220, [0, 2, 1]);  view_default_220 = None
    alias_default_37: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_default_36);  alias_default_36 = None
    permute_default_111: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_216, [0, 2, 1]);  view_default_216 = None
    permute_default_112: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_default_217, [0, 2, 1]);  view_default_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_62: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_221, [0, 2, 1, 3]);  view_default_221 = None
    clone_5: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_125: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_5, [1, 512, 1024]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_126: "f32[512, 1024]" = torch.ops.aten.view.default(view_125, [512, 1024]);  view_125 = None
    permute_63: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
    addmm_33: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_93, view_126, permute_63);  primals_93 = None
    view_127: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_33, [1, 512, 1024]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_17 = torch.ops.aten.native_dropout.default(view_127, 0.1, True);  view_127 = None
    getitem_56: "f32[1, 512, 1024]" = native_dropout_17[0]
    getitem_57: "b8[1, 512, 1024]" = native_dropout_17[1];  native_dropout_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_45: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_41, getitem_56);  add_41 = getitem_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
    getitem_58: "f32[1, 512, 1]" = var_mean_11[0]
    getitem_59: "f32[1, 512, 1]" = var_mean_11[1];  var_mean_11 = None
    add_46: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-12);  getitem_58 = None
    rsqrt_11: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    sub_18: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_45, getitem_59);  getitem_59 = None
    mul_38: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_11);  sub_18 = None
    mul_39: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_38, primals_94)
    add_47: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_39, primals_95);  mul_39 = primals_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_128: "f32[512, 1024]" = torch.ops.aten.view.default(add_47, [512, 1024]);  add_47 = None
    permute_64: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
    addmm_34: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_97, view_128, permute_64);  primals_97 = None
    view_129: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_34, [1, 512, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_40: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_129, 0.5)
    mul_41: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476);  view_129 = None
    erf_5: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
    add_48: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_42: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_40, add_48);  mul_40 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_130: "f32[512, 4096]" = torch.ops.aten.view.default(mul_42, [512, 4096]);  mul_42 = None
    permute_65: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
    addmm_35: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_99, view_130, permute_65);  primals_99 = None
    view_131: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_35, [1, 512, 1024]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_18 = torch.ops.aten.native_dropout.default(view_131, 0.1, True);  view_131 = None
    getitem_60: "f32[1, 512, 1024]" = native_dropout_18[0]
    getitem_61: "b8[1, 512, 1024]" = native_dropout_18[1];  native_dropout_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_49: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_45, getitem_60);  add_45 = getitem_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
    getitem_62: "f32[1, 512, 1]" = var_mean_12[0]
    getitem_63: "f32[1, 512, 1]" = var_mean_12[1];  var_mean_12 = None
    add_50: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-12);  getitem_62 = None
    rsqrt_12: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    sub_19: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_49, getitem_63);  getitem_63 = None
    mul_43: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_12);  sub_19 = None
    mul_44: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_43, primals_100)
    add_51: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_44, primals_101);  mul_44 = primals_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_132: "f32[512, 1024]" = torch.ops.aten.view.default(add_51, [512, 1024]);  add_51 = None
    permute_66: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_102, [1, 0]);  primals_102 = None
    addmm_36: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_103, view_132, permute_66);  primals_103 = None
    view_133: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_36, [1, 512, 1024]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_67: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_104, [1, 0]);  primals_104 = None
    addmm_37: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_105, view_132, permute_67);  primals_105 = None
    view_135: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_37, [1, 512, 1024]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_136: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_135, [1, 512, 16, 64]);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_68: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_69: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
    addmm_38: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_107, view_132, permute_69);  primals_107 = None
    view_138: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_38, [1, 512, 1024]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_139: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_138, [1, 512, 16, 64]);  view_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_70: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_139, [0, 2, 1, 3]);  view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_140: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_133, [1, 512, 16, 64]);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_71: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
    
    # No stacktrace found for following nodes
    clone_default_68: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
    clone_default_69: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
    clone_default_70: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
    mul_scalar_68: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_68, 0.3535533905932738);  clone_default_68 = None
    permute_default_102: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(clone_default_69, [0, 1, 3, 2]);  clone_default_69 = None
    mul_scalar_69: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_102, 0.3535533905932738);  permute_default_102 = None
    expand_default_68: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_68, [1, 16, 512, 64]);  mul_scalar_68 = None
    view_default_204: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_68, [16, 512, 64]);  expand_default_68 = None
    expand_default_69: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_69, [1, 16, 64, 512]);  mul_scalar_69 = None
    view_default_205: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_default_69, [16, 64, 512]);  expand_default_69 = None
    bmm_default_102: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_204, view_default_205)
    view_default_206: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_102, [1, 16, 512, 512]);  bmm_default_102 = None
    amax_default_17: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(view_default_206, [-1], True)
    sub_tensor_34: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_206, amax_default_17);  view_default_206 = amax_default_17 = None
    exp_default_17: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_34);  sub_tensor_34 = None
    sum_dim_int_list_34: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_17, [-1], True)
    div_tensor_17: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_17, sum_dim_int_list_34);  exp_default_17 = sum_dim_int_list_34 = None
    alias_default_34: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_tensor_17)
    native_dropout_default_17 = torch.ops.aten.native_dropout.default(div_tensor_17, 0.1, True);  div_tensor_17 = None
    getitem_280: "f32[1, 16, 512, 512]" = native_dropout_default_17[0]
    getitem_281: "b8[1, 16, 512, 512]" = native_dropout_default_17[1];  native_dropout_default_17 = None
    expand_default_70: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_280, [1, 16, 512, 512]);  getitem_280 = None
    view_default_207: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_default_70, [16, 512, 512]);  expand_default_70 = None
    expand_default_71: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(clone_default_70, [1, 16, 512, 64]);  clone_default_70 = None
    view_default_208: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_71, [16, 512, 64]);  expand_default_71 = None
    bmm_default_103: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_207, view_default_208)
    view_default_209: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_103, [1, 16, 512, 64]);  bmm_default_103 = None
    permute_default_103: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_default_207, [0, 2, 1]);  view_default_207 = None
    permute_default_104: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_208, [0, 2, 1]);  view_default_208 = None
    alias_default_35: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_default_34);  alias_default_34 = None
    permute_default_105: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_204, [0, 2, 1]);  view_default_204 = None
    permute_default_106: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_default_205, [0, 2, 1]);  view_default_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_73: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_209, [0, 2, 1, 3]);  view_default_209 = None
    clone_6: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_147: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_6, [1, 512, 1024]);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_148: "f32[512, 1024]" = torch.ops.aten.view.default(view_147, [512, 1024]);  view_147 = None
    permute_74: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
    addmm_39: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_109, view_148, permute_74);  primals_109 = None
    view_149: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_39, [1, 512, 1024]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_20 = torch.ops.aten.native_dropout.default(view_149, 0.1, True);  view_149 = None
    getitem_66: "f32[1, 512, 1024]" = native_dropout_20[0]
    getitem_67: "b8[1, 512, 1024]" = native_dropout_20[1];  native_dropout_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_53: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_49, getitem_66);  add_49 = getitem_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_13 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
    getitem_68: "f32[1, 512, 1]" = var_mean_13[0]
    getitem_69: "f32[1, 512, 1]" = var_mean_13[1];  var_mean_13 = None
    add_54: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-12);  getitem_68 = None
    rsqrt_13: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
    sub_21: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_53, getitem_69);  getitem_69 = None
    mul_45: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_13);  sub_21 = None
    mul_46: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_45, primals_110)
    add_55: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_46, primals_111);  mul_46 = primals_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_150: "f32[512, 1024]" = torch.ops.aten.view.default(add_55, [512, 1024]);  add_55 = None
    permute_75: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_112, [1, 0]);  primals_112 = None
    addmm_40: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_113, view_150, permute_75);  primals_113 = None
    view_151: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_40, [1, 512, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_47: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_151, 0.5)
    mul_48: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476);  view_151 = None
    erf_6: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
    add_56: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_49: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_47, add_56);  mul_47 = add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_152: "f32[512, 4096]" = torch.ops.aten.view.default(mul_49, [512, 4096]);  mul_49 = None
    permute_76: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
    addmm_41: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_115, view_152, permute_76);  primals_115 = None
    view_153: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_41, [1, 512, 1024]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_21 = torch.ops.aten.native_dropout.default(view_153, 0.1, True);  view_153 = None
    getitem_70: "f32[1, 512, 1024]" = native_dropout_21[0]
    getitem_71: "b8[1, 512, 1024]" = native_dropout_21[1];  native_dropout_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_57: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_53, getitem_70);  add_53 = getitem_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
    getitem_72: "f32[1, 512, 1]" = var_mean_14[0]
    getitem_73: "f32[1, 512, 1]" = var_mean_14[1];  var_mean_14 = None
    add_58: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-12);  getitem_72 = None
    rsqrt_14: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    sub_22: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_57, getitem_73);  getitem_73 = None
    mul_50: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_14);  sub_22 = None
    mul_51: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_50, primals_116)
    add_59: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_51, primals_117);  mul_51 = primals_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_154: "f32[512, 1024]" = torch.ops.aten.view.default(add_59, [512, 1024]);  add_59 = None
    permute_77: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
    addmm_42: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_119, view_154, permute_77);  primals_119 = None
    view_155: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_42, [1, 512, 1024]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_78: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
    addmm_43: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_121, view_154, permute_78);  primals_121 = None
    view_157: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_43, [1, 512, 1024]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_158: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_157, [1, 512, 16, 64]);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_79: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_158, [0, 2, 1, 3]);  view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_80: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
    addmm_44: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_123, view_154, permute_80);  primals_123 = None
    view_160: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_44, [1, 512, 1024]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_161: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_160, [1, 512, 16, 64]);  view_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_81: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_161, [0, 2, 1, 3]);  view_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_162: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_155, [1, 512, 16, 64]);  view_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_82: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
    
    # No stacktrace found for following nodes
    clone_default_64: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
    clone_default_65: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
    clone_default_66: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
    mul_scalar_64: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_64, 0.3535533905932738);  clone_default_64 = None
    permute_default_96: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(clone_default_65, [0, 1, 3, 2]);  clone_default_65 = None
    mul_scalar_65: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_96, 0.3535533905932738);  permute_default_96 = None
    expand_default_64: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_64, [1, 16, 512, 64]);  mul_scalar_64 = None
    view_default_192: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_64, [16, 512, 64]);  expand_default_64 = None
    expand_default_65: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_65, [1, 16, 64, 512]);  mul_scalar_65 = None
    view_default_193: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_default_65, [16, 64, 512]);  expand_default_65 = None
    bmm_default_96: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_192, view_default_193)
    view_default_194: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_96, [1, 16, 512, 512]);  bmm_default_96 = None
    amax_default_16: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(view_default_194, [-1], True)
    sub_tensor_32: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_194, amax_default_16);  view_default_194 = amax_default_16 = None
    exp_default_16: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_32);  sub_tensor_32 = None
    sum_dim_int_list_32: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_16, [-1], True)
    div_tensor_16: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_16, sum_dim_int_list_32);  exp_default_16 = sum_dim_int_list_32 = None
    alias_default_32: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_tensor_16)
    native_dropout_default_16 = torch.ops.aten.native_dropout.default(div_tensor_16, 0.1, True);  div_tensor_16 = None
    getitem_278: "f32[1, 16, 512, 512]" = native_dropout_default_16[0]
    getitem_279: "b8[1, 16, 512, 512]" = native_dropout_default_16[1];  native_dropout_default_16 = None
    expand_default_66: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_278, [1, 16, 512, 512]);  getitem_278 = None
    view_default_195: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_default_66, [16, 512, 512]);  expand_default_66 = None
    expand_default_67: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(clone_default_66, [1, 16, 512, 64]);  clone_default_66 = None
    view_default_196: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_67, [16, 512, 64]);  expand_default_67 = None
    bmm_default_97: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_195, view_default_196)
    view_default_197: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_97, [1, 16, 512, 64]);  bmm_default_97 = None
    permute_default_97: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_default_195, [0, 2, 1]);  view_default_195 = None
    permute_default_98: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_196, [0, 2, 1]);  view_default_196 = None
    alias_default_33: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_default_32);  alias_default_32 = None
    permute_default_99: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_192, [0, 2, 1]);  view_default_192 = None
    permute_default_100: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_default_193, [0, 2, 1]);  view_default_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_84: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_197, [0, 2, 1, 3]);  view_default_197 = None
    clone_7: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_169: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_7, [1, 512, 1024]);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_170: "f32[512, 1024]" = torch.ops.aten.view.default(view_169, [512, 1024]);  view_169 = None
    permute_85: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
    addmm_45: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_125, view_170, permute_85);  primals_125 = None
    view_171: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_45, [1, 512, 1024]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_23 = torch.ops.aten.native_dropout.default(view_171, 0.1, True);  view_171 = None
    getitem_76: "f32[1, 512, 1024]" = native_dropout_23[0]
    getitem_77: "b8[1, 512, 1024]" = native_dropout_23[1];  native_dropout_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_61: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_57, getitem_76);  add_57 = getitem_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_15 = torch.ops.aten.var_mean.correction(add_61, [2], correction = 0, keepdim = True)
    getitem_78: "f32[1, 512, 1]" = var_mean_15[0]
    getitem_79: "f32[1, 512, 1]" = var_mean_15[1];  var_mean_15 = None
    add_62: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-12);  getitem_78 = None
    rsqrt_15: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    sub_24: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_61, getitem_79);  getitem_79 = None
    mul_52: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_15);  sub_24 = None
    mul_53: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_52, primals_126)
    add_63: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_53, primals_127);  mul_53 = primals_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_172: "f32[512, 1024]" = torch.ops.aten.view.default(add_63, [512, 1024]);  add_63 = None
    permute_86: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
    addmm_46: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_129, view_172, permute_86);  primals_129 = None
    view_173: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_46, [1, 512, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_54: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_173, 0.5)
    mul_55: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476);  view_173 = None
    erf_7: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
    add_64: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_56: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_54, add_64);  mul_54 = add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_174: "f32[512, 4096]" = torch.ops.aten.view.default(mul_56, [512, 4096]);  mul_56 = None
    permute_87: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_130, [1, 0]);  primals_130 = None
    addmm_47: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_131, view_174, permute_87);  primals_131 = None
    view_175: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_47, [1, 512, 1024]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_24 = torch.ops.aten.native_dropout.default(view_175, 0.1, True);  view_175 = None
    getitem_80: "f32[1, 512, 1024]" = native_dropout_24[0]
    getitem_81: "b8[1, 512, 1024]" = native_dropout_24[1];  native_dropout_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_65: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_61, getitem_80);  add_61 = getitem_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_16 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
    getitem_82: "f32[1, 512, 1]" = var_mean_16[0]
    getitem_83: "f32[1, 512, 1]" = var_mean_16[1];  var_mean_16 = None
    add_66: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-12);  getitem_82 = None
    rsqrt_16: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    sub_25: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_65, getitem_83);  getitem_83 = None
    mul_57: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_16);  sub_25 = None
    mul_58: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_57, primals_132)
    add_67: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_58, primals_133);  mul_58 = primals_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_176: "f32[512, 1024]" = torch.ops.aten.view.default(add_67, [512, 1024]);  add_67 = None
    permute_88: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_134, [1, 0]);  primals_134 = None
    addmm_48: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_135, view_176, permute_88);  primals_135 = None
    view_177: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_48, [1, 512, 1024]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_89: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
    addmm_49: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_137, view_176, permute_89);  primals_137 = None
    view_179: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_49, [1, 512, 1024]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_180: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_179, [1, 512, 16, 64]);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_90: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_180, [0, 2, 1, 3]);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_91: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
    addmm_50: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_139, view_176, permute_91);  primals_139 = None
    view_182: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_50, [1, 512, 1024]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_183: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_182, [1, 512, 16, 64]);  view_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_92: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_184: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_177, [1, 512, 16, 64]);  view_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_93: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
    
    # No stacktrace found for following nodes
    clone_default_60: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
    clone_default_61: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
    clone_default_62: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
    mul_scalar_60: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_60, 0.3535533905932738);  clone_default_60 = None
    permute_default_90: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(clone_default_61, [0, 1, 3, 2]);  clone_default_61 = None
    mul_scalar_61: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_90, 0.3535533905932738);  permute_default_90 = None
    expand_default_60: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_60, [1, 16, 512, 64]);  mul_scalar_60 = None
    view_default_180: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_60, [16, 512, 64]);  expand_default_60 = None
    expand_default_61: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_61, [1, 16, 64, 512]);  mul_scalar_61 = None
    view_default_181: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_default_61, [16, 64, 512]);  expand_default_61 = None
    bmm_default_90: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_180, view_default_181)
    view_default_182: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_90, [1, 16, 512, 512]);  bmm_default_90 = None
    amax_default_15: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(view_default_182, [-1], True)
    sub_tensor_30: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_182, amax_default_15);  view_default_182 = amax_default_15 = None
    exp_default_15: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_30);  sub_tensor_30 = None
    sum_dim_int_list_30: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_15, [-1], True)
    div_tensor_15: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_15, sum_dim_int_list_30);  exp_default_15 = sum_dim_int_list_30 = None
    alias_default_30: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_tensor_15)
    native_dropout_default_15 = torch.ops.aten.native_dropout.default(div_tensor_15, 0.1, True);  div_tensor_15 = None
    getitem_276: "f32[1, 16, 512, 512]" = native_dropout_default_15[0]
    getitem_277: "b8[1, 16, 512, 512]" = native_dropout_default_15[1];  native_dropout_default_15 = None
    expand_default_62: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_276, [1, 16, 512, 512]);  getitem_276 = None
    view_default_183: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_default_62, [16, 512, 512]);  expand_default_62 = None
    expand_default_63: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(clone_default_62, [1, 16, 512, 64]);  clone_default_62 = None
    view_default_184: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_63, [16, 512, 64]);  expand_default_63 = None
    bmm_default_91: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_183, view_default_184)
    view_default_185: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_91, [1, 16, 512, 64]);  bmm_default_91 = None
    permute_default_91: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_default_183, [0, 2, 1]);  view_default_183 = None
    permute_default_92: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_184, [0, 2, 1]);  view_default_184 = None
    alias_default_31: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_default_30);  alias_default_30 = None
    permute_default_93: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_180, [0, 2, 1]);  view_default_180 = None
    permute_default_94: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_default_181, [0, 2, 1]);  view_default_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_95: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_185, [0, 2, 1, 3]);  view_default_185 = None
    clone_8: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_191: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_8, [1, 512, 1024]);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_192: "f32[512, 1024]" = torch.ops.aten.view.default(view_191, [512, 1024]);  view_191 = None
    permute_96: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_140, [1, 0]);  primals_140 = None
    addmm_51: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_141, view_192, permute_96);  primals_141 = None
    view_193: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_51, [1, 512, 1024]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_26 = torch.ops.aten.native_dropout.default(view_193, 0.1, True);  view_193 = None
    getitem_86: "f32[1, 512, 1024]" = native_dropout_26[0]
    getitem_87: "b8[1, 512, 1024]" = native_dropout_26[1];  native_dropout_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_69: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_65, getitem_86);  add_65 = getitem_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_17 = torch.ops.aten.var_mean.correction(add_69, [2], correction = 0, keepdim = True)
    getitem_88: "f32[1, 512, 1]" = var_mean_17[0]
    getitem_89: "f32[1, 512, 1]" = var_mean_17[1];  var_mean_17 = None
    add_70: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-12);  getitem_88 = None
    rsqrt_17: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    sub_27: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_69, getitem_89);  getitem_89 = None
    mul_59: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_17);  sub_27 = None
    mul_60: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_59, primals_142)
    add_71: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_60, primals_143);  mul_60 = primals_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_194: "f32[512, 1024]" = torch.ops.aten.view.default(add_71, [512, 1024]);  add_71 = None
    permute_97: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
    addmm_52: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_145, view_194, permute_97);  primals_145 = None
    view_195: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_52, [1, 512, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_61: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_195, 0.5)
    mul_62: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476);  view_195 = None
    erf_8: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_62);  mul_62 = None
    add_72: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_63: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_61, add_72);  mul_61 = add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_196: "f32[512, 4096]" = torch.ops.aten.view.default(mul_63, [512, 4096]);  mul_63 = None
    permute_98: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_146, [1, 0]);  primals_146 = None
    addmm_53: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_147, view_196, permute_98);  primals_147 = None
    view_197: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_53, [1, 512, 1024]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_27 = torch.ops.aten.native_dropout.default(view_197, 0.1, True);  view_197 = None
    getitem_90: "f32[1, 512, 1024]" = native_dropout_27[0]
    getitem_91: "b8[1, 512, 1024]" = native_dropout_27[1];  native_dropout_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_73: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_69, getitem_90);  add_69 = getitem_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_18 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
    getitem_92: "f32[1, 512, 1]" = var_mean_18[0]
    getitem_93: "f32[1, 512, 1]" = var_mean_18[1];  var_mean_18 = None
    add_74: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-12);  getitem_92 = None
    rsqrt_18: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_28: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_73, getitem_93);  getitem_93 = None
    mul_64: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_18);  sub_28 = None
    mul_65: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_64, primals_148)
    add_75: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_65, primals_149);  mul_65 = primals_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_198: "f32[512, 1024]" = torch.ops.aten.view.default(add_75, [512, 1024]);  add_75 = None
    permute_99: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
    addmm_54: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_151, view_198, permute_99);  primals_151 = None
    view_199: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_54, [1, 512, 1024]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_100: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
    addmm_55: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_153, view_198, permute_100);  primals_153 = None
    view_201: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_55, [1, 512, 1024]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_202: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_201, [1, 512, 16, 64]);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_101: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_202, [0, 2, 1, 3]);  view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_102: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
    addmm_56: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_155, view_198, permute_102);  primals_155 = None
    view_204: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_56, [1, 512, 1024]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_205: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_204, [1, 512, 16, 64]);  view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_103: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_206: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_199, [1, 512, 16, 64]);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_104: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
    
    # No stacktrace found for following nodes
    clone_default_56: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
    clone_default_57: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_101, memory_format = torch.contiguous_format);  permute_101 = None
    clone_default_58: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_103, memory_format = torch.contiguous_format);  permute_103 = None
    mul_scalar_56: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_56, 0.3535533905932738);  clone_default_56 = None
    permute_default_84: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(clone_default_57, [0, 1, 3, 2]);  clone_default_57 = None
    mul_scalar_57: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_84, 0.3535533905932738);  permute_default_84 = None
    expand_default_56: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_56, [1, 16, 512, 64]);  mul_scalar_56 = None
    view_default_168: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_56, [16, 512, 64]);  expand_default_56 = None
    expand_default_57: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_57, [1, 16, 64, 512]);  mul_scalar_57 = None
    view_default_169: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_default_57, [16, 64, 512]);  expand_default_57 = None
    bmm_default_84: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_168, view_default_169)
    view_default_170: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_84, [1, 16, 512, 512]);  bmm_default_84 = None
    amax_default_14: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(view_default_170, [-1], True)
    sub_tensor_28: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_170, amax_default_14);  view_default_170 = amax_default_14 = None
    exp_default_14: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_28);  sub_tensor_28 = None
    sum_dim_int_list_28: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_14, [-1], True)
    div_tensor_14: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_14, sum_dim_int_list_28);  exp_default_14 = sum_dim_int_list_28 = None
    alias_default_28: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_tensor_14)
    native_dropout_default_14 = torch.ops.aten.native_dropout.default(div_tensor_14, 0.1, True);  div_tensor_14 = None
    getitem_274: "f32[1, 16, 512, 512]" = native_dropout_default_14[0]
    getitem_275: "b8[1, 16, 512, 512]" = native_dropout_default_14[1];  native_dropout_default_14 = None
    expand_default_58: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_274, [1, 16, 512, 512]);  getitem_274 = None
    view_default_171: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_default_58, [16, 512, 512]);  expand_default_58 = None
    expand_default_59: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(clone_default_58, [1, 16, 512, 64]);  clone_default_58 = None
    view_default_172: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_59, [16, 512, 64]);  expand_default_59 = None
    bmm_default_85: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_171, view_default_172)
    view_default_173: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_85, [1, 16, 512, 64]);  bmm_default_85 = None
    permute_default_85: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_default_171, [0, 2, 1]);  view_default_171 = None
    permute_default_86: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_172, [0, 2, 1]);  view_default_172 = None
    alias_default_29: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_default_28);  alias_default_28 = None
    permute_default_87: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_168, [0, 2, 1]);  view_default_168 = None
    permute_default_88: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_default_169, [0, 2, 1]);  view_default_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_106: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_173, [0, 2, 1, 3]);  view_default_173 = None
    clone_9: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_213: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_9, [1, 512, 1024]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_214: "f32[512, 1024]" = torch.ops.aten.view.default(view_213, [512, 1024]);  view_213 = None
    permute_107: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
    addmm_57: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_157, view_214, permute_107);  primals_157 = None
    view_215: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_57, [1, 512, 1024]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_29 = torch.ops.aten.native_dropout.default(view_215, 0.1, True);  view_215 = None
    getitem_96: "f32[1, 512, 1024]" = native_dropout_29[0]
    getitem_97: "b8[1, 512, 1024]" = native_dropout_29[1];  native_dropout_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_77: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_73, getitem_96);  add_73 = getitem_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_19 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
    getitem_98: "f32[1, 512, 1]" = var_mean_19[0]
    getitem_99: "f32[1, 512, 1]" = var_mean_19[1];  var_mean_19 = None
    add_78: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-12);  getitem_98 = None
    rsqrt_19: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    sub_30: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_77, getitem_99);  getitem_99 = None
    mul_66: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_19);  sub_30 = None
    mul_67: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_66, primals_158)
    add_79: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_67, primals_159);  mul_67 = primals_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_216: "f32[512, 1024]" = torch.ops.aten.view.default(add_79, [512, 1024]);  add_79 = None
    permute_108: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None
    addmm_58: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_161, view_216, permute_108);  primals_161 = None
    view_217: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_58, [1, 512, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_68: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_217, 0.5)
    mul_69: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476);  view_217 = None
    erf_9: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
    add_80: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_70: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_68, add_80);  mul_68 = add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_218: "f32[512, 4096]" = torch.ops.aten.view.default(mul_70, [512, 4096]);  mul_70 = None
    permute_109: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
    addmm_59: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_163, view_218, permute_109);  primals_163 = None
    view_219: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_59, [1, 512, 1024]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_30 = torch.ops.aten.native_dropout.default(view_219, 0.1, True);  view_219 = None
    getitem_100: "f32[1, 512, 1024]" = native_dropout_30[0]
    getitem_101: "b8[1, 512, 1024]" = native_dropout_30[1];  native_dropout_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_81: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_77, getitem_100);  add_77 = getitem_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
    getitem_102: "f32[1, 512, 1]" = var_mean_20[0]
    getitem_103: "f32[1, 512, 1]" = var_mean_20[1];  var_mean_20 = None
    add_82: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-12);  getitem_102 = None
    rsqrt_20: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
    sub_31: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_81, getitem_103);  getitem_103 = None
    mul_71: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_20);  sub_31 = None
    mul_72: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_71, primals_164)
    add_83: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_72, primals_165);  mul_72 = primals_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_220: "f32[512, 1024]" = torch.ops.aten.view.default(add_83, [512, 1024]);  add_83 = None
    permute_110: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_166, [1, 0]);  primals_166 = None
    addmm_60: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_167, view_220, permute_110);  primals_167 = None
    view_221: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_60, [1, 512, 1024]);  addmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_111: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_168, [1, 0]);  primals_168 = None
    addmm_61: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_169, view_220, permute_111);  primals_169 = None
    view_223: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_61, [1, 512, 1024]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_224: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_223, [1, 512, 16, 64]);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_112: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_113: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
    addmm_62: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_171, view_220, permute_113);  primals_171 = None
    view_226: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_62, [1, 512, 1024]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_227: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_226, [1, 512, 16, 64]);  view_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_114: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_228: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_221, [1, 512, 16, 64]);  view_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_115: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None
    
    # No stacktrace found for following nodes
    clone_default_52: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
    clone_default_53: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
    clone_default_54: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
    mul_scalar_52: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_52, 0.3535533905932738);  clone_default_52 = None
    permute_default_78: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(clone_default_53, [0, 1, 3, 2]);  clone_default_53 = None
    mul_scalar_53: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_78, 0.3535533905932738);  permute_default_78 = None
    expand_default_52: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_52, [1, 16, 512, 64]);  mul_scalar_52 = None
    view_default_156: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_52, [16, 512, 64]);  expand_default_52 = None
    expand_default_53: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_53, [1, 16, 64, 512]);  mul_scalar_53 = None
    view_default_157: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_default_53, [16, 64, 512]);  expand_default_53 = None
    bmm_default_78: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_156, view_default_157)
    view_default_158: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_78, [1, 16, 512, 512]);  bmm_default_78 = None
    amax_default_13: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(view_default_158, [-1], True)
    sub_tensor_26: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_158, amax_default_13);  view_default_158 = amax_default_13 = None
    exp_default_13: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_26);  sub_tensor_26 = None
    sum_dim_int_list_26: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_13, [-1], True)
    div_tensor_13: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_13, sum_dim_int_list_26);  exp_default_13 = sum_dim_int_list_26 = None
    alias_default_26: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_tensor_13)
    native_dropout_default_13 = torch.ops.aten.native_dropout.default(div_tensor_13, 0.1, True);  div_tensor_13 = None
    getitem_272: "f32[1, 16, 512, 512]" = native_dropout_default_13[0]
    getitem_273: "b8[1, 16, 512, 512]" = native_dropout_default_13[1];  native_dropout_default_13 = None
    expand_default_54: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_272, [1, 16, 512, 512]);  getitem_272 = None
    view_default_159: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_default_54, [16, 512, 512]);  expand_default_54 = None
    expand_default_55: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(clone_default_54, [1, 16, 512, 64]);  clone_default_54 = None
    view_default_160: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_55, [16, 512, 64]);  expand_default_55 = None
    bmm_default_79: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_159, view_default_160)
    view_default_161: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_79, [1, 16, 512, 64]);  bmm_default_79 = None
    permute_default_79: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_default_159, [0, 2, 1]);  view_default_159 = None
    permute_default_80: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_160, [0, 2, 1]);  view_default_160 = None
    alias_default_27: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_default_26);  alias_default_26 = None
    permute_default_81: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_156, [0, 2, 1]);  view_default_156 = None
    permute_default_82: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_default_157, [0, 2, 1]);  view_default_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_117: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_161, [0, 2, 1, 3]);  view_default_161 = None
    clone_10: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_235: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_10, [1, 512, 1024]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_236: "f32[512, 1024]" = torch.ops.aten.view.default(view_235, [512, 1024]);  view_235 = None
    permute_118: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_172, [1, 0]);  primals_172 = None
    addmm_63: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_173, view_236, permute_118);  primals_173 = None
    view_237: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_63, [1, 512, 1024]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_32 = torch.ops.aten.native_dropout.default(view_237, 0.1, True);  view_237 = None
    getitem_106: "f32[1, 512, 1024]" = native_dropout_32[0]
    getitem_107: "b8[1, 512, 1024]" = native_dropout_32[1];  native_dropout_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_85: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_81, getitem_106);  add_81 = getitem_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
    getitem_108: "f32[1, 512, 1]" = var_mean_21[0]
    getitem_109: "f32[1, 512, 1]" = var_mean_21[1];  var_mean_21 = None
    add_86: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-12);  getitem_108 = None
    rsqrt_21: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
    sub_33: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_85, getitem_109);  getitem_109 = None
    mul_73: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_21);  sub_33 = None
    mul_74: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_73, primals_174)
    add_87: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_74, primals_175);  mul_74 = primals_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_238: "f32[512, 1024]" = torch.ops.aten.view.default(add_87, [512, 1024]);  add_87 = None
    permute_119: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_176, [1, 0]);  primals_176 = None
    addmm_64: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_177, view_238, permute_119);  primals_177 = None
    view_239: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_64, [1, 512, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_75: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_239, 0.5)
    mul_76: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476);  view_239 = None
    erf_10: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
    add_88: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_77: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_75, add_88);  mul_75 = add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_240: "f32[512, 4096]" = torch.ops.aten.view.default(mul_77, [512, 4096]);  mul_77 = None
    permute_120: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_178, [1, 0]);  primals_178 = None
    addmm_65: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_179, view_240, permute_120);  primals_179 = None
    view_241: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_65, [1, 512, 1024]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_33 = torch.ops.aten.native_dropout.default(view_241, 0.1, True);  view_241 = None
    getitem_110: "f32[1, 512, 1024]" = native_dropout_33[0]
    getitem_111: "b8[1, 512, 1024]" = native_dropout_33[1];  native_dropout_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_89: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_85, getitem_110);  add_85 = getitem_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
    getitem_112: "f32[1, 512, 1]" = var_mean_22[0]
    getitem_113: "f32[1, 512, 1]" = var_mean_22[1];  var_mean_22 = None
    add_90: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-12);  getitem_112 = None
    rsqrt_22: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
    sub_34: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_89, getitem_113);  getitem_113 = None
    mul_78: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = None
    mul_79: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_78, primals_180)
    add_91: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_79, primals_181);  mul_79 = primals_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_242: "f32[512, 1024]" = torch.ops.aten.view.default(add_91, [512, 1024]);  add_91 = None
    permute_121: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_182, [1, 0]);  primals_182 = None
    addmm_66: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_183, view_242, permute_121);  primals_183 = None
    view_243: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_66, [1, 512, 1024]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_122: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None
    addmm_67: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_185, view_242, permute_122);  primals_185 = None
    view_245: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_67, [1, 512, 1024]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_246: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_245, [1, 512, 16, 64]);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_123: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_124: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_186, [1, 0]);  primals_186 = None
    addmm_68: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_187, view_242, permute_124);  primals_187 = None
    view_248: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_68, [1, 512, 1024]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_249: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_248, [1, 512, 16, 64]);  view_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_125: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_250: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_243, [1, 512, 16, 64]);  view_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_126: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
    
    # No stacktrace found for following nodes
    clone_default_48: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
    clone_default_49: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_123, memory_format = torch.contiguous_format);  permute_123 = None
    clone_default_50: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
    mul_scalar_48: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_48, 0.3535533905932738);  clone_default_48 = None
    permute_default_72: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(clone_default_49, [0, 1, 3, 2]);  clone_default_49 = None
    mul_scalar_49: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_72, 0.3535533905932738);  permute_default_72 = None
    expand_default_48: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_48, [1, 16, 512, 64]);  mul_scalar_48 = None
    view_default_144: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_48, [16, 512, 64]);  expand_default_48 = None
    expand_default_49: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_49, [1, 16, 64, 512]);  mul_scalar_49 = None
    view_default_145: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_default_49, [16, 64, 512]);  expand_default_49 = None
    bmm_default_72: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_144, view_default_145)
    view_default_146: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_72, [1, 16, 512, 512]);  bmm_default_72 = None
    amax_default_12: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(view_default_146, [-1], True)
    sub_tensor_24: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_146, amax_default_12);  view_default_146 = amax_default_12 = None
    exp_default_12: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_24);  sub_tensor_24 = None
    sum_dim_int_list_24: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_12, [-1], True)
    div_tensor_12: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_12, sum_dim_int_list_24);  exp_default_12 = sum_dim_int_list_24 = None
    alias_default_24: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_tensor_12)
    native_dropout_default_12 = torch.ops.aten.native_dropout.default(div_tensor_12, 0.1, True);  div_tensor_12 = None
    getitem_270: "f32[1, 16, 512, 512]" = native_dropout_default_12[0]
    getitem_271: "b8[1, 16, 512, 512]" = native_dropout_default_12[1];  native_dropout_default_12 = None
    expand_default_50: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_270, [1, 16, 512, 512]);  getitem_270 = None
    view_default_147: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_default_50, [16, 512, 512]);  expand_default_50 = None
    expand_default_51: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(clone_default_50, [1, 16, 512, 64]);  clone_default_50 = None
    view_default_148: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_51, [16, 512, 64]);  expand_default_51 = None
    bmm_default_73: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_147, view_default_148)
    view_default_149: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_73, [1, 16, 512, 64]);  bmm_default_73 = None
    permute_default_73: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_default_147, [0, 2, 1]);  view_default_147 = None
    permute_default_74: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_148, [0, 2, 1]);  view_default_148 = None
    alias_default_25: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_default_24);  alias_default_24 = None
    permute_default_75: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_144, [0, 2, 1]);  view_default_144 = None
    permute_default_76: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_default_145, [0, 2, 1]);  view_default_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_128: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_149, [0, 2, 1, 3]);  view_default_149 = None
    clone_11: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_257: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_11, [1, 512, 1024]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_258: "f32[512, 1024]" = torch.ops.aten.view.default(view_257, [512, 1024]);  view_257 = None
    permute_129: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
    addmm_69: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_189, view_258, permute_129);  primals_189 = None
    view_259: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_69, [1, 512, 1024]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_35 = torch.ops.aten.native_dropout.default(view_259, 0.1, True);  view_259 = None
    getitem_116: "f32[1, 512, 1024]" = native_dropout_35[0]
    getitem_117: "b8[1, 512, 1024]" = native_dropout_35[1];  native_dropout_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_93: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_89, getitem_116);  add_89 = getitem_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_93, [2], correction = 0, keepdim = True)
    getitem_118: "f32[1, 512, 1]" = var_mean_23[0]
    getitem_119: "f32[1, 512, 1]" = var_mean_23[1];  var_mean_23 = None
    add_94: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-12);  getitem_118 = None
    rsqrt_23: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
    sub_36: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_93, getitem_119);  getitem_119 = None
    mul_80: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_23);  sub_36 = None
    mul_81: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_80, primals_190)
    add_95: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_81, primals_191);  mul_81 = primals_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_260: "f32[512, 1024]" = torch.ops.aten.view.default(add_95, [512, 1024]);  add_95 = None
    permute_130: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_192, [1, 0]);  primals_192 = None
    addmm_70: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_193, view_260, permute_130);  primals_193 = None
    view_261: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_70, [1, 512, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_82: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_261, 0.5)
    mul_83: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476);  view_261 = None
    erf_11: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
    add_96: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_84: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_82, add_96);  mul_82 = add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_262: "f32[512, 4096]" = torch.ops.aten.view.default(mul_84, [512, 4096]);  mul_84 = None
    permute_131: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_194, [1, 0]);  primals_194 = None
    addmm_71: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_195, view_262, permute_131);  primals_195 = None
    view_263: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_71, [1, 512, 1024]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_36 = torch.ops.aten.native_dropout.default(view_263, 0.1, True);  view_263 = None
    getitem_120: "f32[1, 512, 1024]" = native_dropout_36[0]
    getitem_121: "b8[1, 512, 1024]" = native_dropout_36[1];  native_dropout_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_97: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_93, getitem_120);  add_93 = getitem_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
    getitem_122: "f32[1, 512, 1]" = var_mean_24[0]
    getitem_123: "f32[1, 512, 1]" = var_mean_24[1];  var_mean_24 = None
    add_98: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-12);  getitem_122 = None
    rsqrt_24: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    sub_37: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_97, getitem_123);  getitem_123 = None
    mul_85: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_24);  sub_37 = None
    mul_86: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_85, primals_196)
    add_99: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_86, primals_197);  mul_86 = primals_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_264: "f32[512, 1024]" = torch.ops.aten.view.default(add_99, [512, 1024]);  add_99 = None
    permute_132: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_198, [1, 0]);  primals_198 = None
    addmm_72: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_199, view_264, permute_132);  primals_199 = None
    view_265: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_72, [1, 512, 1024]);  addmm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_133: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_200, [1, 0]);  primals_200 = None
    addmm_73: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_201, view_264, permute_133);  primals_201 = None
    view_267: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_73, [1, 512, 1024]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_268: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_267, [1, 512, 16, 64]);  view_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_134: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_268, [0, 2, 1, 3]);  view_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_135: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_202, [1, 0]);  primals_202 = None
    addmm_74: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_203, view_264, permute_135);  primals_203 = None
    view_270: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_74, [1, 512, 1024]);  addmm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_271: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_270, [1, 512, 16, 64]);  view_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_136: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_271, [0, 2, 1, 3]);  view_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_272: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_265, [1, 512, 16, 64]);  view_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_137: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_272, [0, 2, 1, 3]);  view_272 = None
    
    # No stacktrace found for following nodes
    clone_default_44: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
    clone_default_45: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_134, memory_format = torch.contiguous_format);  permute_134 = None
    clone_default_46: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_136, memory_format = torch.contiguous_format);  permute_136 = None
    mul_scalar_44: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_44, 0.3535533905932738);  clone_default_44 = None
    permute_default_66: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(clone_default_45, [0, 1, 3, 2]);  clone_default_45 = None
    mul_scalar_45: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_66, 0.3535533905932738);  permute_default_66 = None
    expand_default_44: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_44, [1, 16, 512, 64]);  mul_scalar_44 = None
    view_default_132: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_44, [16, 512, 64]);  expand_default_44 = None
    expand_default_45: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_45, [1, 16, 64, 512]);  mul_scalar_45 = None
    view_default_133: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_default_45, [16, 64, 512]);  expand_default_45 = None
    bmm_default_66: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_132, view_default_133)
    view_default_134: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_66, [1, 16, 512, 512]);  bmm_default_66 = None
    amax_default_11: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(view_default_134, [-1], True)
    sub_tensor_22: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_134, amax_default_11);  view_default_134 = amax_default_11 = None
    exp_default_11: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_22);  sub_tensor_22 = None
    sum_dim_int_list_22: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_11, [-1], True)
    div_tensor_11: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_11, sum_dim_int_list_22);  exp_default_11 = sum_dim_int_list_22 = None
    alias_default_22: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_tensor_11)
    native_dropout_default_11 = torch.ops.aten.native_dropout.default(div_tensor_11, 0.1, True);  div_tensor_11 = None
    getitem_268: "f32[1, 16, 512, 512]" = native_dropout_default_11[0]
    getitem_269: "b8[1, 16, 512, 512]" = native_dropout_default_11[1];  native_dropout_default_11 = None
    expand_default_46: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_268, [1, 16, 512, 512]);  getitem_268 = None
    view_default_135: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_default_46, [16, 512, 512]);  expand_default_46 = None
    expand_default_47: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(clone_default_46, [1, 16, 512, 64]);  clone_default_46 = None
    view_default_136: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_47, [16, 512, 64]);  expand_default_47 = None
    bmm_default_67: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_135, view_default_136)
    view_default_137: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_67, [1, 16, 512, 64]);  bmm_default_67 = None
    permute_default_67: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_default_135, [0, 2, 1]);  view_default_135 = None
    permute_default_68: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_136, [0, 2, 1]);  view_default_136 = None
    alias_default_23: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_default_22);  alias_default_22 = None
    permute_default_69: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_132, [0, 2, 1]);  view_default_132 = None
    permute_default_70: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_default_133, [0, 2, 1]);  view_default_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_139: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_137, [0, 2, 1, 3]);  view_default_137 = None
    clone_12: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_279: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_12, [1, 512, 1024]);  clone_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_280: "f32[512, 1024]" = torch.ops.aten.view.default(view_279, [512, 1024]);  view_279 = None
    permute_140: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_204, [1, 0]);  primals_204 = None
    addmm_75: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_205, view_280, permute_140);  primals_205 = None
    view_281: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_75, [1, 512, 1024]);  addmm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_38 = torch.ops.aten.native_dropout.default(view_281, 0.1, True);  view_281 = None
    getitem_126: "f32[1, 512, 1024]" = native_dropout_38[0]
    getitem_127: "b8[1, 512, 1024]" = native_dropout_38[1];  native_dropout_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_101: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_97, getitem_126);  add_97 = getitem_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_25 = torch.ops.aten.var_mean.correction(add_101, [2], correction = 0, keepdim = True)
    getitem_128: "f32[1, 512, 1]" = var_mean_25[0]
    getitem_129: "f32[1, 512, 1]" = var_mean_25[1];  var_mean_25 = None
    add_102: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-12);  getitem_128 = None
    rsqrt_25: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
    sub_39: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_101, getitem_129);  getitem_129 = None
    mul_87: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_25);  sub_39 = None
    mul_88: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_87, primals_206)
    add_103: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_88, primals_207);  mul_88 = primals_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_282: "f32[512, 1024]" = torch.ops.aten.view.default(add_103, [512, 1024]);  add_103 = None
    permute_141: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_208, [1, 0]);  primals_208 = None
    addmm_76: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_209, view_282, permute_141);  primals_209 = None
    view_283: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_76, [1, 512, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_89: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_283, 0.5)
    mul_90: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_283, 0.7071067811865476);  view_283 = None
    erf_12: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_90);  mul_90 = None
    add_104: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_91: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_89, add_104);  mul_89 = add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_284: "f32[512, 4096]" = torch.ops.aten.view.default(mul_91, [512, 4096]);  mul_91 = None
    permute_142: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_210, [1, 0]);  primals_210 = None
    addmm_77: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_211, view_284, permute_142);  primals_211 = None
    view_285: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_77, [1, 512, 1024]);  addmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_39 = torch.ops.aten.native_dropout.default(view_285, 0.1, True);  view_285 = None
    getitem_130: "f32[1, 512, 1024]" = native_dropout_39[0]
    getitem_131: "b8[1, 512, 1024]" = native_dropout_39[1];  native_dropout_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_105: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_101, getitem_130);  add_101 = getitem_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_26 = torch.ops.aten.var_mean.correction(add_105, [2], correction = 0, keepdim = True)
    getitem_132: "f32[1, 512, 1]" = var_mean_26[0]
    getitem_133: "f32[1, 512, 1]" = var_mean_26[1];  var_mean_26 = None
    add_106: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-12);  getitem_132 = None
    rsqrt_26: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
    sub_40: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_105, getitem_133);  getitem_133 = None
    mul_92: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_26);  sub_40 = None
    mul_93: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_92, primals_212)
    add_107: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_93, primals_213);  mul_93 = primals_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_286: "f32[512, 1024]" = torch.ops.aten.view.default(add_107, [512, 1024]);  add_107 = None
    permute_143: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_214, [1, 0]);  primals_214 = None
    addmm_78: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_215, view_286, permute_143);  primals_215 = None
    view_287: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_78, [1, 512, 1024]);  addmm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_144: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_216, [1, 0]);  primals_216 = None
    addmm_79: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_217, view_286, permute_144);  primals_217 = None
    view_289: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_79, [1, 512, 1024]);  addmm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_290: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_289, [1, 512, 16, 64]);  view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_145: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_290, [0, 2, 1, 3]);  view_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_146: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_218, [1, 0]);  primals_218 = None
    addmm_80: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_219, view_286, permute_146);  primals_219 = None
    view_292: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_80, [1, 512, 1024]);  addmm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_293: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_292, [1, 512, 16, 64]);  view_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_147: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_293, [0, 2, 1, 3]);  view_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_294: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_287, [1, 512, 16, 64]);  view_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_148: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None
    
    # No stacktrace found for following nodes
    clone_default_40: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None
    clone_default_41: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_145, memory_format = torch.contiguous_format);  permute_145 = None
    clone_default_42: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None
    mul_scalar_40: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_40, 0.3535533905932738);  clone_default_40 = None
    permute_default_60: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(clone_default_41, [0, 1, 3, 2]);  clone_default_41 = None
    mul_scalar_41: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_60, 0.3535533905932738);  permute_default_60 = None
    expand_default_40: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_40, [1, 16, 512, 64]);  mul_scalar_40 = None
    view_default_120: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_40, [16, 512, 64]);  expand_default_40 = None
    expand_default_41: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_41, [1, 16, 64, 512]);  mul_scalar_41 = None
    view_default_121: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_default_41, [16, 64, 512]);  expand_default_41 = None
    bmm_default_60: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_120, view_default_121)
    view_default_122: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_60, [1, 16, 512, 512]);  bmm_default_60 = None
    amax_default_10: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(view_default_122, [-1], True)
    sub_tensor_20: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_122, amax_default_10);  view_default_122 = amax_default_10 = None
    exp_default_10: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_20);  sub_tensor_20 = None
    sum_dim_int_list_20: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_10, [-1], True)
    div_tensor_10: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_10, sum_dim_int_list_20);  exp_default_10 = sum_dim_int_list_20 = None
    alias_default_20: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_tensor_10)
    native_dropout_default_10 = torch.ops.aten.native_dropout.default(div_tensor_10, 0.1, True);  div_tensor_10 = None
    getitem_266: "f32[1, 16, 512, 512]" = native_dropout_default_10[0]
    getitem_267: "b8[1, 16, 512, 512]" = native_dropout_default_10[1];  native_dropout_default_10 = None
    expand_default_42: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_266, [1, 16, 512, 512]);  getitem_266 = None
    view_default_123: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_default_42, [16, 512, 512]);  expand_default_42 = None
    expand_default_43: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(clone_default_42, [1, 16, 512, 64]);  clone_default_42 = None
    view_default_124: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_43, [16, 512, 64]);  expand_default_43 = None
    bmm_default_61: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_123, view_default_124)
    view_default_125: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_61, [1, 16, 512, 64]);  bmm_default_61 = None
    permute_default_61: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_default_123, [0, 2, 1]);  view_default_123 = None
    permute_default_62: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_124, [0, 2, 1]);  view_default_124 = None
    alias_default_21: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_default_20);  alias_default_20 = None
    permute_default_63: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_120, [0, 2, 1]);  view_default_120 = None
    permute_default_64: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_default_121, [0, 2, 1]);  view_default_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_150: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_125, [0, 2, 1, 3]);  view_default_125 = None
    clone_13: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_301: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_13, [1, 512, 1024]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_302: "f32[512, 1024]" = torch.ops.aten.view.default(view_301, [512, 1024]);  view_301 = None
    permute_151: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_220, [1, 0]);  primals_220 = None
    addmm_81: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_221, view_302, permute_151);  primals_221 = None
    view_303: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_81, [1, 512, 1024]);  addmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_41 = torch.ops.aten.native_dropout.default(view_303, 0.1, True);  view_303 = None
    getitem_136: "f32[1, 512, 1024]" = native_dropout_41[0]
    getitem_137: "b8[1, 512, 1024]" = native_dropout_41[1];  native_dropout_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_109: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_105, getitem_136);  add_105 = getitem_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_27 = torch.ops.aten.var_mean.correction(add_109, [2], correction = 0, keepdim = True)
    getitem_138: "f32[1, 512, 1]" = var_mean_27[0]
    getitem_139: "f32[1, 512, 1]" = var_mean_27[1];  var_mean_27 = None
    add_110: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_138, 1e-12);  getitem_138 = None
    rsqrt_27: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
    sub_42: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_109, getitem_139);  getitem_139 = None
    mul_94: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_27);  sub_42 = None
    mul_95: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_94, primals_222)
    add_111: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_95, primals_223);  mul_95 = primals_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_304: "f32[512, 1024]" = torch.ops.aten.view.default(add_111, [512, 1024]);  add_111 = None
    permute_152: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_224, [1, 0]);  primals_224 = None
    addmm_82: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_225, view_304, permute_152);  primals_225 = None
    view_305: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_82, [1, 512, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_96: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_305, 0.5)
    mul_97: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_305, 0.7071067811865476);  view_305 = None
    erf_13: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_97);  mul_97 = None
    add_112: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_98: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_96, add_112);  mul_96 = add_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_306: "f32[512, 4096]" = torch.ops.aten.view.default(mul_98, [512, 4096]);  mul_98 = None
    permute_153: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_226, [1, 0]);  primals_226 = None
    addmm_83: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_227, view_306, permute_153);  primals_227 = None
    view_307: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_83, [1, 512, 1024]);  addmm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_42 = torch.ops.aten.native_dropout.default(view_307, 0.1, True);  view_307 = None
    getitem_140: "f32[1, 512, 1024]" = native_dropout_42[0]
    getitem_141: "b8[1, 512, 1024]" = native_dropout_42[1];  native_dropout_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_113: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_109, getitem_140);  add_109 = getitem_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_28 = torch.ops.aten.var_mean.correction(add_113, [2], correction = 0, keepdim = True)
    getitem_142: "f32[1, 512, 1]" = var_mean_28[0]
    getitem_143: "f32[1, 512, 1]" = var_mean_28[1];  var_mean_28 = None
    add_114: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_142, 1e-12);  getitem_142 = None
    rsqrt_28: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
    sub_43: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_113, getitem_143);  getitem_143 = None
    mul_99: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_28);  sub_43 = None
    mul_100: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_99, primals_228)
    add_115: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_100, primals_229);  mul_100 = primals_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_308: "f32[512, 1024]" = torch.ops.aten.view.default(add_115, [512, 1024]);  add_115 = None
    permute_154: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_230, [1, 0]);  primals_230 = None
    addmm_84: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_231, view_308, permute_154);  primals_231 = None
    view_309: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_84, [1, 512, 1024]);  addmm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_155: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_232, [1, 0]);  primals_232 = None
    addmm_85: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_233, view_308, permute_155);  primals_233 = None
    view_311: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_85, [1, 512, 1024]);  addmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_312: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_311, [1, 512, 16, 64]);  view_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_156: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_312, [0, 2, 1, 3]);  view_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_157: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_234, [1, 0]);  primals_234 = None
    addmm_86: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_235, view_308, permute_157);  primals_235 = None
    view_314: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_86, [1, 512, 1024]);  addmm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_315: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_314, [1, 512, 16, 64]);  view_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_158: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_315, [0, 2, 1, 3]);  view_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_316: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_309, [1, 512, 16, 64]);  view_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_159: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None
    
    # No stacktrace found for following nodes
    clone_default_36: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None
    clone_default_37: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_156, memory_format = torch.contiguous_format);  permute_156 = None
    clone_default_38: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_158, memory_format = torch.contiguous_format);  permute_158 = None
    mul_scalar_36: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_36, 0.3535533905932738);  clone_default_36 = None
    permute_default_54: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(clone_default_37, [0, 1, 3, 2]);  clone_default_37 = None
    mul_scalar_37: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_54, 0.3535533905932738);  permute_default_54 = None
    expand_default_36: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_36, [1, 16, 512, 64]);  mul_scalar_36 = None
    view_default_108: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_36, [16, 512, 64]);  expand_default_36 = None
    expand_default_37: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_37, [1, 16, 64, 512]);  mul_scalar_37 = None
    view_default_109: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_default_37, [16, 64, 512]);  expand_default_37 = None
    bmm_default_54: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_108, view_default_109)
    view_default_110: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_54, [1, 16, 512, 512]);  bmm_default_54 = None
    amax_default_9: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(view_default_110, [-1], True)
    sub_tensor_18: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_110, amax_default_9);  view_default_110 = amax_default_9 = None
    exp_default_9: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_18);  sub_tensor_18 = None
    sum_dim_int_list_18: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_9, [-1], True)
    div_tensor_9: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_9, sum_dim_int_list_18);  exp_default_9 = sum_dim_int_list_18 = None
    alias_default_18: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_tensor_9)
    native_dropout_default_9 = torch.ops.aten.native_dropout.default(div_tensor_9, 0.1, True);  div_tensor_9 = None
    getitem_264: "f32[1, 16, 512, 512]" = native_dropout_default_9[0]
    getitem_265: "b8[1, 16, 512, 512]" = native_dropout_default_9[1];  native_dropout_default_9 = None
    expand_default_38: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_264, [1, 16, 512, 512]);  getitem_264 = None
    view_default_111: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_default_38, [16, 512, 512]);  expand_default_38 = None
    expand_default_39: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(clone_default_38, [1, 16, 512, 64]);  clone_default_38 = None
    view_default_112: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_39, [16, 512, 64]);  expand_default_39 = None
    bmm_default_55: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_111, view_default_112)
    view_default_113: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_55, [1, 16, 512, 64]);  bmm_default_55 = None
    permute_default_55: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_default_111, [0, 2, 1]);  view_default_111 = None
    permute_default_56: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_112, [0, 2, 1]);  view_default_112 = None
    alias_default_19: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_default_18);  alias_default_18 = None
    permute_default_57: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_108, [0, 2, 1]);  view_default_108 = None
    permute_default_58: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_default_109, [0, 2, 1]);  view_default_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_161: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_113, [0, 2, 1, 3]);  view_default_113 = None
    clone_14: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_323: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_14, [1, 512, 1024]);  clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_324: "f32[512, 1024]" = torch.ops.aten.view.default(view_323, [512, 1024]);  view_323 = None
    permute_162: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_236, [1, 0]);  primals_236 = None
    addmm_87: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_237, view_324, permute_162);  primals_237 = None
    view_325: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_87, [1, 512, 1024]);  addmm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_44 = torch.ops.aten.native_dropout.default(view_325, 0.1, True);  view_325 = None
    getitem_146: "f32[1, 512, 1024]" = native_dropout_44[0]
    getitem_147: "b8[1, 512, 1024]" = native_dropout_44[1];  native_dropout_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_117: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_113, getitem_146);  add_113 = getitem_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_29 = torch.ops.aten.var_mean.correction(add_117, [2], correction = 0, keepdim = True)
    getitem_148: "f32[1, 512, 1]" = var_mean_29[0]
    getitem_149: "f32[1, 512, 1]" = var_mean_29[1];  var_mean_29 = None
    add_118: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_148, 1e-12);  getitem_148 = None
    rsqrt_29: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
    sub_45: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_117, getitem_149);  getitem_149 = None
    mul_101: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_29);  sub_45 = None
    mul_102: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_101, primals_238)
    add_119: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_102, primals_239);  mul_102 = primals_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_326: "f32[512, 1024]" = torch.ops.aten.view.default(add_119, [512, 1024]);  add_119 = None
    permute_163: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_240, [1, 0]);  primals_240 = None
    addmm_88: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_241, view_326, permute_163);  primals_241 = None
    view_327: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_88, [1, 512, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_103: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_327, 0.5)
    mul_104: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_327, 0.7071067811865476);  view_327 = None
    erf_14: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_104);  mul_104 = None
    add_120: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_105: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_103, add_120);  mul_103 = add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_328: "f32[512, 4096]" = torch.ops.aten.view.default(mul_105, [512, 4096]);  mul_105 = None
    permute_164: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_242, [1, 0]);  primals_242 = None
    addmm_89: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_243, view_328, permute_164);  primals_243 = None
    view_329: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_89, [1, 512, 1024]);  addmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_45 = torch.ops.aten.native_dropout.default(view_329, 0.1, True);  view_329 = None
    getitem_150: "f32[1, 512, 1024]" = native_dropout_45[0]
    getitem_151: "b8[1, 512, 1024]" = native_dropout_45[1];  native_dropout_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_121: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_117, getitem_150);  add_117 = getitem_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_30 = torch.ops.aten.var_mean.correction(add_121, [2], correction = 0, keepdim = True)
    getitem_152: "f32[1, 512, 1]" = var_mean_30[0]
    getitem_153: "f32[1, 512, 1]" = var_mean_30[1];  var_mean_30 = None
    add_122: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_152, 1e-12);  getitem_152 = None
    rsqrt_30: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    sub_46: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_121, getitem_153);  getitem_153 = None
    mul_106: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_30);  sub_46 = None
    mul_107: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_106, primals_244)
    add_123: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_107, primals_245);  mul_107 = primals_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_330: "f32[512, 1024]" = torch.ops.aten.view.default(add_123, [512, 1024]);  add_123 = None
    permute_165: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_246, [1, 0]);  primals_246 = None
    addmm_90: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_247, view_330, permute_165);  primals_247 = None
    view_331: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_90, [1, 512, 1024]);  addmm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_166: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_248, [1, 0]);  primals_248 = None
    addmm_91: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_249, view_330, permute_166);  primals_249 = None
    view_333: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_91, [1, 512, 1024]);  addmm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_334: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_333, [1, 512, 16, 64]);  view_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_167: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_334, [0, 2, 1, 3]);  view_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_168: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_250, [1, 0]);  primals_250 = None
    addmm_92: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_251, view_330, permute_168);  primals_251 = None
    view_336: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_92, [1, 512, 1024]);  addmm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_337: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_336, [1, 512, 16, 64]);  view_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_169: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_337, [0, 2, 1, 3]);  view_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_338: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_331, [1, 512, 16, 64]);  view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_170: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None
    
    # No stacktrace found for following nodes
    clone_default_32: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
    clone_default_33: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_167, memory_format = torch.contiguous_format);  permute_167 = None
    clone_default_34: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_169, memory_format = torch.contiguous_format);  permute_169 = None
    mul_scalar_32: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_32, 0.3535533905932738);  clone_default_32 = None
    permute_default_48: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(clone_default_33, [0, 1, 3, 2]);  clone_default_33 = None
    mul_scalar_33: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_48, 0.3535533905932738);  permute_default_48 = None
    expand_default_32: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_32, [1, 16, 512, 64]);  mul_scalar_32 = None
    view_default_96: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_32, [16, 512, 64]);  expand_default_32 = None
    expand_default_33: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_33, [1, 16, 64, 512]);  mul_scalar_33 = None
    view_default_97: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_default_33, [16, 64, 512]);  expand_default_33 = None
    bmm_default_48: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_96, view_default_97)
    view_default_98: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_48, [1, 16, 512, 512]);  bmm_default_48 = None
    amax_default_8: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(view_default_98, [-1], True)
    sub_tensor_16: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_98, amax_default_8);  view_default_98 = amax_default_8 = None
    exp_default_8: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_16);  sub_tensor_16 = None
    sum_dim_int_list_16: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_8, [-1], True)
    div_tensor_8: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_8, sum_dim_int_list_16);  exp_default_8 = sum_dim_int_list_16 = None
    alias_default_16: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_tensor_8)
    native_dropout_default_8 = torch.ops.aten.native_dropout.default(div_tensor_8, 0.1, True);  div_tensor_8 = None
    getitem_262: "f32[1, 16, 512, 512]" = native_dropout_default_8[0]
    getitem_263: "b8[1, 16, 512, 512]" = native_dropout_default_8[1];  native_dropout_default_8 = None
    expand_default_34: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_262, [1, 16, 512, 512]);  getitem_262 = None
    view_default_99: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_default_34, [16, 512, 512]);  expand_default_34 = None
    expand_default_35: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(clone_default_34, [1, 16, 512, 64]);  clone_default_34 = None
    view_default_100: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_35, [16, 512, 64]);  expand_default_35 = None
    bmm_default_49: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_99, view_default_100)
    view_default_101: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_49, [1, 16, 512, 64]);  bmm_default_49 = None
    permute_default_49: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_default_99, [0, 2, 1]);  view_default_99 = None
    permute_default_50: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_100, [0, 2, 1]);  view_default_100 = None
    alias_default_17: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_default_16);  alias_default_16 = None
    permute_default_51: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_96, [0, 2, 1]);  view_default_96 = None
    permute_default_52: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_default_97, [0, 2, 1]);  view_default_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_172: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_101, [0, 2, 1, 3]);  view_default_101 = None
    clone_15: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_345: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_15, [1, 512, 1024]);  clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_346: "f32[512, 1024]" = torch.ops.aten.view.default(view_345, [512, 1024]);  view_345 = None
    permute_173: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_252, [1, 0]);  primals_252 = None
    addmm_93: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_253, view_346, permute_173);  primals_253 = None
    view_347: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_93, [1, 512, 1024]);  addmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_47 = torch.ops.aten.native_dropout.default(view_347, 0.1, True);  view_347 = None
    getitem_156: "f32[1, 512, 1024]" = native_dropout_47[0]
    getitem_157: "b8[1, 512, 1024]" = native_dropout_47[1];  native_dropout_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_125: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_121, getitem_156);  add_121 = getitem_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_31 = torch.ops.aten.var_mean.correction(add_125, [2], correction = 0, keepdim = True)
    getitem_158: "f32[1, 512, 1]" = var_mean_31[0]
    getitem_159: "f32[1, 512, 1]" = var_mean_31[1];  var_mean_31 = None
    add_126: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-12);  getitem_158 = None
    rsqrt_31: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
    sub_48: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_125, getitem_159);  getitem_159 = None
    mul_108: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_31);  sub_48 = None
    mul_109: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_108, primals_254)
    add_127: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_109, primals_255);  mul_109 = primals_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_348: "f32[512, 1024]" = torch.ops.aten.view.default(add_127, [512, 1024]);  add_127 = None
    permute_174: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_256, [1, 0]);  primals_256 = None
    addmm_94: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_257, view_348, permute_174);  primals_257 = None
    view_349: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_94, [1, 512, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_110: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_349, 0.5)
    mul_111: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_349, 0.7071067811865476);  view_349 = None
    erf_15: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_111);  mul_111 = None
    add_128: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_112: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_110, add_128);  mul_110 = add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_350: "f32[512, 4096]" = torch.ops.aten.view.default(mul_112, [512, 4096]);  mul_112 = None
    permute_175: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_258, [1, 0]);  primals_258 = None
    addmm_95: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_259, view_350, permute_175);  primals_259 = None
    view_351: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_95, [1, 512, 1024]);  addmm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_48 = torch.ops.aten.native_dropout.default(view_351, 0.1, True);  view_351 = None
    getitem_160: "f32[1, 512, 1024]" = native_dropout_48[0]
    getitem_161: "b8[1, 512, 1024]" = native_dropout_48[1];  native_dropout_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_129: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_125, getitem_160);  add_125 = getitem_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_32 = torch.ops.aten.var_mean.correction(add_129, [2], correction = 0, keepdim = True)
    getitem_162: "f32[1, 512, 1]" = var_mean_32[0]
    getitem_163: "f32[1, 512, 1]" = var_mean_32[1];  var_mean_32 = None
    add_130: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_162, 1e-12);  getitem_162 = None
    rsqrt_32: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
    sub_49: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_129, getitem_163);  getitem_163 = None
    mul_113: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_32);  sub_49 = None
    mul_114: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_113, primals_260)
    add_131: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_114, primals_261);  mul_114 = primals_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_352: "f32[512, 1024]" = torch.ops.aten.view.default(add_131, [512, 1024]);  add_131 = None
    permute_176: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_262, [1, 0]);  primals_262 = None
    addmm_96: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_263, view_352, permute_176);  primals_263 = None
    view_353: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_96, [1, 512, 1024]);  addmm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_177: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_264, [1, 0]);  primals_264 = None
    addmm_97: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_265, view_352, permute_177);  primals_265 = None
    view_355: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_97, [1, 512, 1024]);  addmm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_356: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_355, [1, 512, 16, 64]);  view_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_178: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_356, [0, 2, 1, 3]);  view_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_179: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_266, [1, 0]);  primals_266 = None
    addmm_98: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_267, view_352, permute_179);  primals_267 = None
    view_358: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_98, [1, 512, 1024]);  addmm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_359: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_358, [1, 512, 16, 64]);  view_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_180: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_359, [0, 2, 1, 3]);  view_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_360: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_353, [1, 512, 16, 64]);  view_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_181: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_360, [0, 2, 1, 3]);  view_360 = None
    
    # No stacktrace found for following nodes
    clone_default_28: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_181, memory_format = torch.contiguous_format);  permute_181 = None
    clone_default_29: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_178, memory_format = torch.contiguous_format);  permute_178 = None
    clone_default_30: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_180, memory_format = torch.contiguous_format);  permute_180 = None
    mul_scalar_28: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_28, 0.3535533905932738);  clone_default_28 = None
    permute_default_42: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(clone_default_29, [0, 1, 3, 2]);  clone_default_29 = None
    mul_scalar_29: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_42, 0.3535533905932738);  permute_default_42 = None
    expand_default_28: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_28, [1, 16, 512, 64]);  mul_scalar_28 = None
    view_default_84: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_28, [16, 512, 64]);  expand_default_28 = None
    expand_default_29: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_29, [1, 16, 64, 512]);  mul_scalar_29 = None
    view_default_85: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_default_29, [16, 64, 512]);  expand_default_29 = None
    bmm_default_42: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_84, view_default_85)
    view_default_86: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_42, [1, 16, 512, 512]);  bmm_default_42 = None
    amax_default_7: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(view_default_86, [-1], True)
    sub_tensor_14: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_86, amax_default_7);  view_default_86 = amax_default_7 = None
    exp_default_7: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_14);  sub_tensor_14 = None
    sum_dim_int_list_14: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_7, [-1], True)
    div_tensor_7: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_7, sum_dim_int_list_14);  exp_default_7 = sum_dim_int_list_14 = None
    alias_default_14: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_tensor_7)
    native_dropout_default_7 = torch.ops.aten.native_dropout.default(div_tensor_7, 0.1, True);  div_tensor_7 = None
    getitem_260: "f32[1, 16, 512, 512]" = native_dropout_default_7[0]
    getitem_261: "b8[1, 16, 512, 512]" = native_dropout_default_7[1];  native_dropout_default_7 = None
    expand_default_30: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_260, [1, 16, 512, 512]);  getitem_260 = None
    view_default_87: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_default_30, [16, 512, 512]);  expand_default_30 = None
    expand_default_31: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(clone_default_30, [1, 16, 512, 64]);  clone_default_30 = None
    view_default_88: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_31, [16, 512, 64]);  expand_default_31 = None
    bmm_default_43: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_87, view_default_88)
    view_default_89: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_43, [1, 16, 512, 64]);  bmm_default_43 = None
    permute_default_43: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_default_87, [0, 2, 1]);  view_default_87 = None
    permute_default_44: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_88, [0, 2, 1]);  view_default_88 = None
    alias_default_15: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_default_14);  alias_default_14 = None
    permute_default_45: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_84, [0, 2, 1]);  view_default_84 = None
    permute_default_46: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_default_85, [0, 2, 1]);  view_default_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_183: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_89, [0, 2, 1, 3]);  view_default_89 = None
    clone_16: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_367: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_16, [1, 512, 1024]);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_368: "f32[512, 1024]" = torch.ops.aten.view.default(view_367, [512, 1024]);  view_367 = None
    permute_184: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_268, [1, 0]);  primals_268 = None
    addmm_99: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_269, view_368, permute_184);  primals_269 = None
    view_369: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_99, [1, 512, 1024]);  addmm_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_50 = torch.ops.aten.native_dropout.default(view_369, 0.1, True);  view_369 = None
    getitem_166: "f32[1, 512, 1024]" = native_dropout_50[0]
    getitem_167: "b8[1, 512, 1024]" = native_dropout_50[1];  native_dropout_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_133: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_129, getitem_166);  add_129 = getitem_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_33 = torch.ops.aten.var_mean.correction(add_133, [2], correction = 0, keepdim = True)
    getitem_168: "f32[1, 512, 1]" = var_mean_33[0]
    getitem_169: "f32[1, 512, 1]" = var_mean_33[1];  var_mean_33 = None
    add_134: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_168, 1e-12);  getitem_168 = None
    rsqrt_33: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
    sub_51: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_133, getitem_169);  getitem_169 = None
    mul_115: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_33);  sub_51 = None
    mul_116: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_115, primals_270)
    add_135: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_116, primals_271);  mul_116 = primals_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_370: "f32[512, 1024]" = torch.ops.aten.view.default(add_135, [512, 1024]);  add_135 = None
    permute_185: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_272, [1, 0]);  primals_272 = None
    addmm_100: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_273, view_370, permute_185);  primals_273 = None
    view_371: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_100, [1, 512, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_117: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_371, 0.5)
    mul_118: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_371, 0.7071067811865476);  view_371 = None
    erf_16: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_118);  mul_118 = None
    add_136: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_119: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_117, add_136);  mul_117 = add_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_372: "f32[512, 4096]" = torch.ops.aten.view.default(mul_119, [512, 4096]);  mul_119 = None
    permute_186: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_274, [1, 0]);  primals_274 = None
    addmm_101: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_275, view_372, permute_186);  primals_275 = None
    view_373: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_101, [1, 512, 1024]);  addmm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_51 = torch.ops.aten.native_dropout.default(view_373, 0.1, True);  view_373 = None
    getitem_170: "f32[1, 512, 1024]" = native_dropout_51[0]
    getitem_171: "b8[1, 512, 1024]" = native_dropout_51[1];  native_dropout_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_137: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_133, getitem_170);  add_133 = getitem_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_34 = torch.ops.aten.var_mean.correction(add_137, [2], correction = 0, keepdim = True)
    getitem_172: "f32[1, 512, 1]" = var_mean_34[0]
    getitem_173: "f32[1, 512, 1]" = var_mean_34[1];  var_mean_34 = None
    add_138: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_172, 1e-12);  getitem_172 = None
    rsqrt_34: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
    sub_52: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_137, getitem_173);  getitem_173 = None
    mul_120: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_34);  sub_52 = None
    mul_121: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_120, primals_276)
    add_139: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_121, primals_277);  mul_121 = primals_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_374: "f32[512, 1024]" = torch.ops.aten.view.default(add_139, [512, 1024]);  add_139 = None
    permute_187: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_278, [1, 0]);  primals_278 = None
    addmm_102: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_279, view_374, permute_187);  primals_279 = None
    view_375: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_102, [1, 512, 1024]);  addmm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_188: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_280, [1, 0]);  primals_280 = None
    addmm_103: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_281, view_374, permute_188);  primals_281 = None
    view_377: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_103, [1, 512, 1024]);  addmm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_378: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_377, [1, 512, 16, 64]);  view_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_189: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_378, [0, 2, 1, 3]);  view_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_190: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_282, [1, 0]);  primals_282 = None
    addmm_104: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_283, view_374, permute_190);  primals_283 = None
    view_380: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_104, [1, 512, 1024]);  addmm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_381: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_380, [1, 512, 16, 64]);  view_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_191: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_381, [0, 2, 1, 3]);  view_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_382: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_375, [1, 512, 16, 64]);  view_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_192: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_382, [0, 2, 1, 3]);  view_382 = None
    
    # No stacktrace found for following nodes
    clone_default_24: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_192, memory_format = torch.contiguous_format);  permute_192 = None
    clone_default_25: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
    clone_default_26: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_191, memory_format = torch.contiguous_format);  permute_191 = None
    mul_scalar_24: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_24, 0.3535533905932738);  clone_default_24 = None
    permute_default_36: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(clone_default_25, [0, 1, 3, 2]);  clone_default_25 = None
    mul_scalar_25: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_36, 0.3535533905932738);  permute_default_36 = None
    expand_default_24: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_24, [1, 16, 512, 64]);  mul_scalar_24 = None
    view_default_72: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_24, [16, 512, 64]);  expand_default_24 = None
    expand_default_25: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_25, [1, 16, 64, 512]);  mul_scalar_25 = None
    view_default_73: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_default_25, [16, 64, 512]);  expand_default_25 = None
    bmm_default_36: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_72, view_default_73)
    view_default_74: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_36, [1, 16, 512, 512]);  bmm_default_36 = None
    amax_default_6: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(view_default_74, [-1], True)
    sub_tensor_12: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_74, amax_default_6);  view_default_74 = amax_default_6 = None
    exp_default_6: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_12);  sub_tensor_12 = None
    sum_dim_int_list_12: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_6, [-1], True)
    div_tensor_6: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_6, sum_dim_int_list_12);  exp_default_6 = sum_dim_int_list_12 = None
    alias_default_12: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_tensor_6)
    native_dropout_default_6 = torch.ops.aten.native_dropout.default(div_tensor_6, 0.1, True);  div_tensor_6 = None
    getitem_258: "f32[1, 16, 512, 512]" = native_dropout_default_6[0]
    getitem_259: "b8[1, 16, 512, 512]" = native_dropout_default_6[1];  native_dropout_default_6 = None
    expand_default_26: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_258, [1, 16, 512, 512]);  getitem_258 = None
    view_default_75: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_default_26, [16, 512, 512]);  expand_default_26 = None
    expand_default_27: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(clone_default_26, [1, 16, 512, 64]);  clone_default_26 = None
    view_default_76: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_27, [16, 512, 64]);  expand_default_27 = None
    bmm_default_37: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_75, view_default_76)
    view_default_77: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_37, [1, 16, 512, 64]);  bmm_default_37 = None
    permute_default_37: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_default_75, [0, 2, 1]);  view_default_75 = None
    permute_default_38: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_76, [0, 2, 1]);  view_default_76 = None
    alias_default_13: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_default_12);  alias_default_12 = None
    permute_default_39: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_72, [0, 2, 1]);  view_default_72 = None
    permute_default_40: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_default_73, [0, 2, 1]);  view_default_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_194: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_77, [0, 2, 1, 3]);  view_default_77 = None
    clone_17: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_389: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_17, [1, 512, 1024]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_390: "f32[512, 1024]" = torch.ops.aten.view.default(view_389, [512, 1024]);  view_389 = None
    permute_195: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_284, [1, 0]);  primals_284 = None
    addmm_105: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_285, view_390, permute_195);  primals_285 = None
    view_391: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_105, [1, 512, 1024]);  addmm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_53 = torch.ops.aten.native_dropout.default(view_391, 0.1, True);  view_391 = None
    getitem_176: "f32[1, 512, 1024]" = native_dropout_53[0]
    getitem_177: "b8[1, 512, 1024]" = native_dropout_53[1];  native_dropout_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_141: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_137, getitem_176);  add_137 = getitem_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_35 = torch.ops.aten.var_mean.correction(add_141, [2], correction = 0, keepdim = True)
    getitem_178: "f32[1, 512, 1]" = var_mean_35[0]
    getitem_179: "f32[1, 512, 1]" = var_mean_35[1];  var_mean_35 = None
    add_142: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_178, 1e-12);  getitem_178 = None
    rsqrt_35: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
    sub_54: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_141, getitem_179);  getitem_179 = None
    mul_122: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_35);  sub_54 = None
    mul_123: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_122, primals_286)
    add_143: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_123, primals_287);  mul_123 = primals_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_392: "f32[512, 1024]" = torch.ops.aten.view.default(add_143, [512, 1024]);  add_143 = None
    permute_196: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_288, [1, 0]);  primals_288 = None
    addmm_106: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_289, view_392, permute_196);  primals_289 = None
    view_393: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_106, [1, 512, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_124: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_393, 0.5)
    mul_125: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_393, 0.7071067811865476);  view_393 = None
    erf_17: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_125);  mul_125 = None
    add_144: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_126: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_124, add_144);  mul_124 = add_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_394: "f32[512, 4096]" = torch.ops.aten.view.default(mul_126, [512, 4096]);  mul_126 = None
    permute_197: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_290, [1, 0]);  primals_290 = None
    addmm_107: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_291, view_394, permute_197);  primals_291 = None
    view_395: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_107, [1, 512, 1024]);  addmm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_54 = torch.ops.aten.native_dropout.default(view_395, 0.1, True);  view_395 = None
    getitem_180: "f32[1, 512, 1024]" = native_dropout_54[0]
    getitem_181: "b8[1, 512, 1024]" = native_dropout_54[1];  native_dropout_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_145: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_141, getitem_180);  add_141 = getitem_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_36 = torch.ops.aten.var_mean.correction(add_145, [2], correction = 0, keepdim = True)
    getitem_182: "f32[1, 512, 1]" = var_mean_36[0]
    getitem_183: "f32[1, 512, 1]" = var_mean_36[1];  var_mean_36 = None
    add_146: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_182, 1e-12);  getitem_182 = None
    rsqrt_36: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
    sub_55: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_145, getitem_183);  getitem_183 = None
    mul_127: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_36);  sub_55 = None
    mul_128: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_127, primals_292)
    add_147: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_128, primals_293);  mul_128 = primals_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_396: "f32[512, 1024]" = torch.ops.aten.view.default(add_147, [512, 1024]);  add_147 = None
    permute_198: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_294, [1, 0]);  primals_294 = None
    addmm_108: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_295, view_396, permute_198);  primals_295 = None
    view_397: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_108, [1, 512, 1024]);  addmm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_199: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_296, [1, 0]);  primals_296 = None
    addmm_109: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_297, view_396, permute_199);  primals_297 = None
    view_399: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_109, [1, 512, 1024]);  addmm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_400: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_399, [1, 512, 16, 64]);  view_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_200: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_400, [0, 2, 1, 3]);  view_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_201: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_298, [1, 0]);  primals_298 = None
    addmm_110: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_299, view_396, permute_201);  primals_299 = None
    view_402: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_110, [1, 512, 1024]);  addmm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_403: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_402, [1, 512, 16, 64]);  view_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_202: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_403, [0, 2, 1, 3]);  view_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_404: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_397, [1, 512, 16, 64]);  view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_203: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_404, [0, 2, 1, 3]);  view_404 = None
    
    # No stacktrace found for following nodes
    clone_default_20: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_203, memory_format = torch.contiguous_format);  permute_203 = None
    clone_default_21: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_200, memory_format = torch.contiguous_format);  permute_200 = None
    clone_default_22: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_202, memory_format = torch.contiguous_format);  permute_202 = None
    mul_scalar_20: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_20, 0.3535533905932738);  clone_default_20 = None
    permute_default_30: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(clone_default_21, [0, 1, 3, 2]);  clone_default_21 = None
    mul_scalar_21: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_30, 0.3535533905932738);  permute_default_30 = None
    expand_default_20: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_20, [1, 16, 512, 64]);  mul_scalar_20 = None
    view_default_60: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_20, [16, 512, 64]);  expand_default_20 = None
    expand_default_21: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_21, [1, 16, 64, 512]);  mul_scalar_21 = None
    view_default_61: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_default_21, [16, 64, 512]);  expand_default_21 = None
    bmm_default_30: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_60, view_default_61)
    view_default_62: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_30, [1, 16, 512, 512]);  bmm_default_30 = None
    amax_default_5: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(view_default_62, [-1], True)
    sub_tensor_10: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_62, amax_default_5);  view_default_62 = amax_default_5 = None
    exp_default_5: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_10);  sub_tensor_10 = None
    sum_dim_int_list_10: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_5, [-1], True)
    div_tensor_5: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_5, sum_dim_int_list_10);  exp_default_5 = sum_dim_int_list_10 = None
    alias_default_10: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_tensor_5)
    native_dropout_default_5 = torch.ops.aten.native_dropout.default(div_tensor_5, 0.1, True);  div_tensor_5 = None
    getitem_256: "f32[1, 16, 512, 512]" = native_dropout_default_5[0]
    getitem_257: "b8[1, 16, 512, 512]" = native_dropout_default_5[1];  native_dropout_default_5 = None
    expand_default_22: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_256, [1, 16, 512, 512]);  getitem_256 = None
    view_default_63: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_default_22, [16, 512, 512]);  expand_default_22 = None
    expand_default_23: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(clone_default_22, [1, 16, 512, 64]);  clone_default_22 = None
    view_default_64: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_23, [16, 512, 64]);  expand_default_23 = None
    bmm_default_31: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_63, view_default_64)
    view_default_65: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_31, [1, 16, 512, 64]);  bmm_default_31 = None
    permute_default_31: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_default_63, [0, 2, 1]);  view_default_63 = None
    permute_default_32: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_64, [0, 2, 1]);  view_default_64 = None
    alias_default_11: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_default_10);  alias_default_10 = None
    permute_default_33: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_60, [0, 2, 1]);  view_default_60 = None
    permute_default_34: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_default_61, [0, 2, 1]);  view_default_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_205: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_65, [0, 2, 1, 3]);  view_default_65 = None
    clone_18: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_411: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_18, [1, 512, 1024]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_412: "f32[512, 1024]" = torch.ops.aten.view.default(view_411, [512, 1024]);  view_411 = None
    permute_206: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_300, [1, 0]);  primals_300 = None
    addmm_111: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_301, view_412, permute_206);  primals_301 = None
    view_413: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_111, [1, 512, 1024]);  addmm_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_56 = torch.ops.aten.native_dropout.default(view_413, 0.1, True);  view_413 = None
    getitem_186: "f32[1, 512, 1024]" = native_dropout_56[0]
    getitem_187: "b8[1, 512, 1024]" = native_dropout_56[1];  native_dropout_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_149: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_145, getitem_186);  add_145 = getitem_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_37 = torch.ops.aten.var_mean.correction(add_149, [2], correction = 0, keepdim = True)
    getitem_188: "f32[1, 512, 1]" = var_mean_37[0]
    getitem_189: "f32[1, 512, 1]" = var_mean_37[1];  var_mean_37 = None
    add_150: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_188, 1e-12);  getitem_188 = None
    rsqrt_37: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
    sub_57: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_149, getitem_189);  getitem_189 = None
    mul_129: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_37);  sub_57 = None
    mul_130: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_129, primals_302)
    add_151: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_130, primals_303);  mul_130 = primals_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_414: "f32[512, 1024]" = torch.ops.aten.view.default(add_151, [512, 1024]);  add_151 = None
    permute_207: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_304, [1, 0]);  primals_304 = None
    addmm_112: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_305, view_414, permute_207);  primals_305 = None
    view_415: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_112, [1, 512, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_131: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_415, 0.5)
    mul_132: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_415, 0.7071067811865476);  view_415 = None
    erf_18: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_132);  mul_132 = None
    add_152: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_133: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_131, add_152);  mul_131 = add_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_416: "f32[512, 4096]" = torch.ops.aten.view.default(mul_133, [512, 4096]);  mul_133 = None
    permute_208: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_306, [1, 0]);  primals_306 = None
    addmm_113: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_307, view_416, permute_208);  primals_307 = None
    view_417: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_113, [1, 512, 1024]);  addmm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_57 = torch.ops.aten.native_dropout.default(view_417, 0.1, True);  view_417 = None
    getitem_190: "f32[1, 512, 1024]" = native_dropout_57[0]
    getitem_191: "b8[1, 512, 1024]" = native_dropout_57[1];  native_dropout_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_153: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_149, getitem_190);  add_149 = getitem_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_38 = torch.ops.aten.var_mean.correction(add_153, [2], correction = 0, keepdim = True)
    getitem_192: "f32[1, 512, 1]" = var_mean_38[0]
    getitem_193: "f32[1, 512, 1]" = var_mean_38[1];  var_mean_38 = None
    add_154: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_192, 1e-12);  getitem_192 = None
    rsqrt_38: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
    sub_58: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_153, getitem_193);  getitem_193 = None
    mul_134: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_38);  sub_58 = None
    mul_135: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_134, primals_308)
    add_155: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_135, primals_309);  mul_135 = primals_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_418: "f32[512, 1024]" = torch.ops.aten.view.default(add_155, [512, 1024]);  add_155 = None
    permute_209: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_310, [1, 0]);  primals_310 = None
    addmm_114: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_311, view_418, permute_209);  primals_311 = None
    view_419: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_114, [1, 512, 1024]);  addmm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_210: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_312, [1, 0]);  primals_312 = None
    addmm_115: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_313, view_418, permute_210);  primals_313 = None
    view_421: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_115, [1, 512, 1024]);  addmm_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_422: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_421, [1, 512, 16, 64]);  view_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_211: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_422, [0, 2, 1, 3]);  view_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_212: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_314, [1, 0]);  primals_314 = None
    addmm_116: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_315, view_418, permute_212);  primals_315 = None
    view_424: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_116, [1, 512, 1024]);  addmm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_425: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_424, [1, 512, 16, 64]);  view_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_213: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_425, [0, 2, 1, 3]);  view_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_426: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_419, [1, 512, 16, 64]);  view_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_214: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_426, [0, 2, 1, 3]);  view_426 = None
    
    # No stacktrace found for following nodes
    clone_default_16: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_214, memory_format = torch.contiguous_format);  permute_214 = None
    clone_default_17: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_211, memory_format = torch.contiguous_format);  permute_211 = None
    clone_default_18: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_213, memory_format = torch.contiguous_format);  permute_213 = None
    mul_scalar_16: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_16, 0.3535533905932738);  clone_default_16 = None
    permute_default_24: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(clone_default_17, [0, 1, 3, 2]);  clone_default_17 = None
    mul_scalar_17: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_24, 0.3535533905932738);  permute_default_24 = None
    expand_default_16: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_16, [1, 16, 512, 64]);  mul_scalar_16 = None
    view_default_48: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_16, [16, 512, 64]);  expand_default_16 = None
    expand_default_17: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_17, [1, 16, 64, 512]);  mul_scalar_17 = None
    view_default_49: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_default_17, [16, 64, 512]);  expand_default_17 = None
    bmm_default_24: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_48, view_default_49)
    view_default_50: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_24, [1, 16, 512, 512]);  bmm_default_24 = None
    amax_default_4: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(view_default_50, [-1], True)
    sub_tensor_8: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_50, amax_default_4);  view_default_50 = amax_default_4 = None
    exp_default_4: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_8);  sub_tensor_8 = None
    sum_dim_int_list_8: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_4, [-1], True)
    div_tensor_4: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_4, sum_dim_int_list_8);  exp_default_4 = sum_dim_int_list_8 = None
    alias_default_8: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_tensor_4)
    native_dropout_default_4 = torch.ops.aten.native_dropout.default(div_tensor_4, 0.1, True);  div_tensor_4 = None
    getitem_254: "f32[1, 16, 512, 512]" = native_dropout_default_4[0]
    getitem_255: "b8[1, 16, 512, 512]" = native_dropout_default_4[1];  native_dropout_default_4 = None
    expand_default_18: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_254, [1, 16, 512, 512]);  getitem_254 = None
    view_default_51: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_default_18, [16, 512, 512]);  expand_default_18 = None
    expand_default_19: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(clone_default_18, [1, 16, 512, 64]);  clone_default_18 = None
    view_default_52: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_19, [16, 512, 64]);  expand_default_19 = None
    bmm_default_25: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_51, view_default_52)
    view_default_53: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_25, [1, 16, 512, 64]);  bmm_default_25 = None
    permute_default_25: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_default_51, [0, 2, 1]);  view_default_51 = None
    permute_default_26: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_52, [0, 2, 1]);  view_default_52 = None
    alias_default_9: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_default_8);  alias_default_8 = None
    permute_default_27: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_48, [0, 2, 1]);  view_default_48 = None
    permute_default_28: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_default_49, [0, 2, 1]);  view_default_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_216: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_53, [0, 2, 1, 3]);  view_default_53 = None
    clone_19: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_433: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_19, [1, 512, 1024]);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_434: "f32[512, 1024]" = torch.ops.aten.view.default(view_433, [512, 1024]);  view_433 = None
    permute_217: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_316, [1, 0]);  primals_316 = None
    addmm_117: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_317, view_434, permute_217);  primals_317 = None
    view_435: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_117, [1, 512, 1024]);  addmm_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_59 = torch.ops.aten.native_dropout.default(view_435, 0.1, True);  view_435 = None
    getitem_196: "f32[1, 512, 1024]" = native_dropout_59[0]
    getitem_197: "b8[1, 512, 1024]" = native_dropout_59[1];  native_dropout_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_157: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_153, getitem_196);  add_153 = getitem_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_39 = torch.ops.aten.var_mean.correction(add_157, [2], correction = 0, keepdim = True)
    getitem_198: "f32[1, 512, 1]" = var_mean_39[0]
    getitem_199: "f32[1, 512, 1]" = var_mean_39[1];  var_mean_39 = None
    add_158: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_198, 1e-12);  getitem_198 = None
    rsqrt_39: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
    sub_60: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_157, getitem_199);  getitem_199 = None
    mul_136: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_39);  sub_60 = None
    mul_137: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_136, primals_318)
    add_159: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_137, primals_319);  mul_137 = primals_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_436: "f32[512, 1024]" = torch.ops.aten.view.default(add_159, [512, 1024]);  add_159 = None
    permute_218: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_320, [1, 0]);  primals_320 = None
    addmm_118: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_321, view_436, permute_218);  primals_321 = None
    view_437: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_118, [1, 512, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_138: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_437, 0.5)
    mul_139: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_437, 0.7071067811865476);  view_437 = None
    erf_19: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_139);  mul_139 = None
    add_160: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_140: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_138, add_160);  mul_138 = add_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_438: "f32[512, 4096]" = torch.ops.aten.view.default(mul_140, [512, 4096]);  mul_140 = None
    permute_219: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_322, [1, 0]);  primals_322 = None
    addmm_119: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_323, view_438, permute_219);  primals_323 = None
    view_439: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_119, [1, 512, 1024]);  addmm_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_60 = torch.ops.aten.native_dropout.default(view_439, 0.1, True);  view_439 = None
    getitem_200: "f32[1, 512, 1024]" = native_dropout_60[0]
    getitem_201: "b8[1, 512, 1024]" = native_dropout_60[1];  native_dropout_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_161: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_157, getitem_200);  add_157 = getitem_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_40 = torch.ops.aten.var_mean.correction(add_161, [2], correction = 0, keepdim = True)
    getitem_202: "f32[1, 512, 1]" = var_mean_40[0]
    getitem_203: "f32[1, 512, 1]" = var_mean_40[1];  var_mean_40 = None
    add_162: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_202, 1e-12);  getitem_202 = None
    rsqrt_40: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
    sub_61: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_161, getitem_203);  getitem_203 = None
    mul_141: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_40);  sub_61 = None
    mul_142: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_141, primals_324)
    add_163: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_142, primals_325);  mul_142 = primals_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_440: "f32[512, 1024]" = torch.ops.aten.view.default(add_163, [512, 1024]);  add_163 = None
    permute_220: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_326, [1, 0]);  primals_326 = None
    addmm_120: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_327, view_440, permute_220);  primals_327 = None
    view_441: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_120, [1, 512, 1024]);  addmm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_221: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_328, [1, 0]);  primals_328 = None
    addmm_121: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_329, view_440, permute_221);  primals_329 = None
    view_443: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_121, [1, 512, 1024]);  addmm_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_444: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_443, [1, 512, 16, 64]);  view_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_222: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_444, [0, 2, 1, 3]);  view_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_223: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_330, [1, 0]);  primals_330 = None
    addmm_122: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_331, view_440, permute_223);  primals_331 = None
    view_446: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_122, [1, 512, 1024]);  addmm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_447: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_446, [1, 512, 16, 64]);  view_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_224: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_447, [0, 2, 1, 3]);  view_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_448: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_441, [1, 512, 16, 64]);  view_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_225: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None
    
    # No stacktrace found for following nodes
    clone_default_12: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_225, memory_format = torch.contiguous_format);  permute_225 = None
    clone_default_13: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_222, memory_format = torch.contiguous_format);  permute_222 = None
    clone_default_14: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_224, memory_format = torch.contiguous_format);  permute_224 = None
    mul_scalar_12: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_12, 0.3535533905932738);  clone_default_12 = None
    permute_default_18: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(clone_default_13, [0, 1, 3, 2]);  clone_default_13 = None
    mul_scalar_13: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_18, 0.3535533905932738);  permute_default_18 = None
    expand_default_12: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_12, [1, 16, 512, 64]);  mul_scalar_12 = None
    view_default_36: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_12, [16, 512, 64]);  expand_default_12 = None
    expand_default_13: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_13, [1, 16, 64, 512]);  mul_scalar_13 = None
    view_default_37: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_default_13, [16, 64, 512]);  expand_default_13 = None
    bmm_default_18: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_36, view_default_37)
    view_default_38: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_18, [1, 16, 512, 512]);  bmm_default_18 = None
    amax_default_3: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(view_default_38, [-1], True)
    sub_tensor_6: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_38, amax_default_3);  view_default_38 = amax_default_3 = None
    exp_default_3: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_6);  sub_tensor_6 = None
    sum_dim_int_list_6: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_3, [-1], True)
    div_tensor_3: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_3, sum_dim_int_list_6);  exp_default_3 = sum_dim_int_list_6 = None
    alias_default_6: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_tensor_3)
    native_dropout_default_3 = torch.ops.aten.native_dropout.default(div_tensor_3, 0.1, True);  div_tensor_3 = None
    getitem_252: "f32[1, 16, 512, 512]" = native_dropout_default_3[0]
    getitem_253: "b8[1, 16, 512, 512]" = native_dropout_default_3[1];  native_dropout_default_3 = None
    expand_default_14: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_252, [1, 16, 512, 512]);  getitem_252 = None
    view_default_39: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_default_14, [16, 512, 512]);  expand_default_14 = None
    expand_default_15: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(clone_default_14, [1, 16, 512, 64]);  clone_default_14 = None
    view_default_40: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_15, [16, 512, 64]);  expand_default_15 = None
    bmm_default_19: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_39, view_default_40)
    view_default_41: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_19, [1, 16, 512, 64]);  bmm_default_19 = None
    permute_default_19: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_default_39, [0, 2, 1]);  view_default_39 = None
    permute_default_20: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_40, [0, 2, 1]);  view_default_40 = None
    alias_default_7: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_default_6);  alias_default_6 = None
    permute_default_21: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_36, [0, 2, 1]);  view_default_36 = None
    permute_default_22: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_default_37, [0, 2, 1]);  view_default_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_227: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_41, [0, 2, 1, 3]);  view_default_41 = None
    clone_20: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_455: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_20, [1, 512, 1024]);  clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_456: "f32[512, 1024]" = torch.ops.aten.view.default(view_455, [512, 1024]);  view_455 = None
    permute_228: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_332, [1, 0]);  primals_332 = None
    addmm_123: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_333, view_456, permute_228);  primals_333 = None
    view_457: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_123, [1, 512, 1024]);  addmm_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_62 = torch.ops.aten.native_dropout.default(view_457, 0.1, True);  view_457 = None
    getitem_206: "f32[1, 512, 1024]" = native_dropout_62[0]
    getitem_207: "b8[1, 512, 1024]" = native_dropout_62[1];  native_dropout_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_165: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_161, getitem_206);  add_161 = getitem_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_41 = torch.ops.aten.var_mean.correction(add_165, [2], correction = 0, keepdim = True)
    getitem_208: "f32[1, 512, 1]" = var_mean_41[0]
    getitem_209: "f32[1, 512, 1]" = var_mean_41[1];  var_mean_41 = None
    add_166: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_208, 1e-12);  getitem_208 = None
    rsqrt_41: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
    sub_63: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_165, getitem_209);  getitem_209 = None
    mul_143: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_41);  sub_63 = None
    mul_144: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_143, primals_334)
    add_167: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_144, primals_335);  mul_144 = primals_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_458: "f32[512, 1024]" = torch.ops.aten.view.default(add_167, [512, 1024]);  add_167 = None
    permute_229: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_336, [1, 0]);  primals_336 = None
    addmm_124: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_337, view_458, permute_229);  primals_337 = None
    view_459: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_124, [1, 512, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_145: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_459, 0.5)
    mul_146: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_459, 0.7071067811865476);  view_459 = None
    erf_20: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_146);  mul_146 = None
    add_168: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_147: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_145, add_168);  mul_145 = add_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_460: "f32[512, 4096]" = torch.ops.aten.view.default(mul_147, [512, 4096]);  mul_147 = None
    permute_230: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_338, [1, 0]);  primals_338 = None
    addmm_125: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_339, view_460, permute_230);  primals_339 = None
    view_461: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_125, [1, 512, 1024]);  addmm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_63 = torch.ops.aten.native_dropout.default(view_461, 0.1, True);  view_461 = None
    getitem_210: "f32[1, 512, 1024]" = native_dropout_63[0]
    getitem_211: "b8[1, 512, 1024]" = native_dropout_63[1];  native_dropout_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_169: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_165, getitem_210);  add_165 = getitem_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_42 = torch.ops.aten.var_mean.correction(add_169, [2], correction = 0, keepdim = True)
    getitem_212: "f32[1, 512, 1]" = var_mean_42[0]
    getitem_213: "f32[1, 512, 1]" = var_mean_42[1];  var_mean_42 = None
    add_170: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_212, 1e-12);  getitem_212 = None
    rsqrt_42: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
    sub_64: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_169, getitem_213);  getitem_213 = None
    mul_148: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_42);  sub_64 = None
    mul_149: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_148, primals_340)
    add_171: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_149, primals_341);  mul_149 = primals_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_462: "f32[512, 1024]" = torch.ops.aten.view.default(add_171, [512, 1024]);  add_171 = None
    permute_231: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_342, [1, 0]);  primals_342 = None
    addmm_126: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_343, view_462, permute_231);  primals_343 = None
    view_463: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_126, [1, 512, 1024]);  addmm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_232: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_344, [1, 0]);  primals_344 = None
    addmm_127: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_345, view_462, permute_232);  primals_345 = None
    view_465: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_127, [1, 512, 1024]);  addmm_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_466: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_465, [1, 512, 16, 64]);  view_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_233: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_466, [0, 2, 1, 3]);  view_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_234: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_346, [1, 0]);  primals_346 = None
    addmm_128: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_347, view_462, permute_234);  primals_347 = None
    view_468: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_128, [1, 512, 1024]);  addmm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_469: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_468, [1, 512, 16, 64]);  view_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_235: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_469, [0, 2, 1, 3]);  view_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_470: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_463, [1, 512, 16, 64]);  view_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_236: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_470, [0, 2, 1, 3]);  view_470 = None
    
    # No stacktrace found for following nodes
    clone_default_8: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_236, memory_format = torch.contiguous_format);  permute_236 = None
    clone_default_9: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_233, memory_format = torch.contiguous_format);  permute_233 = None
    clone_default_10: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_235, memory_format = torch.contiguous_format);  permute_235 = None
    mul_scalar_8: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_8, 0.3535533905932738);  clone_default_8 = None
    permute_default_12: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(clone_default_9, [0, 1, 3, 2]);  clone_default_9 = None
    mul_scalar_9: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_12, 0.3535533905932738);  permute_default_12 = None
    expand_default_8: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_8, [1, 16, 512, 64]);  mul_scalar_8 = None
    view_default_24: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_8, [16, 512, 64]);  expand_default_8 = None
    expand_default_9: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_9, [1, 16, 64, 512]);  mul_scalar_9 = None
    view_default_25: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_default_9, [16, 64, 512]);  expand_default_9 = None
    bmm_default_12: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_24, view_default_25)
    view_default_26: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_12, [1, 16, 512, 512]);  bmm_default_12 = None
    amax_default_2: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(view_default_26, [-1], True)
    sub_tensor_4: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_26, amax_default_2);  view_default_26 = amax_default_2 = None
    exp_default_2: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_4);  sub_tensor_4 = None
    sum_dim_int_list_4: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_2, [-1], True)
    div_tensor_2: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_2, sum_dim_int_list_4);  exp_default_2 = sum_dim_int_list_4 = None
    alias_default_4: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_tensor_2)
    native_dropout_default_2 = torch.ops.aten.native_dropout.default(div_tensor_2, 0.1, True);  div_tensor_2 = None
    getitem_250: "f32[1, 16, 512, 512]" = native_dropout_default_2[0]
    getitem_251: "b8[1, 16, 512, 512]" = native_dropout_default_2[1];  native_dropout_default_2 = None
    expand_default_10: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_250, [1, 16, 512, 512]);  getitem_250 = None
    view_default_27: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_default_10, [16, 512, 512]);  expand_default_10 = None
    expand_default_11: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(clone_default_10, [1, 16, 512, 64]);  clone_default_10 = None
    view_default_28: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_11, [16, 512, 64]);  expand_default_11 = None
    bmm_default_13: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_27, view_default_28)
    view_default_29: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_13, [1, 16, 512, 64]);  bmm_default_13 = None
    permute_default_13: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_default_27, [0, 2, 1]);  view_default_27 = None
    permute_default_14: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_28, [0, 2, 1]);  view_default_28 = None
    alias_default_5: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_default_4);  alias_default_4 = None
    permute_default_15: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_24, [0, 2, 1]);  view_default_24 = None
    permute_default_16: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_default_25, [0, 2, 1]);  view_default_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_238: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_29, [0, 2, 1, 3]);  view_default_29 = None
    clone_21: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_477: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_21, [1, 512, 1024]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_478: "f32[512, 1024]" = torch.ops.aten.view.default(view_477, [512, 1024]);  view_477 = None
    permute_239: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_348, [1, 0]);  primals_348 = None
    addmm_129: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_349, view_478, permute_239);  primals_349 = None
    view_479: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_129, [1, 512, 1024]);  addmm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_65 = torch.ops.aten.native_dropout.default(view_479, 0.1, True);  view_479 = None
    getitem_216: "f32[1, 512, 1024]" = native_dropout_65[0]
    getitem_217: "b8[1, 512, 1024]" = native_dropout_65[1];  native_dropout_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_173: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_169, getitem_216);  add_169 = getitem_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_43 = torch.ops.aten.var_mean.correction(add_173, [2], correction = 0, keepdim = True)
    getitem_218: "f32[1, 512, 1]" = var_mean_43[0]
    getitem_219: "f32[1, 512, 1]" = var_mean_43[1];  var_mean_43 = None
    add_174: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_218, 1e-12);  getitem_218 = None
    rsqrt_43: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
    sub_66: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_173, getitem_219);  getitem_219 = None
    mul_150: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_43);  sub_66 = None
    mul_151: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_150, primals_350)
    add_175: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_151, primals_351);  mul_151 = primals_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_480: "f32[512, 1024]" = torch.ops.aten.view.default(add_175, [512, 1024]);  add_175 = None
    permute_240: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_352, [1, 0]);  primals_352 = None
    addmm_130: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_353, view_480, permute_240);  primals_353 = None
    view_481: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_130, [1, 512, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_152: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_481, 0.5)
    mul_153: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_481, 0.7071067811865476);  view_481 = None
    erf_21: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_153);  mul_153 = None
    add_176: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_154: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_152, add_176);  mul_152 = add_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_482: "f32[512, 4096]" = torch.ops.aten.view.default(mul_154, [512, 4096]);  mul_154 = None
    permute_241: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_354, [1, 0]);  primals_354 = None
    addmm_131: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_355, view_482, permute_241);  primals_355 = None
    view_483: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_131, [1, 512, 1024]);  addmm_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_66 = torch.ops.aten.native_dropout.default(view_483, 0.1, True);  view_483 = None
    getitem_220: "f32[1, 512, 1024]" = native_dropout_66[0]
    getitem_221: "b8[1, 512, 1024]" = native_dropout_66[1];  native_dropout_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_177: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_173, getitem_220);  add_173 = getitem_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_44 = torch.ops.aten.var_mean.correction(add_177, [2], correction = 0, keepdim = True)
    getitem_222: "f32[1, 512, 1]" = var_mean_44[0]
    getitem_223: "f32[1, 512, 1]" = var_mean_44[1];  var_mean_44 = None
    add_178: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_222, 1e-12);  getitem_222 = None
    rsqrt_44: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
    sub_67: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_177, getitem_223);  getitem_223 = None
    mul_155: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_44);  sub_67 = None
    mul_156: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_155, primals_356)
    add_179: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_156, primals_357);  mul_156 = primals_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_484: "f32[512, 1024]" = torch.ops.aten.view.default(add_179, [512, 1024]);  add_179 = None
    permute_242: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_358, [1, 0]);  primals_358 = None
    addmm_132: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_359, view_484, permute_242);  primals_359 = None
    view_485: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_132, [1, 512, 1024]);  addmm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_243: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_360, [1, 0]);  primals_360 = None
    addmm_133: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_361, view_484, permute_243);  primals_361 = None
    view_487: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_133, [1, 512, 1024]);  addmm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_488: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_487, [1, 512, 16, 64]);  view_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_244: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_488, [0, 2, 1, 3]);  view_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_245: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_362, [1, 0]);  primals_362 = None
    addmm_134: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_363, view_484, permute_245);  primals_363 = None
    view_490: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_134, [1, 512, 1024]);  addmm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_491: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_490, [1, 512, 16, 64]);  view_490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_246: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_491, [0, 2, 1, 3]);  view_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_492: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_485, [1, 512, 16, 64]);  view_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_247: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_492, [0, 2, 1, 3]);  view_492 = None
    
    # No stacktrace found for following nodes
    clone_default_4: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_247, memory_format = torch.contiguous_format);  permute_247 = None
    clone_default_5: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_244, memory_format = torch.contiguous_format);  permute_244 = None
    clone_default_6: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_246, memory_format = torch.contiguous_format);  permute_246 = None
    mul_scalar_4: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_4, 0.3535533905932738);  clone_default_4 = None
    permute_default_6: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(clone_default_5, [0, 1, 3, 2]);  clone_default_5 = None
    mul_scalar_5: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_6, 0.3535533905932738);  permute_default_6 = None
    expand_default_4: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_4, [1, 16, 512, 64]);  mul_scalar_4 = None
    view_default_12: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_4, [16, 512, 64]);  expand_default_4 = None
    expand_default_5: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_5, [1, 16, 64, 512]);  mul_scalar_5 = None
    view_default_13: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_default_5, [16, 64, 512]);  expand_default_5 = None
    bmm_default_6: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_12, view_default_13)
    view_default_14: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_6, [1, 16, 512, 512]);  bmm_default_6 = None
    amax_default_1: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(view_default_14, [-1], True)
    sub_tensor_2: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_14, amax_default_1);  view_default_14 = amax_default_1 = None
    exp_default_1: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_2);  sub_tensor_2 = None
    sum_dim_int_list_2: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_1, [-1], True)
    div_tensor_1: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_1, sum_dim_int_list_2);  exp_default_1 = sum_dim_int_list_2 = None
    alias_default_2: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_tensor_1)
    native_dropout_default_1 = torch.ops.aten.native_dropout.default(div_tensor_1, 0.1, True);  div_tensor_1 = None
    getitem_248: "f32[1, 16, 512, 512]" = native_dropout_default_1[0]
    getitem_249: "b8[1, 16, 512, 512]" = native_dropout_default_1[1];  native_dropout_default_1 = None
    expand_default_6: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_248, [1, 16, 512, 512]);  getitem_248 = None
    view_default_15: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_default_6, [16, 512, 512]);  expand_default_6 = None
    expand_default_7: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(clone_default_6, [1, 16, 512, 64]);  clone_default_6 = None
    view_default_16: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_7, [16, 512, 64]);  expand_default_7 = None
    bmm_default_7: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_15, view_default_16)
    view_default_17: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_7, [1, 16, 512, 64]);  bmm_default_7 = None
    permute_default_7: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_default_15, [0, 2, 1]);  view_default_15 = None
    permute_default_8: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_16, [0, 2, 1]);  view_default_16 = None
    alias_default_3: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_default_2);  alias_default_2 = None
    permute_default_9: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_12, [0, 2, 1]);  view_default_12 = None
    permute_default_10: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_default_13, [0, 2, 1]);  view_default_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_249: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_17, [0, 2, 1, 3]);  view_default_17 = None
    clone_22: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_499: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_22, [1, 512, 1024]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_500: "f32[512, 1024]" = torch.ops.aten.view.default(view_499, [512, 1024]);  view_499 = None
    permute_250: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_364, [1, 0]);  primals_364 = None
    addmm_135: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_365, view_500, permute_250);  primals_365 = None
    view_501: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_135, [1, 512, 1024]);  addmm_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_68 = torch.ops.aten.native_dropout.default(view_501, 0.1, True);  view_501 = None
    getitem_226: "f32[1, 512, 1024]" = native_dropout_68[0]
    getitem_227: "b8[1, 512, 1024]" = native_dropout_68[1];  native_dropout_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_181: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_177, getitem_226);  add_177 = getitem_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_45 = torch.ops.aten.var_mean.correction(add_181, [2], correction = 0, keepdim = True)
    getitem_228: "f32[1, 512, 1]" = var_mean_45[0]
    getitem_229: "f32[1, 512, 1]" = var_mean_45[1];  var_mean_45 = None
    add_182: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_228, 1e-12);  getitem_228 = None
    rsqrt_45: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
    sub_69: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_181, getitem_229);  getitem_229 = None
    mul_157: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_45);  sub_69 = None
    mul_158: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_157, primals_366)
    add_183: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_158, primals_367);  mul_158 = primals_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_502: "f32[512, 1024]" = torch.ops.aten.view.default(add_183, [512, 1024]);  add_183 = None
    permute_251: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_368, [1, 0]);  primals_368 = None
    addmm_136: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_369, view_502, permute_251);  primals_369 = None
    view_503: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_136, [1, 512, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_159: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_503, 0.5)
    mul_160: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_503, 0.7071067811865476);  view_503 = None
    erf_22: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_160);  mul_160 = None
    add_184: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_161: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_159, add_184);  mul_159 = add_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_504: "f32[512, 4096]" = torch.ops.aten.view.default(mul_161, [512, 4096]);  mul_161 = None
    permute_252: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_370, [1, 0]);  primals_370 = None
    addmm_137: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_371, view_504, permute_252);  primals_371 = None
    view_505: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_137, [1, 512, 1024]);  addmm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_69 = torch.ops.aten.native_dropout.default(view_505, 0.1, True);  view_505 = None
    getitem_230: "f32[1, 512, 1024]" = native_dropout_69[0]
    getitem_231: "b8[1, 512, 1024]" = native_dropout_69[1];  native_dropout_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_185: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_181, getitem_230);  add_181 = getitem_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_46 = torch.ops.aten.var_mean.correction(add_185, [2], correction = 0, keepdim = True)
    getitem_232: "f32[1, 512, 1]" = var_mean_46[0]
    getitem_233: "f32[1, 512, 1]" = var_mean_46[1];  var_mean_46 = None
    add_186: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_232, 1e-12);  getitem_232 = None
    rsqrt_46: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
    sub_70: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_185, getitem_233);  getitem_233 = None
    mul_162: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_46);  sub_70 = None
    mul_163: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_162, primals_372)
    add_187: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_163, primals_373);  mul_163 = primals_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_506: "f32[512, 1024]" = torch.ops.aten.view.default(add_187, [512, 1024]);  add_187 = None
    permute_253: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_374, [1, 0]);  primals_374 = None
    addmm_138: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_375, view_506, permute_253);  primals_375 = None
    view_507: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_138, [1, 512, 1024]);  addmm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_254: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_376, [1, 0]);  primals_376 = None
    addmm_139: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_377, view_506, permute_254);  primals_377 = None
    view_509: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_139, [1, 512, 1024]);  addmm_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_510: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_509, [1, 512, 16, 64]);  view_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_255: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_510, [0, 2, 1, 3]);  view_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_256: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_378, [1, 0]);  primals_378 = None
    addmm_140: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_379, view_506, permute_256);  primals_379 = None
    view_512: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_140, [1, 512, 1024]);  addmm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_513: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_512, [1, 512, 16, 64]);  view_512 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_257: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_513, [0, 2, 1, 3]);  view_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_514: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_507, [1, 512, 16, 64]);  view_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_258: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_514, [0, 2, 1, 3]);  view_514 = None
    
    # No stacktrace found for following nodes
    clone_default: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_258, memory_format = torch.contiguous_format);  permute_258 = None
    clone_default_1: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_255, memory_format = torch.contiguous_format);  permute_255 = None
    clone_default_2: "f32[1, 16, 512, 64]" = torch.ops.aten.clone.default(permute_257, memory_format = torch.contiguous_format);  permute_257 = None
    mul_scalar: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default, 0.3535533905932738);  clone_default = None
    permute_default: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(clone_default_1, [0, 1, 3, 2]);  clone_default_1 = None
    mul_scalar_1: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default, 0.3535533905932738);  permute_default = None
    expand_default: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_scalar, [1, 16, 512, 64]);  mul_scalar = None
    view_default: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default, [16, 512, 64]);  expand_default = None
    expand_default_1: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_1, [1, 16, 64, 512]);  mul_scalar_1 = None
    view_default_1: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_default_1, [16, 64, 512]);  expand_default_1 = None
    bmm_default: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default, view_default_1)
    view_default_2: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default, [1, 16, 512, 512]);  bmm_default = None
    amax_default: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(view_default_2, [-1], True)
    sub_tensor: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_2, amax_default);  view_default_2 = amax_default = None
    exp_default: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
    sum_dim_int_list: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
    div_tensor: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
    alias_default: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_tensor)
    native_dropout_default = torch.ops.aten.native_dropout.default(div_tensor, 0.1, True);  div_tensor = None
    getitem_246: "f32[1, 16, 512, 512]" = native_dropout_default[0]
    getitem_247: "b8[1, 16, 512, 512]" = native_dropout_default[1];  native_dropout_default = None
    expand_default_2: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_246, [1, 16, 512, 512]);  getitem_246 = None
    view_default_3: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_default_2, [16, 512, 512]);  expand_default_2 = None
    expand_default_3: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(clone_default_2, [1, 16, 512, 64]);  clone_default_2 = None
    view_default_4: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_default_3, [16, 512, 64]);  expand_default_3 = None
    bmm_default_1: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_3, view_default_4)
    view_default_5: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_1, [1, 16, 512, 64]);  bmm_default_1 = None
    permute_default_1: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_default_3, [0, 2, 1]);  view_default_3 = None
    permute_default_2: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default_4, [0, 2, 1]);  view_default_4 = None
    alias_default_1: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_default);  alias_default = None
    permute_default_3: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_default, [0, 2, 1]);  view_default = None
    permute_default_4: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_default_1, [0, 2, 1]);  view_default_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_260: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_5, [0, 2, 1, 3]);  view_default_5 = None
    clone_23: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_521: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_23, [1, 512, 1024]);  clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_522: "f32[512, 1024]" = torch.ops.aten.view.default(view_521, [512, 1024]);  view_521 = None
    permute_261: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_380, [1, 0]);  primals_380 = None
    addmm_141: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_381, view_522, permute_261);  primals_381 = None
    view_523: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_141, [1, 512, 1024]);  addmm_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_71 = torch.ops.aten.native_dropout.default(view_523, 0.1, True);  view_523 = None
    getitem_236: "f32[1, 512, 1024]" = native_dropout_71[0]
    getitem_237: "b8[1, 512, 1024]" = native_dropout_71[1];  native_dropout_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_189: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_185, getitem_236);  add_185 = getitem_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_47 = torch.ops.aten.var_mean.correction(add_189, [2], correction = 0, keepdim = True)
    getitem_238: "f32[1, 512, 1]" = var_mean_47[0]
    getitem_239: "f32[1, 512, 1]" = var_mean_47[1];  var_mean_47 = None
    add_190: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_238, 1e-12);  getitem_238 = None
    rsqrt_47: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_190);  add_190 = None
    sub_72: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_189, getitem_239);  getitem_239 = None
    mul_164: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_47);  sub_72 = None
    mul_165: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_164, primals_382)
    add_191: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_165, primals_383);  mul_165 = primals_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_524: "f32[512, 1024]" = torch.ops.aten.view.default(add_191, [512, 1024]);  add_191 = None
    permute_262: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_384, [1, 0]);  primals_384 = None
    addmm_142: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_385, view_524, permute_262);  primals_385 = None
    view_525: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_142, [1, 512, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_166: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_525, 0.5)
    mul_167: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_525, 0.7071067811865476);  view_525 = None
    erf_23: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_167);  mul_167 = None
    add_192: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_168: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_166, add_192);  mul_166 = add_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_526: "f32[512, 4096]" = torch.ops.aten.view.default(mul_168, [512, 4096]);  mul_168 = None
    permute_263: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_386, [1, 0]);  primals_386 = None
    addmm_143: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_387, view_526, permute_263);  primals_387 = None
    view_527: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_143, [1, 512, 1024]);  addmm_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_72 = torch.ops.aten.native_dropout.default(view_527, 0.1, True);  view_527 = None
    getitem_240: "f32[1, 512, 1024]" = native_dropout_72[0]
    getitem_241: "b8[1, 512, 1024]" = native_dropout_72[1];  native_dropout_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_193: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_189, getitem_240);  add_189 = getitem_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:592, code: hidden_states = self.ln(hidden_states)
    var_mean_48 = torch.ops.aten.var_mean.correction(add_193, [2], correction = 0, keepdim = True)
    getitem_242: "f32[1, 512, 1]" = var_mean_48[0]
    getitem_243: "f32[1, 512, 1]" = var_mean_48[1];  var_mean_48 = None
    add_194: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_242, 1e-12);  getitem_242 = None
    rsqrt_48: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_194);  add_194 = None
    sub_73: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_193, getitem_243);  add_193 = getitem_243 = None
    mul_169: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_48);  sub_73 = None
    mul_170: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_169, primals_388)
    add_195: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_170, primals_389);  mul_170 = primals_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:1804, code: logits = self.qa_outputs(sequence_output)
    view_528: "f32[512, 1024]" = torch.ops.aten.view.default(add_195, [512, 1024]);  add_195 = None
    permute_264: "f32[1024, 2]" = torch.ops.aten.permute.default(primals_390, [1, 0]);  primals_390 = None
    addmm_144: "f32[512, 2]" = torch.ops.aten.addmm.default(primals_391, view_528, permute_264);  primals_391 = None
    view_529: "f32[1, 512, 2]" = torch.ops.aten.view.default(addmm_144, [1, 512, 2]);  addmm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:1805, code: start_logits, end_logits = logits.split(1, dim=-1)
    split_with_sizes = torch.ops.aten.split_with_sizes.default(view_529, [1, 1], 2);  view_529 = None
    getitem_244: "f32[1, 512, 1]" = split_with_sizes[0]
    getitem_245: "f32[1, 512, 1]" = split_with_sizes[1];  split_with_sizes = None
    
    # No stacktrace found for following nodes
    squeeze: "f32[1, 512]" = torch.ops.aten.squeeze.dim(getitem_244, -1);  getitem_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:1806, code: start_logits = start_logits.squeeze(-1).contiguous()
    clone_24: "f32[1, 512]" = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
    
    # No stacktrace found for following nodes
    squeeze_1: "f32[1, 512]" = torch.ops.aten.squeeze.dim(getitem_245, -1);  getitem_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:1807, code: end_logits = end_logits.squeeze(-1).contiguous()
    clone_25: "f32[1, 512]" = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:1818, code: start_positions = start_positions.clamp(0, ignored_index)
    clamp_min: "i64[1]" = torch.ops.aten.clamp_min.default(primals_394, 0);  primals_394 = None
    clamp_max: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min, 512);  clamp_min = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:1819, code: end_positions = end_positions.clamp(0, ignored_index)
    clamp_min_1: "i64[1]" = torch.ops.aten.clamp_min.default(primals_395, 0);  primals_395 = None
    clamp_max_1: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min_1, 512);  clamp_min_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:1822, code: start_loss = loss_fct(start_logits, start_positions)
    amax_24: "f32[1, 1]" = torch.ops.aten.amax.default(clone_24, [1], True)
    sub_74: "f32[1, 512]" = torch.ops.aten.sub.Tensor(clone_24, amax_24);  amax_24 = None
    exp_24: "f32[1, 512]" = torch.ops.aten.exp.default(sub_74)
    sum_25: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
    log: "f32[1, 1]" = torch.ops.aten.log.default(sum_25);  sum_25 = None
    sub_75: "f32[1, 512]" = torch.ops.aten.sub.Tensor(sub_74, log);  sub_74 = log = None
    ne: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max, 512)
    full_default_2: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where: "i64[1]" = torch.ops.aten.where.self(ne, clamp_max, full_default_2)
    unsqueeze_2: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
    gather: "f32[1, 1]" = torch.ops.aten.gather.default(sub_75, 1, unsqueeze_2);  unsqueeze_2 = None
    squeeze_2: "f32[1]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[1]" = torch.ops.aten.neg.default(squeeze_2);  squeeze_2 = None
    full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_1: "f32[1]" = torch.ops.aten.where.self(ne, neg, full_default_3);  neg = None
    sum_26: "i64[]" = torch.ops.aten.sum.default(ne)
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
    sum_27: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
    div_48: "f32[]" = torch.ops.aten.div.Tensor(sum_27, convert_element_type);  sum_27 = convert_element_type = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:1823, code: end_loss = loss_fct(end_logits, end_positions)
    amax_25: "f32[1, 1]" = torch.ops.aten.amax.default(clone_25, [1], True)
    sub_76: "f32[1, 512]" = torch.ops.aten.sub.Tensor(clone_25, amax_25);  amax_25 = None
    exp_25: "f32[1, 512]" = torch.ops.aten.exp.default(sub_76)
    sum_28: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp_25, [1], True);  exp_25 = None
    log_1: "f32[1, 1]" = torch.ops.aten.log.default(sum_28);  sum_28 = None
    sub_77: "f32[1, 512]" = torch.ops.aten.sub.Tensor(sub_76, log_1);  sub_76 = log_1 = None
    ne_3: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_1, 512)
    where_2: "i64[1]" = torch.ops.aten.where.self(ne_3, clamp_max_1, full_default_2)
    unsqueeze_3: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where_2, 1);  where_2 = None
    gather_1: "f32[1, 1]" = torch.ops.aten.gather.default(sub_77, 1, unsqueeze_3);  unsqueeze_3 = None
    squeeze_3: "f32[1]" = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
    neg_1: "f32[1]" = torch.ops.aten.neg.default(squeeze_3);  squeeze_3 = None
    where_3: "f32[1]" = torch.ops.aten.where.self(ne_3, neg_1, full_default_3);  neg_1 = full_default_3 = None
    sum_29: "i64[]" = torch.ops.aten.sum.default(ne_3)
    convert_element_type_1: "f32[]" = torch.ops.prims.convert_element_type.default(sum_29, torch.float32);  sum_29 = None
    sum_30: "f32[]" = torch.ops.aten.sum.default(where_3);  where_3 = None
    div_49: "f32[]" = torch.ops.aten.div.Tensor(sum_30, convert_element_type_1);  sum_30 = convert_element_type_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:1824, code: total_loss = (start_loss + end_loss) / 2
    add_196: "f32[]" = torch.ops.aten.add.Tensor(div_48, div_49);  div_48 = div_49 = None
    div_50: "f32[]" = torch.ops.aten.div.Tensor(add_196, 2);  add_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:1823, code: end_loss = loss_fct(end_logits, end_positions)
    unsqueeze_4: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(clamp_max_1, 1);  clamp_max_1 = None
    ne_6: "b8[1, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_4, 512)
    where_4: "i64[1, 1]" = torch.ops.aten.where.self(ne_6, unsqueeze_4, full_default_2);  unsqueeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:1822, code: start_loss = loss_fct(start_logits, start_positions)
    unsqueeze_5: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(clamp_max, 1);  clamp_max = None
    ne_8: "b8[1, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_5, 512)
    where_6: "i64[1, 1]" = torch.ops.aten.where.self(ne_8, unsqueeze_5, full_default_2);  unsqueeze_5 = full_default_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:1804, code: logits = self.qa_outputs(sequence_output)
    permute_265: "f32[2, 1024]" = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:592, code: hidden_states = self.ln(hidden_states)
    div_54: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_48, 1024);  rsqrt_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    permute_269: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    permute_273: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    div_55: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_47, 1024);  rsqrt_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    permute_277: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_261, [1, 0]);  permute_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_289: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_256, [1, 0]);  permute_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_294: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_254, [1, 0]);  permute_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    permute_298: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_253, [1, 0]);  permute_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    div_57: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_46, 1024);  rsqrt_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    permute_302: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    permute_306: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    div_58: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_45, 1024);  rsqrt_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    permute_310: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_250, [1, 0]);  permute_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_322: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_245, [1, 0]);  permute_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_327: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    permute_331: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    div_60: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 1024);  rsqrt_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    permute_335: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_241, [1, 0]);  permute_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    permute_339: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    div_61: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 1024);  rsqrt_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    permute_343: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_355: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_234, [1, 0]);  permute_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_360: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    permute_364: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_231, [1, 0]);  permute_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    div_63: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_42, 1024);  rsqrt_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    permute_368: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_230, [1, 0]);  permute_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    permute_372: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    div_64: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_41, 1024);  rsqrt_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    permute_376: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_388: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_223, [1, 0]);  permute_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_393: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_221, [1, 0]);  permute_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    permute_397: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    div_66: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 1024);  rsqrt_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    permute_401: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_219, [1, 0]);  permute_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    permute_405: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    div_67: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 1024);  rsqrt_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    permute_409: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_217, [1, 0]);  permute_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_421: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_212, [1, 0]);  permute_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_426: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    permute_430: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_209, [1, 0]);  permute_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    div_69: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 1024);  rsqrt_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    permute_434: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_208, [1, 0]);  permute_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    permute_438: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    div_70: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 1024);  rsqrt_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    permute_442: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_454: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_459: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_199, [1, 0]);  permute_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    permute_463: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_198, [1, 0]);  permute_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    div_72: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 1024);  rsqrt_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    permute_467: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    permute_471: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    div_73: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 1024);  rsqrt_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    permute_475: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_195, [1, 0]);  permute_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_487: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_190, [1, 0]);  permute_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_492: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_188, [1, 0]);  permute_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    permute_496: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    div_75: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 1024);  rsqrt_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    permute_500: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    permute_504: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    div_76: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 1024);  rsqrt_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    permute_508: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_520: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_179, [1, 0]);  permute_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_525: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    permute_529: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    div_78: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 1024);  rsqrt_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    permute_533: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_175, [1, 0]);  permute_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    permute_537: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    div_79: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 1024);  rsqrt_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    permute_541: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_553: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_168, [1, 0]);  permute_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_558: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_166, [1, 0]);  permute_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    permute_562: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    div_81: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 1024);  rsqrt_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    permute_566: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    permute_570: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    div_82: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 1024);  rsqrt_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    permute_574: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_162, [1, 0]);  permute_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_586: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_157, [1, 0]);  permute_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_591: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_155, [1, 0]);  permute_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    permute_595: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    div_84: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 1024);  rsqrt_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    permute_599: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    permute_603: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    div_85: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 1024);  rsqrt_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    permute_607: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_619: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_146, [1, 0]);  permute_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_624: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    permute_628: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    div_87: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 1024);  rsqrt_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    permute_632: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    permute_636: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    div_88: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 1024);  rsqrt_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    permute_640: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_652: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_135, [1, 0]);  permute_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_657: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    permute_661: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    div_90: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 1024);  rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    permute_665: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    permute_669: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    div_91: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 1024);  rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    permute_673: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_685: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_690: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    permute_694: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    div_93: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 1024);  rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    permute_698: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    permute_702: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    div_94: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 1024);  rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    permute_706: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_718: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_113, [1, 0]);  permute_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_723: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    permute_727: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    div_96: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 1024);  rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    permute_731: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    permute_735: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    div_97: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 1024);  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    permute_739: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_751: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_756: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    permute_760: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    div_99: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 1024);  rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    permute_764: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    permute_768: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    div_100: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 1024);  rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    permute_772: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_784: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_91, [1, 0]);  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_789: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    permute_793: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    div_102: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 1024);  rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    permute_797: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    permute_801: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    div_103: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 1024);  rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    permute_805: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_817: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_822: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    permute_826: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    div_105: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 1024);  rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    permute_830: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    permute_834: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    div_106: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 1024);  rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    permute_838: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_850: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_855: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    permute_859: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    div_108: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 1024);  rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    permute_863: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    permute_867: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    div_109: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 1024);  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    permute_871: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_883: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_888: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    permute_892: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    div_111: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 1024);  rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    permute_896: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    permute_900: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    div_112: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 1024);  rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    permute_904: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_916: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_921: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    permute_925: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    div_114: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 1024);  rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    permute_929: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    permute_933: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    div_115: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 1024);  rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    permute_937: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_949: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_954: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    permute_958: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    div_117: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 1024);  rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    permute_962: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    permute_966: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    div_118: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 1024);  rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    permute_970: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_982: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_987: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    permute_991: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    div_120: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 1024);  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    permute_995: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    permute_999: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    div_121: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 1024);  rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    permute_1003: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_1015: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_1020: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    permute_1024: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    div_123: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 1024);  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    permute_1028: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    permute_1032: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    div_124: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 1024);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    permute_1036: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_1048: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_1053: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    permute_1057: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    div_126: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 1024);  rsqrt = None
    return [div_50, clone_24, clone_25, primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_110, primals_116, primals_126, primals_132, primals_142, primals_148, primals_158, primals_164, primals_174, primals_180, primals_190, primals_196, primals_206, primals_212, primals_222, primals_228, primals_238, primals_244, primals_254, primals_260, primals_270, primals_276, primals_286, primals_292, primals_302, primals_308, primals_318, primals_324, primals_334, primals_340, primals_350, primals_356, primals_366, primals_372, primals_382, primals_388, primals_393, full_default, slice_3, getitem_1, mul_1, view, getitem_293, permute_default_139, permute_default_140, alias_default_47, permute_default_141, permute_default_142, view_16, getitem_7, mul_3, view_18, addmm_4, view_20, getitem_11, mul_8, view_22, getitem_291, permute_default_133, permute_default_134, alias_default_45, permute_default_135, permute_default_136, view_38, getitem_17, mul_10, view_40, addmm_10, view_42, getitem_21, mul_15, view_44, getitem_289, permute_default_127, permute_default_128, alias_default_43, permute_default_129, permute_default_130, view_60, getitem_27, mul_17, view_62, addmm_16, view_64, getitem_31, mul_22, view_66, getitem_287, permute_default_121, permute_default_122, alias_default_41, permute_default_123, permute_default_124, view_82, getitem_37, mul_24, view_84, addmm_22, view_86, getitem_41, mul_29, view_88, getitem_285, permute_default_115, permute_default_116, alias_default_39, permute_default_117, permute_default_118, view_104, getitem_47, mul_31, view_106, addmm_28, view_108, getitem_51, mul_36, view_110, getitem_283, permute_default_109, permute_default_110, alias_default_37, permute_default_111, permute_default_112, view_126, getitem_57, mul_38, view_128, addmm_34, view_130, getitem_61, mul_43, view_132, getitem_281, permute_default_103, permute_default_104, alias_default_35, permute_default_105, permute_default_106, view_148, getitem_67, mul_45, view_150, addmm_40, view_152, getitem_71, mul_50, view_154, getitem_279, permute_default_97, permute_default_98, alias_default_33, permute_default_99, permute_default_100, view_170, getitem_77, mul_52, view_172, addmm_46, view_174, getitem_81, mul_57, view_176, getitem_277, permute_default_91, permute_default_92, alias_default_31, permute_default_93, permute_default_94, view_192, getitem_87, mul_59, view_194, addmm_52, view_196, getitem_91, mul_64, view_198, getitem_275, permute_default_85, permute_default_86, alias_default_29, permute_default_87, permute_default_88, view_214, getitem_97, mul_66, view_216, addmm_58, view_218, getitem_101, mul_71, view_220, getitem_273, permute_default_79, permute_default_80, alias_default_27, permute_default_81, permute_default_82, view_236, getitem_107, mul_73, view_238, addmm_64, view_240, getitem_111, mul_78, view_242, getitem_271, permute_default_73, permute_default_74, alias_default_25, permute_default_75, permute_default_76, view_258, getitem_117, mul_80, view_260, addmm_70, view_262, getitem_121, mul_85, view_264, getitem_269, permute_default_67, permute_default_68, alias_default_23, permute_default_69, permute_default_70, view_280, getitem_127, mul_87, view_282, addmm_76, view_284, getitem_131, mul_92, view_286, getitem_267, permute_default_61, permute_default_62, alias_default_21, permute_default_63, permute_default_64, view_302, getitem_137, mul_94, view_304, addmm_82, view_306, getitem_141, mul_99, view_308, getitem_265, permute_default_55, permute_default_56, alias_default_19, permute_default_57, permute_default_58, view_324, getitem_147, mul_101, view_326, addmm_88, view_328, getitem_151, mul_106, view_330, getitem_263, permute_default_49, permute_default_50, alias_default_17, permute_default_51, permute_default_52, view_346, getitem_157, mul_108, view_348, addmm_94, view_350, getitem_161, mul_113, view_352, getitem_261, permute_default_43, permute_default_44, alias_default_15, permute_default_45, permute_default_46, view_368, getitem_167, mul_115, view_370, addmm_100, view_372, getitem_171, mul_120, view_374, getitem_259, permute_default_37, permute_default_38, alias_default_13, permute_default_39, permute_default_40, view_390, getitem_177, mul_122, view_392, addmm_106, view_394, getitem_181, mul_127, view_396, getitem_257, permute_default_31, permute_default_32, alias_default_11, permute_default_33, permute_default_34, view_412, getitem_187, mul_129, view_414, addmm_112, view_416, getitem_191, mul_134, view_418, getitem_255, permute_default_25, permute_default_26, alias_default_9, permute_default_27, permute_default_28, view_434, getitem_197, mul_136, view_436, addmm_118, view_438, getitem_201, mul_141, view_440, getitem_253, permute_default_19, permute_default_20, alias_default_7, permute_default_21, permute_default_22, view_456, getitem_207, mul_143, view_458, addmm_124, view_460, getitem_211, mul_148, view_462, getitem_251, permute_default_13, permute_default_14, alias_default_5, permute_default_15, permute_default_16, view_478, getitem_217, mul_150, view_480, addmm_130, view_482, getitem_221, mul_155, view_484, getitem_249, permute_default_7, permute_default_8, alias_default_3, permute_default_9, permute_default_10, view_500, getitem_227, mul_157, view_502, addmm_136, view_504, getitem_231, mul_162, view_506, getitem_247, permute_default_1, permute_default_2, alias_default_1, permute_default_3, permute_default_4, view_522, getitem_237, mul_164, view_524, addmm_142, view_526, getitem_241, mul_169, view_528, sub_75, ne, sub_77, ne_3, ne_6, where_4, ne_8, where_6, permute_265, div_54, permute_269, permute_273, div_55, permute_277, permute_289, permute_294, permute_298, div_57, permute_302, permute_306, div_58, permute_310, permute_322, permute_327, permute_331, div_60, permute_335, permute_339, div_61, permute_343, permute_355, permute_360, permute_364, div_63, permute_368, permute_372, div_64, permute_376, permute_388, permute_393, permute_397, div_66, permute_401, permute_405, div_67, permute_409, permute_421, permute_426, permute_430, div_69, permute_434, permute_438, div_70, permute_442, permute_454, permute_459, permute_463, div_72, permute_467, permute_471, div_73, permute_475, permute_487, permute_492, permute_496, div_75, permute_500, permute_504, div_76, permute_508, permute_520, permute_525, permute_529, div_78, permute_533, permute_537, div_79, permute_541, permute_553, permute_558, permute_562, div_81, permute_566, permute_570, div_82, permute_574, permute_586, permute_591, permute_595, div_84, permute_599, permute_603, div_85, permute_607, permute_619, permute_624, permute_628, div_87, permute_632, permute_636, div_88, permute_640, permute_652, permute_657, permute_661, div_90, permute_665, permute_669, div_91, permute_673, permute_685, permute_690, permute_694, div_93, permute_698, permute_702, div_94, permute_706, permute_718, permute_723, permute_727, div_96, permute_731, permute_735, div_97, permute_739, permute_751, permute_756, permute_760, div_99, permute_764, permute_768, div_100, permute_772, permute_784, permute_789, permute_793, div_102, permute_797, permute_801, div_103, permute_805, permute_817, permute_822, permute_826, div_105, permute_830, permute_834, div_106, permute_838, permute_850, permute_855, permute_859, div_108, permute_863, permute_867, div_109, permute_871, permute_883, permute_888, permute_892, div_111, permute_896, permute_900, div_112, permute_904, permute_916, permute_921, permute_925, div_114, permute_929, permute_933, div_115, permute_937, permute_949, permute_954, permute_958, div_117, permute_962, permute_966, div_118, permute_970, permute_982, permute_987, permute_991, div_120, permute_995, permute_999, div_121, permute_1003, permute_1015, permute_1020, permute_1024, div_123, permute_1028, permute_1032, div_124, permute_1036, permute_1048, permute_1053, permute_1057, div_126]
    