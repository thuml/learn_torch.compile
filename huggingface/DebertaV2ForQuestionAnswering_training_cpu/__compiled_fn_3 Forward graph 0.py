from __future__ import annotations



def forward(self, primals_1: "f32[128100, 1536]", primals_2: "f32[512, 1536]", primals_3: "f32[1536]", primals_4: "f32[1536]", primals_5: "f32[1536, 1536]", primals_6: "f32[1536]", primals_7: "f32[1536, 1536]", primals_8: "f32[1536]", primals_9: "f32[1536, 1536]", primals_10: "f32[1536]", primals_11: "f32[1536, 1536]", primals_12: "f32[1536]", primals_13: "f32[1536]", primals_14: "f32[1536]", primals_15: "f32[6144, 1536]", primals_16: "f32[6144]", primals_17: "f32[1536, 6144]", primals_18: "f32[1536]", primals_19: "f32[1536]", primals_20: "f32[1536]", primals_21: "f32[1536, 1536]", primals_22: "f32[1536]", primals_23: "f32[1536, 1536]", primals_24: "f32[1536]", primals_25: "f32[1536, 1536]", primals_26: "f32[1536]", primals_27: "f32[1536, 1536]", primals_28: "f32[1536]", primals_29: "f32[1536]", primals_30: "f32[1536]", primals_31: "f32[6144, 1536]", primals_32: "f32[6144]", primals_33: "f32[1536, 6144]", primals_34: "f32[1536]", primals_35: "f32[1536]", primals_36: "f32[1536]", primals_37: "f32[1536, 1536]", primals_38: "f32[1536]", primals_39: "f32[1536, 1536]", primals_40: "f32[1536]", primals_41: "f32[1536, 1536]", primals_42: "f32[1536]", primals_43: "f32[1536, 1536]", primals_44: "f32[1536]", primals_45: "f32[1536]", primals_46: "f32[1536]", primals_47: "f32[6144, 1536]", primals_48: "f32[6144]", primals_49: "f32[1536, 6144]", primals_50: "f32[1536]", primals_51: "f32[1536]", primals_52: "f32[1536]", primals_53: "f32[1536, 1536]", primals_54: "f32[1536]", primals_55: "f32[1536, 1536]", primals_56: "f32[1536]", primals_57: "f32[1536, 1536]", primals_58: "f32[1536]", primals_59: "f32[1536, 1536]", primals_60: "f32[1536]", primals_61: "f32[1536]", primals_62: "f32[1536]", primals_63: "f32[6144, 1536]", primals_64: "f32[6144]", primals_65: "f32[1536, 6144]", primals_66: "f32[1536]", primals_67: "f32[1536]", primals_68: "f32[1536]", primals_69: "f32[1536, 1536]", primals_70: "f32[1536]", primals_71: "f32[1536, 1536]", primals_72: "f32[1536]", primals_73: "f32[1536, 1536]", primals_74: "f32[1536]", primals_75: "f32[1536, 1536]", primals_76: "f32[1536]", primals_77: "f32[1536]", primals_78: "f32[1536]", primals_79: "f32[6144, 1536]", primals_80: "f32[6144]", primals_81: "f32[1536, 6144]", primals_82: "f32[1536]", primals_83: "f32[1536]", primals_84: "f32[1536]", primals_85: "f32[1536, 1536]", primals_86: "f32[1536]", primals_87: "f32[1536, 1536]", primals_88: "f32[1536]", primals_89: "f32[1536, 1536]", primals_90: "f32[1536]", primals_91: "f32[1536, 1536]", primals_92: "f32[1536]", primals_93: "f32[1536]", primals_94: "f32[1536]", primals_95: "f32[6144, 1536]", primals_96: "f32[6144]", primals_97: "f32[1536, 6144]", primals_98: "f32[1536]", primals_99: "f32[1536]", primals_100: "f32[1536]", primals_101: "f32[1536, 1536]", primals_102: "f32[1536]", primals_103: "f32[1536, 1536]", primals_104: "f32[1536]", primals_105: "f32[1536, 1536]", primals_106: "f32[1536]", primals_107: "f32[1536, 1536]", primals_108: "f32[1536]", primals_109: "f32[1536]", primals_110: "f32[1536]", primals_111: "f32[6144, 1536]", primals_112: "f32[6144]", primals_113: "f32[1536, 6144]", primals_114: "f32[1536]", primals_115: "f32[1536]", primals_116: "f32[1536]", primals_117: "f32[1536, 1536]", primals_118: "f32[1536]", primals_119: "f32[1536, 1536]", primals_120: "f32[1536]", primals_121: "f32[1536, 1536]", primals_122: "f32[1536]", primals_123: "f32[1536, 1536]", primals_124: "f32[1536]", primals_125: "f32[1536]", primals_126: "f32[1536]", primals_127: "f32[6144, 1536]", primals_128: "f32[6144]", primals_129: "f32[1536, 6144]", primals_130: "f32[1536]", primals_131: "f32[1536]", primals_132: "f32[1536]", primals_133: "f32[1536, 1536]", primals_134: "f32[1536]", primals_135: "f32[1536, 1536]", primals_136: "f32[1536]", primals_137: "f32[1536, 1536]", primals_138: "f32[1536]", primals_139: "f32[1536, 1536]", primals_140: "f32[1536]", primals_141: "f32[1536]", primals_142: "f32[1536]", primals_143: "f32[6144, 1536]", primals_144: "f32[6144]", primals_145: "f32[1536, 6144]", primals_146: "f32[1536]", primals_147: "f32[1536]", primals_148: "f32[1536]", primals_149: "f32[1536, 1536]", primals_150: "f32[1536]", primals_151: "f32[1536, 1536]", primals_152: "f32[1536]", primals_153: "f32[1536, 1536]", primals_154: "f32[1536]", primals_155: "f32[1536, 1536]", primals_156: "f32[1536]", primals_157: "f32[1536]", primals_158: "f32[1536]", primals_159: "f32[6144, 1536]", primals_160: "f32[6144]", primals_161: "f32[1536, 6144]", primals_162: "f32[1536]", primals_163: "f32[1536]", primals_164: "f32[1536]", primals_165: "f32[1536, 1536]", primals_166: "f32[1536]", primals_167: "f32[1536, 1536]", primals_168: "f32[1536]", primals_169: "f32[1536, 1536]", primals_170: "f32[1536]", primals_171: "f32[1536, 1536]", primals_172: "f32[1536]", primals_173: "f32[1536]", primals_174: "f32[1536]", primals_175: "f32[6144, 1536]", primals_176: "f32[6144]", primals_177: "f32[1536, 6144]", primals_178: "f32[1536]", primals_179: "f32[1536]", primals_180: "f32[1536]", primals_181: "f32[1536, 1536]", primals_182: "f32[1536]", primals_183: "f32[1536, 1536]", primals_184: "f32[1536]", primals_185: "f32[1536, 1536]", primals_186: "f32[1536]", primals_187: "f32[1536, 1536]", primals_188: "f32[1536]", primals_189: "f32[1536]", primals_190: "f32[1536]", primals_191: "f32[6144, 1536]", primals_192: "f32[6144]", primals_193: "f32[1536, 6144]", primals_194: "f32[1536]", primals_195: "f32[1536]", primals_196: "f32[1536]", primals_197: "f32[1536, 1536]", primals_198: "f32[1536]", primals_199: "f32[1536, 1536]", primals_200: "f32[1536]", primals_201: "f32[1536, 1536]", primals_202: "f32[1536]", primals_203: "f32[1536, 1536]", primals_204: "f32[1536]", primals_205: "f32[1536]", primals_206: "f32[1536]", primals_207: "f32[6144, 1536]", primals_208: "f32[6144]", primals_209: "f32[1536, 6144]", primals_210: "f32[1536]", primals_211: "f32[1536]", primals_212: "f32[1536]", primals_213: "f32[1536, 1536]", primals_214: "f32[1536]", primals_215: "f32[1536, 1536]", primals_216: "f32[1536]", primals_217: "f32[1536, 1536]", primals_218: "f32[1536]", primals_219: "f32[1536, 1536]", primals_220: "f32[1536]", primals_221: "f32[1536]", primals_222: "f32[1536]", primals_223: "f32[6144, 1536]", primals_224: "f32[6144]", primals_225: "f32[1536, 6144]", primals_226: "f32[1536]", primals_227: "f32[1536]", primals_228: "f32[1536]", primals_229: "f32[1536, 1536]", primals_230: "f32[1536]", primals_231: "f32[1536, 1536]", primals_232: "f32[1536]", primals_233: "f32[1536, 1536]", primals_234: "f32[1536]", primals_235: "f32[1536, 1536]", primals_236: "f32[1536]", primals_237: "f32[1536]", primals_238: "f32[1536]", primals_239: "f32[6144, 1536]", primals_240: "f32[6144]", primals_241: "f32[1536, 6144]", primals_242: "f32[1536]", primals_243: "f32[1536]", primals_244: "f32[1536]", primals_245: "f32[1536, 1536]", primals_246: "f32[1536]", primals_247: "f32[1536, 1536]", primals_248: "f32[1536]", primals_249: "f32[1536, 1536]", primals_250: "f32[1536]", primals_251: "f32[1536, 1536]", primals_252: "f32[1536]", primals_253: "f32[1536]", primals_254: "f32[1536]", primals_255: "f32[6144, 1536]", primals_256: "f32[6144]", primals_257: "f32[1536, 6144]", primals_258: "f32[1536]", primals_259: "f32[1536]", primals_260: "f32[1536]", primals_261: "f32[1536, 1536]", primals_262: "f32[1536]", primals_263: "f32[1536, 1536]", primals_264: "f32[1536]", primals_265: "f32[1536, 1536]", primals_266: "f32[1536]", primals_267: "f32[1536, 1536]", primals_268: "f32[1536]", primals_269: "f32[1536]", primals_270: "f32[1536]", primals_271: "f32[6144, 1536]", primals_272: "f32[6144]", primals_273: "f32[1536, 6144]", primals_274: "f32[1536]", primals_275: "f32[1536]", primals_276: "f32[1536]", primals_277: "f32[1536, 1536]", primals_278: "f32[1536]", primals_279: "f32[1536, 1536]", primals_280: "f32[1536]", primals_281: "f32[1536, 1536]", primals_282: "f32[1536]", primals_283: "f32[1536, 1536]", primals_284: "f32[1536]", primals_285: "f32[1536]", primals_286: "f32[1536]", primals_287: "f32[6144, 1536]", primals_288: "f32[6144]", primals_289: "f32[1536, 6144]", primals_290: "f32[1536]", primals_291: "f32[1536]", primals_292: "f32[1536]", primals_293: "f32[1536, 1536]", primals_294: "f32[1536]", primals_295: "f32[1536, 1536]", primals_296: "f32[1536]", primals_297: "f32[1536, 1536]", primals_298: "f32[1536]", primals_299: "f32[1536, 1536]", primals_300: "f32[1536]", primals_301: "f32[1536]", primals_302: "f32[1536]", primals_303: "f32[6144, 1536]", primals_304: "f32[6144]", primals_305: "f32[1536, 6144]", primals_306: "f32[1536]", primals_307: "f32[1536]", primals_308: "f32[1536]", primals_309: "f32[1536, 1536]", primals_310: "f32[1536]", primals_311: "f32[1536, 1536]", primals_312: "f32[1536]", primals_313: "f32[1536, 1536]", primals_314: "f32[1536]", primals_315: "f32[1536, 1536]", primals_316: "f32[1536]", primals_317: "f32[1536]", primals_318: "f32[1536]", primals_319: "f32[6144, 1536]", primals_320: "f32[6144]", primals_321: "f32[1536, 6144]", primals_322: "f32[1536]", primals_323: "f32[1536]", primals_324: "f32[1536]", primals_325: "f32[1536, 1536]", primals_326: "f32[1536]", primals_327: "f32[1536, 1536]", primals_328: "f32[1536]", primals_329: "f32[1536, 1536]", primals_330: "f32[1536]", primals_331: "f32[1536, 1536]", primals_332: "f32[1536]", primals_333: "f32[1536]", primals_334: "f32[1536]", primals_335: "f32[6144, 1536]", primals_336: "f32[6144]", primals_337: "f32[1536, 6144]", primals_338: "f32[1536]", primals_339: "f32[1536]", primals_340: "f32[1536]", primals_341: "f32[1536, 1536]", primals_342: "f32[1536]", primals_343: "f32[1536, 1536]", primals_344: "f32[1536]", primals_345: "f32[1536, 1536]", primals_346: "f32[1536]", primals_347: "f32[1536, 1536]", primals_348: "f32[1536]", primals_349: "f32[1536]", primals_350: "f32[1536]", primals_351: "f32[6144, 1536]", primals_352: "f32[6144]", primals_353: "f32[1536, 6144]", primals_354: "f32[1536]", primals_355: "f32[1536]", primals_356: "f32[1536]", primals_357: "f32[1536, 1536]", primals_358: "f32[1536]", primals_359: "f32[1536, 1536]", primals_360: "f32[1536]", primals_361: "f32[1536, 1536]", primals_362: "f32[1536]", primals_363: "f32[1536, 1536]", primals_364: "f32[1536]", primals_365: "f32[1536]", primals_366: "f32[1536]", primals_367: "f32[6144, 1536]", primals_368: "f32[6144]", primals_369: "f32[1536, 6144]", primals_370: "f32[1536]", primals_371: "f32[1536]", primals_372: "f32[1536]", primals_373: "f32[1536, 1536]", primals_374: "f32[1536]", primals_375: "f32[1536, 1536]", primals_376: "f32[1536]", primals_377: "f32[1536, 1536]", primals_378: "f32[1536]", primals_379: "f32[1536, 1536]", primals_380: "f32[1536]", primals_381: "f32[1536]", primals_382: "f32[1536]", primals_383: "f32[6144, 1536]", primals_384: "f32[6144]", primals_385: "f32[1536, 6144]", primals_386: "f32[1536]", primals_387: "f32[1536]", primals_388: "f32[1536]", primals_389: "f32[2, 1536]", primals_390: "f32[2]", primals_391: "i64[1, 512]", primals_392: "i64[1, 512]", primals_393: "i64[1]", primals_394: "i64[1]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:878, code: position_ids = self.position_ids[:, :seq_length]
    slice_1: "i64[1, 512]" = torch.ops.aten.slice.Tensor(primals_391, 0, 0, 9223372036854775807);  primals_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:884, code: inputs_embeds = self.word_embeddings(input_ids)
    embedding: "f32[1, 512, 1536]" = torch.ops.aten.embedding.default(primals_1, primals_392, 0);  primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:887, code: position_embeddings = self.position_embeddings(position_ids.long())
    embedding_1: "f32[1, 512, 1536]" = torch.ops.aten.embedding.default(primals_2, slice_1);  primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:893, code: embeddings += position_embeddings
    add: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:901, code: embeddings = self.LayerNorm(embeddings)
    var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 512, 1]" = var_mean[0]
    getitem_1: "f32[1, 512, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-07);  getitem = None
    rsqrt: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add, getitem_1);  add = getitem_1 = None
    mul: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul, primals_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:910, code: embeddings = embeddings * mask
    add_2: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_1, primals_4);  mul_1 = primals_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    empty: "f32[1, 512, 1536]" = torch.ops.aten.empty.memory_format([1, 512, 1536], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute: "f32[1, 512, 1536]" = torch.ops.aten.permute.default(empty, [0, 1, 2]);  empty = None
    bernoulli: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_1: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli);  bernoulli = None
    convert_element_type: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bool);  sub_1 = None
    full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type, full_default_1, add_2);  add_2 = None
    mul_3: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where, 1.1111111111111112);  where = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view: "f32[512, 1536]" = torch.ops.aten.view.default(mul_3, [512, 1536])
    permute_1: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
    addmm: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_6, view, permute_1);  primals_6 = None
    view_1: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm, [1, 512, 1536]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_2: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_1, [1, 512, 24, -1]);  view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_2: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
    clone: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    view_3: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone, [-1, 512, 64]);  clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_3: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
    addmm_1: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_8, view, permute_3);  primals_8 = None
    view_5: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_1, [1, 512, 1536]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_6: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_5, [1, 512, 24, -1]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_4: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
    clone_1: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    view_7: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_1, [-1, 512, 64]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_5: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
    addmm_2: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_10, view, permute_5);  primals_10 = None
    view_9: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_2, [1, 512, 1536]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_10: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_9, [1, 512, 24, -1]);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_6: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
    clone_2: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
    view_11: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_2, [-1, 512, 64]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    full_default_2: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_7: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None
    div: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_7, full_default_2);  permute_7 = None
    bmm: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_3, div)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_12: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm, [-1, 24, 512, 512]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    full_default_3: "b8[1, 1, 512, 512]" = torch.ops.aten.full.default([1, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    full_default_4: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_1: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_4, view_12);  view_12 = None
    amax: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_1, [-1], True)
    sub_2: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_1, amax);  where_1 = amax = None
    exp: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
    sum_1: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_1: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    where_2: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_1, div_1);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    empty_1: "f32[1, 24, 512, 512]" = torch.ops.aten.empty.memory_format([1, 24, 512, 512], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_8: "f32[1, 24, 512, 512]" = torch.ops.aten.permute.default(empty_1, [0, 1, 2, 3]);  empty_1 = None
    bernoulli_1: "f32[1, 24, 512, 512]" = torch.ops.aten.bernoulli.p(permute_8, 0.9)
    sub_3: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_1);  bernoulli_1 = None
    convert_element_type_2: "b8[1, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_3, torch.bool);  sub_3 = None
    where_3: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_2, full_default_1, where_2)
    mul_6: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_3, 1.1111111111111112);  where_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_13: "f32[24, 512, 512]" = torch.ops.aten.view.default(mul_6, [-1, 512, 512]);  mul_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_1: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_13, view_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_14: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_1, [-1, 24, 512, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_9: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_3: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_9, memory_format = torch.contiguous_format);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_15: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_3, [1, 512, -1]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_16: "f32[512, 1536]" = torch.ops.aten.view.default(view_15, [512, 1536]);  view_15 = None
    permute_10: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
    addmm_3: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_12, view_16, permute_10);  primals_12 = None
    view_17: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_3, [1, 512, 1536]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_2: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_4: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_2);  bernoulli_2 = None
    convert_element_type_3: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_4, torch.bool);  sub_4 = None
    where_4: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_3, full_default_1, view_17);  view_17 = None
    mul_7: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_4, 1.1111111111111112);  where_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_3: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_7, mul_3);  mul_7 = mul_3 = None
    var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
    getitem_2: "f32[1, 512, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 512, 1]" = var_mean_1[1];  var_mean_1 = None
    add_4: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-07);  getitem_2 = None
    rsqrt_1: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_5: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_3, getitem_3);  add_3 = getitem_3 = None
    mul_8: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_1);  sub_5 = None
    mul_9: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_8, primals_13)
    add_5: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_9, primals_14);  mul_9 = primals_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_18: "f32[512, 1536]" = torch.ops.aten.view.default(add_5, [512, 1536])
    permute_12: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
    addmm_4: "f32[512, 6144]" = torch.ops.aten.addmm.default(primals_16, view_18, permute_12);  primals_16 = None
    view_19: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_4, [1, 512, 6144])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_10: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_19, 0.5)
    mul_11: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476);  view_19 = None
    erf: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_11);  mul_11 = None
    add_6: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_12: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_10, add_6);  mul_10 = add_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_20: "f32[512, 6144]" = torch.ops.aten.view.default(mul_12, [512, 6144]);  mul_12 = None
    permute_13: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
    addmm_5: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_18, view_20, permute_13);  primals_18 = None
    view_21: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_5, [1, 512, 1536]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_3: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_6: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_3);  bernoulli_3 = None
    convert_element_type_4: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_6, torch.bool);  sub_6 = None
    where_5: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_4, full_default_1, view_21);  view_21 = None
    mul_13: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_5, 1.1111111111111112);  where_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_7: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_13, add_5);  mul_13 = add_5 = None
    var_mean_2 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 512, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 512, 1]" = var_mean_2[1];  var_mean_2 = None
    add_8: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-07);  getitem_4 = None
    rsqrt_2: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
    sub_7: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_7, getitem_5);  add_7 = getitem_5 = None
    mul_14: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_2);  sub_7 = None
    mul_15: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_14, primals_19)
    add_9: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_15, primals_20);  mul_15 = primals_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_22: "f32[512, 1536]" = torch.ops.aten.view.default(add_9, [512, 1536])
    permute_15: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
    addmm_6: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_22, view_22, permute_15);  primals_22 = None
    view_23: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_6, [1, 512, 1536]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_24: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_23, [1, 512, 24, -1]);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_16: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
    clone_4: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
    view_25: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_4, [-1, 512, 64]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_17: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_23, [1, 0]);  primals_23 = None
    addmm_7: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_24, view_22, permute_17);  primals_24 = None
    view_27: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_7, [1, 512, 1536]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_28: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_27, [1, 512, 24, -1]);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_18: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
    clone_5: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    view_29: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_5, [-1, 512, 64]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_19: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
    addmm_8: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_26, view_22, permute_19);  primals_26 = None
    view_31: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_8, [1, 512, 1536]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_32: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_31, [1, 512, 24, -1]);  view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_20: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
    clone_6: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_20, memory_format = torch.contiguous_format);  permute_20 = None
    view_33: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_6, [-1, 512, 64]);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_21: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_29, [0, 2, 1]);  view_29 = None
    div_2: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_21, full_default_2);  permute_21 = None
    bmm_2: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_25, div_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_34: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_2, [-1, 24, 512, 512]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    where_6: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_4, view_34);  view_34 = None
    amax_1: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_6, [-1], True)
    sub_8: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_6, amax_1);  where_6 = amax_1 = None
    exp_1: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
    sum_2: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_3: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    where_7: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_1, div_3);  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_4: "f32[1, 24, 512, 512]" = torch.ops.aten.bernoulli.p(permute_8, 0.9)
    sub_9: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_4);  bernoulli_4 = None
    convert_element_type_6: "b8[1, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_9, torch.bool);  sub_9 = None
    where_8: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_6, full_default_1, where_7)
    mul_17: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_8, 1.1111111111111112);  where_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_35: "f32[24, 512, 512]" = torch.ops.aten.view.default(mul_17, [-1, 512, 512]);  mul_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_3: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_35, view_33)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_36: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_3, [-1, 24, 512, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_23: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_7: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_37: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_7, [1, 512, -1]);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_38: "f32[512, 1536]" = torch.ops.aten.view.default(view_37, [512, 1536]);  view_37 = None
    permute_24: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
    addmm_9: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_28, view_38, permute_24);  primals_28 = None
    view_39: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_9, [1, 512, 1536]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_5: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_10: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_5);  bernoulli_5 = None
    convert_element_type_7: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_10, torch.bool);  sub_10 = None
    where_9: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_7, full_default_1, view_39);  view_39 = None
    mul_18: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_9, 1.1111111111111112);  where_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_10: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_18, add_9);  mul_18 = add_9 = None
    var_mean_3 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
    getitem_6: "f32[1, 512, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 512, 1]" = var_mean_3[1];  var_mean_3 = None
    add_11: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-07);  getitem_6 = None
    rsqrt_3: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_11: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_10, getitem_7);  add_10 = getitem_7 = None
    mul_19: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_3);  sub_11 = None
    mul_20: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_19, primals_29)
    add_12: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_20, primals_30);  mul_20 = primals_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_40: "f32[512, 1536]" = torch.ops.aten.view.default(add_12, [512, 1536])
    permute_26: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_31, [1, 0]);  primals_31 = None
    addmm_10: "f32[512, 6144]" = torch.ops.aten.addmm.default(primals_32, view_40, permute_26);  primals_32 = None
    view_41: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_10, [1, 512, 6144])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_21: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_41, 0.5)
    mul_22: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476);  view_41 = None
    erf_1: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_22);  mul_22 = None
    add_13: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_23: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_21, add_13);  mul_21 = add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_42: "f32[512, 6144]" = torch.ops.aten.view.default(mul_23, [512, 6144]);  mul_23 = None
    permute_27: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_33, [1, 0]);  primals_33 = None
    addmm_11: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_34, view_42, permute_27);  primals_34 = None
    view_43: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_11, [1, 512, 1536]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_6: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_12: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_6);  bernoulli_6 = None
    convert_element_type_8: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_12, torch.bool);  sub_12 = None
    where_10: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_8, full_default_1, view_43);  view_43 = None
    mul_24: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_10, 1.1111111111111112);  where_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_14: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_24, add_12);  mul_24 = add_12 = None
    var_mean_4 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 512, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 512, 1]" = var_mean_4[1];  var_mean_4 = None
    add_15: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-07);  getitem_8 = None
    rsqrt_4: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
    sub_13: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_14, getitem_9);  add_14 = getitem_9 = None
    mul_25: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_4);  sub_13 = None
    mul_26: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_25, primals_35)
    add_16: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_26, primals_36);  mul_26 = primals_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_44: "f32[512, 1536]" = torch.ops.aten.view.default(add_16, [512, 1536])
    permute_29: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_37, [1, 0]);  primals_37 = None
    addmm_12: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_38, view_44, permute_29);  primals_38 = None
    view_45: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_12, [1, 512, 1536]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_46: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_45, [1, 512, 24, -1]);  view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_30: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
    clone_8: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
    view_47: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_8, [-1, 512, 64]);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_31: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_39, [1, 0]);  primals_39 = None
    addmm_13: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_40, view_44, permute_31);  primals_40 = None
    view_49: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_13, [1, 512, 1536]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_50: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_49, [1, 512, 24, -1]);  view_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_32: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
    clone_9: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
    view_51: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_9, [-1, 512, 64]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_33: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_41, [1, 0]);  primals_41 = None
    addmm_14: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_42, view_44, permute_33);  primals_42 = None
    view_53: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_14, [1, 512, 1536]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_54: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_53, [1, 512, 24, -1]);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_34: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
    clone_10: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
    view_55: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_10, [-1, 512, 64]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_35: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_51, [0, 2, 1]);  view_51 = None
    div_4: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_35, full_default_2);  permute_35 = None
    bmm_4: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_47, div_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_56: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_4, [-1, 24, 512, 512]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    where_11: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_4, view_56);  view_56 = None
    amax_2: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_11, [-1], True)
    sub_14: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_11, amax_2);  where_11 = amax_2 = None
    exp_2: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
    sum_3: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_5: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    where_12: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_1, div_5);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_7: "f32[1, 24, 512, 512]" = torch.ops.aten.bernoulli.p(permute_8, 0.9)
    sub_15: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_7);  bernoulli_7 = None
    convert_element_type_10: "b8[1, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_15, torch.bool);  sub_15 = None
    where_13: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_10, full_default_1, where_12)
    mul_28: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_13, 1.1111111111111112);  where_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_57: "f32[24, 512, 512]" = torch.ops.aten.view.default(mul_28, [-1, 512, 512]);  mul_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_5: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_57, view_55)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_58: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_5, [-1, 24, 512, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_37: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_11: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_59: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_11, [1, 512, -1]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_60: "f32[512, 1536]" = torch.ops.aten.view.default(view_59, [512, 1536]);  view_59 = None
    permute_38: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
    addmm_15: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_44, view_60, permute_38);  primals_44 = None
    view_61: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_15, [1, 512, 1536]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_8: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_16: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_8);  bernoulli_8 = None
    convert_element_type_11: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_16, torch.bool);  sub_16 = None
    where_14: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_11, full_default_1, view_61);  view_61 = None
    mul_29: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_14, 1.1111111111111112);  where_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_17: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_29, add_16);  mul_29 = add_16 = None
    var_mean_5 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
    getitem_10: "f32[1, 512, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 512, 1]" = var_mean_5[1];  var_mean_5 = None
    add_18: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-07);  getitem_10 = None
    rsqrt_5: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_17: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_17, getitem_11);  add_17 = getitem_11 = None
    mul_30: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_5);  sub_17 = None
    mul_31: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_30, primals_45)
    add_19: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_31, primals_46);  mul_31 = primals_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_62: "f32[512, 1536]" = torch.ops.aten.view.default(add_19, [512, 1536])
    permute_40: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_47, [1, 0]);  primals_47 = None
    addmm_16: "f32[512, 6144]" = torch.ops.aten.addmm.default(primals_48, view_62, permute_40);  primals_48 = None
    view_63: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_16, [1, 512, 6144])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_32: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_63, 0.5)
    mul_33: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476);  view_63 = None
    erf_2: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
    add_20: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_34: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_32, add_20);  mul_32 = add_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_64: "f32[512, 6144]" = torch.ops.aten.view.default(mul_34, [512, 6144]);  mul_34 = None
    permute_41: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
    addmm_17: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_50, view_64, permute_41);  primals_50 = None
    view_65: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_17, [1, 512, 1536]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_9: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_18: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_9);  bernoulli_9 = None
    convert_element_type_12: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_18, torch.bool);  sub_18 = None
    where_15: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_12, full_default_1, view_65);  view_65 = None
    mul_35: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_15, 1.1111111111111112);  where_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_21: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_35, add_19);  mul_35 = add_19 = None
    var_mean_6 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 512, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 512, 1]" = var_mean_6[1];  var_mean_6 = None
    add_22: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-07);  getitem_12 = None
    rsqrt_6: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
    sub_19: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_21, getitem_13);  add_21 = getitem_13 = None
    mul_36: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_6);  sub_19 = None
    mul_37: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_36, primals_51)
    add_23: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_37, primals_52);  mul_37 = primals_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_66: "f32[512, 1536]" = torch.ops.aten.view.default(add_23, [512, 1536])
    permute_43: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
    addmm_18: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_54, view_66, permute_43);  primals_54 = None
    view_67: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_18, [1, 512, 1536]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_68: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_67, [1, 512, 24, -1]);  view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_44: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
    clone_12: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
    view_69: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_12, [-1, 512, 64]);  clone_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_45: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
    addmm_19: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_56, view_66, permute_45);  primals_56 = None
    view_71: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_19, [1, 512, 1536]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_72: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_71, [1, 512, 24, -1]);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_46: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
    clone_13: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
    view_73: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_13, [-1, 512, 64]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_47: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
    addmm_20: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_58, view_66, permute_47);  primals_58 = None
    view_75: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_20, [1, 512, 1536]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_76: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_75, [1, 512, 24, -1]);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_48: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
    clone_14: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
    view_77: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_14, [-1, 512, 64]);  clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_49: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_73, [0, 2, 1]);  view_73 = None
    div_6: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_49, full_default_2);  permute_49 = None
    bmm_6: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_69, div_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_78: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_6, [-1, 24, 512, 512]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    where_16: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_4, view_78);  view_78 = None
    amax_3: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_16, [-1], True)
    sub_20: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_16, amax_3);  where_16 = amax_3 = None
    exp_3: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
    sum_4: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_7: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    where_17: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_1, div_7);  div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_10: "f32[1, 24, 512, 512]" = torch.ops.aten.bernoulli.p(permute_8, 0.9)
    sub_21: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_10);  bernoulli_10 = None
    convert_element_type_14: "b8[1, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_21, torch.bool);  sub_21 = None
    where_18: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_14, full_default_1, where_17)
    mul_39: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_18, 1.1111111111111112);  where_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_79: "f32[24, 512, 512]" = torch.ops.aten.view.default(mul_39, [-1, 512, 512]);  mul_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_7: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_79, view_77)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_80: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_7, [-1, 24, 512, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_51: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_15: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_81: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_15, [1, 512, -1]);  clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_82: "f32[512, 1536]" = torch.ops.aten.view.default(view_81, [512, 1536]);  view_81 = None
    permute_52: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
    addmm_21: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_60, view_82, permute_52);  primals_60 = None
    view_83: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_21, [1, 512, 1536]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_11: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_22: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_11);  bernoulli_11 = None
    convert_element_type_15: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_22, torch.bool);  sub_22 = None
    where_19: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_15, full_default_1, view_83);  view_83 = None
    mul_40: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_19, 1.1111111111111112);  where_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_24: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_40, add_23);  mul_40 = add_23 = None
    var_mean_7 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
    getitem_14: "f32[1, 512, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 512, 1]" = var_mean_7[1];  var_mean_7 = None
    add_25: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-07);  getitem_14 = None
    rsqrt_7: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    sub_23: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_24, getitem_15);  add_24 = getitem_15 = None
    mul_41: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_7);  sub_23 = None
    mul_42: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_41, primals_61)
    add_26: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_42, primals_62);  mul_42 = primals_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_84: "f32[512, 1536]" = torch.ops.aten.view.default(add_26, [512, 1536])
    permute_54: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_63, [1, 0]);  primals_63 = None
    addmm_22: "f32[512, 6144]" = torch.ops.aten.addmm.default(primals_64, view_84, permute_54);  primals_64 = None
    view_85: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_22, [1, 512, 6144])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_43: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_85, 0.5)
    mul_44: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476);  view_85 = None
    erf_3: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_44);  mul_44 = None
    add_27: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_45: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_43, add_27);  mul_43 = add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_86: "f32[512, 6144]" = torch.ops.aten.view.default(mul_45, [512, 6144]);  mul_45 = None
    permute_55: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_65, [1, 0]);  primals_65 = None
    addmm_23: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_66, view_86, permute_55);  primals_66 = None
    view_87: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_23, [1, 512, 1536]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_12: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_24: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_12);  bernoulli_12 = None
    convert_element_type_16: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_24, torch.bool);  sub_24 = None
    where_20: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_16, full_default_1, view_87);  view_87 = None
    mul_46: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_20, 1.1111111111111112);  where_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_28: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_46, add_26);  mul_46 = add_26 = None
    var_mean_8 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
    getitem_16: "f32[1, 512, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 512, 1]" = var_mean_8[1];  var_mean_8 = None
    add_29: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-07);  getitem_16 = None
    rsqrt_8: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
    sub_25: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_28, getitem_17);  add_28 = getitem_17 = None
    mul_47: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_8);  sub_25 = None
    mul_48: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_47, primals_67)
    add_30: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_48, primals_68);  mul_48 = primals_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_88: "f32[512, 1536]" = torch.ops.aten.view.default(add_30, [512, 1536])
    permute_57: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_69, [1, 0]);  primals_69 = None
    addmm_24: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_70, view_88, permute_57);  primals_70 = None
    view_89: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_24, [1, 512, 1536]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_90: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_89, [1, 512, 24, -1]);  view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_58: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None
    clone_16: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_58, memory_format = torch.contiguous_format);  permute_58 = None
    view_91: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_16, [-1, 512, 64]);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_59: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
    addmm_25: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_72, view_88, permute_59);  primals_72 = None
    view_93: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_25, [1, 512, 1536]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_94: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_93, [1, 512, 24, -1]);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_60: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
    clone_17: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
    view_95: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_17, [-1, 512, 64]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_61: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
    addmm_26: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_74, view_88, permute_61);  primals_74 = None
    view_97: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_26, [1, 512, 1536]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_98: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_97, [1, 512, 24, -1]);  view_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_62: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
    clone_18: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    view_99: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_18, [-1, 512, 64]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_63: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_95, [0, 2, 1]);  view_95 = None
    div_8: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_63, full_default_2);  permute_63 = None
    bmm_8: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_91, div_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_100: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_8, [-1, 24, 512, 512]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    where_21: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_4, view_100);  view_100 = None
    amax_4: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_21, [-1], True)
    sub_26: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_21, amax_4);  where_21 = amax_4 = None
    exp_4: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
    sum_5: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_9: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    where_22: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_1, div_9);  div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_13: "f32[1, 24, 512, 512]" = torch.ops.aten.bernoulli.p(permute_8, 0.9)
    sub_27: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_13);  bernoulli_13 = None
    convert_element_type_18: "b8[1, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_27, torch.bool);  sub_27 = None
    where_23: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_18, full_default_1, where_22)
    mul_50: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_23, 1.1111111111111112);  where_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_101: "f32[24, 512, 512]" = torch.ops.aten.view.default(mul_50, [-1, 512, 512]);  mul_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_9: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_101, view_99)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_102: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_9, [-1, 24, 512, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_65: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_19: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_65, memory_format = torch.contiguous_format);  permute_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_103: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_19, [1, 512, -1]);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_104: "f32[512, 1536]" = torch.ops.aten.view.default(view_103, [512, 1536]);  view_103 = None
    permute_66: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
    addmm_27: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_76, view_104, permute_66);  primals_76 = None
    view_105: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_27, [1, 512, 1536]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_14: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_28: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_14);  bernoulli_14 = None
    convert_element_type_19: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_28, torch.bool);  sub_28 = None
    where_24: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_19, full_default_1, view_105);  view_105 = None
    mul_51: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_24, 1.1111111111111112);  where_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_31: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_51, add_30);  mul_51 = add_30 = None
    var_mean_9 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 512, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 512, 1]" = var_mean_9[1];  var_mean_9 = None
    add_32: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-07);  getitem_18 = None
    rsqrt_9: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_29: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_31, getitem_19);  add_31 = getitem_19 = None
    mul_52: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_9);  sub_29 = None
    mul_53: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_52, primals_77)
    add_33: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_53, primals_78);  mul_53 = primals_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_106: "f32[512, 1536]" = torch.ops.aten.view.default(add_33, [512, 1536])
    permute_68: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
    addmm_28: "f32[512, 6144]" = torch.ops.aten.addmm.default(primals_80, view_106, permute_68);  primals_80 = None
    view_107: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_28, [1, 512, 6144])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_54: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_107, 0.5)
    mul_55: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476);  view_107 = None
    erf_4: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
    add_34: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_56: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_54, add_34);  mul_54 = add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_108: "f32[512, 6144]" = torch.ops.aten.view.default(mul_56, [512, 6144]);  mul_56 = None
    permute_69: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_81, [1, 0]);  primals_81 = None
    addmm_29: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_82, view_108, permute_69);  primals_82 = None
    view_109: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_29, [1, 512, 1536]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_15: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_30: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_15);  bernoulli_15 = None
    convert_element_type_20: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_30, torch.bool);  sub_30 = None
    where_25: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_20, full_default_1, view_109);  view_109 = None
    mul_57: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_25, 1.1111111111111112);  where_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_35: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_57, add_33);  mul_57 = add_33 = None
    var_mean_10 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
    getitem_20: "f32[1, 512, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 512, 1]" = var_mean_10[1];  var_mean_10 = None
    add_36: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-07);  getitem_20 = None
    rsqrt_10: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_31: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_35, getitem_21);  add_35 = getitem_21 = None
    mul_58: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_10);  sub_31 = None
    mul_59: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_58, primals_83)
    add_37: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_59, primals_84);  mul_59 = primals_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_110: "f32[512, 1536]" = torch.ops.aten.view.default(add_37, [512, 1536])
    permute_71: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
    addmm_30: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_86, view_110, permute_71);  primals_86 = None
    view_111: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_30, [1, 512, 1536]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_112: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_111, [1, 512, 24, -1]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_72: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
    clone_20: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format);  permute_72 = None
    view_113: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_20, [-1, 512, 64]);  clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_73: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
    addmm_31: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_88, view_110, permute_73);  primals_88 = None
    view_115: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_31, [1, 512, 1536]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_116: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_115, [1, 512, 24, -1]);  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_74: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
    clone_21: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format);  permute_74 = None
    view_117: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_21, [-1, 512, 64]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_75: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_89, [1, 0]);  primals_89 = None
    addmm_32: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_90, view_110, permute_75);  primals_90 = None
    view_119: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_32, [1, 512, 1536]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_120: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_119, [1, 512, 24, -1]);  view_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_76: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
    clone_22: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_76, memory_format = torch.contiguous_format);  permute_76 = None
    view_121: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_22, [-1, 512, 64]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_77: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_117, [0, 2, 1]);  view_117 = None
    div_10: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_77, full_default_2);  permute_77 = None
    bmm_10: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_113, div_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_122: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_10, [-1, 24, 512, 512]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    where_26: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_4, view_122);  view_122 = None
    amax_5: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_26, [-1], True)
    sub_32: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_26, amax_5);  where_26 = amax_5 = None
    exp_5: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_32);  sub_32 = None
    sum_6: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_11: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    where_27: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_1, div_11);  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_16: "f32[1, 24, 512, 512]" = torch.ops.aten.bernoulli.p(permute_8, 0.9)
    sub_33: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_16);  bernoulli_16 = None
    convert_element_type_22: "b8[1, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_33, torch.bool);  sub_33 = None
    where_28: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_22, full_default_1, where_27)
    mul_61: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_28, 1.1111111111111112);  where_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_123: "f32[24, 512, 512]" = torch.ops.aten.view.default(mul_61, [-1, 512, 512]);  mul_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_11: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_123, view_121)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_124: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_11, [-1, 24, 512, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_79: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_23: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_125: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_23, [1, 512, -1]);  clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_126: "f32[512, 1536]" = torch.ops.aten.view.default(view_125, [512, 1536]);  view_125 = None
    permute_80: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
    addmm_33: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_92, view_126, permute_80);  primals_92 = None
    view_127: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_33, [1, 512, 1536]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_17: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_34: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_17);  bernoulli_17 = None
    convert_element_type_23: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_34, torch.bool);  sub_34 = None
    where_29: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_23, full_default_1, view_127);  view_127 = None
    mul_62: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_29, 1.1111111111111112);  where_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_38: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_62, add_37);  mul_62 = add_37 = None
    var_mean_11 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 512, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 512, 1]" = var_mean_11[1];  var_mean_11 = None
    add_39: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-07);  getitem_22 = None
    rsqrt_11: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
    sub_35: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_38, getitem_23);  add_38 = getitem_23 = None
    mul_63: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_11);  sub_35 = None
    mul_64: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_63, primals_93)
    add_40: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_64, primals_94);  mul_64 = primals_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_128: "f32[512, 1536]" = torch.ops.aten.view.default(add_40, [512, 1536])
    permute_82: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_95, [1, 0]);  primals_95 = None
    addmm_34: "f32[512, 6144]" = torch.ops.aten.addmm.default(primals_96, view_128, permute_82);  primals_96 = None
    view_129: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_34, [1, 512, 6144])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_65: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_129, 0.5)
    mul_66: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476);  view_129 = None
    erf_5: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_66);  mul_66 = None
    add_41: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_67: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_65, add_41);  mul_65 = add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_130: "f32[512, 6144]" = torch.ops.aten.view.default(mul_67, [512, 6144]);  mul_67 = None
    permute_83: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
    addmm_35: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_98, view_130, permute_83);  primals_98 = None
    view_131: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_35, [1, 512, 1536]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_18: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_36: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_18);  bernoulli_18 = None
    convert_element_type_24: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_36, torch.bool);  sub_36 = None
    where_30: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_24, full_default_1, view_131);  view_131 = None
    mul_68: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_30, 1.1111111111111112);  where_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_42: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_68, add_40);  mul_68 = add_40 = None
    var_mean_12 = torch.ops.aten.var_mean.correction(add_42, [2], correction = 0, keepdim = True)
    getitem_24: "f32[1, 512, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 512, 1]" = var_mean_12[1];  var_mean_12 = None
    add_43: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-07);  getitem_24 = None
    rsqrt_12: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
    sub_37: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_42, getitem_25);  add_42 = getitem_25 = None
    mul_69: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_12);  sub_37 = None
    mul_70: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_69, primals_99)
    add_44: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_70, primals_100);  mul_70 = primals_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_132: "f32[512, 1536]" = torch.ops.aten.view.default(add_44, [512, 1536])
    permute_85: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
    addmm_36: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_102, view_132, permute_85);  primals_102 = None
    view_133: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_36, [1, 512, 1536]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_134: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_133, [1, 512, 24, -1]);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_86: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None
    clone_24: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_86, memory_format = torch.contiguous_format);  permute_86 = None
    view_135: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_24, [-1, 512, 64]);  clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_87: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
    addmm_37: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_104, view_132, permute_87);  primals_104 = None
    view_137: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_37, [1, 512, 1536]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_138: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_137, [1, 512, 24, -1]);  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_88: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
    clone_25: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_88, memory_format = torch.contiguous_format);  permute_88 = None
    view_139: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_25, [-1, 512, 64]);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_89: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
    addmm_38: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_106, view_132, permute_89);  primals_106 = None
    view_141: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_38, [1, 512, 1536]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_142: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_141, [1, 512, 24, -1]);  view_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_90: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None
    clone_26: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
    view_143: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_26, [-1, 512, 64]);  clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_91: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_139, [0, 2, 1]);  view_139 = None
    div_12: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_91, full_default_2);  permute_91 = None
    bmm_12: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_135, div_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_144: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_12, [-1, 24, 512, 512]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    where_31: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_4, view_144);  view_144 = None
    amax_6: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_31, [-1], True)
    sub_38: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_31, amax_6);  where_31 = amax_6 = None
    exp_6: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_38);  sub_38 = None
    sum_7: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_13: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    where_32: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_1, div_13);  div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_19: "f32[1, 24, 512, 512]" = torch.ops.aten.bernoulli.p(permute_8, 0.9)
    sub_39: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_19);  bernoulli_19 = None
    convert_element_type_26: "b8[1, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_39, torch.bool);  sub_39 = None
    where_33: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_26, full_default_1, where_32)
    mul_72: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_33, 1.1111111111111112);  where_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_145: "f32[24, 512, 512]" = torch.ops.aten.view.default(mul_72, [-1, 512, 512]);  mul_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_13: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_145, view_143)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_146: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_13, [-1, 24, 512, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_93: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_27: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_147: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_27, [1, 512, -1]);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_148: "f32[512, 1536]" = torch.ops.aten.view.default(view_147, [512, 1536]);  view_147 = None
    permute_94: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_107, [1, 0]);  primals_107 = None
    addmm_39: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_108, view_148, permute_94);  primals_108 = None
    view_149: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_39, [1, 512, 1536]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_20: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_40: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_20);  bernoulli_20 = None
    convert_element_type_27: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_40, torch.bool);  sub_40 = None
    where_34: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_27, full_default_1, view_149);  view_149 = None
    mul_73: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_34, 1.1111111111111112);  where_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_45: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_73, add_44);  mul_73 = add_44 = None
    var_mean_13 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
    getitem_26: "f32[1, 512, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 512, 1]" = var_mean_13[1];  var_mean_13 = None
    add_46: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-07);  getitem_26 = None
    rsqrt_13: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    sub_41: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_45, getitem_27);  add_45 = getitem_27 = None
    mul_74: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_13);  sub_41 = None
    mul_75: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_74, primals_109)
    add_47: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_75, primals_110);  mul_75 = primals_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_150: "f32[512, 1536]" = torch.ops.aten.view.default(add_47, [512, 1536])
    permute_96: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
    addmm_40: "f32[512, 6144]" = torch.ops.aten.addmm.default(primals_112, view_150, permute_96);  primals_112 = None
    view_151: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_40, [1, 512, 6144])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_76: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_151, 0.5)
    mul_77: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476);  view_151 = None
    erf_6: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_77);  mul_77 = None
    add_48: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_78: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_76, add_48);  mul_76 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_152: "f32[512, 6144]" = torch.ops.aten.view.default(mul_78, [512, 6144]);  mul_78 = None
    permute_97: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_113, [1, 0]);  primals_113 = None
    addmm_41: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_114, view_152, permute_97);  primals_114 = None
    view_153: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_41, [1, 512, 1536]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_21: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_42: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_21);  bernoulli_21 = None
    convert_element_type_28: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_42, torch.bool);  sub_42 = None
    where_35: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_28, full_default_1, view_153);  view_153 = None
    mul_79: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_35, 1.1111111111111112);  where_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_49: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_79, add_47);  mul_79 = add_47 = None
    var_mean_14 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 512, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 512, 1]" = var_mean_14[1];  var_mean_14 = None
    add_50: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-07);  getitem_28 = None
    rsqrt_14: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    sub_43: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_49, getitem_29);  add_49 = getitem_29 = None
    mul_80: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_14);  sub_43 = None
    mul_81: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_80, primals_115)
    add_51: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_81, primals_116);  mul_81 = primals_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_154: "f32[512, 1536]" = torch.ops.aten.view.default(add_51, [512, 1536])
    permute_99: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_117, [1, 0]);  primals_117 = None
    addmm_42: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_118, view_154, permute_99);  primals_118 = None
    view_155: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_42, [1, 512, 1536]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_156: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_155, [1, 512, 24, -1]);  view_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_100: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None
    clone_28: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_100, memory_format = torch.contiguous_format);  permute_100 = None
    view_157: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_28, [-1, 512, 64]);  clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_101: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
    addmm_43: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_120, view_154, permute_101);  primals_120 = None
    view_159: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_43, [1, 512, 1536]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_160: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_159, [1, 512, 24, -1]);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_102: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None
    clone_29: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None
    view_161: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_29, [-1, 512, 64]);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_103: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
    addmm_44: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_122, view_154, permute_103);  primals_122 = None
    view_163: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_44, [1, 512, 1536]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_164: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_163, [1, 512, 24, -1]);  view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_104: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None
    clone_30: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
    view_165: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_30, [-1, 512, 64]);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_105: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_161, [0, 2, 1]);  view_161 = None
    div_14: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_105, full_default_2);  permute_105 = None
    bmm_14: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_157, div_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_166: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_14, [-1, 24, 512, 512]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    where_36: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_4, view_166);  view_166 = None
    amax_7: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_36, [-1], True)
    sub_44: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_36, amax_7);  where_36 = amax_7 = None
    exp_7: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_44);  sub_44 = None
    sum_8: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_15: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    where_37: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_1, div_15);  div_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_22: "f32[1, 24, 512, 512]" = torch.ops.aten.bernoulli.p(permute_8, 0.9)
    sub_45: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_22);  bernoulli_22 = None
    convert_element_type_30: "b8[1, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_45, torch.bool);  sub_45 = None
    where_38: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_30, full_default_1, where_37)
    mul_83: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_38, 1.1111111111111112);  where_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_167: "f32[24, 512, 512]" = torch.ops.aten.view.default(mul_83, [-1, 512, 512]);  mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_15: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_167, view_165)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_168: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_15, [-1, 24, 512, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_107: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_31: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_107, memory_format = torch.contiguous_format);  permute_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_169: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_31, [1, 512, -1]);  clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_170: "f32[512, 1536]" = torch.ops.aten.view.default(view_169, [512, 1536]);  view_169 = None
    permute_108: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_123, [1, 0]);  primals_123 = None
    addmm_45: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_124, view_170, permute_108);  primals_124 = None
    view_171: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_45, [1, 512, 1536]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_23: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_46: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_23);  bernoulli_23 = None
    convert_element_type_31: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_46, torch.bool);  sub_46 = None
    where_39: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_31, full_default_1, view_171);  view_171 = None
    mul_84: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_39, 1.1111111111111112);  where_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_52: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_84, add_51);  mul_84 = add_51 = None
    var_mean_15 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
    getitem_30: "f32[1, 512, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 512, 1]" = var_mean_15[1];  var_mean_15 = None
    add_53: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-07);  getitem_30 = None
    rsqrt_15: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    sub_47: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_52, getitem_31);  add_52 = getitem_31 = None
    mul_85: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_15);  sub_47 = None
    mul_86: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_85, primals_125)
    add_54: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_86, primals_126);  mul_86 = primals_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_172: "f32[512, 1536]" = torch.ops.aten.view.default(add_54, [512, 1536])
    permute_110: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
    addmm_46: "f32[512, 6144]" = torch.ops.aten.addmm.default(primals_128, view_172, permute_110);  primals_128 = None
    view_173: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_46, [1, 512, 6144])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_87: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_173, 0.5)
    mul_88: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476);  view_173 = None
    erf_7: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
    add_55: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_89: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_87, add_55);  mul_87 = add_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_174: "f32[512, 6144]" = torch.ops.aten.view.default(mul_89, [512, 6144]);  mul_89 = None
    permute_111: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
    addmm_47: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_130, view_174, permute_111);  primals_130 = None
    view_175: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_47, [1, 512, 1536]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_24: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_48: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_24);  bernoulli_24 = None
    convert_element_type_32: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_48, torch.bool);  sub_48 = None
    where_40: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_32, full_default_1, view_175);  view_175 = None
    mul_90: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_40, 1.1111111111111112);  where_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_56: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_90, add_54);  mul_90 = add_54 = None
    var_mean_16 = torch.ops.aten.var_mean.correction(add_56, [2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 512, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 512, 1]" = var_mean_16[1];  var_mean_16 = None
    add_57: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-07);  getitem_32 = None
    rsqrt_16: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
    sub_49: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_56, getitem_33);  add_56 = getitem_33 = None
    mul_91: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_16);  sub_49 = None
    mul_92: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_91, primals_131)
    add_58: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_92, primals_132);  mul_92 = primals_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_176: "f32[512, 1536]" = torch.ops.aten.view.default(add_58, [512, 1536])
    permute_113: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_133, [1, 0]);  primals_133 = None
    addmm_48: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_134, view_176, permute_113);  primals_134 = None
    view_177: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_48, [1, 512, 1536]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_178: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_177, [1, 512, 24, -1]);  view_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_114: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None
    clone_32: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
    view_179: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_32, [-1, 512, 64]);  clone_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_115: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_135, [1, 0]);  primals_135 = None
    addmm_49: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_136, view_176, permute_115);  primals_136 = None
    view_181: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_49, [1, 512, 1536]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_182: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_181, [1, 512, 24, -1]);  view_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_116: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
    clone_33: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_116, memory_format = torch.contiguous_format);  permute_116 = None
    view_183: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_33, [-1, 512, 64]);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_117: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_137, [1, 0]);  primals_137 = None
    addmm_50: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_138, view_176, permute_117);  primals_138 = None
    view_185: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_50, [1, 512, 1536]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_186: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_185, [1, 512, 24, -1]);  view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_118: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
    clone_34: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_118, memory_format = torch.contiguous_format);  permute_118 = None
    view_187: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_34, [-1, 512, 64]);  clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_119: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_183, [0, 2, 1]);  view_183 = None
    div_16: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_119, full_default_2);  permute_119 = None
    bmm_16: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_179, div_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_188: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_16, [-1, 24, 512, 512]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    where_41: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_4, view_188);  view_188 = None
    amax_8: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_41, [-1], True)
    sub_50: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_41, amax_8);  where_41 = amax_8 = None
    exp_8: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_50);  sub_50 = None
    sum_9: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_17: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    where_42: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_1, div_17);  div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_25: "f32[1, 24, 512, 512]" = torch.ops.aten.bernoulli.p(permute_8, 0.9)
    sub_51: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_25);  bernoulli_25 = None
    convert_element_type_34: "b8[1, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_51, torch.bool);  sub_51 = None
    where_43: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_34, full_default_1, where_42)
    mul_94: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_43, 1.1111111111111112);  where_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_189: "f32[24, 512, 512]" = torch.ops.aten.view.default(mul_94, [-1, 512, 512]);  mul_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_17: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_189, view_187)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_190: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_17, [-1, 24, 512, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_121: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_35: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_121, memory_format = torch.contiguous_format);  permute_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_191: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_35, [1, 512, -1]);  clone_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_192: "f32[512, 1536]" = torch.ops.aten.view.default(view_191, [512, 1536]);  view_191 = None
    permute_122: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_139, [1, 0]);  primals_139 = None
    addmm_51: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_140, view_192, permute_122);  primals_140 = None
    view_193: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_51, [1, 512, 1536]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_26: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_52: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_26);  bernoulli_26 = None
    convert_element_type_35: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_52, torch.bool);  sub_52 = None
    where_44: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_35, full_default_1, view_193);  view_193 = None
    mul_95: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_44, 1.1111111111111112);  where_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_59: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_95, add_58);  mul_95 = add_58 = None
    var_mean_17 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
    getitem_34: "f32[1, 512, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 512, 1]" = var_mean_17[1];  var_mean_17 = None
    add_60: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-07);  getitem_34 = None
    rsqrt_17: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_53: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_59, getitem_35);  add_59 = getitem_35 = None
    mul_96: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_17);  sub_53 = None
    mul_97: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_96, primals_141)
    add_61: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_97, primals_142);  mul_97 = primals_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_194: "f32[512, 1536]" = torch.ops.aten.view.default(add_61, [512, 1536])
    permute_124: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_143, [1, 0]);  primals_143 = None
    addmm_52: "f32[512, 6144]" = torch.ops.aten.addmm.default(primals_144, view_194, permute_124);  primals_144 = None
    view_195: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_52, [1, 512, 6144])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_98: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_195, 0.5)
    mul_99: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476);  view_195 = None
    erf_8: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_99);  mul_99 = None
    add_62: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_100: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_98, add_62);  mul_98 = add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_196: "f32[512, 6144]" = torch.ops.aten.view.default(mul_100, [512, 6144]);  mul_100 = None
    permute_125: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_145, [1, 0]);  primals_145 = None
    addmm_53: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_146, view_196, permute_125);  primals_146 = None
    view_197: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_53, [1, 512, 1536]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_27: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_54: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_27);  bernoulli_27 = None
    convert_element_type_36: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_54, torch.bool);  sub_54 = None
    where_45: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_36, full_default_1, view_197);  view_197 = None
    mul_101: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_45, 1.1111111111111112);  where_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_63: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_101, add_61);  mul_101 = add_61 = None
    var_mean_18 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
    getitem_36: "f32[1, 512, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 512, 1]" = var_mean_18[1];  var_mean_18 = None
    add_64: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-07);  getitem_36 = None
    rsqrt_18: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
    sub_55: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_63, getitem_37);  add_63 = getitem_37 = None
    mul_102: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_18);  sub_55 = None
    mul_103: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_102, primals_147)
    add_65: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_103, primals_148);  mul_103 = primals_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_198: "f32[512, 1536]" = torch.ops.aten.view.default(add_65, [512, 1536])
    permute_127: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_149, [1, 0]);  primals_149 = None
    addmm_54: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_150, view_198, permute_127);  primals_150 = None
    view_199: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_54, [1, 512, 1536]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_200: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_199, [1, 512, 24, -1]);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_128: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None
    clone_36: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    view_201: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_36, [-1, 512, 64]);  clone_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_129: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_151, [1, 0]);  primals_151 = None
    addmm_55: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_152, view_198, permute_129);  primals_152 = None
    view_203: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_55, [1, 512, 1536]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_204: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_203, [1, 512, 24, -1]);  view_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_130: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None
    clone_37: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_130, memory_format = torch.contiguous_format);  permute_130 = None
    view_205: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_37, [-1, 512, 64]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_131: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_153, [1, 0]);  primals_153 = None
    addmm_56: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_154, view_198, permute_131);  primals_154 = None
    view_207: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_56, [1, 512, 1536]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_208: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_207, [1, 512, 24, -1]);  view_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_132: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
    clone_38: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_132, memory_format = torch.contiguous_format);  permute_132 = None
    view_209: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_38, [-1, 512, 64]);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_133: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_205, [0, 2, 1]);  view_205 = None
    div_18: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_133, full_default_2);  permute_133 = None
    bmm_18: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_201, div_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_210: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_18, [-1, 24, 512, 512]);  bmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    where_46: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_4, view_210);  view_210 = None
    amax_9: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_46, [-1], True)
    sub_56: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_46, amax_9);  where_46 = amax_9 = None
    exp_9: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_56);  sub_56 = None
    sum_10: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_19: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    where_47: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_1, div_19);  div_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_28: "f32[1, 24, 512, 512]" = torch.ops.aten.bernoulli.p(permute_8, 0.9)
    sub_57: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_28);  bernoulli_28 = None
    convert_element_type_38: "b8[1, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_57, torch.bool);  sub_57 = None
    where_48: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_38, full_default_1, where_47)
    mul_105: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_48, 1.1111111111111112);  where_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_211: "f32[24, 512, 512]" = torch.ops.aten.view.default(mul_105, [-1, 512, 512]);  mul_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_19: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_211, view_209)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_212: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_19, [-1, 24, 512, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_135: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_39: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_213: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_39, [1, 512, -1]);  clone_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_214: "f32[512, 1536]" = torch.ops.aten.view.default(view_213, [512, 1536]);  view_213 = None
    permute_136: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_155, [1, 0]);  primals_155 = None
    addmm_57: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_156, view_214, permute_136);  primals_156 = None
    view_215: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_57, [1, 512, 1536]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_29: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_58: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_29);  bernoulli_29 = None
    convert_element_type_39: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_58, torch.bool);  sub_58 = None
    where_49: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_39, full_default_1, view_215);  view_215 = None
    mul_106: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_49, 1.1111111111111112);  where_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_66: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_106, add_65);  mul_106 = add_65 = None
    var_mean_19 = torch.ops.aten.var_mean.correction(add_66, [2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 512, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 512, 1]" = var_mean_19[1];  var_mean_19 = None
    add_67: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-07);  getitem_38 = None
    rsqrt_19: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    sub_59: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_66, getitem_39);  add_66 = getitem_39 = None
    mul_107: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_19);  sub_59 = None
    mul_108: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_107, primals_157)
    add_68: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_108, primals_158);  mul_108 = primals_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_216: "f32[512, 1536]" = torch.ops.aten.view.default(add_68, [512, 1536])
    permute_138: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_159, [1, 0]);  primals_159 = None
    addmm_58: "f32[512, 6144]" = torch.ops.aten.addmm.default(primals_160, view_216, permute_138);  primals_160 = None
    view_217: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_58, [1, 512, 6144])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_109: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_217, 0.5)
    mul_110: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476);  view_217 = None
    erf_9: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_110);  mul_110 = None
    add_69: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_111: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_109, add_69);  mul_109 = add_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_218: "f32[512, 6144]" = torch.ops.aten.view.default(mul_111, [512, 6144]);  mul_111 = None
    permute_139: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_161, [1, 0]);  primals_161 = None
    addmm_59: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_162, view_218, permute_139);  primals_162 = None
    view_219: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_59, [1, 512, 1536]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_30: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_60: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_30);  bernoulli_30 = None
    convert_element_type_40: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_60, torch.bool);  sub_60 = None
    where_50: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_40, full_default_1, view_219);  view_219 = None
    mul_112: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_50, 1.1111111111111112);  where_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_70: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_112, add_68);  mul_112 = add_68 = None
    var_mean_20 = torch.ops.aten.var_mean.correction(add_70, [2], correction = 0, keepdim = True)
    getitem_40: "f32[1, 512, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 512, 1]" = var_mean_20[1];  var_mean_20 = None
    add_71: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-07);  getitem_40 = None
    rsqrt_20: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
    sub_61: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_70, getitem_41);  add_70 = getitem_41 = None
    mul_113: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_20);  sub_61 = None
    mul_114: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_113, primals_163)
    add_72: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_114, primals_164);  mul_114 = primals_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_220: "f32[512, 1536]" = torch.ops.aten.view.default(add_72, [512, 1536])
    permute_141: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_165, [1, 0]);  primals_165 = None
    addmm_60: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_166, view_220, permute_141);  primals_166 = None
    view_221: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_60, [1, 512, 1536]);  addmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_222: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_221, [1, 512, 24, -1]);  view_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_142: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None
    clone_40: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_142, memory_format = torch.contiguous_format);  permute_142 = None
    view_223: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_40, [-1, 512, 64]);  clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_143: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_167, [1, 0]);  primals_167 = None
    addmm_61: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_168, view_220, permute_143);  primals_168 = None
    view_225: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_61, [1, 512, 1536]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_226: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_225, [1, 512, 24, -1]);  view_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_144: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
    clone_41: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None
    view_227: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_41, [-1, 512, 64]);  clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_145: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_169, [1, 0]);  primals_169 = None
    addmm_62: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_170, view_220, permute_145);  primals_170 = None
    view_229: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_62, [1, 512, 1536]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_230: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_229, [1, 512, 24, -1]);  view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_146: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_230, [0, 2, 1, 3]);  view_230 = None
    clone_42: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_146, memory_format = torch.contiguous_format);  permute_146 = None
    view_231: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_42, [-1, 512, 64]);  clone_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_147: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_227, [0, 2, 1]);  view_227 = None
    div_20: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_147, full_default_2);  permute_147 = None
    bmm_20: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_223, div_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_232: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_20, [-1, 24, 512, 512]);  bmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    where_51: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_4, view_232);  view_232 = None
    amax_10: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_51, [-1], True)
    sub_62: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_51, amax_10);  where_51 = amax_10 = None
    exp_10: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_62);  sub_62 = None
    sum_11: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_21: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    where_52: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_1, div_21);  div_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_31: "f32[1, 24, 512, 512]" = torch.ops.aten.bernoulli.p(permute_8, 0.9)
    sub_63: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_31);  bernoulli_31 = None
    convert_element_type_42: "b8[1, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_63, torch.bool);  sub_63 = None
    where_53: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_42, full_default_1, where_52)
    mul_116: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_53, 1.1111111111111112);  where_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_233: "f32[24, 512, 512]" = torch.ops.aten.view.default(mul_116, [-1, 512, 512]);  mul_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_21: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_233, view_231)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_234: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_21, [-1, 24, 512, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_149: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_43: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_149, memory_format = torch.contiguous_format);  permute_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_235: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_43, [1, 512, -1]);  clone_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_236: "f32[512, 1536]" = torch.ops.aten.view.default(view_235, [512, 1536]);  view_235 = None
    permute_150: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_171, [1, 0]);  primals_171 = None
    addmm_63: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_172, view_236, permute_150);  primals_172 = None
    view_237: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_63, [1, 512, 1536]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_32: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_64: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_32);  bernoulli_32 = None
    convert_element_type_43: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_64, torch.bool);  sub_64 = None
    where_54: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_43, full_default_1, view_237);  view_237 = None
    mul_117: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_54, 1.1111111111111112);  where_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_73: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_117, add_72);  mul_117 = add_72 = None
    var_mean_21 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 512, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 512, 1]" = var_mean_21[1];  var_mean_21 = None
    add_74: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-07);  getitem_42 = None
    rsqrt_21: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_65: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_73, getitem_43);  add_73 = getitem_43 = None
    mul_118: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_21);  sub_65 = None
    mul_119: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_118, primals_173)
    add_75: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_119, primals_174);  mul_119 = primals_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_238: "f32[512, 1536]" = torch.ops.aten.view.default(add_75, [512, 1536])
    permute_152: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_175, [1, 0]);  primals_175 = None
    addmm_64: "f32[512, 6144]" = torch.ops.aten.addmm.default(primals_176, view_238, permute_152);  primals_176 = None
    view_239: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_64, [1, 512, 6144])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_120: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_239, 0.5)
    mul_121: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476);  view_239 = None
    erf_10: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_121);  mul_121 = None
    add_76: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_122: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_120, add_76);  mul_120 = add_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_240: "f32[512, 6144]" = torch.ops.aten.view.default(mul_122, [512, 6144]);  mul_122 = None
    permute_153: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_177, [1, 0]);  primals_177 = None
    addmm_65: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_178, view_240, permute_153);  primals_178 = None
    view_241: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_65, [1, 512, 1536]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_33: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_66: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_33);  bernoulli_33 = None
    convert_element_type_44: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_66, torch.bool);  sub_66 = None
    where_55: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_44, full_default_1, view_241);  view_241 = None
    mul_123: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_55, 1.1111111111111112);  where_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_77: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_123, add_75);  mul_123 = add_75 = None
    var_mean_22 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
    getitem_44: "f32[1, 512, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 512, 1]" = var_mean_22[1];  var_mean_22 = None
    add_78: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-07);  getitem_44 = None
    rsqrt_22: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    sub_67: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_77, getitem_45);  add_77 = getitem_45 = None
    mul_124: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_22);  sub_67 = None
    mul_125: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_124, primals_179)
    add_79: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_125, primals_180);  mul_125 = primals_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_242: "f32[512, 1536]" = torch.ops.aten.view.default(add_79, [512, 1536])
    permute_155: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_181, [1, 0]);  primals_181 = None
    addmm_66: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_182, view_242, permute_155);  primals_182 = None
    view_243: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_66, [1, 512, 1536]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_244: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_243, [1, 512, 24, -1]);  view_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_156: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None
    clone_44: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_156, memory_format = torch.contiguous_format);  permute_156 = None
    view_245: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_44, [-1, 512, 64]);  clone_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_157: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_183, [1, 0]);  primals_183 = None
    addmm_67: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_184, view_242, permute_157);  primals_184 = None
    view_247: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_67, [1, 512, 1536]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_248: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_247, [1, 512, 24, -1]);  view_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_158: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_248, [0, 2, 1, 3]);  view_248 = None
    clone_45: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_158, memory_format = torch.contiguous_format);  permute_158 = None
    view_249: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_45, [-1, 512, 64]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_159: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_185, [1, 0]);  primals_185 = None
    addmm_68: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_186, view_242, permute_159);  primals_186 = None
    view_251: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_68, [1, 512, 1536]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_252: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_251, [1, 512, 24, -1]);  view_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_160: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None
    clone_46: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_160, memory_format = torch.contiguous_format);  permute_160 = None
    view_253: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_46, [-1, 512, 64]);  clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_161: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_249, [0, 2, 1]);  view_249 = None
    div_22: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_161, full_default_2);  permute_161 = None
    bmm_22: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_245, div_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_254: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_22, [-1, 24, 512, 512]);  bmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    where_56: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_4, view_254);  view_254 = None
    amax_11: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_56, [-1], True)
    sub_68: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_56, amax_11);  where_56 = amax_11 = None
    exp_11: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_68);  sub_68 = None
    sum_12: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_23: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    where_57: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_1, div_23);  div_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_34: "f32[1, 24, 512, 512]" = torch.ops.aten.bernoulli.p(permute_8, 0.9)
    sub_69: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_34);  bernoulli_34 = None
    convert_element_type_46: "b8[1, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_69, torch.bool);  sub_69 = None
    where_58: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_46, full_default_1, where_57)
    mul_127: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_58, 1.1111111111111112);  where_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_255: "f32[24, 512, 512]" = torch.ops.aten.view.default(mul_127, [-1, 512, 512]);  mul_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_23: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_255, view_253)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_256: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_23, [-1, 24, 512, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_163: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_47: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_163, memory_format = torch.contiguous_format);  permute_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_257: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_47, [1, 512, -1]);  clone_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_258: "f32[512, 1536]" = torch.ops.aten.view.default(view_257, [512, 1536]);  view_257 = None
    permute_164: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_187, [1, 0]);  primals_187 = None
    addmm_69: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_188, view_258, permute_164);  primals_188 = None
    view_259: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_69, [1, 512, 1536]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_35: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_70: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_35);  bernoulli_35 = None
    convert_element_type_47: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_70, torch.bool);  sub_70 = None
    where_59: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_47, full_default_1, view_259);  view_259 = None
    mul_128: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_59, 1.1111111111111112);  where_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_80: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_128, add_79);  mul_128 = add_79 = None
    var_mean_23 = torch.ops.aten.var_mean.correction(add_80, [2], correction = 0, keepdim = True)
    getitem_46: "f32[1, 512, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 512, 1]" = var_mean_23[1];  var_mean_23 = None
    add_81: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-07);  getitem_46 = None
    rsqrt_23: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    sub_71: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_80, getitem_47);  add_80 = getitem_47 = None
    mul_129: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_23);  sub_71 = None
    mul_130: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_129, primals_189)
    add_82: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_130, primals_190);  mul_130 = primals_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_260: "f32[512, 1536]" = torch.ops.aten.view.default(add_82, [512, 1536])
    permute_166: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_191, [1, 0]);  primals_191 = None
    addmm_70: "f32[512, 6144]" = torch.ops.aten.addmm.default(primals_192, view_260, permute_166);  primals_192 = None
    view_261: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_70, [1, 512, 6144])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_131: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_261, 0.5)
    mul_132: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476);  view_261 = None
    erf_11: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_132);  mul_132 = None
    add_83: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_133: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_131, add_83);  mul_131 = add_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_262: "f32[512, 6144]" = torch.ops.aten.view.default(mul_133, [512, 6144]);  mul_133 = None
    permute_167: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_193, [1, 0]);  primals_193 = None
    addmm_71: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_194, view_262, permute_167);  primals_194 = None
    view_263: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_71, [1, 512, 1536]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_36: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_72: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_36);  bernoulli_36 = None
    convert_element_type_48: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_72, torch.bool);  sub_72 = None
    where_60: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_48, full_default_1, view_263);  view_263 = None
    mul_134: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_60, 1.1111111111111112);  where_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_84: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_134, add_82);  mul_134 = add_82 = None
    var_mean_24 = torch.ops.aten.var_mean.correction(add_84, [2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 512, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 512, 1]" = var_mean_24[1];  var_mean_24 = None
    add_85: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-07);  getitem_48 = None
    rsqrt_24: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
    sub_73: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_84, getitem_49);  add_84 = getitem_49 = None
    mul_135: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_24);  sub_73 = None
    mul_136: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_135, primals_195)
    add_86: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_136, primals_196);  mul_136 = primals_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_264: "f32[512, 1536]" = torch.ops.aten.view.default(add_86, [512, 1536])
    permute_169: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_197, [1, 0]);  primals_197 = None
    addmm_72: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_198, view_264, permute_169);  primals_198 = None
    view_265: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_72, [1, 512, 1536]);  addmm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_266: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_265, [1, 512, 24, -1]);  view_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_170: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_266, [0, 2, 1, 3]);  view_266 = None
    clone_48: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
    view_267: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_48, [-1, 512, 64]);  clone_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_171: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_199, [1, 0]);  primals_199 = None
    addmm_73: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_200, view_264, permute_171);  primals_200 = None
    view_269: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_73, [1, 512, 1536]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_270: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_269, [1, 512, 24, -1]);  view_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_172: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_270, [0, 2, 1, 3]);  view_270 = None
    clone_49: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
    view_271: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_49, [-1, 512, 64]);  clone_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_173: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_201, [1, 0]);  primals_201 = None
    addmm_74: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_202, view_264, permute_173);  primals_202 = None
    view_273: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_74, [1, 512, 1536]);  addmm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_274: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_273, [1, 512, 24, -1]);  view_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_174: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_274, [0, 2, 1, 3]);  view_274 = None
    clone_50: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_174, memory_format = torch.contiguous_format);  permute_174 = None
    view_275: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_50, [-1, 512, 64]);  clone_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_175: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_271, [0, 2, 1]);  view_271 = None
    div_24: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_175, full_default_2);  permute_175 = None
    bmm_24: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_267, div_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_276: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_24, [-1, 24, 512, 512]);  bmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    where_61: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_4, view_276);  view_276 = None
    amax_12: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_61, [-1], True)
    sub_74: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_61, amax_12);  where_61 = amax_12 = None
    exp_12: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_74);  sub_74 = None
    sum_13: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_25: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    where_62: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_1, div_25);  div_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_37: "f32[1, 24, 512, 512]" = torch.ops.aten.bernoulli.p(permute_8, 0.9)
    sub_75: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_37);  bernoulli_37 = None
    convert_element_type_50: "b8[1, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_75, torch.bool);  sub_75 = None
    where_63: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_50, full_default_1, where_62)
    mul_138: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_63, 1.1111111111111112);  where_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_277: "f32[24, 512, 512]" = torch.ops.aten.view.default(mul_138, [-1, 512, 512]);  mul_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_25: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_277, view_275)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_278: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_25, [-1, 24, 512, 64]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_177: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_278, [0, 2, 1, 3]);  view_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_51: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_279: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_51, [1, 512, -1]);  clone_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_280: "f32[512, 1536]" = torch.ops.aten.view.default(view_279, [512, 1536]);  view_279 = None
    permute_178: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_203, [1, 0]);  primals_203 = None
    addmm_75: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_204, view_280, permute_178);  primals_204 = None
    view_281: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_75, [1, 512, 1536]);  addmm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_38: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_76: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_38);  bernoulli_38 = None
    convert_element_type_51: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_76, torch.bool);  sub_76 = None
    where_64: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_51, full_default_1, view_281);  view_281 = None
    mul_139: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_64, 1.1111111111111112);  where_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_87: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_139, add_86);  mul_139 = add_86 = None
    var_mean_25 = torch.ops.aten.var_mean.correction(add_87, [2], correction = 0, keepdim = True)
    getitem_50: "f32[1, 512, 1]" = var_mean_25[0]
    getitem_51: "f32[1, 512, 1]" = var_mean_25[1];  var_mean_25 = None
    add_88: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-07);  getitem_50 = None
    rsqrt_25: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
    sub_77: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_87, getitem_51);  add_87 = getitem_51 = None
    mul_140: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_25);  sub_77 = None
    mul_141: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_140, primals_205)
    add_89: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_141, primals_206);  mul_141 = primals_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_282: "f32[512, 1536]" = torch.ops.aten.view.default(add_89, [512, 1536])
    permute_180: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_207, [1, 0]);  primals_207 = None
    addmm_76: "f32[512, 6144]" = torch.ops.aten.addmm.default(primals_208, view_282, permute_180);  primals_208 = None
    view_283: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_76, [1, 512, 6144])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_142: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_283, 0.5)
    mul_143: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_283, 0.7071067811865476);  view_283 = None
    erf_12: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_143);  mul_143 = None
    add_90: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_144: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_142, add_90);  mul_142 = add_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_284: "f32[512, 6144]" = torch.ops.aten.view.default(mul_144, [512, 6144]);  mul_144 = None
    permute_181: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_209, [1, 0]);  primals_209 = None
    addmm_77: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_210, view_284, permute_181);  primals_210 = None
    view_285: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_77, [1, 512, 1536]);  addmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_39: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_78: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_39);  bernoulli_39 = None
    convert_element_type_52: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_78, torch.bool);  sub_78 = None
    where_65: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_52, full_default_1, view_285);  view_285 = None
    mul_145: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_65, 1.1111111111111112);  where_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_91: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_145, add_89);  mul_145 = add_89 = None
    var_mean_26 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
    getitem_52: "f32[1, 512, 1]" = var_mean_26[0]
    getitem_53: "f32[1, 512, 1]" = var_mean_26[1];  var_mean_26 = None
    add_92: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-07);  getitem_52 = None
    rsqrt_26: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
    sub_79: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_91, getitem_53);  add_91 = getitem_53 = None
    mul_146: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_26);  sub_79 = None
    mul_147: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_146, primals_211)
    add_93: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_147, primals_212);  mul_147 = primals_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_286: "f32[512, 1536]" = torch.ops.aten.view.default(add_93, [512, 1536])
    permute_183: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_213, [1, 0]);  primals_213 = None
    addmm_78: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_214, view_286, permute_183);  primals_214 = None
    view_287: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_78, [1, 512, 1536]);  addmm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_288: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_287, [1, 512, 24, -1]);  view_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_184: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_288, [0, 2, 1, 3]);  view_288 = None
    clone_52: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_184, memory_format = torch.contiguous_format);  permute_184 = None
    view_289: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_52, [-1, 512, 64]);  clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_185: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_215, [1, 0]);  primals_215 = None
    addmm_79: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_216, view_286, permute_185);  primals_216 = None
    view_291: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_79, [1, 512, 1536]);  addmm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_292: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_291, [1, 512, 24, -1]);  view_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_186: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None
    clone_53: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_186, memory_format = torch.contiguous_format);  permute_186 = None
    view_293: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_53, [-1, 512, 64]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_187: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_217, [1, 0]);  primals_217 = None
    addmm_80: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_218, view_286, permute_187);  primals_218 = None
    view_295: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_80, [1, 512, 1536]);  addmm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_296: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_295, [1, 512, 24, -1]);  view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_188: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_296, [0, 2, 1, 3]);  view_296 = None
    clone_54: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
    view_297: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_54, [-1, 512, 64]);  clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_189: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_293, [0, 2, 1]);  view_293 = None
    div_26: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_189, full_default_2);  permute_189 = None
    bmm_26: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_289, div_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_298: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_26, [-1, 24, 512, 512]);  bmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    where_66: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_4, view_298);  view_298 = None
    amax_13: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_66, [-1], True)
    sub_80: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_66, amax_13);  where_66 = amax_13 = None
    exp_13: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_80);  sub_80 = None
    sum_14: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_27: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
    where_67: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_1, div_27);  div_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_40: "f32[1, 24, 512, 512]" = torch.ops.aten.bernoulli.p(permute_8, 0.9)
    sub_81: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_40);  bernoulli_40 = None
    convert_element_type_54: "b8[1, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_81, torch.bool);  sub_81 = None
    where_68: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_54, full_default_1, where_67)
    mul_149: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_68, 1.1111111111111112);  where_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_299: "f32[24, 512, 512]" = torch.ops.aten.view.default(mul_149, [-1, 512, 512]);  mul_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_27: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_299, view_297)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_300: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_27, [-1, 24, 512, 64]);  bmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_191: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_300, [0, 2, 1, 3]);  view_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_55: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_191, memory_format = torch.contiguous_format);  permute_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_301: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_55, [1, 512, -1]);  clone_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_302: "f32[512, 1536]" = torch.ops.aten.view.default(view_301, [512, 1536]);  view_301 = None
    permute_192: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_219, [1, 0]);  primals_219 = None
    addmm_81: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_220, view_302, permute_192);  primals_220 = None
    view_303: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_81, [1, 512, 1536]);  addmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_41: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_82: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_41);  bernoulli_41 = None
    convert_element_type_55: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_82, torch.bool);  sub_82 = None
    where_69: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_55, full_default_1, view_303);  view_303 = None
    mul_150: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_69, 1.1111111111111112);  where_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_94: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_150, add_93);  mul_150 = add_93 = None
    var_mean_27 = torch.ops.aten.var_mean.correction(add_94, [2], correction = 0, keepdim = True)
    getitem_54: "f32[1, 512, 1]" = var_mean_27[0]
    getitem_55: "f32[1, 512, 1]" = var_mean_27[1];  var_mean_27 = None
    add_95: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-07);  getitem_54 = None
    rsqrt_27: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
    sub_83: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_94, getitem_55);  add_94 = getitem_55 = None
    mul_151: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_27);  sub_83 = None
    mul_152: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_151, primals_221)
    add_96: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_152, primals_222);  mul_152 = primals_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_304: "f32[512, 1536]" = torch.ops.aten.view.default(add_96, [512, 1536])
    permute_194: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_223, [1, 0]);  primals_223 = None
    addmm_82: "f32[512, 6144]" = torch.ops.aten.addmm.default(primals_224, view_304, permute_194);  primals_224 = None
    view_305: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_82, [1, 512, 6144])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_153: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_305, 0.5)
    mul_154: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_305, 0.7071067811865476);  view_305 = None
    erf_13: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_154);  mul_154 = None
    add_97: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_155: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_153, add_97);  mul_153 = add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_306: "f32[512, 6144]" = torch.ops.aten.view.default(mul_155, [512, 6144]);  mul_155 = None
    permute_195: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_225, [1, 0]);  primals_225 = None
    addmm_83: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_226, view_306, permute_195);  primals_226 = None
    view_307: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_83, [1, 512, 1536]);  addmm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_42: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_84: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_42);  bernoulli_42 = None
    convert_element_type_56: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_84, torch.bool);  sub_84 = None
    where_70: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_56, full_default_1, view_307);  view_307 = None
    mul_156: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_70, 1.1111111111111112);  where_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_98: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_156, add_96);  mul_156 = add_96 = None
    var_mean_28 = torch.ops.aten.var_mean.correction(add_98, [2], correction = 0, keepdim = True)
    getitem_56: "f32[1, 512, 1]" = var_mean_28[0]
    getitem_57: "f32[1, 512, 1]" = var_mean_28[1];  var_mean_28 = None
    add_99: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-07);  getitem_56 = None
    rsqrt_28: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
    sub_85: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_98, getitem_57);  add_98 = getitem_57 = None
    mul_157: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_28);  sub_85 = None
    mul_158: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_157, primals_227)
    add_100: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_158, primals_228);  mul_158 = primals_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_308: "f32[512, 1536]" = torch.ops.aten.view.default(add_100, [512, 1536])
    permute_197: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_229, [1, 0]);  primals_229 = None
    addmm_84: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_230, view_308, permute_197);  primals_230 = None
    view_309: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_84, [1, 512, 1536]);  addmm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_310: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_309, [1, 512, 24, -1]);  view_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_198: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_310, [0, 2, 1, 3]);  view_310 = None
    clone_56: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_198, memory_format = torch.contiguous_format);  permute_198 = None
    view_311: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_56, [-1, 512, 64]);  clone_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_199: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_231, [1, 0]);  primals_231 = None
    addmm_85: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_232, view_308, permute_199);  primals_232 = None
    view_313: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_85, [1, 512, 1536]);  addmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_314: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_313, [1, 512, 24, -1]);  view_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_200: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_314, [0, 2, 1, 3]);  view_314 = None
    clone_57: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_200, memory_format = torch.contiguous_format);  permute_200 = None
    view_315: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_57, [-1, 512, 64]);  clone_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_201: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_233, [1, 0]);  primals_233 = None
    addmm_86: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_234, view_308, permute_201);  primals_234 = None
    view_317: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_86, [1, 512, 1536]);  addmm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_318: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_317, [1, 512, 24, -1]);  view_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_202: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_318, [0, 2, 1, 3]);  view_318 = None
    clone_58: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_202, memory_format = torch.contiguous_format);  permute_202 = None
    view_319: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_58, [-1, 512, 64]);  clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_203: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_315, [0, 2, 1]);  view_315 = None
    div_28: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_203, full_default_2);  permute_203 = None
    bmm_28: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_311, div_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_320: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_28, [-1, 24, 512, 512]);  bmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    where_71: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_4, view_320);  view_320 = None
    amax_14: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_71, [-1], True)
    sub_86: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_71, amax_14);  where_71 = amax_14 = None
    exp_14: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_86);  sub_86 = None
    sum_15: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_29: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
    where_72: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_1, div_29);  div_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_43: "f32[1, 24, 512, 512]" = torch.ops.aten.bernoulli.p(permute_8, 0.9)
    sub_87: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_43);  bernoulli_43 = None
    convert_element_type_58: "b8[1, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_87, torch.bool);  sub_87 = None
    where_73: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_58, full_default_1, where_72)
    mul_160: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_73, 1.1111111111111112);  where_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_321: "f32[24, 512, 512]" = torch.ops.aten.view.default(mul_160, [-1, 512, 512]);  mul_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_29: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_321, view_319)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_322: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_29, [-1, 24, 512, 64]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_205: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_322, [0, 2, 1, 3]);  view_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_59: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_323: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_59, [1, 512, -1]);  clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_324: "f32[512, 1536]" = torch.ops.aten.view.default(view_323, [512, 1536]);  view_323 = None
    permute_206: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_235, [1, 0]);  primals_235 = None
    addmm_87: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_236, view_324, permute_206);  primals_236 = None
    view_325: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_87, [1, 512, 1536]);  addmm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_44: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_88: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_44);  bernoulli_44 = None
    convert_element_type_59: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_88, torch.bool);  sub_88 = None
    where_74: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_59, full_default_1, view_325);  view_325 = None
    mul_161: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_74, 1.1111111111111112);  where_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_101: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_161, add_100);  mul_161 = add_100 = None
    var_mean_29 = torch.ops.aten.var_mean.correction(add_101, [2], correction = 0, keepdim = True)
    getitem_58: "f32[1, 512, 1]" = var_mean_29[0]
    getitem_59: "f32[1, 512, 1]" = var_mean_29[1];  var_mean_29 = None
    add_102: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-07);  getitem_58 = None
    rsqrt_29: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
    sub_89: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_101, getitem_59);  add_101 = getitem_59 = None
    mul_162: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_29);  sub_89 = None
    mul_163: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_162, primals_237)
    add_103: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_163, primals_238);  mul_163 = primals_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_326: "f32[512, 1536]" = torch.ops.aten.view.default(add_103, [512, 1536])
    permute_208: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_239, [1, 0]);  primals_239 = None
    addmm_88: "f32[512, 6144]" = torch.ops.aten.addmm.default(primals_240, view_326, permute_208);  primals_240 = None
    view_327: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_88, [1, 512, 6144])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_164: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_327, 0.5)
    mul_165: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_327, 0.7071067811865476);  view_327 = None
    erf_14: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_165);  mul_165 = None
    add_104: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_166: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_164, add_104);  mul_164 = add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_328: "f32[512, 6144]" = torch.ops.aten.view.default(mul_166, [512, 6144]);  mul_166 = None
    permute_209: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_241, [1, 0]);  primals_241 = None
    addmm_89: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_242, view_328, permute_209);  primals_242 = None
    view_329: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_89, [1, 512, 1536]);  addmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_45: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_90: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_45);  bernoulli_45 = None
    convert_element_type_60: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_90, torch.bool);  sub_90 = None
    where_75: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_60, full_default_1, view_329);  view_329 = None
    mul_167: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_75, 1.1111111111111112);  where_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_105: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_167, add_103);  mul_167 = add_103 = None
    var_mean_30 = torch.ops.aten.var_mean.correction(add_105, [2], correction = 0, keepdim = True)
    getitem_60: "f32[1, 512, 1]" = var_mean_30[0]
    getitem_61: "f32[1, 512, 1]" = var_mean_30[1];  var_mean_30 = None
    add_106: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-07);  getitem_60 = None
    rsqrt_30: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
    sub_91: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_105, getitem_61);  add_105 = getitem_61 = None
    mul_168: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_30);  sub_91 = None
    mul_169: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_168, primals_243)
    add_107: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_169, primals_244);  mul_169 = primals_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_330: "f32[512, 1536]" = torch.ops.aten.view.default(add_107, [512, 1536])
    permute_211: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_245, [1, 0]);  primals_245 = None
    addmm_90: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_246, view_330, permute_211);  primals_246 = None
    view_331: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_90, [1, 512, 1536]);  addmm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_332: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_331, [1, 512, 24, -1]);  view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_212: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_332, [0, 2, 1, 3]);  view_332 = None
    clone_60: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_212, memory_format = torch.contiguous_format);  permute_212 = None
    view_333: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_60, [-1, 512, 64]);  clone_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_213: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_247, [1, 0]);  primals_247 = None
    addmm_91: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_248, view_330, permute_213);  primals_248 = None
    view_335: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_91, [1, 512, 1536]);  addmm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_336: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_335, [1, 512, 24, -1]);  view_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_214: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_336, [0, 2, 1, 3]);  view_336 = None
    clone_61: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_214, memory_format = torch.contiguous_format);  permute_214 = None
    view_337: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_61, [-1, 512, 64]);  clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_215: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_249, [1, 0]);  primals_249 = None
    addmm_92: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_250, view_330, permute_215);  primals_250 = None
    view_339: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_92, [1, 512, 1536]);  addmm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_340: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_339, [1, 512, 24, -1]);  view_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_216: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_340, [0, 2, 1, 3]);  view_340 = None
    clone_62: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None
    view_341: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_62, [-1, 512, 64]);  clone_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_217: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_337, [0, 2, 1]);  view_337 = None
    div_30: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_217, full_default_2);  permute_217 = None
    bmm_30: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_333, div_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_342: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_30, [-1, 24, 512, 512]);  bmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    where_76: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_4, view_342);  view_342 = None
    amax_15: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_76, [-1], True)
    sub_92: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_76, amax_15);  where_76 = amax_15 = None
    exp_15: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_92);  sub_92 = None
    sum_16: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
    div_31: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
    where_77: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_1, div_31);  div_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_46: "f32[1, 24, 512, 512]" = torch.ops.aten.bernoulli.p(permute_8, 0.9)
    sub_93: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_46);  bernoulli_46 = None
    convert_element_type_62: "b8[1, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_93, torch.bool);  sub_93 = None
    where_78: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_62, full_default_1, where_77)
    mul_171: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_78, 1.1111111111111112);  where_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_343: "f32[24, 512, 512]" = torch.ops.aten.view.default(mul_171, [-1, 512, 512]);  mul_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_31: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_343, view_341)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_344: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_31, [-1, 24, 512, 64]);  bmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_219: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_63: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_219, memory_format = torch.contiguous_format);  permute_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_345: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_63, [1, 512, -1]);  clone_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_346: "f32[512, 1536]" = torch.ops.aten.view.default(view_345, [512, 1536]);  view_345 = None
    permute_220: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_251, [1, 0]);  primals_251 = None
    addmm_93: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_252, view_346, permute_220);  primals_252 = None
    view_347: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_93, [1, 512, 1536]);  addmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_47: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_94: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_47);  bernoulli_47 = None
    convert_element_type_63: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_94, torch.bool);  sub_94 = None
    where_79: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_63, full_default_1, view_347);  view_347 = None
    mul_172: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_79, 1.1111111111111112);  where_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_108: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_172, add_107);  mul_172 = add_107 = None
    var_mean_31 = torch.ops.aten.var_mean.correction(add_108, [2], correction = 0, keepdim = True)
    getitem_62: "f32[1, 512, 1]" = var_mean_31[0]
    getitem_63: "f32[1, 512, 1]" = var_mean_31[1];  var_mean_31 = None
    add_109: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-07);  getitem_62 = None
    rsqrt_31: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
    sub_95: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_108, getitem_63);  add_108 = getitem_63 = None
    mul_173: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_31);  sub_95 = None
    mul_174: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_173, primals_253)
    add_110: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_174, primals_254);  mul_174 = primals_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_348: "f32[512, 1536]" = torch.ops.aten.view.default(add_110, [512, 1536])
    permute_222: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_255, [1, 0]);  primals_255 = None
    addmm_94: "f32[512, 6144]" = torch.ops.aten.addmm.default(primals_256, view_348, permute_222);  primals_256 = None
    view_349: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_94, [1, 512, 6144])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_175: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_349, 0.5)
    mul_176: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_349, 0.7071067811865476);  view_349 = None
    erf_15: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_176);  mul_176 = None
    add_111: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_177: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_175, add_111);  mul_175 = add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_350: "f32[512, 6144]" = torch.ops.aten.view.default(mul_177, [512, 6144]);  mul_177 = None
    permute_223: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_257, [1, 0]);  primals_257 = None
    addmm_95: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_258, view_350, permute_223);  primals_258 = None
    view_351: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_95, [1, 512, 1536]);  addmm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_48: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_96: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_48);  bernoulli_48 = None
    convert_element_type_64: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_96, torch.bool);  sub_96 = None
    where_80: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_64, full_default_1, view_351);  view_351 = None
    mul_178: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_80, 1.1111111111111112);  where_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_112: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_178, add_110);  mul_178 = add_110 = None
    var_mean_32 = torch.ops.aten.var_mean.correction(add_112, [2], correction = 0, keepdim = True)
    getitem_64: "f32[1, 512, 1]" = var_mean_32[0]
    getitem_65: "f32[1, 512, 1]" = var_mean_32[1];  var_mean_32 = None
    add_113: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-07);  getitem_64 = None
    rsqrt_32: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
    sub_97: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_112, getitem_65);  add_112 = getitem_65 = None
    mul_179: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_97, rsqrt_32);  sub_97 = None
    mul_180: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_179, primals_259)
    add_114: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_180, primals_260);  mul_180 = primals_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_352: "f32[512, 1536]" = torch.ops.aten.view.default(add_114, [512, 1536])
    permute_225: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_261, [1, 0]);  primals_261 = None
    addmm_96: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_262, view_352, permute_225);  primals_262 = None
    view_353: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_96, [1, 512, 1536]);  addmm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_354: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_353, [1, 512, 24, -1]);  view_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_226: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_354, [0, 2, 1, 3]);  view_354 = None
    clone_64: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_226, memory_format = torch.contiguous_format);  permute_226 = None
    view_355: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_64, [-1, 512, 64]);  clone_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_227: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_263, [1, 0]);  primals_263 = None
    addmm_97: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_264, view_352, permute_227);  primals_264 = None
    view_357: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_97, [1, 512, 1536]);  addmm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_358: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_357, [1, 512, 24, -1]);  view_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_228: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_358, [0, 2, 1, 3]);  view_358 = None
    clone_65: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_228, memory_format = torch.contiguous_format);  permute_228 = None
    view_359: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_65, [-1, 512, 64]);  clone_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_229: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_265, [1, 0]);  primals_265 = None
    addmm_98: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_266, view_352, permute_229);  primals_266 = None
    view_361: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_98, [1, 512, 1536]);  addmm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_362: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_361, [1, 512, 24, -1]);  view_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_230: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_362, [0, 2, 1, 3]);  view_362 = None
    clone_66: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_230, memory_format = torch.contiguous_format);  permute_230 = None
    view_363: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_66, [-1, 512, 64]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_231: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_359, [0, 2, 1]);  view_359 = None
    div_32: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_231, full_default_2);  permute_231 = None
    bmm_32: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_355, div_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_364: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_32, [-1, 24, 512, 512]);  bmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    where_81: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_4, view_364);  view_364 = None
    amax_16: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_81, [-1], True)
    sub_98: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_81, amax_16);  where_81 = amax_16 = None
    exp_16: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_98);  sub_98 = None
    sum_17: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_33: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
    where_82: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_1, div_33);  div_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_49: "f32[1, 24, 512, 512]" = torch.ops.aten.bernoulli.p(permute_8, 0.9)
    sub_99: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_49);  bernoulli_49 = None
    convert_element_type_66: "b8[1, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_99, torch.bool);  sub_99 = None
    where_83: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_66, full_default_1, where_82)
    mul_182: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_83, 1.1111111111111112);  where_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_365: "f32[24, 512, 512]" = torch.ops.aten.view.default(mul_182, [-1, 512, 512]);  mul_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_33: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_365, view_363)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_366: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_33, [-1, 24, 512, 64]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_233: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_366, [0, 2, 1, 3]);  view_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_67: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_233, memory_format = torch.contiguous_format);  permute_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_367: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_67, [1, 512, -1]);  clone_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_368: "f32[512, 1536]" = torch.ops.aten.view.default(view_367, [512, 1536]);  view_367 = None
    permute_234: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_267, [1, 0]);  primals_267 = None
    addmm_99: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_268, view_368, permute_234);  primals_268 = None
    view_369: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_99, [1, 512, 1536]);  addmm_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_50: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_100: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_50);  bernoulli_50 = None
    convert_element_type_67: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_100, torch.bool);  sub_100 = None
    where_84: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_67, full_default_1, view_369);  view_369 = None
    mul_183: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_84, 1.1111111111111112);  where_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_115: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_183, add_114);  mul_183 = add_114 = None
    var_mean_33 = torch.ops.aten.var_mean.correction(add_115, [2], correction = 0, keepdim = True)
    getitem_66: "f32[1, 512, 1]" = var_mean_33[0]
    getitem_67: "f32[1, 512, 1]" = var_mean_33[1];  var_mean_33 = None
    add_116: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-07);  getitem_66 = None
    rsqrt_33: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
    sub_101: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_115, getitem_67);  add_115 = getitem_67 = None
    mul_184: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_101, rsqrt_33);  sub_101 = None
    mul_185: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_184, primals_269)
    add_117: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_185, primals_270);  mul_185 = primals_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_370: "f32[512, 1536]" = torch.ops.aten.view.default(add_117, [512, 1536])
    permute_236: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_271, [1, 0]);  primals_271 = None
    addmm_100: "f32[512, 6144]" = torch.ops.aten.addmm.default(primals_272, view_370, permute_236);  primals_272 = None
    view_371: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_100, [1, 512, 6144])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_186: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_371, 0.5)
    mul_187: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_371, 0.7071067811865476);  view_371 = None
    erf_16: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_187);  mul_187 = None
    add_118: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_188: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_186, add_118);  mul_186 = add_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_372: "f32[512, 6144]" = torch.ops.aten.view.default(mul_188, [512, 6144]);  mul_188 = None
    permute_237: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_273, [1, 0]);  primals_273 = None
    addmm_101: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_274, view_372, permute_237);  primals_274 = None
    view_373: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_101, [1, 512, 1536]);  addmm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_51: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_102: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_51);  bernoulli_51 = None
    convert_element_type_68: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_102, torch.bool);  sub_102 = None
    where_85: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_68, full_default_1, view_373);  view_373 = None
    mul_189: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_85, 1.1111111111111112);  where_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_119: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_189, add_117);  mul_189 = add_117 = None
    var_mean_34 = torch.ops.aten.var_mean.correction(add_119, [2], correction = 0, keepdim = True)
    getitem_68: "f32[1, 512, 1]" = var_mean_34[0]
    getitem_69: "f32[1, 512, 1]" = var_mean_34[1];  var_mean_34 = None
    add_120: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-07);  getitem_68 = None
    rsqrt_34: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
    sub_103: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_119, getitem_69);  add_119 = getitem_69 = None
    mul_190: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_103, rsqrt_34);  sub_103 = None
    mul_191: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_190, primals_275)
    add_121: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_191, primals_276);  mul_191 = primals_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_374: "f32[512, 1536]" = torch.ops.aten.view.default(add_121, [512, 1536])
    permute_239: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_277, [1, 0]);  primals_277 = None
    addmm_102: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_278, view_374, permute_239);  primals_278 = None
    view_375: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_102, [1, 512, 1536]);  addmm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_376: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_375, [1, 512, 24, -1]);  view_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_240: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_376, [0, 2, 1, 3]);  view_376 = None
    clone_68: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_240, memory_format = torch.contiguous_format);  permute_240 = None
    view_377: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_68, [-1, 512, 64]);  clone_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_241: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_279, [1, 0]);  primals_279 = None
    addmm_103: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_280, view_374, permute_241);  primals_280 = None
    view_379: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_103, [1, 512, 1536]);  addmm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_380: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_379, [1, 512, 24, -1]);  view_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_242: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_380, [0, 2, 1, 3]);  view_380 = None
    clone_69: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_242, memory_format = torch.contiguous_format);  permute_242 = None
    view_381: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_69, [-1, 512, 64]);  clone_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_243: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_281, [1, 0]);  primals_281 = None
    addmm_104: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_282, view_374, permute_243);  primals_282 = None
    view_383: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_104, [1, 512, 1536]);  addmm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_384: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_383, [1, 512, 24, -1]);  view_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_244: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_384, [0, 2, 1, 3]);  view_384 = None
    clone_70: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_244, memory_format = torch.contiguous_format);  permute_244 = None
    view_385: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_70, [-1, 512, 64]);  clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_245: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_381, [0, 2, 1]);  view_381 = None
    div_34: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_245, full_default_2);  permute_245 = None
    bmm_34: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_377, div_34)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_386: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_34, [-1, 24, 512, 512]);  bmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    where_86: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_4, view_386);  view_386 = None
    amax_17: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_86, [-1], True)
    sub_104: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_86, amax_17);  where_86 = amax_17 = None
    exp_17: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_104);  sub_104 = None
    sum_18: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
    div_35: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
    where_87: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_1, div_35);  div_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_52: "f32[1, 24, 512, 512]" = torch.ops.aten.bernoulli.p(permute_8, 0.9)
    sub_105: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_52);  bernoulli_52 = None
    convert_element_type_70: "b8[1, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_105, torch.bool);  sub_105 = None
    where_88: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_70, full_default_1, where_87)
    mul_193: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_88, 1.1111111111111112);  where_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_387: "f32[24, 512, 512]" = torch.ops.aten.view.default(mul_193, [-1, 512, 512]);  mul_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_35: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_387, view_385)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_388: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_35, [-1, 24, 512, 64]);  bmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_247: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_388, [0, 2, 1, 3]);  view_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_71: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_247, memory_format = torch.contiguous_format);  permute_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_389: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_71, [1, 512, -1]);  clone_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_390: "f32[512, 1536]" = torch.ops.aten.view.default(view_389, [512, 1536]);  view_389 = None
    permute_248: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_283, [1, 0]);  primals_283 = None
    addmm_105: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_284, view_390, permute_248);  primals_284 = None
    view_391: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_105, [1, 512, 1536]);  addmm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_53: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_106: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_53);  bernoulli_53 = None
    convert_element_type_71: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_106, torch.bool);  sub_106 = None
    where_89: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_71, full_default_1, view_391);  view_391 = None
    mul_194: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_89, 1.1111111111111112);  where_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_122: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_194, add_121);  mul_194 = add_121 = None
    var_mean_35 = torch.ops.aten.var_mean.correction(add_122, [2], correction = 0, keepdim = True)
    getitem_70: "f32[1, 512, 1]" = var_mean_35[0]
    getitem_71: "f32[1, 512, 1]" = var_mean_35[1];  var_mean_35 = None
    add_123: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-07);  getitem_70 = None
    rsqrt_35: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
    sub_107: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_122, getitem_71);  add_122 = getitem_71 = None
    mul_195: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_107, rsqrt_35);  sub_107 = None
    mul_196: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_195, primals_285)
    add_124: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_196, primals_286);  mul_196 = primals_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_392: "f32[512, 1536]" = torch.ops.aten.view.default(add_124, [512, 1536])
    permute_250: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_287, [1, 0]);  primals_287 = None
    addmm_106: "f32[512, 6144]" = torch.ops.aten.addmm.default(primals_288, view_392, permute_250);  primals_288 = None
    view_393: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_106, [1, 512, 6144])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_197: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_393, 0.5)
    mul_198: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_393, 0.7071067811865476);  view_393 = None
    erf_17: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_198);  mul_198 = None
    add_125: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_199: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_197, add_125);  mul_197 = add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_394: "f32[512, 6144]" = torch.ops.aten.view.default(mul_199, [512, 6144]);  mul_199 = None
    permute_251: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_289, [1, 0]);  primals_289 = None
    addmm_107: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_290, view_394, permute_251);  primals_290 = None
    view_395: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_107, [1, 512, 1536]);  addmm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_54: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_108: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_54);  bernoulli_54 = None
    convert_element_type_72: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_108, torch.bool);  sub_108 = None
    where_90: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_72, full_default_1, view_395);  view_395 = None
    mul_200: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_90, 1.1111111111111112);  where_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_126: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_200, add_124);  mul_200 = add_124 = None
    var_mean_36 = torch.ops.aten.var_mean.correction(add_126, [2], correction = 0, keepdim = True)
    getitem_72: "f32[1, 512, 1]" = var_mean_36[0]
    getitem_73: "f32[1, 512, 1]" = var_mean_36[1];  var_mean_36 = None
    add_127: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-07);  getitem_72 = None
    rsqrt_36: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
    sub_109: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_126, getitem_73);  add_126 = getitem_73 = None
    mul_201: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_109, rsqrt_36);  sub_109 = None
    mul_202: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_201, primals_291)
    add_128: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_202, primals_292);  mul_202 = primals_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_396: "f32[512, 1536]" = torch.ops.aten.view.default(add_128, [512, 1536])
    permute_253: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_293, [1, 0]);  primals_293 = None
    addmm_108: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_294, view_396, permute_253);  primals_294 = None
    view_397: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_108, [1, 512, 1536]);  addmm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_398: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_397, [1, 512, 24, -1]);  view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_254: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_398, [0, 2, 1, 3]);  view_398 = None
    clone_72: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_254, memory_format = torch.contiguous_format);  permute_254 = None
    view_399: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_72, [-1, 512, 64]);  clone_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_255: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_295, [1, 0]);  primals_295 = None
    addmm_109: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_296, view_396, permute_255);  primals_296 = None
    view_401: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_109, [1, 512, 1536]);  addmm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_402: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_401, [1, 512, 24, -1]);  view_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_256: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
    clone_73: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_256, memory_format = torch.contiguous_format);  permute_256 = None
    view_403: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_73, [-1, 512, 64]);  clone_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_257: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_297, [1, 0]);  primals_297 = None
    addmm_110: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_298, view_396, permute_257);  primals_298 = None
    view_405: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_110, [1, 512, 1536]);  addmm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_406: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_405, [1, 512, 24, -1]);  view_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_258: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_406, [0, 2, 1, 3]);  view_406 = None
    clone_74: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_258, memory_format = torch.contiguous_format);  permute_258 = None
    view_407: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_74, [-1, 512, 64]);  clone_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_259: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_403, [0, 2, 1]);  view_403 = None
    div_36: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_259, full_default_2);  permute_259 = None
    bmm_36: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_399, div_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_408: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_36, [-1, 24, 512, 512]);  bmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    where_91: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_4, view_408);  view_408 = None
    amax_18: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_91, [-1], True)
    sub_110: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_91, amax_18);  where_91 = amax_18 = None
    exp_18: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_110);  sub_110 = None
    sum_19: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
    div_37: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
    where_92: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_1, div_37);  div_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_55: "f32[1, 24, 512, 512]" = torch.ops.aten.bernoulli.p(permute_8, 0.9)
    sub_111: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_55);  bernoulli_55 = None
    convert_element_type_74: "b8[1, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_111, torch.bool);  sub_111 = None
    where_93: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_74, full_default_1, where_92)
    mul_204: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_93, 1.1111111111111112);  where_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_409: "f32[24, 512, 512]" = torch.ops.aten.view.default(mul_204, [-1, 512, 512]);  mul_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_37: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_409, view_407)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_410: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_37, [-1, 24, 512, 64]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_261: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_410, [0, 2, 1, 3]);  view_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_75: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_261, memory_format = torch.contiguous_format);  permute_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_411: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_75, [1, 512, -1]);  clone_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_412: "f32[512, 1536]" = torch.ops.aten.view.default(view_411, [512, 1536]);  view_411 = None
    permute_262: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_299, [1, 0]);  primals_299 = None
    addmm_111: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_300, view_412, permute_262);  primals_300 = None
    view_413: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_111, [1, 512, 1536]);  addmm_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_56: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_112: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_56);  bernoulli_56 = None
    convert_element_type_75: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_112, torch.bool);  sub_112 = None
    where_94: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_75, full_default_1, view_413);  view_413 = None
    mul_205: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_94, 1.1111111111111112);  where_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_129: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_205, add_128);  mul_205 = add_128 = None
    var_mean_37 = torch.ops.aten.var_mean.correction(add_129, [2], correction = 0, keepdim = True)
    getitem_74: "f32[1, 512, 1]" = var_mean_37[0]
    getitem_75: "f32[1, 512, 1]" = var_mean_37[1];  var_mean_37 = None
    add_130: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-07);  getitem_74 = None
    rsqrt_37: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
    sub_113: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_129, getitem_75);  add_129 = getitem_75 = None
    mul_206: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_113, rsqrt_37);  sub_113 = None
    mul_207: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_206, primals_301)
    add_131: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_207, primals_302);  mul_207 = primals_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_414: "f32[512, 1536]" = torch.ops.aten.view.default(add_131, [512, 1536])
    permute_264: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_303, [1, 0]);  primals_303 = None
    addmm_112: "f32[512, 6144]" = torch.ops.aten.addmm.default(primals_304, view_414, permute_264);  primals_304 = None
    view_415: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_112, [1, 512, 6144])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_208: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_415, 0.5)
    mul_209: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_415, 0.7071067811865476);  view_415 = None
    erf_18: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_209);  mul_209 = None
    add_132: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_210: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_208, add_132);  mul_208 = add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_416: "f32[512, 6144]" = torch.ops.aten.view.default(mul_210, [512, 6144]);  mul_210 = None
    permute_265: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_305, [1, 0]);  primals_305 = None
    addmm_113: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_306, view_416, permute_265);  primals_306 = None
    view_417: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_113, [1, 512, 1536]);  addmm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_57: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_114: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_57);  bernoulli_57 = None
    convert_element_type_76: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_114, torch.bool);  sub_114 = None
    where_95: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_76, full_default_1, view_417);  view_417 = None
    mul_211: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_95, 1.1111111111111112);  where_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_133: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_211, add_131);  mul_211 = add_131 = None
    var_mean_38 = torch.ops.aten.var_mean.correction(add_133, [2], correction = 0, keepdim = True)
    getitem_76: "f32[1, 512, 1]" = var_mean_38[0]
    getitem_77: "f32[1, 512, 1]" = var_mean_38[1];  var_mean_38 = None
    add_134: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-07);  getitem_76 = None
    rsqrt_38: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
    sub_115: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_133, getitem_77);  add_133 = getitem_77 = None
    mul_212: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_115, rsqrt_38);  sub_115 = None
    mul_213: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_212, primals_307)
    add_135: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_213, primals_308);  mul_213 = primals_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_418: "f32[512, 1536]" = torch.ops.aten.view.default(add_135, [512, 1536])
    permute_267: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_309, [1, 0]);  primals_309 = None
    addmm_114: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_310, view_418, permute_267);  primals_310 = None
    view_419: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_114, [1, 512, 1536]);  addmm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_420: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_419, [1, 512, 24, -1]);  view_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_268: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_420, [0, 2, 1, 3]);  view_420 = None
    clone_76: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_268, memory_format = torch.contiguous_format);  permute_268 = None
    view_421: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_76, [-1, 512, 64]);  clone_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_269: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_311, [1, 0]);  primals_311 = None
    addmm_115: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_312, view_418, permute_269);  primals_312 = None
    view_423: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_115, [1, 512, 1536]);  addmm_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_424: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_423, [1, 512, 24, -1]);  view_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_270: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_424, [0, 2, 1, 3]);  view_424 = None
    clone_77: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_270, memory_format = torch.contiguous_format);  permute_270 = None
    view_425: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_77, [-1, 512, 64]);  clone_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_271: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_313, [1, 0]);  primals_313 = None
    addmm_116: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_314, view_418, permute_271);  primals_314 = None
    view_427: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_116, [1, 512, 1536]);  addmm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_428: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_427, [1, 512, 24, -1]);  view_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_272: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_428, [0, 2, 1, 3]);  view_428 = None
    clone_78: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_272, memory_format = torch.contiguous_format);  permute_272 = None
    view_429: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_78, [-1, 512, 64]);  clone_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_273: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_425, [0, 2, 1]);  view_425 = None
    div_38: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_273, full_default_2);  permute_273 = None
    bmm_38: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_421, div_38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_430: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_38, [-1, 24, 512, 512]);  bmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    where_96: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_4, view_430);  view_430 = None
    amax_19: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_96, [-1], True)
    sub_116: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_96, amax_19);  where_96 = amax_19 = None
    exp_19: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_116);  sub_116 = None
    sum_20: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
    div_39: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
    where_97: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_1, div_39);  div_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_58: "f32[1, 24, 512, 512]" = torch.ops.aten.bernoulli.p(permute_8, 0.9)
    sub_117: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_58);  bernoulli_58 = None
    convert_element_type_78: "b8[1, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_117, torch.bool);  sub_117 = None
    where_98: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_78, full_default_1, where_97)
    mul_215: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_98, 1.1111111111111112);  where_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_431: "f32[24, 512, 512]" = torch.ops.aten.view.default(mul_215, [-1, 512, 512]);  mul_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_39: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_431, view_429)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_432: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_39, [-1, 24, 512, 64]);  bmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_275: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_432, [0, 2, 1, 3]);  view_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_79: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_275, memory_format = torch.contiguous_format);  permute_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_433: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_79, [1, 512, -1]);  clone_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_434: "f32[512, 1536]" = torch.ops.aten.view.default(view_433, [512, 1536]);  view_433 = None
    permute_276: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_315, [1, 0]);  primals_315 = None
    addmm_117: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_316, view_434, permute_276);  primals_316 = None
    view_435: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_117, [1, 512, 1536]);  addmm_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_59: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_118: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_59);  bernoulli_59 = None
    convert_element_type_79: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_118, torch.bool);  sub_118 = None
    where_99: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_79, full_default_1, view_435);  view_435 = None
    mul_216: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_99, 1.1111111111111112);  where_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_136: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_216, add_135);  mul_216 = add_135 = None
    var_mean_39 = torch.ops.aten.var_mean.correction(add_136, [2], correction = 0, keepdim = True)
    getitem_78: "f32[1, 512, 1]" = var_mean_39[0]
    getitem_79: "f32[1, 512, 1]" = var_mean_39[1];  var_mean_39 = None
    add_137: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-07);  getitem_78 = None
    rsqrt_39: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
    sub_119: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_136, getitem_79);  add_136 = getitem_79 = None
    mul_217: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_119, rsqrt_39);  sub_119 = None
    mul_218: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_217, primals_317)
    add_138: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_218, primals_318);  mul_218 = primals_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_436: "f32[512, 1536]" = torch.ops.aten.view.default(add_138, [512, 1536])
    permute_278: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_319, [1, 0]);  primals_319 = None
    addmm_118: "f32[512, 6144]" = torch.ops.aten.addmm.default(primals_320, view_436, permute_278);  primals_320 = None
    view_437: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_118, [1, 512, 6144])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_219: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_437, 0.5)
    mul_220: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_437, 0.7071067811865476);  view_437 = None
    erf_19: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_220);  mul_220 = None
    add_139: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_221: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_219, add_139);  mul_219 = add_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_438: "f32[512, 6144]" = torch.ops.aten.view.default(mul_221, [512, 6144]);  mul_221 = None
    permute_279: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_321, [1, 0]);  primals_321 = None
    addmm_119: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_322, view_438, permute_279);  primals_322 = None
    view_439: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_119, [1, 512, 1536]);  addmm_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_60: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_120: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_60);  bernoulli_60 = None
    convert_element_type_80: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_120, torch.bool);  sub_120 = None
    where_100: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_80, full_default_1, view_439);  view_439 = None
    mul_222: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_100, 1.1111111111111112);  where_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_140: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_222, add_138);  mul_222 = add_138 = None
    var_mean_40 = torch.ops.aten.var_mean.correction(add_140, [2], correction = 0, keepdim = True)
    getitem_80: "f32[1, 512, 1]" = var_mean_40[0]
    getitem_81: "f32[1, 512, 1]" = var_mean_40[1];  var_mean_40 = None
    add_141: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-07);  getitem_80 = None
    rsqrt_40: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
    sub_121: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_140, getitem_81);  add_140 = getitem_81 = None
    mul_223: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_121, rsqrt_40);  sub_121 = None
    mul_224: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_223, primals_323)
    add_142: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_224, primals_324);  mul_224 = primals_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_440: "f32[512, 1536]" = torch.ops.aten.view.default(add_142, [512, 1536])
    permute_281: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_325, [1, 0]);  primals_325 = None
    addmm_120: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_326, view_440, permute_281);  primals_326 = None
    view_441: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_120, [1, 512, 1536]);  addmm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_442: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_441, [1, 512, 24, -1]);  view_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_282: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_442, [0, 2, 1, 3]);  view_442 = None
    clone_80: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_282, memory_format = torch.contiguous_format);  permute_282 = None
    view_443: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_80, [-1, 512, 64]);  clone_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_283: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_327, [1, 0]);  primals_327 = None
    addmm_121: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_328, view_440, permute_283);  primals_328 = None
    view_445: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_121, [1, 512, 1536]);  addmm_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_446: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_445, [1, 512, 24, -1]);  view_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_284: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_446, [0, 2, 1, 3]);  view_446 = None
    clone_81: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_284, memory_format = torch.contiguous_format);  permute_284 = None
    view_447: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_81, [-1, 512, 64]);  clone_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_285: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_329, [1, 0]);  primals_329 = None
    addmm_122: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_330, view_440, permute_285);  primals_330 = None
    view_449: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_122, [1, 512, 1536]);  addmm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_450: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_449, [1, 512, 24, -1]);  view_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_286: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_450, [0, 2, 1, 3]);  view_450 = None
    clone_82: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_286, memory_format = torch.contiguous_format);  permute_286 = None
    view_451: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_82, [-1, 512, 64]);  clone_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_287: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_447, [0, 2, 1]);  view_447 = None
    div_40: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_287, full_default_2);  permute_287 = None
    bmm_40: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_443, div_40)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_452: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_40, [-1, 24, 512, 512]);  bmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    where_101: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_4, view_452);  view_452 = None
    amax_20: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_101, [-1], True)
    sub_122: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_101, amax_20);  where_101 = amax_20 = None
    exp_20: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_122);  sub_122 = None
    sum_21: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
    div_41: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
    where_102: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_1, div_41);  div_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_61: "f32[1, 24, 512, 512]" = torch.ops.aten.bernoulli.p(permute_8, 0.9)
    sub_123: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_61);  bernoulli_61 = None
    convert_element_type_82: "b8[1, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_123, torch.bool);  sub_123 = None
    where_103: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_82, full_default_1, where_102)
    mul_226: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_103, 1.1111111111111112);  where_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_453: "f32[24, 512, 512]" = torch.ops.aten.view.default(mul_226, [-1, 512, 512]);  mul_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_41: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_453, view_451)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_454: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_41, [-1, 24, 512, 64]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_289: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_454, [0, 2, 1, 3]);  view_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_83: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_289, memory_format = torch.contiguous_format);  permute_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_455: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_83, [1, 512, -1]);  clone_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_456: "f32[512, 1536]" = torch.ops.aten.view.default(view_455, [512, 1536]);  view_455 = None
    permute_290: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_331, [1, 0]);  primals_331 = None
    addmm_123: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_332, view_456, permute_290);  primals_332 = None
    view_457: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_123, [1, 512, 1536]);  addmm_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_62: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_124: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_62);  bernoulli_62 = None
    convert_element_type_83: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_124, torch.bool);  sub_124 = None
    where_104: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_83, full_default_1, view_457);  view_457 = None
    mul_227: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_104, 1.1111111111111112);  where_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_143: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_227, add_142);  mul_227 = add_142 = None
    var_mean_41 = torch.ops.aten.var_mean.correction(add_143, [2], correction = 0, keepdim = True)
    getitem_82: "f32[1, 512, 1]" = var_mean_41[0]
    getitem_83: "f32[1, 512, 1]" = var_mean_41[1];  var_mean_41 = None
    add_144: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-07);  getitem_82 = None
    rsqrt_41: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    sub_125: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_143, getitem_83);  add_143 = getitem_83 = None
    mul_228: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_125, rsqrt_41);  sub_125 = None
    mul_229: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_228, primals_333)
    add_145: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_229, primals_334);  mul_229 = primals_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_458: "f32[512, 1536]" = torch.ops.aten.view.default(add_145, [512, 1536])
    permute_292: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_335, [1, 0]);  primals_335 = None
    addmm_124: "f32[512, 6144]" = torch.ops.aten.addmm.default(primals_336, view_458, permute_292);  primals_336 = None
    view_459: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_124, [1, 512, 6144])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_230: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_459, 0.5)
    mul_231: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_459, 0.7071067811865476);  view_459 = None
    erf_20: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_231);  mul_231 = None
    add_146: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_232: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_230, add_146);  mul_230 = add_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_460: "f32[512, 6144]" = torch.ops.aten.view.default(mul_232, [512, 6144]);  mul_232 = None
    permute_293: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_337, [1, 0]);  primals_337 = None
    addmm_125: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_338, view_460, permute_293);  primals_338 = None
    view_461: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_125, [1, 512, 1536]);  addmm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_63: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_126: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_63);  bernoulli_63 = None
    convert_element_type_84: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_126, torch.bool);  sub_126 = None
    where_105: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_84, full_default_1, view_461);  view_461 = None
    mul_233: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_105, 1.1111111111111112);  where_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_147: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_233, add_145);  mul_233 = add_145 = None
    var_mean_42 = torch.ops.aten.var_mean.correction(add_147, [2], correction = 0, keepdim = True)
    getitem_84: "f32[1, 512, 1]" = var_mean_42[0]
    getitem_85: "f32[1, 512, 1]" = var_mean_42[1];  var_mean_42 = None
    add_148: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-07);  getitem_84 = None
    rsqrt_42: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
    sub_127: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_147, getitem_85);  add_147 = getitem_85 = None
    mul_234: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_127, rsqrt_42);  sub_127 = None
    mul_235: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_234, primals_339)
    add_149: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_235, primals_340);  mul_235 = primals_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_462: "f32[512, 1536]" = torch.ops.aten.view.default(add_149, [512, 1536])
    permute_295: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_341, [1, 0]);  primals_341 = None
    addmm_126: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_342, view_462, permute_295);  primals_342 = None
    view_463: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_126, [1, 512, 1536]);  addmm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_464: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_463, [1, 512, 24, -1]);  view_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_296: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_464, [0, 2, 1, 3]);  view_464 = None
    clone_84: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_296, memory_format = torch.contiguous_format);  permute_296 = None
    view_465: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_84, [-1, 512, 64]);  clone_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_297: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_343, [1, 0]);  primals_343 = None
    addmm_127: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_344, view_462, permute_297);  primals_344 = None
    view_467: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_127, [1, 512, 1536]);  addmm_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_468: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_467, [1, 512, 24, -1]);  view_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_298: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_468, [0, 2, 1, 3]);  view_468 = None
    clone_85: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_298, memory_format = torch.contiguous_format);  permute_298 = None
    view_469: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_85, [-1, 512, 64]);  clone_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_299: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_345, [1, 0]);  primals_345 = None
    addmm_128: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_346, view_462, permute_299);  primals_346 = None
    view_471: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_128, [1, 512, 1536]);  addmm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_472: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_471, [1, 512, 24, -1]);  view_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_300: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_472, [0, 2, 1, 3]);  view_472 = None
    clone_86: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_300, memory_format = torch.contiguous_format);  permute_300 = None
    view_473: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_86, [-1, 512, 64]);  clone_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_301: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_469, [0, 2, 1]);  view_469 = None
    div_42: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_301, full_default_2);  permute_301 = None
    bmm_42: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_465, div_42)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_474: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_42, [-1, 24, 512, 512]);  bmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    where_106: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_4, view_474);  view_474 = None
    amax_21: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_106, [-1], True)
    sub_128: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_106, amax_21);  where_106 = amax_21 = None
    exp_21: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_128);  sub_128 = None
    sum_22: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
    div_43: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
    where_107: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_1, div_43);  div_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_64: "f32[1, 24, 512, 512]" = torch.ops.aten.bernoulli.p(permute_8, 0.9)
    sub_129: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_64);  bernoulli_64 = None
    convert_element_type_86: "b8[1, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_129, torch.bool);  sub_129 = None
    where_108: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_86, full_default_1, where_107)
    mul_237: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_108, 1.1111111111111112);  where_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_475: "f32[24, 512, 512]" = torch.ops.aten.view.default(mul_237, [-1, 512, 512]);  mul_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_43: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_475, view_473)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_476: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_43, [-1, 24, 512, 64]);  bmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_303: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_476, [0, 2, 1, 3]);  view_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_87: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_303, memory_format = torch.contiguous_format);  permute_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_477: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_87, [1, 512, -1]);  clone_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_478: "f32[512, 1536]" = torch.ops.aten.view.default(view_477, [512, 1536]);  view_477 = None
    permute_304: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_347, [1, 0]);  primals_347 = None
    addmm_129: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_348, view_478, permute_304);  primals_348 = None
    view_479: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_129, [1, 512, 1536]);  addmm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_65: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_130: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_65);  bernoulli_65 = None
    convert_element_type_87: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_130, torch.bool);  sub_130 = None
    where_109: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_87, full_default_1, view_479);  view_479 = None
    mul_238: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_109, 1.1111111111111112);  where_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_150: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_238, add_149);  mul_238 = add_149 = None
    var_mean_43 = torch.ops.aten.var_mean.correction(add_150, [2], correction = 0, keepdim = True)
    getitem_86: "f32[1, 512, 1]" = var_mean_43[0]
    getitem_87: "f32[1, 512, 1]" = var_mean_43[1];  var_mean_43 = None
    add_151: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-07);  getitem_86 = None
    rsqrt_43: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
    sub_131: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_150, getitem_87);  add_150 = getitem_87 = None
    mul_239: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_131, rsqrt_43);  sub_131 = None
    mul_240: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_239, primals_349)
    add_152: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_240, primals_350);  mul_240 = primals_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_480: "f32[512, 1536]" = torch.ops.aten.view.default(add_152, [512, 1536])
    permute_306: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_351, [1, 0]);  primals_351 = None
    addmm_130: "f32[512, 6144]" = torch.ops.aten.addmm.default(primals_352, view_480, permute_306);  primals_352 = None
    view_481: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_130, [1, 512, 6144])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_241: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_481, 0.5)
    mul_242: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_481, 0.7071067811865476);  view_481 = None
    erf_21: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_242);  mul_242 = None
    add_153: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_243: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_241, add_153);  mul_241 = add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_482: "f32[512, 6144]" = torch.ops.aten.view.default(mul_243, [512, 6144]);  mul_243 = None
    permute_307: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_353, [1, 0]);  primals_353 = None
    addmm_131: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_354, view_482, permute_307);  primals_354 = None
    view_483: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_131, [1, 512, 1536]);  addmm_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_66: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_132: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_66);  bernoulli_66 = None
    convert_element_type_88: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_132, torch.bool);  sub_132 = None
    where_110: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_88, full_default_1, view_483);  view_483 = None
    mul_244: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_110, 1.1111111111111112);  where_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_154: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_244, add_152);  mul_244 = add_152 = None
    var_mean_44 = torch.ops.aten.var_mean.correction(add_154, [2], correction = 0, keepdim = True)
    getitem_88: "f32[1, 512, 1]" = var_mean_44[0]
    getitem_89: "f32[1, 512, 1]" = var_mean_44[1];  var_mean_44 = None
    add_155: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-07);  getitem_88 = None
    rsqrt_44: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
    sub_133: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_154, getitem_89);  add_154 = getitem_89 = None
    mul_245: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_133, rsqrt_44);  sub_133 = None
    mul_246: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_245, primals_355)
    add_156: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_246, primals_356);  mul_246 = primals_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_484: "f32[512, 1536]" = torch.ops.aten.view.default(add_156, [512, 1536])
    permute_309: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_357, [1, 0]);  primals_357 = None
    addmm_132: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_358, view_484, permute_309);  primals_358 = None
    view_485: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_132, [1, 512, 1536]);  addmm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_486: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_485, [1, 512, 24, -1]);  view_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_310: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_486, [0, 2, 1, 3]);  view_486 = None
    clone_88: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_310, memory_format = torch.contiguous_format);  permute_310 = None
    view_487: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_88, [-1, 512, 64]);  clone_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_311: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_359, [1, 0]);  primals_359 = None
    addmm_133: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_360, view_484, permute_311);  primals_360 = None
    view_489: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_133, [1, 512, 1536]);  addmm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_490: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_489, [1, 512, 24, -1]);  view_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_312: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_490, [0, 2, 1, 3]);  view_490 = None
    clone_89: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_312, memory_format = torch.contiguous_format);  permute_312 = None
    view_491: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_89, [-1, 512, 64]);  clone_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_313: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_361, [1, 0]);  primals_361 = None
    addmm_134: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_362, view_484, permute_313);  primals_362 = None
    view_493: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_134, [1, 512, 1536]);  addmm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_494: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_493, [1, 512, 24, -1]);  view_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_314: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_494, [0, 2, 1, 3]);  view_494 = None
    clone_90: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_314, memory_format = torch.contiguous_format);  permute_314 = None
    view_495: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_90, [-1, 512, 64]);  clone_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_315: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_491, [0, 2, 1]);  view_491 = None
    div_44: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_315, full_default_2);  permute_315 = None
    bmm_44: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_487, div_44)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_496: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_44, [-1, 24, 512, 512]);  bmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    where_111: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_4, view_496);  view_496 = None
    amax_22: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_111, [-1], True)
    sub_134: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_111, amax_22);  where_111 = amax_22 = None
    exp_22: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_134);  sub_134 = None
    sum_23: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
    div_45: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
    where_112: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_1, div_45);  div_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_67: "f32[1, 24, 512, 512]" = torch.ops.aten.bernoulli.p(permute_8, 0.9)
    sub_135: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_67);  bernoulli_67 = None
    convert_element_type_90: "b8[1, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_135, torch.bool);  sub_135 = None
    where_113: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_90, full_default_1, where_112)
    mul_248: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_113, 1.1111111111111112);  where_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_497: "f32[24, 512, 512]" = torch.ops.aten.view.default(mul_248, [-1, 512, 512]);  mul_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_45: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_497, view_495)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_498: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_45, [-1, 24, 512, 64]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_317: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_498, [0, 2, 1, 3]);  view_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_91: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_317, memory_format = torch.contiguous_format);  permute_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_499: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_91, [1, 512, -1]);  clone_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_500: "f32[512, 1536]" = torch.ops.aten.view.default(view_499, [512, 1536]);  view_499 = None
    permute_318: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_363, [1, 0]);  primals_363 = None
    addmm_135: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_364, view_500, permute_318);  primals_364 = None
    view_501: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_135, [1, 512, 1536]);  addmm_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_68: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_136: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_68);  bernoulli_68 = None
    convert_element_type_91: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_136, torch.bool);  sub_136 = None
    where_114: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_91, full_default_1, view_501);  view_501 = None
    mul_249: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_114, 1.1111111111111112);  where_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_157: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_249, add_156);  mul_249 = add_156 = None
    var_mean_45 = torch.ops.aten.var_mean.correction(add_157, [2], correction = 0, keepdim = True)
    getitem_90: "f32[1, 512, 1]" = var_mean_45[0]
    getitem_91: "f32[1, 512, 1]" = var_mean_45[1];  var_mean_45 = None
    add_158: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-07);  getitem_90 = None
    rsqrt_45: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
    sub_137: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_157, getitem_91);  add_157 = getitem_91 = None
    mul_250: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_137, rsqrt_45);  sub_137 = None
    mul_251: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_250, primals_365)
    add_159: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_251, primals_366);  mul_251 = primals_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_502: "f32[512, 1536]" = torch.ops.aten.view.default(add_159, [512, 1536])
    permute_320: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_367, [1, 0]);  primals_367 = None
    addmm_136: "f32[512, 6144]" = torch.ops.aten.addmm.default(primals_368, view_502, permute_320);  primals_368 = None
    view_503: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_136, [1, 512, 6144])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_252: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_503, 0.5)
    mul_253: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_503, 0.7071067811865476);  view_503 = None
    erf_22: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_253);  mul_253 = None
    add_160: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_254: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_252, add_160);  mul_252 = add_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_504: "f32[512, 6144]" = torch.ops.aten.view.default(mul_254, [512, 6144]);  mul_254 = None
    permute_321: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_369, [1, 0]);  primals_369 = None
    addmm_137: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_370, view_504, permute_321);  primals_370 = None
    view_505: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_137, [1, 512, 1536]);  addmm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_69: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_138: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_69);  bernoulli_69 = None
    convert_element_type_92: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_138, torch.bool);  sub_138 = None
    where_115: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_92, full_default_1, view_505);  view_505 = None
    mul_255: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_115, 1.1111111111111112);  where_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_161: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_255, add_159);  mul_255 = add_159 = None
    var_mean_46 = torch.ops.aten.var_mean.correction(add_161, [2], correction = 0, keepdim = True)
    getitem_92: "f32[1, 512, 1]" = var_mean_46[0]
    getitem_93: "f32[1, 512, 1]" = var_mean_46[1];  var_mean_46 = None
    add_162: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-07);  getitem_92 = None
    rsqrt_46: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
    sub_139: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_161, getitem_93);  add_161 = getitem_93 = None
    mul_256: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_139, rsqrt_46);  sub_139 = None
    mul_257: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_256, primals_371)
    add_163: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_257, primals_372);  mul_257 = primals_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_506: "f32[512, 1536]" = torch.ops.aten.view.default(add_163, [512, 1536])
    permute_323: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_373, [1, 0]);  primals_373 = None
    addmm_138: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_374, view_506, permute_323);  primals_374 = None
    view_507: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_138, [1, 512, 1536]);  addmm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_508: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_507, [1, 512, 24, -1]);  view_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_324: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_508, [0, 2, 1, 3]);  view_508 = None
    clone_92: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_324, memory_format = torch.contiguous_format);  permute_324 = None
    view_509: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_92, [-1, 512, 64]);  clone_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_325: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_375, [1, 0]);  primals_375 = None
    addmm_139: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_376, view_506, permute_325);  primals_376 = None
    view_511: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_139, [1, 512, 1536]);  addmm_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_512: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_511, [1, 512, 24, -1]);  view_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_326: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_512, [0, 2, 1, 3]);  view_512 = None
    clone_93: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_326, memory_format = torch.contiguous_format);  permute_326 = None
    view_513: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_93, [-1, 512, 64]);  clone_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_327: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_377, [1, 0]);  primals_377 = None
    addmm_140: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_378, view_506, permute_327);  primals_378 = None
    view_515: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_140, [1, 512, 1536]);  addmm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_516: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_515, [1, 512, 24, -1]);  view_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_328: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_516, [0, 2, 1, 3]);  view_516 = None
    clone_94: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_328, memory_format = torch.contiguous_format);  permute_328 = None
    view_517: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_94, [-1, 512, 64]);  clone_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_329: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_513, [0, 2, 1]);  view_513 = None
    div_46: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_329, full_default_2);  permute_329 = full_default_2 = None
    bmm_46: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_509, div_46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_518: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_46, [-1, 24, 512, 512]);  bmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    where_116: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_4, view_518);  full_default_4 = view_518 = None
    amax_23: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_116, [-1], True)
    sub_140: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_116, amax_23);  where_116 = amax_23 = None
    exp_23: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_140);  sub_140 = None
    sum_24: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
    div_47: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
    where_117: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_1, div_47);  full_default_3 = div_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_70: "f32[1, 24, 512, 512]" = torch.ops.aten.bernoulli.p(permute_8, 0.9);  permute_8 = None
    sub_141: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_70);  bernoulli_70 = None
    convert_element_type_94: "b8[1, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_141, torch.bool);  sub_141 = None
    where_118: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_94, full_default_1, where_117)
    mul_259: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_118, 1.1111111111111112);  where_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_519: "f32[24, 512, 512]" = torch.ops.aten.view.default(mul_259, [-1, 512, 512]);  mul_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_47: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_519, view_517)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_520: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_47, [-1, 24, 512, 64]);  bmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_331: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_520, [0, 2, 1, 3]);  view_520 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_95: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_331, memory_format = torch.contiguous_format);  permute_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_521: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_95, [1, 512, -1]);  clone_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_522: "f32[512, 1536]" = torch.ops.aten.view.default(view_521, [512, 1536]);  view_521 = None
    permute_332: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_379, [1, 0]);  primals_379 = None
    addmm_141: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_380, view_522, permute_332);  primals_380 = None
    view_523: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_141, [1, 512, 1536]);  addmm_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_71: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_142: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_71);  bernoulli_71 = None
    convert_element_type_95: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_142, torch.bool);  sub_142 = None
    where_119: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_95, full_default_1, view_523);  view_523 = None
    mul_260: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_119, 1.1111111111111112);  where_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_164: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_260, add_163);  mul_260 = add_163 = None
    var_mean_47 = torch.ops.aten.var_mean.correction(add_164, [2], correction = 0, keepdim = True)
    getitem_94: "f32[1, 512, 1]" = var_mean_47[0]
    getitem_95: "f32[1, 512, 1]" = var_mean_47[1];  var_mean_47 = None
    add_165: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-07);  getitem_94 = None
    rsqrt_47: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
    sub_143: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_164, getitem_95);  add_164 = getitem_95 = None
    mul_261: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_143, rsqrt_47);  sub_143 = None
    mul_262: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_261, primals_381)
    add_166: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_262, primals_382);  mul_262 = primals_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_524: "f32[512, 1536]" = torch.ops.aten.view.default(add_166, [512, 1536])
    permute_334: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_383, [1, 0]);  primals_383 = None
    addmm_142: "f32[512, 6144]" = torch.ops.aten.addmm.default(primals_384, view_524, permute_334);  primals_384 = None
    view_525: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_142, [1, 512, 6144])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_263: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_525, 0.5)
    mul_264: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_525, 0.7071067811865476);  view_525 = None
    erf_23: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_264);  mul_264 = None
    add_167: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_265: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_263, add_167);  mul_263 = add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_526: "f32[512, 6144]" = torch.ops.aten.view.default(mul_265, [512, 6144]);  mul_265 = None
    permute_335: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_385, [1, 0]);  primals_385 = None
    addmm_143: "f32[512, 1536]" = torch.ops.aten.addmm.default(primals_386, view_526, permute_335);  primals_386 = None
    view_527: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_143, [1, 512, 1536]);  addmm_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    bernoulli_72: "f32[1, 512, 1536]" = torch.ops.aten.bernoulli.p(permute, 0.9);  permute = None
    sub_144: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(1, bernoulli_72);  bernoulli_72 = None
    convert_element_type_96: "b8[1, 512, 1536]" = torch.ops.prims.convert_element_type.default(sub_144, torch.bool);  sub_144 = None
    where_120: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_96, full_default_1, view_527);  view_527 = None
    mul_266: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_120, 1.1111111111111112);  where_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_168: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_266, add_166);  mul_266 = add_166 = None
    var_mean_48 = torch.ops.aten.var_mean.correction(add_168, [2], correction = 0, keepdim = True)
    getitem_96: "f32[1, 512, 1]" = var_mean_48[0]
    getitem_97: "f32[1, 512, 1]" = var_mean_48[1];  var_mean_48 = None
    add_169: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-07);  getitem_96 = None
    rsqrt_48: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_169);  add_169 = None
    sub_145: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_168, getitem_97);  add_168 = getitem_97 = None
    mul_267: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_145, rsqrt_48);  sub_145 = None
    mul_268: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_267, primals_387)
    add_170: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_268, primals_388);  mul_268 = primals_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1513, code: logits = self.qa_outputs(sequence_output)
    view_528: "f32[512, 1536]" = torch.ops.aten.view.default(add_170, [512, 1536]);  add_170 = None
    permute_337: "f32[1536, 2]" = torch.ops.aten.permute.default(primals_389, [1, 0]);  primals_389 = None
    addmm_144: "f32[512, 2]" = torch.ops.aten.addmm.default(primals_390, view_528, permute_337);  primals_390 = None
    view_529: "f32[1, 512, 2]" = torch.ops.aten.view.default(addmm_144, [1, 512, 2]);  addmm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1514, code: start_logits, end_logits = logits.split(1, dim=-1)
    split_with_sizes = torch.ops.aten.split_with_sizes.default(view_529, [1, 1], 2);  view_529 = None
    getitem_98: "f32[1, 512, 1]" = split_with_sizes[0]
    getitem_99: "f32[1, 512, 1]" = split_with_sizes[1];  split_with_sizes = None
    
    # No stacktrace found for following nodes
    squeeze_1: "f32[1, 512]" = torch.ops.aten.squeeze.dim(getitem_98, -1);  getitem_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1515, code: start_logits = start_logits.squeeze(-1).contiguous()
    clone_96: "f32[1, 512]" = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None
    
    # No stacktrace found for following nodes
    squeeze_2: "f32[1, 512]" = torch.ops.aten.squeeze.dim(getitem_99, -1);  getitem_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1516, code: end_logits = end_logits.squeeze(-1).contiguous()
    clone_97: "f32[1, 512]" = torch.ops.aten.clone.default(squeeze_2, memory_format = torch.contiguous_format);  squeeze_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1527, code: start_positions = start_positions.clamp(0, ignored_index)
    clamp_min: "i64[1]" = torch.ops.aten.clamp_min.default(primals_393, 0);  primals_393 = None
    clamp_max: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min, 512);  clamp_min = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1528, code: end_positions = end_positions.clamp(0, ignored_index)
    clamp_min_1: "i64[1]" = torch.ops.aten.clamp_min.default(primals_394, 0);  primals_394 = None
    clamp_max_1: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min_1, 512);  clamp_min_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1531, code: start_loss = loss_fct(start_logits, start_positions)
    amax_24: "f32[1, 1]" = torch.ops.aten.amax.default(clone_96, [1], True)
    sub_146: "f32[1, 512]" = torch.ops.aten.sub.Tensor(clone_96, amax_24);  amax_24 = None
    exp_24: "f32[1, 512]" = torch.ops.aten.exp.default(sub_146)
    sum_25: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
    log: "f32[1, 1]" = torch.ops.aten.log.default(sum_25);  sum_25 = None
    sub_147: "f32[1, 512]" = torch.ops.aten.sub.Tensor(sub_146, log);  sub_146 = log = None
    ne: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max, 512)
    full_default_170: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_121: "i64[1]" = torch.ops.aten.where.self(ne, clamp_max, full_default_170)
    unsqueeze_4: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where_121, 1);  where_121 = None
    gather: "f32[1, 1]" = torch.ops.aten.gather.default(sub_147, 1, unsqueeze_4);  unsqueeze_4 = None
    squeeze_3: "f32[1]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[1]" = torch.ops.aten.neg.default(squeeze_3);  squeeze_3 = None
    where_122: "f32[1]" = torch.ops.aten.where.self(ne, neg, full_default_1);  neg = None
    sum_26: "i64[]" = torch.ops.aten.sum.default(ne)
    convert_element_type_97: "f32[]" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
    sum_27: "f32[]" = torch.ops.aten.sum.default(where_122);  where_122 = None
    div_48: "f32[]" = torch.ops.aten.div.Tensor(sum_27, convert_element_type_97);  sum_27 = convert_element_type_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1532, code: end_loss = loss_fct(end_logits, end_positions)
    amax_25: "f32[1, 1]" = torch.ops.aten.amax.default(clone_97, [1], True)
    sub_148: "f32[1, 512]" = torch.ops.aten.sub.Tensor(clone_97, amax_25);  amax_25 = None
    exp_25: "f32[1, 512]" = torch.ops.aten.exp.default(sub_148)
    sum_28: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp_25, [1], True);  exp_25 = None
    log_1: "f32[1, 1]" = torch.ops.aten.log.default(sum_28);  sum_28 = None
    sub_149: "f32[1, 512]" = torch.ops.aten.sub.Tensor(sub_148, log_1);  sub_148 = log_1 = None
    ne_3: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_1, 512)
    where_123: "i64[1]" = torch.ops.aten.where.self(ne_3, clamp_max_1, full_default_170)
    unsqueeze_5: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where_123, 1);  where_123 = None
    gather_1: "f32[1, 1]" = torch.ops.aten.gather.default(sub_149, 1, unsqueeze_5);  unsqueeze_5 = None
    squeeze_4: "f32[1]" = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
    neg_1: "f32[1]" = torch.ops.aten.neg.default(squeeze_4);  squeeze_4 = None
    where_124: "f32[1]" = torch.ops.aten.where.self(ne_3, neg_1, full_default_1);  neg_1 = full_default_1 = None
    sum_29: "i64[]" = torch.ops.aten.sum.default(ne_3)
    convert_element_type_98: "f32[]" = torch.ops.prims.convert_element_type.default(sum_29, torch.float32);  sum_29 = None
    sum_30: "f32[]" = torch.ops.aten.sum.default(where_124);  where_124 = None
    div_49: "f32[]" = torch.ops.aten.div.Tensor(sum_30, convert_element_type_98);  sum_30 = convert_element_type_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1533, code: total_loss = (start_loss + end_loss) / 2
    add_171: "f32[]" = torch.ops.aten.add.Tensor(div_48, div_49);  div_48 = div_49 = None
    div_50: "f32[]" = torch.ops.aten.div.Tensor(add_171, 2);  add_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1532, code: end_loss = loss_fct(end_logits, end_positions)
    unsqueeze_6: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(clamp_max_1, 1);  clamp_max_1 = None
    ne_6: "b8[1, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_6, 512)
    where_125: "i64[1, 1]" = torch.ops.aten.where.self(ne_6, unsqueeze_6, full_default_170);  unsqueeze_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1531, code: start_loss = loss_fct(start_logits, start_positions)
    unsqueeze_7: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(clamp_max, 1);  clamp_max = None
    ne_8: "b8[1, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_7, 512)
    where_127: "i64[1, 1]" = torch.ops.aten.where.self(ne_8, unsqueeze_7, full_default_170);  unsqueeze_7 = full_default_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1513, code: logits = self.qa_outputs(sequence_output)
    permute_338: "f32[2, 1536]" = torch.ops.aten.permute.default(permute_337, [1, 0]);  permute_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_54: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_48, 1536);  rsqrt_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    permute_342: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_335, [1, 0]);  permute_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    permute_346: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_334, [1, 0]);  permute_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_55: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_47, 1536);  rsqrt_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    permute_350: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_332, [1, 0]);  permute_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    permute_355: "f32[24, 512, 512]" = torch.ops.aten.permute.default(view_519, [0, 2, 1]);  view_519 = None
    permute_356: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_517, [0, 2, 1]);  view_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_29: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(where_117);  where_117 = None
    alias_30: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(alias_29);  alias_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_357: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_509, [0, 2, 1]);  view_509 = None
    permute_358: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_46, [0, 2, 1]);  div_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_361: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_327, [1, 0]);  permute_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_366: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_325, [1, 0]);  permute_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    permute_371: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_323, [1, 0]);  permute_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_57: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_46, 1536);  rsqrt_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    permute_375: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_321, [1, 0]);  permute_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    permute_379: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_320, [1, 0]);  permute_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_58: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_45, 1536);  rsqrt_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    permute_383: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_318, [1, 0]);  permute_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    permute_388: "f32[24, 512, 512]" = torch.ops.aten.permute.default(view_497, [0, 2, 1]);  view_497 = None
    permute_389: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_495, [0, 2, 1]);  view_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_32: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(where_112);  where_112 = None
    alias_33: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(alias_32);  alias_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_390: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_487, [0, 2, 1]);  view_487 = None
    permute_391: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_44, [0, 2, 1]);  div_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_394: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_313, [1, 0]);  permute_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_399: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_311, [1, 0]);  permute_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    permute_404: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_309, [1, 0]);  permute_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_60: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 1536);  rsqrt_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    permute_408: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_307, [1, 0]);  permute_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    permute_412: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_306, [1, 0]);  permute_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_61: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 1536);  rsqrt_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    permute_416: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_304, [1, 0]);  permute_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    permute_421: "f32[24, 512, 512]" = torch.ops.aten.permute.default(view_475, [0, 2, 1]);  view_475 = None
    permute_422: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_473, [0, 2, 1]);  view_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_35: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(where_107);  where_107 = None
    alias_36: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(alias_35);  alias_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_423: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_465, [0, 2, 1]);  view_465 = None
    permute_424: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_42, [0, 2, 1]);  div_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_427: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_299, [1, 0]);  permute_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_432: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_297, [1, 0]);  permute_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    permute_437: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_295, [1, 0]);  permute_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_63: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_42, 1536);  rsqrt_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    permute_441: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_293, [1, 0]);  permute_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    permute_445: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_292, [1, 0]);  permute_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_64: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_41, 1536);  rsqrt_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    permute_449: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_290, [1, 0]);  permute_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    permute_454: "f32[24, 512, 512]" = torch.ops.aten.permute.default(view_453, [0, 2, 1]);  view_453 = None
    permute_455: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_451, [0, 2, 1]);  view_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_38: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(where_102);  where_102 = None
    alias_39: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(alias_38);  alias_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_456: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_443, [0, 2, 1]);  view_443 = None
    permute_457: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_40, [0, 2, 1]);  div_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_460: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_285, [1, 0]);  permute_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_465: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_283, [1, 0]);  permute_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    permute_470: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_281, [1, 0]);  permute_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_66: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 1536);  rsqrt_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    permute_474: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_279, [1, 0]);  permute_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    permute_478: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_278, [1, 0]);  permute_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_67: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 1536);  rsqrt_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    permute_482: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_276, [1, 0]);  permute_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    permute_487: "f32[24, 512, 512]" = torch.ops.aten.permute.default(view_431, [0, 2, 1]);  view_431 = None
    permute_488: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_429, [0, 2, 1]);  view_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_41: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(where_97);  where_97 = None
    alias_42: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(alias_41);  alias_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_489: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_421, [0, 2, 1]);  view_421 = None
    permute_490: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_38, [0, 2, 1]);  div_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_493: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_271, [1, 0]);  permute_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_498: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_269, [1, 0]);  permute_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    permute_503: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_267, [1, 0]);  permute_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_69: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 1536);  rsqrt_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    permute_507: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_265, [1, 0]);  permute_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    permute_511: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_70: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 1536);  rsqrt_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    permute_515: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    permute_520: "f32[24, 512, 512]" = torch.ops.aten.permute.default(view_409, [0, 2, 1]);  view_409 = None
    permute_521: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_407, [0, 2, 1]);  view_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_44: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(where_92);  where_92 = None
    alias_45: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(alias_44);  alias_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_522: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_399, [0, 2, 1]);  view_399 = None
    permute_523: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_36, [0, 2, 1]);  div_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_526: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_257, [1, 0]);  permute_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_531: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_255, [1, 0]);  permute_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    permute_536: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_253, [1, 0]);  permute_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_72: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 1536);  rsqrt_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    permute_540: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    permute_544: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_250, [1, 0]);  permute_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_73: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 1536);  rsqrt_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    permute_548: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_248, [1, 0]);  permute_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    permute_553: "f32[24, 512, 512]" = torch.ops.aten.permute.default(view_387, [0, 2, 1]);  view_387 = None
    permute_554: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_385, [0, 2, 1]);  view_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_47: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(where_87);  where_87 = None
    alias_48: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(alias_47);  alias_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_555: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_377, [0, 2, 1]);  view_377 = None
    permute_556: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_34, [0, 2, 1]);  div_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_559: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_564: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_241, [1, 0]);  permute_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    permute_569: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_75: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 1536);  rsqrt_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    permute_573: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_237, [1, 0]);  permute_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    permute_577: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_236, [1, 0]);  permute_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_76: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 1536);  rsqrt_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    permute_581: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_234, [1, 0]);  permute_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    permute_586: "f32[24, 512, 512]" = torch.ops.aten.permute.default(view_365, [0, 2, 1]);  view_365 = None
    permute_587: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_363, [0, 2, 1]);  view_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_50: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(where_82);  where_82 = None
    alias_51: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(alias_50);  alias_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_588: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_355, [0, 2, 1]);  view_355 = None
    permute_589: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_32, [0, 2, 1]);  div_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_592: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_597: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_227, [1, 0]);  permute_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    permute_602: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_225, [1, 0]);  permute_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_78: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 1536);  rsqrt_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    permute_606: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_223, [1, 0]);  permute_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    permute_610: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_79: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 1536);  rsqrt_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    permute_614: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    permute_619: "f32[24, 512, 512]" = torch.ops.aten.permute.default(view_343, [0, 2, 1]);  view_343 = None
    permute_620: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_341, [0, 2, 1]);  view_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_53: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(where_77);  where_77 = None
    alias_54: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(alias_53);  alias_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_621: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_333, [0, 2, 1]);  view_333 = None
    permute_622: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_30, [0, 2, 1]);  div_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_625: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_215, [1, 0]);  permute_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_630: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_213, [1, 0]);  permute_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    permute_635: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_81: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 1536);  rsqrt_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    permute_639: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_209, [1, 0]);  permute_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    permute_643: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_208, [1, 0]);  permute_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_82: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 1536);  rsqrt_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    permute_647: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    permute_652: "f32[24, 512, 512]" = torch.ops.aten.permute.default(view_321, [0, 2, 1]);  view_321 = None
    permute_653: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_319, [0, 2, 1]);  view_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_56: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(where_72);  where_72 = None
    alias_57: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(alias_56);  alias_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_654: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_311, [0, 2, 1]);  view_311 = None
    permute_655: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_28, [0, 2, 1]);  div_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_658: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_663: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_199, [1, 0]);  permute_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    permute_668: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_84: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 1536);  rsqrt_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    permute_672: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_195, [1, 0]);  permute_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    permute_676: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_194, [1, 0]);  permute_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_85: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 1536);  rsqrt_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    permute_680: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_192, [1, 0]);  permute_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    permute_685: "f32[24, 512, 512]" = torch.ops.aten.permute.default(view_299, [0, 2, 1]);  view_299 = None
    permute_686: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_297, [0, 2, 1]);  view_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_59: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(where_67);  where_67 = None
    alias_60: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(alias_59);  alias_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_687: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_289, [0, 2, 1]);  view_289 = None
    permute_688: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_26, [0, 2, 1]);  div_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_691: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_696: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    permute_701: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_183, [1, 0]);  permute_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_87: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 1536);  rsqrt_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    permute_705: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    permute_709: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_180, [1, 0]);  permute_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_88: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 1536);  rsqrt_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    permute_713: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    permute_718: "f32[24, 512, 512]" = torch.ops.aten.permute.default(view_277, [0, 2, 1]);  view_277 = None
    permute_719: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_275, [0, 2, 1]);  view_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_62: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(where_62);  where_62 = None
    alias_63: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(alias_62);  alias_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_720: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_267, [0, 2, 1]);  view_267 = None
    permute_721: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_24, [0, 2, 1]);  div_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_724: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_729: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_171, [1, 0]);  permute_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    permute_734: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_169, [1, 0]);  permute_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_90: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 1536);  rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    permute_738: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    permute_742: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_166, [1, 0]);  permute_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_91: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 1536);  rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    permute_746: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    permute_751: "f32[24, 512, 512]" = torch.ops.aten.permute.default(view_255, [0, 2, 1]);  view_255 = None
    permute_752: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_253, [0, 2, 1]);  view_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_65: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(where_57);  where_57 = None
    alias_66: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(alias_65);  alias_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_753: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_245, [0, 2, 1]);  view_245 = None
    permute_754: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_22, [0, 2, 1]);  div_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_757: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_159, [1, 0]);  permute_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_762: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_157, [1, 0]);  permute_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    permute_767: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_155, [1, 0]);  permute_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_93: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 1536);  rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    permute_771: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    permute_775: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_94: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 1536);  rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    permute_779: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_150, [1, 0]);  permute_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    permute_784: "f32[24, 512, 512]" = torch.ops.aten.permute.default(view_233, [0, 2, 1]);  view_233 = None
    permute_785: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_231, [0, 2, 1]);  view_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_68: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(where_52);  where_52 = None
    alias_69: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(alias_68);  alias_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_786: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_223, [0, 2, 1]);  view_223 = None
    permute_787: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_20, [0, 2, 1]);  div_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_790: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_795: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    permute_800: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_96: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 1536);  rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    permute_804: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_139, [1, 0]);  permute_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    permute_808: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_138, [1, 0]);  permute_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_97: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 1536);  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    permute_812: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    permute_817: "f32[24, 512, 512]" = torch.ops.aten.permute.default(view_211, [0, 2, 1]);  view_211 = None
    permute_818: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_209, [0, 2, 1]);  view_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_71: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(where_47);  where_47 = None
    alias_72: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(alias_71);  alias_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_819: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_201, [0, 2, 1]);  view_201 = None
    permute_820: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_18, [0, 2, 1]);  div_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_823: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_828: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    permute_833: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_127, [1, 0]);  permute_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_99: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 1536);  rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    permute_837: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    permute_841: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_100: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 1536);  rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    permute_845: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    permute_850: "f32[24, 512, 512]" = torch.ops.aten.permute.default(view_189, [0, 2, 1]);  view_189 = None
    permute_851: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_187, [0, 2, 1]);  view_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_74: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(where_42);  where_42 = None
    alias_75: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(alias_74);  alias_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_852: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_179, [0, 2, 1]);  view_179 = None
    permute_853: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_16, [0, 2, 1]);  div_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_856: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_117, [1, 0]);  permute_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_861: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_115, [1, 0]);  permute_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    permute_866: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_113, [1, 0]);  permute_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_102: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 1536);  rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    permute_870: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    permute_874: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_103: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 1536);  rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    permute_878: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    permute_883: "f32[24, 512, 512]" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None
    permute_884: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_165, [0, 2, 1]);  view_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_77: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(where_37);  where_37 = None
    alias_78: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(alias_77);  alias_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_885: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_157, [0, 2, 1]);  view_157 = None
    permute_886: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_14, [0, 2, 1]);  div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_889: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_894: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    permute_899: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_105: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 1536);  rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    permute_903: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    permute_907: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_106: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 1536);  rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    permute_911: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    permute_916: "f32[24, 512, 512]" = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None
    permute_917: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_143, [0, 2, 1]);  view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_80: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(where_32);  where_32 = None
    alias_81: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(alias_80);  alias_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_918: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_135, [0, 2, 1]);  view_135 = None
    permute_919: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_12, [0, 2, 1]);  div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_922: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_927: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    permute_932: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_108: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 1536);  rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    permute_936: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    permute_940: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_82, [1, 0]);  permute_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_109: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 1536);  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    permute_944: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    permute_949: "f32[24, 512, 512]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
    permute_950: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_121, [0, 2, 1]);  view_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_83: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(where_27);  where_27 = None
    alias_84: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(alias_83);  alias_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_951: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_113, [0, 2, 1]);  view_113 = None
    permute_952: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_10, [0, 2, 1]);  div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_955: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_960: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    permute_965: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_111: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 1536);  rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    permute_969: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    permute_973: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_112: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 1536);  rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    permute_977: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    permute_982: "f32[24, 512, 512]" = torch.ops.aten.permute.default(view_101, [0, 2, 1]);  view_101 = None
    permute_983: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_99, [0, 2, 1]);  view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_86: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(where_22);  where_22 = None
    alias_87: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(alias_86);  alias_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_984: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_91, [0, 2, 1]);  view_91 = None
    permute_985: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_8, [0, 2, 1]);  div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_988: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_993: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    permute_998: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_114: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 1536);  rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    permute_1002: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    permute_1006: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_115: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 1536);  rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    permute_1010: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    permute_1015: "f32[24, 512, 512]" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None
    permute_1016: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_77, [0, 2, 1]);  view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_89: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(where_17);  where_17 = None
    alias_90: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(alias_89);  alias_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_1017: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_69, [0, 2, 1]);  view_69 = None
    permute_1018: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_6, [0, 2, 1]);  div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_1021: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_1026: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    permute_1031: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_117: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 1536);  rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    permute_1035: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    permute_1039: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_118: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 1536);  rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    permute_1043: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    permute_1048: "f32[24, 512, 512]" = torch.ops.aten.permute.default(view_57, [0, 2, 1]);  view_57 = None
    permute_1049: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_55, [0, 2, 1]);  view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_92: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(where_12);  where_12 = None
    alias_93: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(alias_92);  alias_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_1050: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_47, [0, 2, 1]);  view_47 = None
    permute_1051: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_4, [0, 2, 1]);  div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_1054: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_1059: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    permute_1064: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_120: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 1536);  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    permute_1068: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    permute_1072: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_121: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 1536);  rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    permute_1076: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    permute_1081: "f32[24, 512, 512]" = torch.ops.aten.permute.default(view_35, [0, 2, 1]);  view_35 = None
    permute_1082: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_95: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(where_7);  where_7 = None
    alias_96: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(alias_95);  alias_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_1083: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_25, [0, 2, 1]);  view_25 = None
    permute_1084: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_2, [0, 2, 1]);  div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_1087: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_1092: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    permute_1097: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_123: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 1536);  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    permute_1101: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    permute_1105: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_124: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 1536);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    permute_1109: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    permute_1114: "f32[24, 512, 512]" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
    permute_1115: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_98: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(where_2);  where_2 = None
    alias_99: "f32[1, 24, 512, 512]" = torch.ops.aten.alias.default(alias_98);  alias_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_1116: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_3, [0, 2, 1]);  view_3 = None
    permute_1117: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div, [0, 2, 1]);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    permute_1120: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    permute_1125: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    permute_1130: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:901, code: embeddings = self.LayerNorm(embeddings)
    div_126: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 1536);  rsqrt = None
    return [div_50, clone_96, clone_97, primals_3, primals_13, primals_19, primals_29, primals_35, primals_45, primals_51, primals_61, primals_67, primals_77, primals_83, primals_93, primals_99, primals_109, primals_115, primals_125, primals_131, primals_141, primals_147, primals_157, primals_163, primals_173, primals_179, primals_189, primals_195, primals_205, primals_211, primals_221, primals_227, primals_237, primals_243, primals_253, primals_259, primals_269, primals_275, primals_285, primals_291, primals_301, primals_307, primals_317, primals_323, primals_333, primals_339, primals_349, primals_355, primals_365, primals_371, primals_381, primals_387, primals_392, slice_1, mul, convert_element_type, view, convert_element_type_2, view_16, convert_element_type_3, mul_8, view_18, addmm_4, view_20, convert_element_type_4, mul_14, view_22, convert_element_type_6, view_38, convert_element_type_7, mul_19, view_40, addmm_10, view_42, convert_element_type_8, mul_25, view_44, convert_element_type_10, view_60, convert_element_type_11, mul_30, view_62, addmm_16, view_64, convert_element_type_12, mul_36, view_66, convert_element_type_14, view_82, convert_element_type_15, mul_41, view_84, addmm_22, view_86, convert_element_type_16, mul_47, view_88, convert_element_type_18, view_104, convert_element_type_19, mul_52, view_106, addmm_28, view_108, convert_element_type_20, mul_58, view_110, convert_element_type_22, view_126, convert_element_type_23, mul_63, view_128, addmm_34, view_130, convert_element_type_24, mul_69, view_132, convert_element_type_26, view_148, convert_element_type_27, mul_74, view_150, addmm_40, view_152, convert_element_type_28, mul_80, view_154, convert_element_type_30, view_170, convert_element_type_31, mul_85, view_172, addmm_46, view_174, convert_element_type_32, mul_91, view_176, convert_element_type_34, view_192, convert_element_type_35, mul_96, view_194, addmm_52, view_196, convert_element_type_36, mul_102, view_198, convert_element_type_38, view_214, convert_element_type_39, mul_107, view_216, addmm_58, view_218, convert_element_type_40, mul_113, view_220, convert_element_type_42, view_236, convert_element_type_43, mul_118, view_238, addmm_64, view_240, convert_element_type_44, mul_124, view_242, convert_element_type_46, view_258, convert_element_type_47, mul_129, view_260, addmm_70, view_262, convert_element_type_48, mul_135, view_264, convert_element_type_50, view_280, convert_element_type_51, mul_140, view_282, addmm_76, view_284, convert_element_type_52, mul_146, view_286, convert_element_type_54, view_302, convert_element_type_55, mul_151, view_304, addmm_82, view_306, convert_element_type_56, mul_157, view_308, convert_element_type_58, view_324, convert_element_type_59, mul_162, view_326, addmm_88, view_328, convert_element_type_60, mul_168, view_330, convert_element_type_62, view_346, convert_element_type_63, mul_173, view_348, addmm_94, view_350, convert_element_type_64, mul_179, view_352, convert_element_type_66, view_368, convert_element_type_67, mul_184, view_370, addmm_100, view_372, convert_element_type_68, mul_190, view_374, convert_element_type_70, view_390, convert_element_type_71, mul_195, view_392, addmm_106, view_394, convert_element_type_72, mul_201, view_396, convert_element_type_74, view_412, convert_element_type_75, mul_206, view_414, addmm_112, view_416, convert_element_type_76, mul_212, view_418, convert_element_type_78, view_434, convert_element_type_79, mul_217, view_436, addmm_118, view_438, convert_element_type_80, mul_223, view_440, convert_element_type_82, view_456, convert_element_type_83, mul_228, view_458, addmm_124, view_460, convert_element_type_84, mul_234, view_462, convert_element_type_86, view_478, convert_element_type_87, mul_239, view_480, addmm_130, view_482, convert_element_type_88, mul_245, view_484, convert_element_type_90, view_500, convert_element_type_91, mul_250, view_502, addmm_136, view_504, convert_element_type_92, mul_256, view_506, convert_element_type_94, view_522, convert_element_type_95, mul_261, view_524, addmm_142, view_526, convert_element_type_96, mul_267, view_528, sub_147, ne, sub_149, ne_3, ne_6, where_125, ne_8, where_127, permute_338, div_54, permute_342, permute_346, div_55, permute_350, permute_355, permute_356, alias_30, permute_357, permute_358, permute_361, permute_366, permute_371, div_57, permute_375, permute_379, div_58, permute_383, permute_388, permute_389, alias_33, permute_390, permute_391, permute_394, permute_399, permute_404, div_60, permute_408, permute_412, div_61, permute_416, permute_421, permute_422, alias_36, permute_423, permute_424, permute_427, permute_432, permute_437, div_63, permute_441, permute_445, div_64, permute_449, permute_454, permute_455, alias_39, permute_456, permute_457, permute_460, permute_465, permute_470, div_66, permute_474, permute_478, div_67, permute_482, permute_487, permute_488, alias_42, permute_489, permute_490, permute_493, permute_498, permute_503, div_69, permute_507, permute_511, div_70, permute_515, permute_520, permute_521, alias_45, permute_522, permute_523, permute_526, permute_531, permute_536, div_72, permute_540, permute_544, div_73, permute_548, permute_553, permute_554, alias_48, permute_555, permute_556, permute_559, permute_564, permute_569, div_75, permute_573, permute_577, div_76, permute_581, permute_586, permute_587, alias_51, permute_588, permute_589, permute_592, permute_597, permute_602, div_78, permute_606, permute_610, div_79, permute_614, permute_619, permute_620, alias_54, permute_621, permute_622, permute_625, permute_630, permute_635, div_81, permute_639, permute_643, div_82, permute_647, permute_652, permute_653, alias_57, permute_654, permute_655, permute_658, permute_663, permute_668, div_84, permute_672, permute_676, div_85, permute_680, permute_685, permute_686, alias_60, permute_687, permute_688, permute_691, permute_696, permute_701, div_87, permute_705, permute_709, div_88, permute_713, permute_718, permute_719, alias_63, permute_720, permute_721, permute_724, permute_729, permute_734, div_90, permute_738, permute_742, div_91, permute_746, permute_751, permute_752, alias_66, permute_753, permute_754, permute_757, permute_762, permute_767, div_93, permute_771, permute_775, div_94, permute_779, permute_784, permute_785, alias_69, permute_786, permute_787, permute_790, permute_795, permute_800, div_96, permute_804, permute_808, div_97, permute_812, permute_817, permute_818, alias_72, permute_819, permute_820, permute_823, permute_828, permute_833, div_99, permute_837, permute_841, div_100, permute_845, permute_850, permute_851, alias_75, permute_852, permute_853, permute_856, permute_861, permute_866, div_102, permute_870, permute_874, div_103, permute_878, permute_883, permute_884, alias_78, permute_885, permute_886, permute_889, permute_894, permute_899, div_105, permute_903, permute_907, div_106, permute_911, permute_916, permute_917, alias_81, permute_918, permute_919, permute_922, permute_927, permute_932, div_108, permute_936, permute_940, div_109, permute_944, permute_949, permute_950, alias_84, permute_951, permute_952, permute_955, permute_960, permute_965, div_111, permute_969, permute_973, div_112, permute_977, permute_982, permute_983, alias_87, permute_984, permute_985, permute_988, permute_993, permute_998, div_114, permute_1002, permute_1006, div_115, permute_1010, permute_1015, permute_1016, alias_90, permute_1017, permute_1018, permute_1021, permute_1026, permute_1031, div_117, permute_1035, permute_1039, div_118, permute_1043, permute_1048, permute_1049, alias_93, permute_1050, permute_1051, permute_1054, permute_1059, permute_1064, div_120, permute_1068, permute_1072, div_121, permute_1076, permute_1081, permute_1082, alias_96, permute_1083, permute_1084, permute_1087, permute_1092, permute_1097, div_123, permute_1101, permute_1105, div_124, permute_1109, permute_1114, permute_1115, alias_99, permute_1116, permute_1117, permute_1120, permute_1125, permute_1130, div_126]
    