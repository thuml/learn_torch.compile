from __future__ import annotations



def forward(self, primals_1: "f32[50400, 4096]", primals_2: "f32[4096]", primals_3: "f32[4096]", primals_4: "f32[4096, 4096]", primals_5: "f32[4096, 4096]", primals_6: "f32[4096, 4096]", primals_7: "f32[4096, 4096]", primals_8: "f32[16384, 4096]", primals_9: "f32[16384]", primals_10: "f32[4096, 16384]", primals_11: "f32[4096]", primals_12: "f32[4096]", primals_13: "f32[4096]", primals_14: "f32[4096, 4096]", primals_15: "f32[4096, 4096]", primals_16: "f32[4096, 4096]", primals_17: "f32[4096, 4096]", primals_18: "f32[16384, 4096]", primals_19: "f32[16384]", primals_20: "f32[4096, 16384]", primals_21: "f32[4096]", primals_22: "f32[4096]", primals_23: "f32[4096]", primals_24: "f32[4096, 4096]", primals_25: "f32[4096, 4096]", primals_26: "f32[4096, 4096]", primals_27: "f32[4096, 4096]", primals_28: "f32[16384, 4096]", primals_29: "f32[16384]", primals_30: "f32[4096, 16384]", primals_31: "f32[4096]", primals_32: "f32[4096]", primals_33: "f32[4096]", primals_34: "f32[4096, 4096]", primals_35: "f32[4096, 4096]", primals_36: "f32[4096, 4096]", primals_37: "f32[4096, 4096]", primals_38: "f32[16384, 4096]", primals_39: "f32[16384]", primals_40: "f32[4096, 16384]", primals_41: "f32[4096]", primals_42: "f32[4096]", primals_43: "f32[4096]", primals_44: "f32[4096, 4096]", primals_45: "f32[4096, 4096]", primals_46: "f32[4096, 4096]", primals_47: "f32[4096, 4096]", primals_48: "f32[16384, 4096]", primals_49: "f32[16384]", primals_50: "f32[4096, 16384]", primals_51: "f32[4096]", primals_52: "f32[4096]", primals_53: "f32[4096]", primals_54: "f32[4096, 4096]", primals_55: "f32[4096, 4096]", primals_56: "f32[4096, 4096]", primals_57: "f32[4096, 4096]", primals_58: "f32[16384, 4096]", primals_59: "f32[16384]", primals_60: "f32[4096, 16384]", primals_61: "f32[4096]", primals_62: "f32[4096]", primals_63: "f32[4096]", primals_64: "f32[4096, 4096]", primals_65: "f32[4096, 4096]", primals_66: "f32[4096, 4096]", primals_67: "f32[4096, 4096]", primals_68: "f32[16384, 4096]", primals_69: "f32[16384]", primals_70: "f32[4096, 16384]", primals_71: "f32[4096]", primals_72: "f32[4096]", primals_73: "f32[4096]", primals_74: "f32[4096, 4096]", primals_75: "f32[4096, 4096]", primals_76: "f32[4096, 4096]", primals_77: "f32[4096, 4096]", primals_78: "f32[16384, 4096]", primals_79: "f32[16384]", primals_80: "f32[4096, 16384]", primals_81: "f32[4096]", primals_82: "f32[4096]", primals_83: "f32[4096]", primals_84: "f32[4096, 4096]", primals_85: "f32[4096, 4096]", primals_86: "f32[4096, 4096]", primals_87: "f32[4096, 4096]", primals_88: "f32[16384, 4096]", primals_89: "f32[16384]", primals_90: "f32[4096, 16384]", primals_91: "f32[4096]", primals_92: "f32[4096]", primals_93: "f32[4096]", primals_94: "f32[4096, 4096]", primals_95: "f32[4096, 4096]", primals_96: "f32[4096, 4096]", primals_97: "f32[4096, 4096]", primals_98: "f32[16384, 4096]", primals_99: "f32[16384]", primals_100: "f32[4096, 16384]", primals_101: "f32[4096]", primals_102: "f32[4096]", primals_103: "f32[4096]", primals_104: "f32[4096, 4096]", primals_105: "f32[4096, 4096]", primals_106: "f32[4096, 4096]", primals_107: "f32[4096, 4096]", primals_108: "f32[16384, 4096]", primals_109: "f32[16384]", primals_110: "f32[4096, 16384]", primals_111: "f32[4096]", primals_112: "f32[4096]", primals_113: "f32[4096]", primals_114: "f32[4096, 4096]", primals_115: "f32[4096, 4096]", primals_116: "f32[4096, 4096]", primals_117: "f32[4096, 4096]", primals_118: "f32[16384, 4096]", primals_119: "f32[16384]", primals_120: "f32[4096, 16384]", primals_121: "f32[4096]", primals_122: "f32[4096]", primals_123: "f32[4096]", primals_124: "f32[4096, 4096]", primals_125: "f32[4096, 4096]", primals_126: "f32[4096, 4096]", primals_127: "f32[4096, 4096]", primals_128: "f32[16384, 4096]", primals_129: "f32[16384]", primals_130: "f32[4096, 16384]", primals_131: "f32[4096]", primals_132: "f32[4096]", primals_133: "f32[4096]", primals_134: "f32[4096, 4096]", primals_135: "f32[4096, 4096]", primals_136: "f32[4096, 4096]", primals_137: "f32[4096, 4096]", primals_138: "f32[16384, 4096]", primals_139: "f32[16384]", primals_140: "f32[4096, 16384]", primals_141: "f32[4096]", primals_142: "f32[4096]", primals_143: "f32[4096]", primals_144: "f32[4096, 4096]", primals_145: "f32[4096, 4096]", primals_146: "f32[4096, 4096]", primals_147: "f32[4096, 4096]", primals_148: "f32[16384, 4096]", primals_149: "f32[16384]", primals_150: "f32[4096, 16384]", primals_151: "f32[4096]", primals_152: "f32[4096]", primals_153: "f32[4096]", primals_154: "f32[4096, 4096]", primals_155: "f32[4096, 4096]", primals_156: "f32[4096, 4096]", primals_157: "f32[4096, 4096]", primals_158: "f32[16384, 4096]", primals_159: "f32[16384]", primals_160: "f32[4096, 16384]", primals_161: "f32[4096]", primals_162: "f32[4096]", primals_163: "f32[4096]", primals_164: "f32[4096, 4096]", primals_165: "f32[4096, 4096]", primals_166: "f32[4096, 4096]", primals_167: "f32[4096, 4096]", primals_168: "f32[16384, 4096]", primals_169: "f32[16384]", primals_170: "f32[4096, 16384]", primals_171: "f32[4096]", primals_172: "f32[4096]", primals_173: "f32[4096]", primals_174: "f32[4096, 4096]", primals_175: "f32[4096, 4096]", primals_176: "f32[4096, 4096]", primals_177: "f32[4096, 4096]", primals_178: "f32[16384, 4096]", primals_179: "f32[16384]", primals_180: "f32[4096, 16384]", primals_181: "f32[4096]", primals_182: "f32[4096]", primals_183: "f32[4096]", primals_184: "f32[4096, 4096]", primals_185: "f32[4096, 4096]", primals_186: "f32[4096, 4096]", primals_187: "f32[4096, 4096]", primals_188: "f32[16384, 4096]", primals_189: "f32[16384]", primals_190: "f32[4096, 16384]", primals_191: "f32[4096]", primals_192: "f32[4096]", primals_193: "f32[4096]", primals_194: "f32[4096, 4096]", primals_195: "f32[4096, 4096]", primals_196: "f32[4096, 4096]", primals_197: "f32[4096, 4096]", primals_198: "f32[16384, 4096]", primals_199: "f32[16384]", primals_200: "f32[4096, 16384]", primals_201: "f32[4096]", primals_202: "f32[4096]", primals_203: "f32[4096]", primals_204: "f32[4096, 4096]", primals_205: "f32[4096, 4096]", primals_206: "f32[4096, 4096]", primals_207: "f32[4096, 4096]", primals_208: "f32[16384, 4096]", primals_209: "f32[16384]", primals_210: "f32[4096, 16384]", primals_211: "f32[4096]", primals_212: "f32[4096]", primals_213: "f32[4096]", primals_214: "f32[4096, 4096]", primals_215: "f32[4096, 4096]", primals_216: "f32[4096, 4096]", primals_217: "f32[4096, 4096]", primals_218: "f32[16384, 4096]", primals_219: "f32[16384]", primals_220: "f32[4096, 16384]", primals_221: "f32[4096]", primals_222: "f32[4096]", primals_223: "f32[4096]", primals_224: "f32[4096, 4096]", primals_225: "f32[4096, 4096]", primals_226: "f32[4096, 4096]", primals_227: "f32[4096, 4096]", primals_228: "f32[16384, 4096]", primals_229: "f32[16384]", primals_230: "f32[4096, 16384]", primals_231: "f32[4096]", primals_232: "f32[4096]", primals_233: "f32[4096]", primals_234: "f32[4096, 4096]", primals_235: "f32[4096, 4096]", primals_236: "f32[4096, 4096]", primals_237: "f32[4096, 4096]", primals_238: "f32[16384, 4096]", primals_239: "f32[16384]", primals_240: "f32[4096, 16384]", primals_241: "f32[4096]", primals_242: "f32[4096]", primals_243: "f32[4096]", primals_244: "f32[4096, 4096]", primals_245: "f32[4096, 4096]", primals_246: "f32[4096, 4096]", primals_247: "f32[4096, 4096]", primals_248: "f32[16384, 4096]", primals_249: "f32[16384]", primals_250: "f32[4096, 16384]", primals_251: "f32[4096]", primals_252: "f32[4096]", primals_253: "f32[4096]", primals_254: "f32[4096, 4096]", primals_255: "f32[4096, 4096]", primals_256: "f32[4096, 4096]", primals_257: "f32[4096, 4096]", primals_258: "f32[16384, 4096]", primals_259: "f32[16384]", primals_260: "f32[4096, 16384]", primals_261: "f32[4096]", primals_262: "f32[4096]", primals_263: "f32[4096]", primals_264: "f32[4096, 4096]", primals_265: "f32[4096, 4096]", primals_266: "f32[4096, 4096]", primals_267: "f32[4096, 4096]", primals_268: "f32[16384, 4096]", primals_269: "f32[16384]", primals_270: "f32[4096, 16384]", primals_271: "f32[4096]", primals_272: "f32[4096]", primals_273: "f32[4096]", primals_274: "f32[4096, 4096]", primals_275: "f32[4096, 4096]", primals_276: "f32[4096, 4096]", primals_277: "f32[4096, 4096]", primals_278: "f32[16384, 4096]", primals_279: "f32[16384]", primals_280: "f32[4096, 16384]", primals_281: "f32[4096]", primals_282: "f32[4096]", primals_283: "f32[4096]", primals_284: "f32[2, 4096]", primals_285: "f32[2]", primals_286: "f32[2048, 64]", primals_287: "b8[1, 1, 2048, 2048]", primals_288: "f32[]", primals_289: "f32[2048, 64]", primals_290: "b8[1, 1, 2048, 2048]", primals_291: "f32[]", primals_292: "f32[2048, 64]", primals_293: "b8[1, 1, 2048, 2048]", primals_294: "f32[]", primals_295: "f32[2048, 64]", primals_296: "b8[1, 1, 2048, 2048]", primals_297: "f32[]", primals_298: "f32[2048, 64]", primals_299: "b8[1, 1, 2048, 2048]", primals_300: "f32[]", primals_301: "f32[2048, 64]", primals_302: "b8[1, 1, 2048, 2048]", primals_303: "f32[]", primals_304: "f32[2048, 64]", primals_305: "b8[1, 1, 2048, 2048]", primals_306: "f32[]", primals_307: "f32[2048, 64]", primals_308: "b8[1, 1, 2048, 2048]", primals_309: "f32[]", primals_310: "f32[2048, 64]", primals_311: "b8[1, 1, 2048, 2048]", primals_312: "f32[]", primals_313: "f32[2048, 64]", primals_314: "b8[1, 1, 2048, 2048]", primals_315: "f32[]", primals_316: "f32[2048, 64]", primals_317: "b8[1, 1, 2048, 2048]", primals_318: "f32[]", primals_319: "f32[2048, 64]", primals_320: "b8[1, 1, 2048, 2048]", primals_321: "f32[]", primals_322: "f32[2048, 64]", primals_323: "b8[1, 1, 2048, 2048]", primals_324: "f32[]", primals_325: "f32[2048, 64]", primals_326: "b8[1, 1, 2048, 2048]", primals_327: "f32[]", primals_328: "f32[2048, 64]", primals_329: "b8[1, 1, 2048, 2048]", primals_330: "f32[]", primals_331: "f32[2048, 64]", primals_332: "b8[1, 1, 2048, 2048]", primals_333: "f32[]", primals_334: "f32[2048, 64]", primals_335: "b8[1, 1, 2048, 2048]", primals_336: "f32[]", primals_337: "f32[2048, 64]", primals_338: "b8[1, 1, 2048, 2048]", primals_339: "f32[]", primals_340: "f32[2048, 64]", primals_341: "b8[1, 1, 2048, 2048]", primals_342: "f32[]", primals_343: "f32[2048, 64]", primals_344: "b8[1, 1, 2048, 2048]", primals_345: "f32[]", primals_346: "f32[2048, 64]", primals_347: "b8[1, 1, 2048, 2048]", primals_348: "f32[]", primals_349: "f32[2048, 64]", primals_350: "b8[1, 1, 2048, 2048]", primals_351: "f32[]", primals_352: "f32[2048, 64]", primals_353: "b8[1, 1, 2048, 2048]", primals_354: "f32[]", primals_355: "f32[2048, 64]", primals_356: "b8[1, 1, 2048, 2048]", primals_357: "f32[]", primals_358: "f32[2048, 64]", primals_359: "b8[1, 1, 2048, 2048]", primals_360: "f32[]", primals_361: "f32[2048, 64]", primals_362: "b8[1, 1, 2048, 2048]", primals_363: "f32[]", primals_364: "f32[2048, 64]", primals_365: "b8[1, 1, 2048, 2048]", primals_366: "f32[]", primals_367: "f32[2048, 64]", primals_368: "b8[1, 1, 2048, 2048]", primals_369: "f32[]", primals_370: "i64[1, 128]", primals_371: "i64[1]", primals_372: "i64[1]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:582, code: input_ids = input_ids.view(-1, input_shape[-1])
    view: "i64[1, 128]" = torch.ops.aten.reshape.default(primals_370, [-1, 128]);  primals_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:605, code: position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
    iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:606, code: position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
    unsqueeze: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
    view_1: "i64[1, 128]" = torch.ops.aten.reshape.default(unsqueeze, [-1, 128]);  unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:635, code: inputs_embeds = self.wte(input_ids)
    embedding: "f32[1, 128, 4096]" = torch.ops.aten.embedding.default(primals_1, view);  primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(embedding, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 128, 1]" = var_mean[0]
    getitem_1: "f32[1, 128, 1]" = var_mean[1];  var_mean = None
    add: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
    sub: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(embedding, getitem_1)
    mul: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul, primals_2);  mul = None
    add_1: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_1, primals_3);  mul_1 = primals_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
    view_2: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_1, [128, 4096]);  add_1 = None
    mm: "f32[128, 4096]" = torch.ops.aten.mm.default(view_2, permute)
    view_3: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm, [1, 128, 4096]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_1: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
    mm_1: "f32[128, 4096]" = torch.ops.aten.mm.default(view_2, permute_1)
    view_5: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_1, [1, 128, 4096]);  mm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_2: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
    mm_2: "f32[128, 4096]" = torch.ops.aten.mm.default(view_2, permute_2)
    view_7: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_2, [1, 128, 4096]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_8: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_3, [1, 128, 16, 256]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_9: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_5, [1, 128, 16, 256]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_10: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_7, [1, 128, 16, 256]);  view_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_3: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(primals_286, [1, 1, 1]);  primals_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_1: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1);  view_1 = None
    repeat_1: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_1, [1, 1, 64]);  unsqueeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat, 1, repeat_1);  repeat = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes = torch.ops.aten.split_with_sizes.default(gather, [32, 32], 2);  gather = None
    getitem_2: "f32[1, 128, 32]" = split_with_sizes[0]
    getitem_3: "f32[1, 128, 32]" = split_with_sizes[1];  split_with_sizes = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_4: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_9, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_8: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_9, 3, 64, 9223372036854775807);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_12: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_8, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_16: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_8, 3, 64, 9223372036854775807);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    slice_17: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_2, 0, 0, 9223372036854775807);  getitem_2 = None
    slice_18: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_17, 1, 0, 9223372036854775807);  slice_17 = None
    unsqueeze_2: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_18, 2);  slice_18 = None
    slice_19: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_2, 3, 0, 9223372036854775807);  unsqueeze_2 = None
    unsqueeze_3: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_19, 4);  slice_19 = None
    expand: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_3, [1, 128, 1, 32, 2])
    clone_1: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
    view_11: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_1, [1, 128, 1, 64]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    slice_20: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_3, 0, 0, 9223372036854775807);  getitem_3 = None
    slice_21: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_20, 1, 0, 9223372036854775807);  slice_20 = None
    unsqueeze_4: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_21, 2);  slice_21 = None
    slice_22: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_4, 3, 0, 9223372036854775807);  unsqueeze_4 = None
    unsqueeze_5: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_22, 4);  slice_22 = None
    expand_1: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_5, [1, 128, 1, 32, 2])
    clone_2: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_12: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_2, [1, 128, 1, 64]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_2: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_4, view_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_26: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_4, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_30: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_4, 3, 1, 9223372036854775807, 2);  slice_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_30);  slice_30 = None
    unsqueeze_6: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg, 4);  neg = None
    unsqueeze_7: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_26, 4);  slice_26 = None
    cat: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_6, unsqueeze_7], 4);  unsqueeze_6 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_13: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat, [1, 128, 16, 64]);  cat = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_3: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_13, view_11);  view_13 = None
    add_2: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_4: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_12, view_12);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_40: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_12, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_44: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_12, 3, 1, 9223372036854775807, 2);  slice_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_1: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_44);  slice_44 = None
    unsqueeze_12: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_1, 4);  neg_1 = None
    unsqueeze_13: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_40, 4);  slice_40 = None
    cat_1: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_12, unsqueeze_13], 4);  unsqueeze_12 = unsqueeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_16: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_1, [1, 128, 16, 64]);  cat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_5: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_16, view_11);  view_16 = view_11 = None
    add_3: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_2: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_2, slice_8], 3);  add_2 = slice_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_3: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_3, slice_16], 3);  add_3 = slice_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_4: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_2, [0, 2, 1, 3]);  cat_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_5: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_3, [0, 2, 1, 3]);  cat_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_45: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_287, 0, 0, 9223372036854775807);  primals_287 = None
    slice_46: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_45, 1, 0, 9223372036854775807);  slice_45 = None
    slice_47: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_46, 2, 0, 128);  slice_46 = None
    slice_48: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_47, 3, 0, 128);  slice_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_6: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_4, [0, 1, 3, 2]);  permute_4 = None
    expand_4: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_5, [1, 16, 128, 256]);  permute_5 = None
    view_17: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_4, [16, 128, 256]);  expand_4 = None
    expand_5: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_6, [1, 16, 256, 128]);  permute_6 = None
    view_18: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_5, [16, 256, 128]);  expand_5 = None
    bmm: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_17, view_18)
    view_19: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm, [1, 16, 128, 128]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_48, view_19, full_default);  view_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where, primals_288);  where = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div, [-1], True)
    sub_1: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div, amax);  div = amax = None
    exp: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_1: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_1: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    clone_5: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_1);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_6: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_5, [1, 16, 128, 128]);  clone_5 = None
    view_20: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_6, [16, 128, 128]);  expand_6 = None
    expand_7: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_3, [1, 16, 128, 256]);  permute_3 = None
    view_21: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_7, [16, 128, 256]);  expand_7 = None
    bmm_1: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_20, view_21)
    view_22: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_1, [1, 16, 128, 256]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_7: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
    clone_6: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_23: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_6, [1, 128, 4096]);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_8: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
    view_24: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_23, [128, 4096]);  view_23 = None
    mm_3: "f32[128, 4096]" = torch.ops.aten.mm.default(view_24, permute_8)
    view_25: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_3, [1, 128, 4096]);  mm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_9: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
    addmm: "f32[128, 16384]" = torch.ops.aten.addmm.default(primals_9, view_2, permute_9);  primals_9 = None
    view_27: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm, [1, 128, 16384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_6: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_27, 0.5)
    pow_1: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_27, 3.0)
    mul_7: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
    add_4: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_27, mul_7);  view_27 = mul_7 = None
    mul_8: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_4, 0.7978845608028654);  add_4 = None
    tanh: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_8);  mul_8 = None
    add_5: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh, 1.0)
    mul_9: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_6, add_5);  mul_6 = add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_28: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_9, [128, 16384]);  mul_9 = None
    permute_10: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
    addmm_1: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_11, view_28, permute_10);  primals_11 = None
    view_29: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_1, [1, 128, 4096]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_6: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_25, view_29);  view_25 = view_29 = None
    add_7: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_6, embedding);  add_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 128, 1]" = var_mean_1[0]
    getitem_5: "f32[1, 128, 1]" = var_mean_1[1];  var_mean_1 = None
    add_8: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_1: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
    sub_2: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_7, getitem_5);  getitem_5 = None
    mul_10: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
    mul_11: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_10, primals_12)
    add_9: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_11, primals_13);  mul_11 = primals_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_11: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
    view_30: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_9, [128, 4096]);  add_9 = None
    mm_4: "f32[128, 4096]" = torch.ops.aten.mm.default(view_30, permute_11)
    view_31: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_4, [1, 128, 4096]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_12: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
    mm_5: "f32[128, 4096]" = torch.ops.aten.mm.default(view_30, permute_12)
    view_33: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_5, [1, 128, 4096]);  mm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_13: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
    mm_6: "f32[128, 4096]" = torch.ops.aten.mm.default(view_30, permute_13)
    view_35: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_6, [1, 128, 4096]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_36: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_31, [1, 128, 16, 256]);  view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_37: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_33, [1, 128, 16, 256]);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_38: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_35, [1, 128, 16, 256]);  view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_14: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_2: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(primals_289, [1, 1, 1]);  primals_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_1: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_2, 1, repeat_1);  repeat_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(gather_1, [32, 32], 2);  gather_1 = None
    getitem_6: "f32[1, 128, 32]" = split_with_sizes_1[0]
    getitem_7: "f32[1, 128, 32]" = split_with_sizes_1[1];  split_with_sizes_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_52: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_37, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_56: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_37, 3, 64, 9223372036854775807);  view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_60: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_36, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_64: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_36, 3, 64, 9223372036854775807);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    slice_65: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_6, 0, 0, 9223372036854775807);  getitem_6 = None
    slice_66: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_65, 1, 0, 9223372036854775807);  slice_65 = None
    unsqueeze_15: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_66, 2);  slice_66 = None
    slice_67: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_15, 3, 0, 9223372036854775807);  unsqueeze_15 = None
    unsqueeze_16: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_67, 4);  slice_67 = None
    expand_8: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_16, [1, 128, 1, 32, 2])
    clone_9: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
    view_39: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_9, [1, 128, 1, 64]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    slice_68: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_7, 0, 0, 9223372036854775807);  getitem_7 = None
    slice_69: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_68, 1, 0, 9223372036854775807);  slice_68 = None
    unsqueeze_17: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_69, 2);  slice_69 = None
    slice_70: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_17, 3, 0, 9223372036854775807);  unsqueeze_17 = None
    unsqueeze_18: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_70, 4);  slice_70 = None
    expand_9: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_18, [1, 128, 1, 32, 2])
    clone_10: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    view_40: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_10, [1, 128, 1, 64]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_12: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_52, view_40)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_74: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_52, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_78: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_52, 3, 1, 9223372036854775807, 2);  slice_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_2: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_78);  slice_78 = None
    unsqueeze_19: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_2, 4);  neg_2 = None
    unsqueeze_20: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_74, 4);  slice_74 = None
    cat_4: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_19, unsqueeze_20], 4);  unsqueeze_19 = unsqueeze_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_41: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_4, [1, 128, 16, 64]);  cat_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_13: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_41, view_39);  view_41 = None
    add_10: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_12, mul_13);  mul_12 = mul_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_14: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_60, view_40);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_88: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_60, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_92: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_60, 3, 1, 9223372036854775807, 2);  slice_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_3: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_92);  slice_92 = None
    unsqueeze_25: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_3, 4);  neg_3 = None
    unsqueeze_26: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_88, 4);  slice_88 = None
    cat_5: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_25, unsqueeze_26], 4);  unsqueeze_25 = unsqueeze_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_44: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_5, [1, 128, 16, 64]);  cat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_15: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_44, view_39);  view_44 = view_39 = None
    add_11: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_14, mul_15);  mul_14 = mul_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_6: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_10, slice_56], 3);  add_10 = slice_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_7: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_11, slice_64], 3);  add_11 = slice_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_15: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_6, [0, 2, 1, 3]);  cat_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_16: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_7, [0, 2, 1, 3]);  cat_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_93: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_290, 0, 0, 9223372036854775807);  primals_290 = None
    slice_94: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_93, 1, 0, 9223372036854775807);  slice_93 = None
    slice_95: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_94, 2, 0, 128);  slice_94 = None
    slice_96: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_95, 3, 0, 128);  slice_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_17: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_15, [0, 1, 3, 2]);  permute_15 = None
    expand_12: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_16, [1, 16, 128, 256]);  permute_16 = None
    view_45: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_12, [16, 128, 256]);  expand_12 = None
    expand_13: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_17, [1, 16, 256, 128]);  permute_17 = None
    view_46: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_13, [16, 256, 128]);  expand_13 = None
    bmm_2: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_45, view_46)
    view_47: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_2, [1, 16, 128, 128]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_1: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_96, view_47, full_default);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_2: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_1, primals_291);  where_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_1: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_2, [-1], True)
    sub_3: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_2, amax_1);  div_2 = amax_1 = None
    exp_1: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
    sum_2: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_3: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_2: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    clone_13: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_3);  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_14: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_13, [1, 16, 128, 128]);  clone_13 = None
    view_48: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_14, [16, 128, 128]);  expand_14 = None
    expand_15: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_14, [1, 16, 128, 256]);  permute_14 = None
    view_49: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_15, [16, 128, 256]);  expand_15 = None
    bmm_3: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_48, view_49)
    view_50: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_3, [1, 16, 128, 256]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_18: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
    clone_14: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_51: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_14, [1, 128, 4096]);  clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_19: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
    view_52: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_51, [128, 4096]);  view_51 = None
    mm_7: "f32[128, 4096]" = torch.ops.aten.mm.default(view_52, permute_19)
    view_53: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_7, [1, 128, 4096]);  mm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_20: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
    addmm_2: "f32[128, 16384]" = torch.ops.aten.addmm.default(primals_19, view_30, permute_20);  primals_19 = None
    view_55: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_2, [1, 128, 16384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_16: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_55, 0.5)
    pow_2: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_55, 3.0)
    mul_17: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
    add_12: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_55, mul_17);  view_55 = mul_17 = None
    mul_18: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_12, 0.7978845608028654);  add_12 = None
    tanh_1: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_18);  mul_18 = None
    add_13: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_1, 1.0)
    mul_19: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_16, add_13);  mul_16 = add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_56: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_19, [128, 16384]);  mul_19 = None
    permute_21: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
    addmm_3: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_21, view_56, permute_21);  primals_21 = None
    view_57: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_3, [1, 128, 4096]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_14: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_53, view_57);  view_53 = view_57 = None
    add_15: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_14, add_7);  add_14 = add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 128, 1]" = var_mean_2[0]
    getitem_9: "f32[1, 128, 1]" = var_mean_2[1];  var_mean_2 = None
    add_16: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_2: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_4: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_15, getitem_9);  getitem_9 = None
    mul_20: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = None
    mul_21: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_20, primals_22)
    add_17: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_21, primals_23);  mul_21 = primals_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_22: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
    view_58: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_17, [128, 4096]);  add_17 = None
    mm_8: "f32[128, 4096]" = torch.ops.aten.mm.default(view_58, permute_22)
    view_59: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_8, [1, 128, 4096]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_23: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
    mm_9: "f32[128, 4096]" = torch.ops.aten.mm.default(view_58, permute_23)
    view_61: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_9, [1, 128, 4096]);  mm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_24: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
    mm_10: "f32[128, 4096]" = torch.ops.aten.mm.default(view_58, permute_24)
    view_63: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_10, [1, 128, 4096]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_64: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_59, [1, 128, 16, 256]);  view_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_65: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_61, [1, 128, 16, 256]);  view_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_66: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_63, [1, 128, 16, 256]);  view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_25: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_4: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(primals_292, [1, 1, 1]);  primals_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_2: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_4, 1, repeat_1);  repeat_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(gather_2, [32, 32], 2);  gather_2 = None
    getitem_10: "f32[1, 128, 32]" = split_with_sizes_2[0]
    getitem_11: "f32[1, 128, 32]" = split_with_sizes_2[1];  split_with_sizes_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_100: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_65, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_104: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_65, 3, 64, 9223372036854775807);  view_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_108: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_64, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_112: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_64, 3, 64, 9223372036854775807);  view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    slice_113: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_10, 0, 0, 9223372036854775807);  getitem_10 = None
    slice_114: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_113, 1, 0, 9223372036854775807);  slice_113 = None
    unsqueeze_28: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_114, 2);  slice_114 = None
    slice_115: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_28, 3, 0, 9223372036854775807);  unsqueeze_28 = None
    unsqueeze_29: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_115, 4);  slice_115 = None
    expand_16: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_29, [1, 128, 1, 32, 2])
    clone_17: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
    view_67: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_17, [1, 128, 1, 64]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    slice_116: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_11, 0, 0, 9223372036854775807);  getitem_11 = None
    slice_117: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_116, 1, 0, 9223372036854775807);  slice_116 = None
    unsqueeze_30: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_117, 2);  slice_117 = None
    slice_118: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_30, 3, 0, 9223372036854775807);  unsqueeze_30 = None
    unsqueeze_31: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_118, 4);  slice_118 = None
    expand_17: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_31, [1, 128, 1, 32, 2])
    clone_18: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    view_68: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_18, [1, 128, 1, 64]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_22: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_100, view_68)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_122: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_100, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_126: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_100, 3, 1, 9223372036854775807, 2);  slice_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_4: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_126);  slice_126 = None
    unsqueeze_32: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_4, 4);  neg_4 = None
    unsqueeze_33: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_122, 4);  slice_122 = None
    cat_8: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_32, unsqueeze_33], 4);  unsqueeze_32 = unsqueeze_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_69: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_8, [1, 128, 16, 64]);  cat_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_23: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_69, view_67);  view_69 = None
    add_18: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_24: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_108, view_68);  view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_136: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_108, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_140: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_108, 3, 1, 9223372036854775807, 2);  slice_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_5: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_140);  slice_140 = None
    unsqueeze_38: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_5, 4);  neg_5 = None
    unsqueeze_39: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_136, 4);  slice_136 = None
    cat_9: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_38, unsqueeze_39], 4);  unsqueeze_38 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_72: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_9, [1, 128, 16, 64]);  cat_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_25: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_72, view_67);  view_72 = view_67 = None
    add_19: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_24, mul_25);  mul_24 = mul_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_10: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_18, slice_104], 3);  add_18 = slice_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_11: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_19, slice_112], 3);  add_19 = slice_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_26: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_10, [0, 2, 1, 3]);  cat_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_27: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_11, [0, 2, 1, 3]);  cat_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_141: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_293, 0, 0, 9223372036854775807);  primals_293 = None
    slice_142: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_141, 1, 0, 9223372036854775807);  slice_141 = None
    slice_143: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_142, 2, 0, 128);  slice_142 = None
    slice_144: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_143, 3, 0, 128);  slice_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_28: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_26, [0, 1, 3, 2]);  permute_26 = None
    expand_20: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_27, [1, 16, 128, 256]);  permute_27 = None
    view_73: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_20, [16, 128, 256]);  expand_20 = None
    expand_21: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_28, [1, 16, 256, 128]);  permute_28 = None
    view_74: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_21, [16, 256, 128]);  expand_21 = None
    bmm_4: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_73, view_74)
    view_75: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_4, [1, 16, 128, 128]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_2: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_144, view_75, full_default);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_4: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_2, primals_294);  where_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_2: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_4, [-1], True)
    sub_5: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_4, amax_2);  div_4 = amax_2 = None
    exp_2: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
    sum_3: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_5: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_4: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    clone_21: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_5);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_22: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_21, [1, 16, 128, 128]);  clone_21 = None
    view_76: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_22, [16, 128, 128]);  expand_22 = None
    expand_23: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_25, [1, 16, 128, 256]);  permute_25 = None
    view_77: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_23, [16, 128, 256]);  expand_23 = None
    bmm_5: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_76, view_77)
    view_78: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_5, [1, 16, 128, 256]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_29: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
    clone_22: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_79: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_22, [1, 128, 4096]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_30: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
    view_80: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_79, [128, 4096]);  view_79 = None
    mm_11: "f32[128, 4096]" = torch.ops.aten.mm.default(view_80, permute_30)
    view_81: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_11, [1, 128, 4096]);  mm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_31: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
    addmm_4: "f32[128, 16384]" = torch.ops.aten.addmm.default(primals_29, view_58, permute_31);  primals_29 = None
    view_83: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_4, [1, 128, 16384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_26: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_83, 0.5)
    pow_3: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_83, 3.0)
    mul_27: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
    add_20: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_83, mul_27);  view_83 = mul_27 = None
    mul_28: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_20, 0.7978845608028654);  add_20 = None
    tanh_2: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_28);  mul_28 = None
    add_21: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_2, 1.0)
    mul_29: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_26, add_21);  mul_26 = add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_84: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_29, [128, 16384]);  mul_29 = None
    permute_32: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_30, [1, 0]);  primals_30 = None
    addmm_5: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_31, view_84, permute_32);  primals_31 = None
    view_85: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_5, [1, 128, 4096]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_22: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_81, view_85);  view_81 = view_85 = None
    add_23: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_22, add_15);  add_22 = add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_23, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 128, 1]" = var_mean_3[0]
    getitem_13: "f32[1, 128, 1]" = var_mean_3[1];  var_mean_3 = None
    add_24: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_3: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
    sub_6: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_23, getitem_13);  getitem_13 = None
    mul_30: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_3);  sub_6 = None
    mul_31: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_30, primals_32)
    add_25: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_31, primals_33);  mul_31 = primals_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_33: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
    view_86: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_25, [128, 4096]);  add_25 = None
    mm_12: "f32[128, 4096]" = torch.ops.aten.mm.default(view_86, permute_33)
    view_87: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_12, [1, 128, 4096]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_34: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_35, [1, 0]);  primals_35 = None
    mm_13: "f32[128, 4096]" = torch.ops.aten.mm.default(view_86, permute_34)
    view_89: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_13, [1, 128, 4096]);  mm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_35: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_36, [1, 0]);  primals_36 = None
    mm_14: "f32[128, 4096]" = torch.ops.aten.mm.default(view_86, permute_35)
    view_91: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_14, [1, 128, 4096]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_92: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_87, [1, 128, 16, 256]);  view_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_93: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_89, [1, 128, 16, 256]);  view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_94: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_91, [1, 128, 16, 256]);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_36: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_6: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(primals_295, [1, 1, 1]);  primals_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_3: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_6, 1, repeat_1);  repeat_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_3 = torch.ops.aten.split_with_sizes.default(gather_3, [32, 32], 2);  gather_3 = None
    getitem_14: "f32[1, 128, 32]" = split_with_sizes_3[0]
    getitem_15: "f32[1, 128, 32]" = split_with_sizes_3[1];  split_with_sizes_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_148: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_93, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_152: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_93, 3, 64, 9223372036854775807);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_156: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_92, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_160: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_92, 3, 64, 9223372036854775807);  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    slice_161: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_14, 0, 0, 9223372036854775807);  getitem_14 = None
    slice_162: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_161, 1, 0, 9223372036854775807);  slice_161 = None
    unsqueeze_41: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_162, 2);  slice_162 = None
    slice_163: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_41, 3, 0, 9223372036854775807);  unsqueeze_41 = None
    unsqueeze_42: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_163, 4);  slice_163 = None
    expand_24: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_42, [1, 128, 1, 32, 2])
    clone_25: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
    view_95: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_25, [1, 128, 1, 64]);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    slice_164: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_15, 0, 0, 9223372036854775807);  getitem_15 = None
    slice_165: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_164, 1, 0, 9223372036854775807);  slice_164 = None
    unsqueeze_43: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_165, 2);  slice_165 = None
    slice_166: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_43, 3, 0, 9223372036854775807);  unsqueeze_43 = None
    unsqueeze_44: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_166, 4);  slice_166 = None
    expand_25: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_44, [1, 128, 1, 32, 2])
    clone_26: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    view_96: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_26, [1, 128, 1, 64]);  clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_32: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_148, view_96)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_170: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_148, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_174: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_148, 3, 1, 9223372036854775807, 2);  slice_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_6: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_174);  slice_174 = None
    unsqueeze_45: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_6, 4);  neg_6 = None
    unsqueeze_46: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_170, 4);  slice_170 = None
    cat_12: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_45, unsqueeze_46], 4);  unsqueeze_45 = unsqueeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_97: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_12, [1, 128, 16, 64]);  cat_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_33: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_97, view_95);  view_97 = None
    add_26: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_34: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_156, view_96);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_184: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_156, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_188: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_156, 3, 1, 9223372036854775807, 2);  slice_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_7: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_188);  slice_188 = None
    unsqueeze_51: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_7, 4);  neg_7 = None
    unsqueeze_52: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_184, 4);  slice_184 = None
    cat_13: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_51, unsqueeze_52], 4);  unsqueeze_51 = unsqueeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_100: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_13, [1, 128, 16, 64]);  cat_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_35: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_100, view_95);  view_100 = view_95 = None
    add_27: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_34, mul_35);  mul_34 = mul_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_14: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_26, slice_152], 3);  add_26 = slice_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_15: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_27, slice_160], 3);  add_27 = slice_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_37: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_14, [0, 2, 1, 3]);  cat_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_38: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_15, [0, 2, 1, 3]);  cat_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_189: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_296, 0, 0, 9223372036854775807);  primals_296 = None
    slice_190: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_189, 1, 0, 9223372036854775807);  slice_189 = None
    slice_191: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_190, 2, 0, 128);  slice_190 = None
    slice_192: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_191, 3, 0, 128);  slice_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_39: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_37, [0, 1, 3, 2]);  permute_37 = None
    expand_28: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_38, [1, 16, 128, 256]);  permute_38 = None
    view_101: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_28, [16, 128, 256]);  expand_28 = None
    expand_29: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_39, [1, 16, 256, 128]);  permute_39 = None
    view_102: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_29, [16, 256, 128]);  expand_29 = None
    bmm_6: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_101, view_102)
    view_103: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_6, [1, 16, 128, 128]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_3: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_192, view_103, full_default);  view_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_6: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_3, primals_297);  where_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_3: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_6, [-1], True)
    sub_7: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_6, amax_3);  div_6 = amax_3 = None
    exp_3: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_4: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_7: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_6: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    clone_29: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_7);  div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_30: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_29, [1, 16, 128, 128]);  clone_29 = None
    view_104: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_30, [16, 128, 128]);  expand_30 = None
    expand_31: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_36, [1, 16, 128, 256]);  permute_36 = None
    view_105: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_31, [16, 128, 256]);  expand_31 = None
    bmm_7: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_104, view_105)
    view_106: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_7, [1, 16, 128, 256]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_40: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None
    clone_30: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_107: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_30, [1, 128, 4096]);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_41: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_37, [1, 0]);  primals_37 = None
    view_108: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_107, [128, 4096]);  view_107 = None
    mm_15: "f32[128, 4096]" = torch.ops.aten.mm.default(view_108, permute_41)
    view_109: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_15, [1, 128, 4096]);  mm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_42: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
    addmm_6: "f32[128, 16384]" = torch.ops.aten.addmm.default(primals_39, view_86, permute_42);  primals_39 = None
    view_111: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_6, [1, 128, 16384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_36: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_111, 0.5)
    pow_4: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_111, 3.0)
    mul_37: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
    add_28: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_111, mul_37);  view_111 = mul_37 = None
    mul_38: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_28, 0.7978845608028654);  add_28 = None
    tanh_3: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
    add_29: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_3, 1.0)
    mul_39: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_36, add_29);  mul_36 = add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_112: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_39, [128, 16384]);  mul_39 = None
    permute_43: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
    addmm_7: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_41, view_112, permute_43);  primals_41 = None
    view_113: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_7, [1, 128, 4096]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_30: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_109, view_113);  view_109 = view_113 = None
    add_31: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_30, add_23);  add_30 = add_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
    getitem_16: "f32[1, 128, 1]" = var_mean_4[0]
    getitem_17: "f32[1, 128, 1]" = var_mean_4[1];  var_mean_4 = None
    add_32: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_4: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_8: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_31, getitem_17);  getitem_17 = None
    mul_40: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_4);  sub_8 = None
    mul_41: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_40, primals_42)
    add_33: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_41, primals_43);  mul_41 = primals_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_44: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
    view_114: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_33, [128, 4096]);  add_33 = None
    mm_16: "f32[128, 4096]" = torch.ops.aten.mm.default(view_114, permute_44)
    view_115: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_16, [1, 128, 4096]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_45: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
    mm_17: "f32[128, 4096]" = torch.ops.aten.mm.default(view_114, permute_45)
    view_117: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_17, [1, 128, 4096]);  mm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_46: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_46, [1, 0]);  primals_46 = None
    mm_18: "f32[128, 4096]" = torch.ops.aten.mm.default(view_114, permute_46)
    view_119: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_18, [1, 128, 4096]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_120: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_115, [1, 128, 16, 256]);  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_121: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_117, [1, 128, 16, 256]);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_122: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_119, [1, 128, 16, 256]);  view_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_47: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_122, [0, 2, 1, 3]);  view_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_8: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(primals_298, [1, 1, 1]);  primals_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_4: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_8, 1, repeat_1);  repeat_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_4 = torch.ops.aten.split_with_sizes.default(gather_4, [32, 32], 2);  gather_4 = None
    getitem_18: "f32[1, 128, 32]" = split_with_sizes_4[0]
    getitem_19: "f32[1, 128, 32]" = split_with_sizes_4[1];  split_with_sizes_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_196: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_121, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_200: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_121, 3, 64, 9223372036854775807);  view_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_204: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_120, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_208: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_120, 3, 64, 9223372036854775807);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    slice_209: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_18, 0, 0, 9223372036854775807);  getitem_18 = None
    slice_210: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_209, 1, 0, 9223372036854775807);  slice_209 = None
    unsqueeze_54: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_210, 2);  slice_210 = None
    slice_211: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_54, 3, 0, 9223372036854775807);  unsqueeze_54 = None
    unsqueeze_55: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_211, 4);  slice_211 = None
    expand_32: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_55, [1, 128, 1, 32, 2])
    clone_33: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
    view_123: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_33, [1, 128, 1, 64]);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    slice_212: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_19, 0, 0, 9223372036854775807);  getitem_19 = None
    slice_213: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_212, 1, 0, 9223372036854775807);  slice_212 = None
    unsqueeze_56: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_213, 2);  slice_213 = None
    slice_214: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_56, 3, 0, 9223372036854775807);  unsqueeze_56 = None
    unsqueeze_57: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_214, 4);  slice_214 = None
    expand_33: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_57, [1, 128, 1, 32, 2])
    clone_34: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
    view_124: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_34, [1, 128, 1, 64]);  clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_42: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_196, view_124)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_218: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_196, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_222: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_196, 3, 1, 9223372036854775807, 2);  slice_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_8: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_222);  slice_222 = None
    unsqueeze_58: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_8, 4);  neg_8 = None
    unsqueeze_59: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_218, 4);  slice_218 = None
    cat_16: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_58, unsqueeze_59], 4);  unsqueeze_58 = unsqueeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_125: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_16, [1, 128, 16, 64]);  cat_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_43: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_125, view_123);  view_125 = None
    add_34: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_42, mul_43);  mul_42 = mul_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_44: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_204, view_124);  view_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_232: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_204, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_236: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_204, 3, 1, 9223372036854775807, 2);  slice_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_9: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_236);  slice_236 = None
    unsqueeze_64: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_9, 4);  neg_9 = None
    unsqueeze_65: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_232, 4);  slice_232 = None
    cat_17: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_64, unsqueeze_65], 4);  unsqueeze_64 = unsqueeze_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_128: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_17, [1, 128, 16, 64]);  cat_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_45: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_128, view_123);  view_128 = view_123 = None
    add_35: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_44, mul_45);  mul_44 = mul_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_18: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_34, slice_200], 3);  add_34 = slice_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_19: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_35, slice_208], 3);  add_35 = slice_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_48: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_18, [0, 2, 1, 3]);  cat_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_49: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_19, [0, 2, 1, 3]);  cat_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_237: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_299, 0, 0, 9223372036854775807);  primals_299 = None
    slice_238: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_237, 1, 0, 9223372036854775807);  slice_237 = None
    slice_239: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_238, 2, 0, 128);  slice_238 = None
    slice_240: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_239, 3, 0, 128);  slice_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_50: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_48, [0, 1, 3, 2]);  permute_48 = None
    expand_36: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_49, [1, 16, 128, 256]);  permute_49 = None
    view_129: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_36, [16, 128, 256]);  expand_36 = None
    expand_37: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_50, [1, 16, 256, 128]);  permute_50 = None
    view_130: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_37, [16, 256, 128]);  expand_37 = None
    bmm_8: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_129, view_130)
    view_131: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_8, [1, 16, 128, 128]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_4: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_240, view_131, full_default);  view_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_8: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_4, primals_300);  where_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_4: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_8, [-1], True)
    sub_9: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_8, amax_4);  div_8 = amax_4 = None
    exp_4: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
    sum_5: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_9: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    alias_8: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    clone_37: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_9);  div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_38: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_37, [1, 16, 128, 128]);  clone_37 = None
    view_132: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_38, [16, 128, 128]);  expand_38 = None
    expand_39: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_47, [1, 16, 128, 256]);  permute_47 = None
    view_133: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_39, [16, 128, 256]);  expand_39 = None
    bmm_9: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_132, view_133)
    view_134: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_9, [1, 16, 128, 256]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_51: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None
    clone_38: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_135: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_38, [1, 128, 4096]);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_52: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_47, [1, 0]);  primals_47 = None
    view_136: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_135, [128, 4096]);  view_135 = None
    mm_19: "f32[128, 4096]" = torch.ops.aten.mm.default(view_136, permute_52)
    view_137: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_19, [1, 128, 4096]);  mm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_53: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
    addmm_8: "f32[128, 16384]" = torch.ops.aten.addmm.default(primals_49, view_114, permute_53);  primals_49 = None
    view_139: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_8, [1, 128, 16384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_46: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_139, 0.5)
    pow_5: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_139, 3.0)
    mul_47: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
    add_36: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_139, mul_47);  view_139 = mul_47 = None
    mul_48: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_36, 0.7978845608028654);  add_36 = None
    tanh_4: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_48);  mul_48 = None
    add_37: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_4, 1.0)
    mul_49: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_46, add_37);  mul_46 = add_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_140: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_49, [128, 16384]);  mul_49 = None
    permute_54: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
    addmm_9: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_51, view_140, permute_54);  primals_51 = None
    view_141: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_9, [1, 128, 4096]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_38: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_137, view_141);  view_137 = view_141 = None
    add_39: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_38, add_31);  add_38 = add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
    getitem_20: "f32[1, 128, 1]" = var_mean_5[0]
    getitem_21: "f32[1, 128, 1]" = var_mean_5[1];  var_mean_5 = None
    add_40: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
    rsqrt_5: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
    sub_10: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_39, getitem_21);  getitem_21 = None
    mul_50: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_5);  sub_10 = None
    mul_51: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_50, primals_52)
    add_41: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_51, primals_53);  mul_51 = primals_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_55: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_54, [1, 0]);  primals_54 = None
    view_142: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_41, [128, 4096]);  add_41 = None
    mm_20: "f32[128, 4096]" = torch.ops.aten.mm.default(view_142, permute_55)
    view_143: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_20, [1, 128, 4096]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_56: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
    mm_21: "f32[128, 4096]" = torch.ops.aten.mm.default(view_142, permute_56)
    view_145: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_21, [1, 128, 4096]);  mm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_57: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
    mm_22: "f32[128, 4096]" = torch.ops.aten.mm.default(view_142, permute_57)
    view_147: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_22, [1, 128, 4096]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_148: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_143, [1, 128, 16, 256]);  view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_149: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_145, [1, 128, 16, 256]);  view_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_150: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_147, [1, 128, 16, 256]);  view_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_58: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_150, [0, 2, 1, 3]);  view_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_10: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(primals_301, [1, 1, 1]);  primals_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_5: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_10, 1, repeat_1);  repeat_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_5 = torch.ops.aten.split_with_sizes.default(gather_5, [32, 32], 2);  gather_5 = None
    getitem_22: "f32[1, 128, 32]" = split_with_sizes_5[0]
    getitem_23: "f32[1, 128, 32]" = split_with_sizes_5[1];  split_with_sizes_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_244: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_149, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_248: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_149, 3, 64, 9223372036854775807);  view_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_252: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_148, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_256: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_148, 3, 64, 9223372036854775807);  view_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    slice_257: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_22, 0, 0, 9223372036854775807);  getitem_22 = None
    slice_258: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_257, 1, 0, 9223372036854775807);  slice_257 = None
    unsqueeze_67: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_258, 2);  slice_258 = None
    slice_259: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_67, 3, 0, 9223372036854775807);  unsqueeze_67 = None
    unsqueeze_68: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_259, 4);  slice_259 = None
    expand_40: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_68, [1, 128, 1, 32, 2])
    clone_41: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
    view_151: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_41, [1, 128, 1, 64]);  clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    slice_260: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_23, 0, 0, 9223372036854775807);  getitem_23 = None
    slice_261: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_260, 1, 0, 9223372036854775807);  slice_260 = None
    unsqueeze_69: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_261, 2);  slice_261 = None
    slice_262: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_69, 3, 0, 9223372036854775807);  unsqueeze_69 = None
    unsqueeze_70: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_262, 4);  slice_262 = None
    expand_41: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_70, [1, 128, 1, 32, 2])
    clone_42: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
    view_152: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_42, [1, 128, 1, 64]);  clone_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_52: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_244, view_152)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_266: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_244, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_270: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_244, 3, 1, 9223372036854775807, 2);  slice_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_10: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_270);  slice_270 = None
    unsqueeze_71: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_10, 4);  neg_10 = None
    unsqueeze_72: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_266, 4);  slice_266 = None
    cat_20: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_71, unsqueeze_72], 4);  unsqueeze_71 = unsqueeze_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_153: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_20, [1, 128, 16, 64]);  cat_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_53: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_153, view_151);  view_153 = None
    add_42: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_52, mul_53);  mul_52 = mul_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_54: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_252, view_152);  view_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_280: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_252, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_284: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_252, 3, 1, 9223372036854775807, 2);  slice_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_11: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_284);  slice_284 = None
    unsqueeze_77: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_11, 4);  neg_11 = None
    unsqueeze_78: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_280, 4);  slice_280 = None
    cat_21: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_77, unsqueeze_78], 4);  unsqueeze_77 = unsqueeze_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_156: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_21, [1, 128, 16, 64]);  cat_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_55: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_156, view_151);  view_156 = view_151 = None
    add_43: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_54, mul_55);  mul_54 = mul_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_22: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_42, slice_248], 3);  add_42 = slice_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_23: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_43, slice_256], 3);  add_43 = slice_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_59: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_22, [0, 2, 1, 3]);  cat_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_60: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_23, [0, 2, 1, 3]);  cat_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_285: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_302, 0, 0, 9223372036854775807);  primals_302 = None
    slice_286: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_285, 1, 0, 9223372036854775807);  slice_285 = None
    slice_287: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_286, 2, 0, 128);  slice_286 = None
    slice_288: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_287, 3, 0, 128);  slice_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_61: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_59, [0, 1, 3, 2]);  permute_59 = None
    expand_44: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_60, [1, 16, 128, 256]);  permute_60 = None
    view_157: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_44, [16, 128, 256]);  expand_44 = None
    expand_45: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_61, [1, 16, 256, 128]);  permute_61 = None
    view_158: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_45, [16, 256, 128]);  expand_45 = None
    bmm_10: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_157, view_158)
    view_159: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_10, [1, 16, 128, 128]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_5: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_288, view_159, full_default);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_10: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_5, primals_303);  where_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_5: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_10, [-1], True)
    sub_11: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_10, amax_5);  div_10 = amax_5 = None
    exp_5: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
    sum_6: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_11: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    alias_10: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    clone_45: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_11);  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_46: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_45, [1, 16, 128, 128]);  clone_45 = None
    view_160: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_46, [16, 128, 128]);  expand_46 = None
    expand_47: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_58, [1, 16, 128, 256]);  permute_58 = None
    view_161: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_47, [16, 128, 256]);  expand_47 = None
    bmm_11: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_160, view_161)
    view_162: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_11, [1, 16, 128, 256]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_62: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
    clone_46: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_163: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_46, [1, 128, 4096]);  clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_63: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
    view_164: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_163, [128, 4096]);  view_163 = None
    mm_23: "f32[128, 4096]" = torch.ops.aten.mm.default(view_164, permute_63)
    view_165: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_23, [1, 128, 4096]);  mm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_64: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
    addmm_10: "f32[128, 16384]" = torch.ops.aten.addmm.default(primals_59, view_142, permute_64);  primals_59 = None
    view_167: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_10, [1, 128, 16384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_56: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_167, 0.5)
    pow_6: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_167, 3.0)
    mul_57: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
    add_44: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_167, mul_57);  view_167 = mul_57 = None
    mul_58: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_44, 0.7978845608028654);  add_44 = None
    tanh_5: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_58);  mul_58 = None
    add_45: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_5, 1.0)
    mul_59: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_56, add_45);  mul_56 = add_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_168: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_59, [128, 16384]);  mul_59 = None
    permute_65: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
    addmm_11: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_61, view_168, permute_65);  primals_61 = None
    view_169: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_11, [1, 128, 4096]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_46: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_165, view_169);  view_165 = view_169 = None
    add_47: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_46, add_39);  add_46 = add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_47, [2], correction = 0, keepdim = True)
    getitem_24: "f32[1, 128, 1]" = var_mean_6[0]
    getitem_25: "f32[1, 128, 1]" = var_mean_6[1];  var_mean_6 = None
    add_48: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
    rsqrt_6: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
    sub_12: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_47, getitem_25);  getitem_25 = None
    mul_60: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_6);  sub_12 = None
    mul_61: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_60, primals_62)
    add_49: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_61, primals_63);  mul_61 = primals_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_66: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
    view_170: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_49, [128, 4096]);  add_49 = None
    mm_24: "f32[128, 4096]" = torch.ops.aten.mm.default(view_170, permute_66)
    view_171: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_24, [1, 128, 4096]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_67: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_65, [1, 0]);  primals_65 = None
    mm_25: "f32[128, 4096]" = torch.ops.aten.mm.default(view_170, permute_67)
    view_173: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_25, [1, 128, 4096]);  mm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_68: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
    mm_26: "f32[128, 4096]" = torch.ops.aten.mm.default(view_170, permute_68)
    view_175: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_26, [1, 128, 4096]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_176: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_171, [1, 128, 16, 256]);  view_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_177: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_173, [1, 128, 16, 256]);  view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_178: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_175, [1, 128, 16, 256]);  view_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_69: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_12: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(primals_304, [1, 1, 1]);  primals_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_6: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_12, 1, repeat_1);  repeat_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_6 = torch.ops.aten.split_with_sizes.default(gather_6, [32, 32], 2);  gather_6 = None
    getitem_26: "f32[1, 128, 32]" = split_with_sizes_6[0]
    getitem_27: "f32[1, 128, 32]" = split_with_sizes_6[1];  split_with_sizes_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_292: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_177, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_296: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_177, 3, 64, 9223372036854775807);  view_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_300: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_176, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_304: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_176, 3, 64, 9223372036854775807);  view_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    slice_305: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_26, 0, 0, 9223372036854775807);  getitem_26 = None
    slice_306: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_305, 1, 0, 9223372036854775807);  slice_305 = None
    unsqueeze_80: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_306, 2);  slice_306 = None
    slice_307: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_80, 3, 0, 9223372036854775807);  unsqueeze_80 = None
    unsqueeze_81: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_307, 4);  slice_307 = None
    expand_48: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_81, [1, 128, 1, 32, 2])
    clone_49: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
    view_179: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_49, [1, 128, 1, 64]);  clone_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    slice_308: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_27, 0, 0, 9223372036854775807);  getitem_27 = None
    slice_309: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_308, 1, 0, 9223372036854775807);  slice_308 = None
    unsqueeze_82: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_309, 2);  slice_309 = None
    slice_310: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_82, 3, 0, 9223372036854775807);  unsqueeze_82 = None
    unsqueeze_83: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_310, 4);  slice_310 = None
    expand_49: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_83, [1, 128, 1, 32, 2])
    clone_50: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
    view_180: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_50, [1, 128, 1, 64]);  clone_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_62: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_292, view_180)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_314: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_292, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_318: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_292, 3, 1, 9223372036854775807, 2);  slice_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_12: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_318);  slice_318 = None
    unsqueeze_84: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_12, 4);  neg_12 = None
    unsqueeze_85: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_314, 4);  slice_314 = None
    cat_24: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_84, unsqueeze_85], 4);  unsqueeze_84 = unsqueeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_181: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_24, [1, 128, 16, 64]);  cat_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_63: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_181, view_179);  view_181 = None
    add_50: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_62, mul_63);  mul_62 = mul_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_64: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_300, view_180);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_328: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_300, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_332: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_300, 3, 1, 9223372036854775807, 2);  slice_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_13: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_332);  slice_332 = None
    unsqueeze_90: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_13, 4);  neg_13 = None
    unsqueeze_91: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_328, 4);  slice_328 = None
    cat_25: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_90, unsqueeze_91], 4);  unsqueeze_90 = unsqueeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_184: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_25, [1, 128, 16, 64]);  cat_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_65: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_184, view_179);  view_184 = view_179 = None
    add_51: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_26: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_50, slice_296], 3);  add_50 = slice_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_27: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_51, slice_304], 3);  add_51 = slice_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_70: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_26, [0, 2, 1, 3]);  cat_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_71: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_27, [0, 2, 1, 3]);  cat_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_333: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_305, 0, 0, 9223372036854775807);  primals_305 = None
    slice_334: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_333, 1, 0, 9223372036854775807);  slice_333 = None
    slice_335: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_334, 2, 0, 128);  slice_334 = None
    slice_336: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_335, 3, 0, 128);  slice_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_72: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_70, [0, 1, 3, 2]);  permute_70 = None
    expand_52: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_71, [1, 16, 128, 256]);  permute_71 = None
    view_185: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_52, [16, 128, 256]);  expand_52 = None
    expand_53: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_72, [1, 16, 256, 128]);  permute_72 = None
    view_186: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_53, [16, 256, 128]);  expand_53 = None
    bmm_12: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_185, view_186)
    view_187: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_12, [1, 16, 128, 128]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_6: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_336, view_187, full_default);  view_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_12: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_6, primals_306);  where_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_6: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_12, [-1], True)
    sub_13: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_12, amax_6);  div_12 = amax_6 = None
    exp_6: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_7: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_13: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    alias_12: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    clone_53: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_13);  div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_54: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_53, [1, 16, 128, 128]);  clone_53 = None
    view_188: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_54, [16, 128, 128]);  expand_54 = None
    expand_55: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_69, [1, 16, 128, 256]);  permute_69 = None
    view_189: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_55, [16, 128, 256]);  expand_55 = None
    bmm_13: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_188, view_189)
    view_190: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_13, [1, 16, 128, 256]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_73: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
    clone_54: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_191: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_54, [1, 128, 4096]);  clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_74: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
    view_192: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_191, [128, 4096]);  view_191 = None
    mm_27: "f32[128, 4096]" = torch.ops.aten.mm.default(view_192, permute_74)
    view_193: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_27, [1, 128, 4096]);  mm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_75: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_68, [1, 0]);  primals_68 = None
    addmm_12: "f32[128, 16384]" = torch.ops.aten.addmm.default(primals_69, view_170, permute_75);  primals_69 = None
    view_195: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_12, [1, 128, 16384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_66: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_195, 0.5)
    pow_7: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_195, 3.0)
    mul_67: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
    add_52: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_195, mul_67);  view_195 = mul_67 = None
    mul_68: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_52, 0.7978845608028654);  add_52 = None
    tanh_6: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_68);  mul_68 = None
    add_53: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_6, 1.0)
    mul_69: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_66, add_53);  mul_66 = add_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_196: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_69, [128, 16384]);  mul_69 = None
    permute_76: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
    addmm_13: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_71, view_196, permute_76);  primals_71 = None
    view_197: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_13, [1, 128, 4096]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_54: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_193, view_197);  view_193 = view_197 = None
    add_55: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_54, add_47);  add_54 = add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 128, 1]" = var_mean_7[0]
    getitem_29: "f32[1, 128, 1]" = var_mean_7[1];  var_mean_7 = None
    add_56: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_7: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_14: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_55, getitem_29);  getitem_29 = None
    mul_70: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_7);  sub_14 = None
    mul_71: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_70, primals_72)
    add_57: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_71, primals_73);  mul_71 = primals_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_77: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
    view_198: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_57, [128, 4096]);  add_57 = None
    mm_28: "f32[128, 4096]" = torch.ops.aten.mm.default(view_198, permute_77)
    view_199: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_28, [1, 128, 4096]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_78: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
    mm_29: "f32[128, 4096]" = torch.ops.aten.mm.default(view_198, permute_78)
    view_201: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_29, [1, 128, 4096]);  mm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_79: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
    mm_30: "f32[128, 4096]" = torch.ops.aten.mm.default(view_198, permute_79)
    view_203: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_30, [1, 128, 4096]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_204: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_199, [1, 128, 16, 256]);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_205: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_201, [1, 128, 16, 256]);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_206: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_203, [1, 128, 16, 256]);  view_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_80: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_14: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(primals_307, [1, 1, 1]);  primals_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_7: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_14, 1, repeat_1);  repeat_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_7 = torch.ops.aten.split_with_sizes.default(gather_7, [32, 32], 2);  gather_7 = None
    getitem_30: "f32[1, 128, 32]" = split_with_sizes_7[0]
    getitem_31: "f32[1, 128, 32]" = split_with_sizes_7[1];  split_with_sizes_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_340: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_205, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_344: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_205, 3, 64, 9223372036854775807);  view_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_348: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_204, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_352: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_204, 3, 64, 9223372036854775807);  view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    slice_353: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_30, 0, 0, 9223372036854775807);  getitem_30 = None
    slice_354: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_353, 1, 0, 9223372036854775807);  slice_353 = None
    unsqueeze_93: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_354, 2);  slice_354 = None
    slice_355: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_93, 3, 0, 9223372036854775807);  unsqueeze_93 = None
    unsqueeze_94: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_355, 4);  slice_355 = None
    expand_56: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_94, [1, 128, 1, 32, 2])
    clone_57: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
    view_207: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_57, [1, 128, 1, 64]);  clone_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    slice_356: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_31, 0, 0, 9223372036854775807);  getitem_31 = None
    slice_357: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_356, 1, 0, 9223372036854775807);  slice_356 = None
    unsqueeze_95: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_357, 2);  slice_357 = None
    slice_358: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_95, 3, 0, 9223372036854775807);  unsqueeze_95 = None
    unsqueeze_96: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_358, 4);  slice_358 = None
    expand_57: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_96, [1, 128, 1, 32, 2])
    clone_58: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
    view_208: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_58, [1, 128, 1, 64]);  clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_72: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_340, view_208)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_362: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_340, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_366: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_340, 3, 1, 9223372036854775807, 2);  slice_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_14: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_366);  slice_366 = None
    unsqueeze_97: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_14, 4);  neg_14 = None
    unsqueeze_98: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_362, 4);  slice_362 = None
    cat_28: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_97, unsqueeze_98], 4);  unsqueeze_97 = unsqueeze_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_209: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_28, [1, 128, 16, 64]);  cat_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_73: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_209, view_207);  view_209 = None
    add_58: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_72, mul_73);  mul_72 = mul_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_74: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_348, view_208);  view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_376: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_348, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_380: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_348, 3, 1, 9223372036854775807, 2);  slice_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_15: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_380);  slice_380 = None
    unsqueeze_103: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_15, 4);  neg_15 = None
    unsqueeze_104: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_376, 4);  slice_376 = None
    cat_29: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_103, unsqueeze_104], 4);  unsqueeze_103 = unsqueeze_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_212: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_29, [1, 128, 16, 64]);  cat_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_75: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_212, view_207);  view_212 = view_207 = None
    add_59: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_30: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_58, slice_344], 3);  add_58 = slice_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_31: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_59, slice_352], 3);  add_59 = slice_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_81: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_30, [0, 2, 1, 3]);  cat_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_82: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_31, [0, 2, 1, 3]);  cat_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_381: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_308, 0, 0, 9223372036854775807);  primals_308 = None
    slice_382: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_381, 1, 0, 9223372036854775807);  slice_381 = None
    slice_383: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_382, 2, 0, 128);  slice_382 = None
    slice_384: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_383, 3, 0, 128);  slice_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_83: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_81, [0, 1, 3, 2]);  permute_81 = None
    expand_60: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_82, [1, 16, 128, 256]);  permute_82 = None
    view_213: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_60, [16, 128, 256]);  expand_60 = None
    expand_61: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_83, [1, 16, 256, 128]);  permute_83 = None
    view_214: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_61, [16, 256, 128]);  expand_61 = None
    bmm_14: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_213, view_214)
    view_215: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_14, [1, 16, 128, 128]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_7: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_384, view_215, full_default);  view_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_14: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_7, primals_309);  where_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_7: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_14, [-1], True)
    sub_15: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_14, amax_7);  div_14 = amax_7 = None
    exp_7: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
    sum_8: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_15: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    alias_14: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    clone_61: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_15);  div_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_62: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_61, [1, 16, 128, 128]);  clone_61 = None
    view_216: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_62, [16, 128, 128]);  expand_62 = None
    expand_63: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_80, [1, 16, 128, 256]);  permute_80 = None
    view_217: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_63, [16, 128, 256]);  expand_63 = None
    bmm_15: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_216, view_217)
    view_218: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_15, [1, 16, 128, 256]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_84: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_218, [0, 2, 1, 3]);  view_218 = None
    clone_62: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_219: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_62, [1, 128, 4096]);  clone_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_85: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
    view_220: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_219, [128, 4096]);  view_219 = None
    mm_31: "f32[128, 4096]" = torch.ops.aten.mm.default(view_220, permute_85)
    view_221: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_31, [1, 128, 4096]);  mm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_86: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_78, [1, 0]);  primals_78 = None
    addmm_14: "f32[128, 16384]" = torch.ops.aten.addmm.default(primals_79, view_198, permute_86);  primals_79 = None
    view_223: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_14, [1, 128, 16384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_76: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_223, 0.5)
    pow_8: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_223, 3.0)
    mul_77: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
    add_60: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_223, mul_77);  view_223 = mul_77 = None
    mul_78: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_60, 0.7978845608028654);  add_60 = None
    tanh_7: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
    add_61: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_7, 1.0)
    mul_79: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_76, add_61);  mul_76 = add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_224: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_79, [128, 16384]);  mul_79 = None
    permute_87: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
    addmm_15: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_81, view_224, permute_87);  primals_81 = None
    view_225: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_15, [1, 128, 4096]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_62: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_221, view_225);  view_221 = view_225 = None
    add_63: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_62, add_55);  add_62 = add_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 128, 1]" = var_mean_8[0]
    getitem_33: "f32[1, 128, 1]" = var_mean_8[1];  var_mean_8 = None
    add_64: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
    rsqrt_8: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
    sub_16: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_63, getitem_33);  getitem_33 = None
    mul_80: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_8);  sub_16 = None
    mul_81: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_80, primals_82)
    add_65: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_81, primals_83);  mul_81 = primals_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_88: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_84, [1, 0]);  primals_84 = None
    view_226: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_65, [128, 4096]);  add_65 = None
    mm_32: "f32[128, 4096]" = torch.ops.aten.mm.default(view_226, permute_88)
    view_227: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_32, [1, 128, 4096]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_89: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
    mm_33: "f32[128, 4096]" = torch.ops.aten.mm.default(view_226, permute_89)
    view_229: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_33, [1, 128, 4096]);  mm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_90: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
    mm_34: "f32[128, 4096]" = torch.ops.aten.mm.default(view_226, permute_90)
    view_231: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_34, [1, 128, 4096]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_232: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_227, [1, 128, 16, 256]);  view_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_233: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_229, [1, 128, 16, 256]);  view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_234: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_231, [1, 128, 16, 256]);  view_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_91: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_16: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(primals_310, [1, 1, 1]);  primals_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_8: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_16, 1, repeat_1);  repeat_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_8 = torch.ops.aten.split_with_sizes.default(gather_8, [32, 32], 2);  gather_8 = None
    getitem_34: "f32[1, 128, 32]" = split_with_sizes_8[0]
    getitem_35: "f32[1, 128, 32]" = split_with_sizes_8[1];  split_with_sizes_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_388: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_233, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_392: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_233, 3, 64, 9223372036854775807);  view_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_396: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_232, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_400: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_232, 3, 64, 9223372036854775807);  view_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    slice_401: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_34, 0, 0, 9223372036854775807);  getitem_34 = None
    slice_402: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_401, 1, 0, 9223372036854775807);  slice_401 = None
    unsqueeze_106: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_402, 2);  slice_402 = None
    slice_403: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_106, 3, 0, 9223372036854775807);  unsqueeze_106 = None
    unsqueeze_107: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_403, 4);  slice_403 = None
    expand_64: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_107, [1, 128, 1, 32, 2])
    clone_65: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
    view_235: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_65, [1, 128, 1, 64]);  clone_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    slice_404: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_35, 0, 0, 9223372036854775807);  getitem_35 = None
    slice_405: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_404, 1, 0, 9223372036854775807);  slice_404 = None
    unsqueeze_108: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_405, 2);  slice_405 = None
    slice_406: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_108, 3, 0, 9223372036854775807);  unsqueeze_108 = None
    unsqueeze_109: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_406, 4);  slice_406 = None
    expand_65: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_109, [1, 128, 1, 32, 2])
    clone_66: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
    view_236: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_66, [1, 128, 1, 64]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_82: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_388, view_236)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_410: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_388, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_414: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_388, 3, 1, 9223372036854775807, 2);  slice_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_16: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_414);  slice_414 = None
    unsqueeze_110: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_16, 4);  neg_16 = None
    unsqueeze_111: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_410, 4);  slice_410 = None
    cat_32: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_110, unsqueeze_111], 4);  unsqueeze_110 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_237: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_32, [1, 128, 16, 64]);  cat_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_83: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_237, view_235);  view_237 = None
    add_66: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_82, mul_83);  mul_82 = mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_84: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_396, view_236);  view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_424: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_396, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_428: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_396, 3, 1, 9223372036854775807, 2);  slice_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_17: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_428);  slice_428 = None
    unsqueeze_116: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_17, 4);  neg_17 = None
    unsqueeze_117: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_424, 4);  slice_424 = None
    cat_33: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_116, unsqueeze_117], 4);  unsqueeze_116 = unsqueeze_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_240: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_33, [1, 128, 16, 64]);  cat_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_85: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_240, view_235);  view_240 = view_235 = None
    add_67: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_84, mul_85);  mul_84 = mul_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_34: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_66, slice_392], 3);  add_66 = slice_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_35: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_67, slice_400], 3);  add_67 = slice_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_92: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_34, [0, 2, 1, 3]);  cat_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_93: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_35, [0, 2, 1, 3]);  cat_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_429: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_311, 0, 0, 9223372036854775807);  primals_311 = None
    slice_430: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_429, 1, 0, 9223372036854775807);  slice_429 = None
    slice_431: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_430, 2, 0, 128);  slice_430 = None
    slice_432: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_431, 3, 0, 128);  slice_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_94: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_92, [0, 1, 3, 2]);  permute_92 = None
    expand_68: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_93, [1, 16, 128, 256]);  permute_93 = None
    view_241: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_68, [16, 128, 256]);  expand_68 = None
    expand_69: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_94, [1, 16, 256, 128]);  permute_94 = None
    view_242: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_69, [16, 256, 128]);  expand_69 = None
    bmm_16: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_241, view_242)
    view_243: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_16, [1, 16, 128, 128]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_8: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_432, view_243, full_default);  view_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_16: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_8, primals_312);  where_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_8: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_16, [-1], True)
    sub_17: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_16, amax_8);  div_16 = amax_8 = None
    exp_8: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
    sum_9: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_17: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    alias_16: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    clone_69: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_17);  div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_70: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_69, [1, 16, 128, 128]);  clone_69 = None
    view_244: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_70, [16, 128, 128]);  expand_70 = None
    expand_71: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_91, [1, 16, 128, 256]);  permute_91 = None
    view_245: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_71, [16, 128, 256]);  expand_71 = None
    bmm_17: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_244, view_245)
    view_246: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_17, [1, 16, 128, 256]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_95: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
    clone_70: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_247: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_70, [1, 128, 4096]);  clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_96: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
    view_248: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_247, [128, 4096]);  view_247 = None
    mm_35: "f32[128, 4096]" = torch.ops.aten.mm.default(view_248, permute_96)
    view_249: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_35, [1, 128, 4096]);  mm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_97: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
    addmm_16: "f32[128, 16384]" = torch.ops.aten.addmm.default(primals_89, view_226, permute_97);  primals_89 = None
    view_251: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_16, [1, 128, 16384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_86: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_251, 0.5)
    pow_9: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_251, 3.0)
    mul_87: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
    add_68: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_251, mul_87);  view_251 = mul_87 = None
    mul_88: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_68, 0.7978845608028654);  add_68 = None
    tanh_8: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_88);  mul_88 = None
    add_69: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_8, 1.0)
    mul_89: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_86, add_69);  mul_86 = add_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_252: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_89, [128, 16384]);  mul_89 = None
    permute_98: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
    addmm_17: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_91, view_252, permute_98);  primals_91 = None
    view_253: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_17, [1, 128, 4096]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_70: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_249, view_253);  view_249 = view_253 = None
    add_71: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_70, add_63);  add_70 = add_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_9 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
    getitem_36: "f32[1, 128, 1]" = var_mean_9[0]
    getitem_37: "f32[1, 128, 1]" = var_mean_9[1];  var_mean_9 = None
    add_72: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
    rsqrt_9: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
    sub_18: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_71, getitem_37);  getitem_37 = None
    mul_90: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_9);  sub_18 = None
    mul_91: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_90, primals_92)
    add_73: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_91, primals_93);  mul_91 = primals_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_99: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_94, [1, 0]);  primals_94 = None
    view_254: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_73, [128, 4096]);  add_73 = None
    mm_36: "f32[128, 4096]" = torch.ops.aten.mm.default(view_254, permute_99)
    view_255: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_36, [1, 128, 4096]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_100: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_95, [1, 0]);  primals_95 = None
    mm_37: "f32[128, 4096]" = torch.ops.aten.mm.default(view_254, permute_100)
    view_257: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_37, [1, 128, 4096]);  mm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_101: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
    mm_38: "f32[128, 4096]" = torch.ops.aten.mm.default(view_254, permute_101)
    view_259: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_38, [1, 128, 4096]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_260: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_255, [1, 128, 16, 256]);  view_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_261: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_257, [1, 128, 16, 256]);  view_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_262: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_259, [1, 128, 16, 256]);  view_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_102: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_262, [0, 2, 1, 3]);  view_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_18: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(primals_313, [1, 1, 1]);  primals_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_9: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_18, 1, repeat_1);  repeat_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_9 = torch.ops.aten.split_with_sizes.default(gather_9, [32, 32], 2);  gather_9 = None
    getitem_38: "f32[1, 128, 32]" = split_with_sizes_9[0]
    getitem_39: "f32[1, 128, 32]" = split_with_sizes_9[1];  split_with_sizes_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_436: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_261, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_440: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_261, 3, 64, 9223372036854775807);  view_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_444: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_260, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_448: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_260, 3, 64, 9223372036854775807);  view_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    slice_449: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_38, 0, 0, 9223372036854775807);  getitem_38 = None
    slice_450: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_449, 1, 0, 9223372036854775807);  slice_449 = None
    unsqueeze_119: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_450, 2);  slice_450 = None
    slice_451: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_119, 3, 0, 9223372036854775807);  unsqueeze_119 = None
    unsqueeze_120: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_451, 4);  slice_451 = None
    expand_72: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_120, [1, 128, 1, 32, 2])
    clone_73: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
    view_263: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_73, [1, 128, 1, 64]);  clone_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    slice_452: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_39, 0, 0, 9223372036854775807);  getitem_39 = None
    slice_453: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_452, 1, 0, 9223372036854775807);  slice_452 = None
    unsqueeze_121: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_453, 2);  slice_453 = None
    slice_454: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_121, 3, 0, 9223372036854775807);  unsqueeze_121 = None
    unsqueeze_122: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_454, 4);  slice_454 = None
    expand_73: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_122, [1, 128, 1, 32, 2])
    clone_74: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
    view_264: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_74, [1, 128, 1, 64]);  clone_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_92: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_436, view_264)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_458: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_436, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_462: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_436, 3, 1, 9223372036854775807, 2);  slice_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_18: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_462);  slice_462 = None
    unsqueeze_123: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_18, 4);  neg_18 = None
    unsqueeze_124: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_458, 4);  slice_458 = None
    cat_36: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_123, unsqueeze_124], 4);  unsqueeze_123 = unsqueeze_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_265: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_36, [1, 128, 16, 64]);  cat_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_93: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_265, view_263);  view_265 = None
    add_74: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_94: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_444, view_264);  view_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_472: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_444, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_476: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_444, 3, 1, 9223372036854775807, 2);  slice_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_19: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_476);  slice_476 = None
    unsqueeze_129: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_19, 4);  neg_19 = None
    unsqueeze_130: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_472, 4);  slice_472 = None
    cat_37: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_129, unsqueeze_130], 4);  unsqueeze_129 = unsqueeze_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_268: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_37, [1, 128, 16, 64]);  cat_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_95: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_268, view_263);  view_268 = view_263 = None
    add_75: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_94, mul_95);  mul_94 = mul_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_38: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_74, slice_440], 3);  add_74 = slice_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_39: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_75, slice_448], 3);  add_75 = slice_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_103: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_38, [0, 2, 1, 3]);  cat_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_104: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_39, [0, 2, 1, 3]);  cat_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_477: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_314, 0, 0, 9223372036854775807);  primals_314 = None
    slice_478: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_477, 1, 0, 9223372036854775807);  slice_477 = None
    slice_479: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_478, 2, 0, 128);  slice_478 = None
    slice_480: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_479, 3, 0, 128);  slice_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_105: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_103, [0, 1, 3, 2]);  permute_103 = None
    expand_76: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_104, [1, 16, 128, 256]);  permute_104 = None
    view_269: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_76, [16, 128, 256]);  expand_76 = None
    expand_77: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_105, [1, 16, 256, 128]);  permute_105 = None
    view_270: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_77, [16, 256, 128]);  expand_77 = None
    bmm_18: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_269, view_270)
    view_271: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_18, [1, 16, 128, 128]);  bmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_9: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_480, view_271, full_default);  view_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_18: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_9, primals_315);  where_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_9: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_18, [-1], True)
    sub_19: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_18, amax_9);  div_18 = amax_9 = None
    exp_9: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_10: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_19: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    alias_18: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    clone_77: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_19);  div_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_78: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_77, [1, 16, 128, 128]);  clone_77 = None
    view_272: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_78, [16, 128, 128]);  expand_78 = None
    expand_79: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_102, [1, 16, 128, 256]);  permute_102 = None
    view_273: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_79, [16, 128, 256]);  expand_79 = None
    bmm_19: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_272, view_273)
    view_274: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_19, [1, 16, 128, 256]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_106: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_274, [0, 2, 1, 3]);  view_274 = None
    clone_78: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_275: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_78, [1, 128, 4096]);  clone_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_107: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
    view_276: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_275, [128, 4096]);  view_275 = None
    mm_39: "f32[128, 4096]" = torch.ops.aten.mm.default(view_276, permute_107)
    view_277: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_39, [1, 128, 4096]);  mm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_108: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
    addmm_18: "f32[128, 16384]" = torch.ops.aten.addmm.default(primals_99, view_254, permute_108);  primals_99 = None
    view_279: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_18, [1, 128, 16384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_96: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_279, 0.5)
    pow_10: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_279, 3.0)
    mul_97: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
    add_76: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_279, mul_97);  view_279 = mul_97 = None
    mul_98: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_76, 0.7978845608028654);  add_76 = None
    tanh_9: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_98);  mul_98 = None
    add_77: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_9, 1.0)
    mul_99: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_96, add_77);  mul_96 = add_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_280: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_99, [128, 16384]);  mul_99 = None
    permute_109: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_100, [1, 0]);  primals_100 = None
    addmm_19: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_101, view_280, permute_109);  primals_101 = None
    view_281: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_19, [1, 128, 4096]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_78: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_277, view_281);  view_277 = view_281 = None
    add_79: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_78, add_71);  add_78 = add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_79, [2], correction = 0, keepdim = True)
    getitem_40: "f32[1, 128, 1]" = var_mean_10[0]
    getitem_41: "f32[1, 128, 1]" = var_mean_10[1];  var_mean_10 = None
    add_80: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
    rsqrt_10: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
    sub_20: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_79, getitem_41);  getitem_41 = None
    mul_100: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_10);  sub_20 = None
    mul_101: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_100, primals_102)
    add_81: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_101, primals_103);  mul_101 = primals_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_110: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_104, [1, 0]);  primals_104 = None
    view_282: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_81, [128, 4096]);  add_81 = None
    mm_40: "f32[128, 4096]" = torch.ops.aten.mm.default(view_282, permute_110)
    view_283: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_40, [1, 128, 4096]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_111: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
    mm_41: "f32[128, 4096]" = torch.ops.aten.mm.default(view_282, permute_111)
    view_285: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_41, [1, 128, 4096]);  mm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_112: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
    mm_42: "f32[128, 4096]" = torch.ops.aten.mm.default(view_282, permute_112)
    view_287: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_42, [1, 128, 4096]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_288: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_283, [1, 128, 16, 256]);  view_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_289: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_285, [1, 128, 16, 256]);  view_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_290: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_287, [1, 128, 16, 256]);  view_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_113: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_290, [0, 2, 1, 3]);  view_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_20: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(primals_316, [1, 1, 1]);  primals_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_10: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_20, 1, repeat_1);  repeat_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_10 = torch.ops.aten.split_with_sizes.default(gather_10, [32, 32], 2);  gather_10 = None
    getitem_42: "f32[1, 128, 32]" = split_with_sizes_10[0]
    getitem_43: "f32[1, 128, 32]" = split_with_sizes_10[1];  split_with_sizes_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_484: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_289, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_488: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_289, 3, 64, 9223372036854775807);  view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_492: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_288, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_496: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_288, 3, 64, 9223372036854775807);  view_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    slice_497: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_42, 0, 0, 9223372036854775807);  getitem_42 = None
    slice_498: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_497, 1, 0, 9223372036854775807);  slice_497 = None
    unsqueeze_132: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_498, 2);  slice_498 = None
    slice_499: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_132, 3, 0, 9223372036854775807);  unsqueeze_132 = None
    unsqueeze_133: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_499, 4);  slice_499 = None
    expand_80: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_133, [1, 128, 1, 32, 2])
    clone_81: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
    view_291: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_81, [1, 128, 1, 64]);  clone_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    slice_500: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_43, 0, 0, 9223372036854775807);  getitem_43 = None
    slice_501: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_500, 1, 0, 9223372036854775807);  slice_500 = None
    unsqueeze_134: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_501, 2);  slice_501 = None
    slice_502: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_134, 3, 0, 9223372036854775807);  unsqueeze_134 = None
    unsqueeze_135: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_502, 4);  slice_502 = None
    expand_81: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_135, [1, 128, 1, 32, 2])
    clone_82: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
    view_292: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_82, [1, 128, 1, 64]);  clone_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_102: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_484, view_292)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_506: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_484, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_510: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_484, 3, 1, 9223372036854775807, 2);  slice_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_20: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_510);  slice_510 = None
    unsqueeze_136: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_20, 4);  neg_20 = None
    unsqueeze_137: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_506, 4);  slice_506 = None
    cat_40: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_136, unsqueeze_137], 4);  unsqueeze_136 = unsqueeze_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_293: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_40, [1, 128, 16, 64]);  cat_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_103: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_293, view_291);  view_293 = None
    add_82: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_104: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_492, view_292);  view_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_520: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_492, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_524: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_492, 3, 1, 9223372036854775807, 2);  slice_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_21: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_524);  slice_524 = None
    unsqueeze_142: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_21, 4);  neg_21 = None
    unsqueeze_143: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_520, 4);  slice_520 = None
    cat_41: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_142, unsqueeze_143], 4);  unsqueeze_142 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_296: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_41, [1, 128, 16, 64]);  cat_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_105: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_296, view_291);  view_296 = view_291 = None
    add_83: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_104, mul_105);  mul_104 = mul_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_42: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_82, slice_488], 3);  add_82 = slice_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_43: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_83, slice_496], 3);  add_83 = slice_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_114: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_42, [0, 2, 1, 3]);  cat_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_115: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_43, [0, 2, 1, 3]);  cat_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_525: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_317, 0, 0, 9223372036854775807);  primals_317 = None
    slice_526: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_525, 1, 0, 9223372036854775807);  slice_525 = None
    slice_527: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_526, 2, 0, 128);  slice_526 = None
    slice_528: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_527, 3, 0, 128);  slice_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_116: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_114, [0, 1, 3, 2]);  permute_114 = None
    expand_84: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_115, [1, 16, 128, 256]);  permute_115 = None
    view_297: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_84, [16, 128, 256]);  expand_84 = None
    expand_85: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_116, [1, 16, 256, 128]);  permute_116 = None
    view_298: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_85, [16, 256, 128]);  expand_85 = None
    bmm_20: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_297, view_298)
    view_299: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_20, [1, 16, 128, 128]);  bmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_10: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_528, view_299, full_default);  view_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_20: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_10, primals_318);  where_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_10: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_20, [-1], True)
    sub_21: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_20, amax_10);  div_20 = amax_10 = None
    exp_10: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
    sum_11: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_21: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    alias_20: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    clone_85: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_21);  div_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_86: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_85, [1, 16, 128, 128]);  clone_85 = None
    view_300: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_86, [16, 128, 128]);  expand_86 = None
    expand_87: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_113, [1, 16, 128, 256]);  permute_113 = None
    view_301: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_87, [16, 128, 256]);  expand_87 = None
    bmm_21: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_300, view_301)
    view_302: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_21, [1, 16, 128, 256]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_117: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_302, [0, 2, 1, 3]);  view_302 = None
    clone_86: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_303: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_86, [1, 128, 4096]);  clone_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_118: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_107, [1, 0]);  primals_107 = None
    view_304: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_303, [128, 4096]);  view_303 = None
    mm_43: "f32[128, 4096]" = torch.ops.aten.mm.default(view_304, permute_118)
    view_305: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_43, [1, 128, 4096]);  mm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_119: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
    addmm_20: "f32[128, 16384]" = torch.ops.aten.addmm.default(primals_109, view_282, permute_119);  primals_109 = None
    view_307: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_20, [1, 128, 16384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_106: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_307, 0.5)
    pow_11: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_307, 3.0)
    mul_107: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
    add_84: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_307, mul_107);  view_307 = mul_107 = None
    mul_108: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_84, 0.7978845608028654);  add_84 = None
    tanh_10: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_108);  mul_108 = None
    add_85: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_10, 1.0)
    mul_109: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_106, add_85);  mul_106 = add_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_308: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_109, [128, 16384]);  mul_109 = None
    permute_120: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_110, [1, 0]);  primals_110 = None
    addmm_21: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_111, view_308, permute_120);  primals_111 = None
    view_309: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_21, [1, 128, 4096]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_86: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_305, view_309);  view_305 = view_309 = None
    add_87: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_86, add_79);  add_86 = add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_87, [2], correction = 0, keepdim = True)
    getitem_44: "f32[1, 128, 1]" = var_mean_11[0]
    getitem_45: "f32[1, 128, 1]" = var_mean_11[1];  var_mean_11 = None
    add_88: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
    rsqrt_11: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
    sub_22: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_87, getitem_45);  getitem_45 = None
    mul_110: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_11);  sub_22 = None
    mul_111: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_110, primals_112)
    add_89: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_111, primals_113);  mul_111 = primals_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_121: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
    view_310: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_89, [128, 4096]);  add_89 = None
    mm_44: "f32[128, 4096]" = torch.ops.aten.mm.default(view_310, permute_121)
    view_311: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_44, [1, 128, 4096]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_122: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
    mm_45: "f32[128, 4096]" = torch.ops.aten.mm.default(view_310, permute_122)
    view_313: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_45, [1, 128, 4096]);  mm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_123: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_116, [1, 0]);  primals_116 = None
    mm_46: "f32[128, 4096]" = torch.ops.aten.mm.default(view_310, permute_123)
    view_315: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_46, [1, 128, 4096]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_316: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_311, [1, 128, 16, 256]);  view_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_317: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_313, [1, 128, 16, 256]);  view_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_318: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_315, [1, 128, 16, 256]);  view_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_124: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_318, [0, 2, 1, 3]);  view_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_22: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(primals_319, [1, 1, 1]);  primals_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_11: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_22, 1, repeat_1);  repeat_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_11 = torch.ops.aten.split_with_sizes.default(gather_11, [32, 32], 2);  gather_11 = None
    getitem_46: "f32[1, 128, 32]" = split_with_sizes_11[0]
    getitem_47: "f32[1, 128, 32]" = split_with_sizes_11[1];  split_with_sizes_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_532: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_317, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_536: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_317, 3, 64, 9223372036854775807);  view_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_540: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_316, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_544: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_316, 3, 64, 9223372036854775807);  view_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    slice_545: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_46, 0, 0, 9223372036854775807);  getitem_46 = None
    slice_546: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_545, 1, 0, 9223372036854775807);  slice_545 = None
    unsqueeze_145: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_546, 2);  slice_546 = None
    slice_547: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_145, 3, 0, 9223372036854775807);  unsqueeze_145 = None
    unsqueeze_146: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_547, 4);  slice_547 = None
    expand_88: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_146, [1, 128, 1, 32, 2])
    clone_89: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
    view_319: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_89, [1, 128, 1, 64]);  clone_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    slice_548: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_47, 0, 0, 9223372036854775807);  getitem_47 = None
    slice_549: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_548, 1, 0, 9223372036854775807);  slice_548 = None
    unsqueeze_147: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_549, 2);  slice_549 = None
    slice_550: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_147, 3, 0, 9223372036854775807);  unsqueeze_147 = None
    unsqueeze_148: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_550, 4);  slice_550 = None
    expand_89: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_148, [1, 128, 1, 32, 2])
    clone_90: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
    view_320: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_90, [1, 128, 1, 64]);  clone_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_112: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_532, view_320)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_554: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_532, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_558: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_532, 3, 1, 9223372036854775807, 2);  slice_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_22: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_558);  slice_558 = None
    unsqueeze_149: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_22, 4);  neg_22 = None
    unsqueeze_150: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_554, 4);  slice_554 = None
    cat_44: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_149, unsqueeze_150], 4);  unsqueeze_149 = unsqueeze_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_321: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_44, [1, 128, 16, 64]);  cat_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_113: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_321, view_319);  view_321 = None
    add_90: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_112, mul_113);  mul_112 = mul_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_114: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_540, view_320);  view_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_568: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_540, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_572: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_540, 3, 1, 9223372036854775807, 2);  slice_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_23: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_572);  slice_572 = None
    unsqueeze_155: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_23, 4);  neg_23 = None
    unsqueeze_156: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_568, 4);  slice_568 = None
    cat_45: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_155, unsqueeze_156], 4);  unsqueeze_155 = unsqueeze_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_324: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_45, [1, 128, 16, 64]);  cat_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_115: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_324, view_319);  view_324 = view_319 = None
    add_91: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_114, mul_115);  mul_114 = mul_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_46: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_90, slice_536], 3);  add_90 = slice_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_47: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_91, slice_544], 3);  add_91 = slice_544 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_125: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_46, [0, 2, 1, 3]);  cat_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_126: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_47, [0, 2, 1, 3]);  cat_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_573: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_320, 0, 0, 9223372036854775807);  primals_320 = None
    slice_574: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_573, 1, 0, 9223372036854775807);  slice_573 = None
    slice_575: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_574, 2, 0, 128);  slice_574 = None
    slice_576: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_575, 3, 0, 128);  slice_575 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_127: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_125, [0, 1, 3, 2]);  permute_125 = None
    expand_92: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_126, [1, 16, 128, 256]);  permute_126 = None
    view_325: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_92, [16, 128, 256]);  expand_92 = None
    expand_93: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_127, [1, 16, 256, 128]);  permute_127 = None
    view_326: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_93, [16, 256, 128]);  expand_93 = None
    bmm_22: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_325, view_326)
    view_327: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_22, [1, 16, 128, 128]);  bmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_11: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_576, view_327, full_default);  view_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_22: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_11, primals_321);  where_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_11: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_22, [-1], True)
    sub_23: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_22, amax_11);  div_22 = amax_11 = None
    exp_11: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_23);  sub_23 = None
    sum_12: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_23: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    alias_22: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    clone_93: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_23);  div_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_94: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_93, [1, 16, 128, 128]);  clone_93 = None
    view_328: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_94, [16, 128, 128]);  expand_94 = None
    expand_95: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_124, [1, 16, 128, 256]);  permute_124 = None
    view_329: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_95, [16, 128, 256]);  expand_95 = None
    bmm_23: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_328, view_329)
    view_330: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_23, [1, 16, 128, 256]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_128: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None
    clone_94: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_331: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_94, [1, 128, 4096]);  clone_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_129: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_117, [1, 0]);  primals_117 = None
    view_332: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_331, [128, 4096]);  view_331 = None
    mm_47: "f32[128, 4096]" = torch.ops.aten.mm.default(view_332, permute_129)
    view_333: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_47, [1, 128, 4096]);  mm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_130: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
    addmm_22: "f32[128, 16384]" = torch.ops.aten.addmm.default(primals_119, view_310, permute_130);  primals_119 = None
    view_335: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_22, [1, 128, 16384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_116: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_335, 0.5)
    pow_12: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_335, 3.0)
    mul_117: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
    add_92: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_335, mul_117);  view_335 = mul_117 = None
    mul_118: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_92, 0.7978845608028654);  add_92 = None
    tanh_11: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_118);  mul_118 = None
    add_93: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_11, 1.0)
    mul_119: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_116, add_93);  mul_116 = add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_336: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_119, [128, 16384]);  mul_119 = None
    permute_131: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
    addmm_23: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_121, view_336, permute_131);  primals_121 = None
    view_337: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_23, [1, 128, 4096]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_94: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_333, view_337);  view_333 = view_337 = None
    add_95: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_94, add_87);  add_94 = add_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_95, [2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 128, 1]" = var_mean_12[0]
    getitem_49: "f32[1, 128, 1]" = var_mean_12[1];  var_mean_12 = None
    add_96: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
    rsqrt_12: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
    sub_24: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_95, getitem_49);  getitem_49 = None
    mul_120: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_12);  sub_24 = None
    mul_121: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_120, primals_122)
    add_97: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_121, primals_123);  mul_121 = primals_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_132: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
    view_338: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_97, [128, 4096]);  add_97 = None
    mm_48: "f32[128, 4096]" = torch.ops.aten.mm.default(view_338, permute_132)
    view_339: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_48, [1, 128, 4096]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_133: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
    mm_49: "f32[128, 4096]" = torch.ops.aten.mm.default(view_338, permute_133)
    view_341: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_49, [1, 128, 4096]);  mm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_134: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_126, [1, 0]);  primals_126 = None
    mm_50: "f32[128, 4096]" = torch.ops.aten.mm.default(view_338, permute_134)
    view_343: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_50, [1, 128, 4096]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_344: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_339, [1, 128, 16, 256]);  view_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_345: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_341, [1, 128, 16, 256]);  view_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_346: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_343, [1, 128, 16, 256]);  view_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_135: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_346, [0, 2, 1, 3]);  view_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_24: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(primals_322, [1, 1, 1]);  primals_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_12: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_24, 1, repeat_1);  repeat_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_12 = torch.ops.aten.split_with_sizes.default(gather_12, [32, 32], 2);  gather_12 = None
    getitem_50: "f32[1, 128, 32]" = split_with_sizes_12[0]
    getitem_51: "f32[1, 128, 32]" = split_with_sizes_12[1];  split_with_sizes_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_580: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_345, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_584: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_345, 3, 64, 9223372036854775807);  view_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_588: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_344, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_592: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_344, 3, 64, 9223372036854775807);  view_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    slice_593: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_50, 0, 0, 9223372036854775807);  getitem_50 = None
    slice_594: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_593, 1, 0, 9223372036854775807);  slice_593 = None
    unsqueeze_158: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_594, 2);  slice_594 = None
    slice_595: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_158, 3, 0, 9223372036854775807);  unsqueeze_158 = None
    unsqueeze_159: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_595, 4);  slice_595 = None
    expand_96: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_159, [1, 128, 1, 32, 2])
    clone_97: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_96, memory_format = torch.contiguous_format);  expand_96 = None
    view_347: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_97, [1, 128, 1, 64]);  clone_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    slice_596: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_51, 0, 0, 9223372036854775807);  getitem_51 = None
    slice_597: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_596, 1, 0, 9223372036854775807);  slice_596 = None
    unsqueeze_160: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_597, 2);  slice_597 = None
    slice_598: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_160, 3, 0, 9223372036854775807);  unsqueeze_160 = None
    unsqueeze_161: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_598, 4);  slice_598 = None
    expand_97: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_161, [1, 128, 1, 32, 2])
    clone_98: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_97, memory_format = torch.contiguous_format);  expand_97 = None
    view_348: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_98, [1, 128, 1, 64]);  clone_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_122: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_580, view_348)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_602: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_580, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_606: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_580, 3, 1, 9223372036854775807, 2);  slice_580 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_24: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_606);  slice_606 = None
    unsqueeze_162: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_24, 4);  neg_24 = None
    unsqueeze_163: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_602, 4);  slice_602 = None
    cat_48: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_162, unsqueeze_163], 4);  unsqueeze_162 = unsqueeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_349: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_48, [1, 128, 16, 64]);  cat_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_123: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_349, view_347);  view_349 = None
    add_98: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_122, mul_123);  mul_122 = mul_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_124: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_588, view_348);  view_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_616: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_588, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_620: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_588, 3, 1, 9223372036854775807, 2);  slice_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_25: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_620);  slice_620 = None
    unsqueeze_168: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_25, 4);  neg_25 = None
    unsqueeze_169: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_616, 4);  slice_616 = None
    cat_49: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_168, unsqueeze_169], 4);  unsqueeze_168 = unsqueeze_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_352: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_49, [1, 128, 16, 64]);  cat_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_125: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_352, view_347);  view_352 = view_347 = None
    add_99: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_124, mul_125);  mul_124 = mul_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_50: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_98, slice_584], 3);  add_98 = slice_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_51: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_99, slice_592], 3);  add_99 = slice_592 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_136: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_50, [0, 2, 1, 3]);  cat_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_137: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_51, [0, 2, 1, 3]);  cat_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_621: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_323, 0, 0, 9223372036854775807);  primals_323 = None
    slice_622: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_621, 1, 0, 9223372036854775807);  slice_621 = None
    slice_623: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_622, 2, 0, 128);  slice_622 = None
    slice_624: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_623, 3, 0, 128);  slice_623 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_138: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_136, [0, 1, 3, 2]);  permute_136 = None
    expand_100: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_137, [1, 16, 128, 256]);  permute_137 = None
    view_353: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_100, [16, 128, 256]);  expand_100 = None
    expand_101: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_138, [1, 16, 256, 128]);  permute_138 = None
    view_354: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_101, [16, 256, 128]);  expand_101 = None
    bmm_24: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_353, view_354)
    view_355: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_24, [1, 16, 128, 128]);  bmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_12: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_624, view_355, full_default);  view_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_24: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_12, primals_324);  where_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_12: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_24, [-1], True)
    sub_25: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_24, amax_12);  div_24 = amax_12 = None
    exp_12: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
    sum_13: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_25: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    alias_24: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    clone_101: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_25);  div_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_102: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_101, [1, 16, 128, 128]);  clone_101 = None
    view_356: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_102, [16, 128, 128]);  expand_102 = None
    expand_103: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_135, [1, 16, 128, 256]);  permute_135 = None
    view_357: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_103, [16, 128, 256]);  expand_103 = None
    bmm_25: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_356, view_357)
    view_358: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_25, [1, 16, 128, 256]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_139: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_358, [0, 2, 1, 3]);  view_358 = None
    clone_102: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_359: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_102, [1, 128, 4096]);  clone_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_140: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
    view_360: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_359, [128, 4096]);  view_359 = None
    mm_51: "f32[128, 4096]" = torch.ops.aten.mm.default(view_360, permute_140)
    view_361: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_51, [1, 128, 4096]);  mm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_141: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
    addmm_24: "f32[128, 16384]" = torch.ops.aten.addmm.default(primals_129, view_338, permute_141);  primals_129 = None
    view_363: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_24, [1, 128, 16384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_126: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_363, 0.5)
    pow_13: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_363, 3.0)
    mul_127: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
    add_100: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_363, mul_127);  view_363 = mul_127 = None
    mul_128: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_100, 0.7978845608028654);  add_100 = None
    tanh_12: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_128);  mul_128 = None
    add_101: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_12, 1.0)
    mul_129: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_126, add_101);  mul_126 = add_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_364: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_129, [128, 16384]);  mul_129 = None
    permute_142: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_130, [1, 0]);  primals_130 = None
    addmm_25: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_131, view_364, permute_142);  primals_131 = None
    view_365: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_25, [1, 128, 4096]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_102: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_361, view_365);  view_361 = view_365 = None
    add_103: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_102, add_95);  add_102 = add_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_13 = torch.ops.aten.var_mean.correction(add_103, [2], correction = 0, keepdim = True)
    getitem_52: "f32[1, 128, 1]" = var_mean_13[0]
    getitem_53: "f32[1, 128, 1]" = var_mean_13[1];  var_mean_13 = None
    add_104: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
    rsqrt_13: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
    sub_26: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_103, getitem_53);  getitem_53 = None
    mul_130: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_13);  sub_26 = None
    mul_131: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_130, primals_132)
    add_105: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_131, primals_133);  mul_131 = primals_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_143: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_134, [1, 0]);  primals_134 = None
    view_366: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_105, [128, 4096]);  add_105 = None
    mm_52: "f32[128, 4096]" = torch.ops.aten.mm.default(view_366, permute_143)
    view_367: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_52, [1, 128, 4096]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_144: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_135, [1, 0]);  primals_135 = None
    mm_53: "f32[128, 4096]" = torch.ops.aten.mm.default(view_366, permute_144)
    view_369: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_53, [1, 128, 4096]);  mm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_145: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
    mm_54: "f32[128, 4096]" = torch.ops.aten.mm.default(view_366, permute_145)
    view_371: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_54, [1, 128, 4096]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_372: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_367, [1, 128, 16, 256]);  view_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_373: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_369, [1, 128, 16, 256]);  view_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_374: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_371, [1, 128, 16, 256]);  view_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_146: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_374, [0, 2, 1, 3]);  view_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_26: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(primals_325, [1, 1, 1]);  primals_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_13: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_26, 1, repeat_1);  repeat_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_13 = torch.ops.aten.split_with_sizes.default(gather_13, [32, 32], 2);  gather_13 = None
    getitem_54: "f32[1, 128, 32]" = split_with_sizes_13[0]
    getitem_55: "f32[1, 128, 32]" = split_with_sizes_13[1];  split_with_sizes_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_628: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_373, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_632: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_373, 3, 64, 9223372036854775807);  view_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_636: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_372, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_640: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_372, 3, 64, 9223372036854775807);  view_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    slice_641: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_54, 0, 0, 9223372036854775807);  getitem_54 = None
    slice_642: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_641, 1, 0, 9223372036854775807);  slice_641 = None
    unsqueeze_171: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_642, 2);  slice_642 = None
    slice_643: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_171, 3, 0, 9223372036854775807);  unsqueeze_171 = None
    unsqueeze_172: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_643, 4);  slice_643 = None
    expand_104: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_172, [1, 128, 1, 32, 2])
    clone_105: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_104, memory_format = torch.contiguous_format);  expand_104 = None
    view_375: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_105, [1, 128, 1, 64]);  clone_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    slice_644: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_55, 0, 0, 9223372036854775807);  getitem_55 = None
    slice_645: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_644, 1, 0, 9223372036854775807);  slice_644 = None
    unsqueeze_173: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_645, 2);  slice_645 = None
    slice_646: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_173, 3, 0, 9223372036854775807);  unsqueeze_173 = None
    unsqueeze_174: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_646, 4);  slice_646 = None
    expand_105: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_174, [1, 128, 1, 32, 2])
    clone_106: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_105, memory_format = torch.contiguous_format);  expand_105 = None
    view_376: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_106, [1, 128, 1, 64]);  clone_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_132: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_628, view_376)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_650: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_628, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_654: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_628, 3, 1, 9223372036854775807, 2);  slice_628 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_26: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_654);  slice_654 = None
    unsqueeze_175: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_26, 4);  neg_26 = None
    unsqueeze_176: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_650, 4);  slice_650 = None
    cat_52: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_175, unsqueeze_176], 4);  unsqueeze_175 = unsqueeze_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_377: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_52, [1, 128, 16, 64]);  cat_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_133: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_377, view_375);  view_377 = None
    add_106: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_132, mul_133);  mul_132 = mul_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_134: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_636, view_376);  view_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_664: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_636, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_668: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_636, 3, 1, 9223372036854775807, 2);  slice_636 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_27: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_668);  slice_668 = None
    unsqueeze_181: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_27, 4);  neg_27 = None
    unsqueeze_182: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_664, 4);  slice_664 = None
    cat_53: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_181, unsqueeze_182], 4);  unsqueeze_181 = unsqueeze_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_380: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_53, [1, 128, 16, 64]);  cat_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_135: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_380, view_375);  view_380 = view_375 = None
    add_107: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_54: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_106, slice_632], 3);  add_106 = slice_632 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_55: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_107, slice_640], 3);  add_107 = slice_640 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_147: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_54, [0, 2, 1, 3]);  cat_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_148: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_55, [0, 2, 1, 3]);  cat_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_669: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_326, 0, 0, 9223372036854775807);  primals_326 = None
    slice_670: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_669, 1, 0, 9223372036854775807);  slice_669 = None
    slice_671: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_670, 2, 0, 128);  slice_670 = None
    slice_672: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_671, 3, 0, 128);  slice_671 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_149: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_147, [0, 1, 3, 2]);  permute_147 = None
    expand_108: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_148, [1, 16, 128, 256]);  permute_148 = None
    view_381: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_108, [16, 128, 256]);  expand_108 = None
    expand_109: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_149, [1, 16, 256, 128]);  permute_149 = None
    view_382: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_109, [16, 256, 128]);  expand_109 = None
    bmm_26: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_381, view_382)
    view_383: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_26, [1, 16, 128, 128]);  bmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_13: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_672, view_383, full_default);  view_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_26: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_13, primals_327);  where_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_13: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_26, [-1], True)
    sub_27: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_26, amax_13);  div_26 = amax_13 = None
    exp_13: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_27);  sub_27 = None
    sum_14: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_27: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
    alias_26: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    clone_109: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_27);  div_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_110: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_109, [1, 16, 128, 128]);  clone_109 = None
    view_384: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_110, [16, 128, 128]);  expand_110 = None
    expand_111: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_146, [1, 16, 128, 256]);  permute_146 = None
    view_385: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_111, [16, 128, 256]);  expand_111 = None
    bmm_27: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_384, view_385)
    view_386: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_27, [1, 16, 128, 256]);  bmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_150: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_386, [0, 2, 1, 3]);  view_386 = None
    clone_110: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_387: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_110, [1, 128, 4096]);  clone_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_151: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_137, [1, 0]);  primals_137 = None
    view_388: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_387, [128, 4096]);  view_387 = None
    mm_55: "f32[128, 4096]" = torch.ops.aten.mm.default(view_388, permute_151)
    view_389: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_55, [1, 128, 4096]);  mm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_152: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
    addmm_26: "f32[128, 16384]" = torch.ops.aten.addmm.default(primals_139, view_366, permute_152);  primals_139 = None
    view_391: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_26, [1, 128, 16384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_136: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_391, 0.5)
    pow_14: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_391, 3.0)
    mul_137: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_14, 0.044715);  pow_14 = None
    add_108: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_391, mul_137);  view_391 = mul_137 = None
    mul_138: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_108, 0.7978845608028654);  add_108 = None
    tanh_13: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_138);  mul_138 = None
    add_109: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_13, 1.0)
    mul_139: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_136, add_109);  mul_136 = add_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_392: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_139, [128, 16384]);  mul_139 = None
    permute_153: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_140, [1, 0]);  primals_140 = None
    addmm_27: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_141, view_392, permute_153);  primals_141 = None
    view_393: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_27, [1, 128, 4096]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_110: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_389, view_393);  view_389 = view_393 = None
    add_111: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_110, add_103);  add_110 = add_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_111, [2], correction = 0, keepdim = True)
    getitem_56: "f32[1, 128, 1]" = var_mean_14[0]
    getitem_57: "f32[1, 128, 1]" = var_mean_14[1];  var_mean_14 = None
    add_112: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
    rsqrt_14: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
    sub_28: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_111, getitem_57);  getitem_57 = None
    mul_140: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_14);  sub_28 = None
    mul_141: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_140, primals_142)
    add_113: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_141, primals_143);  mul_141 = primals_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_154: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
    view_394: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_113, [128, 4096]);  add_113 = None
    mm_56: "f32[128, 4096]" = torch.ops.aten.mm.default(view_394, permute_154)
    view_395: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_56, [1, 128, 4096]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_155: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_145, [1, 0]);  primals_145 = None
    mm_57: "f32[128, 4096]" = torch.ops.aten.mm.default(view_394, permute_155)
    view_397: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_57, [1, 128, 4096]);  mm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_156: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_146, [1, 0]);  primals_146 = None
    mm_58: "f32[128, 4096]" = torch.ops.aten.mm.default(view_394, permute_156)
    view_399: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_58, [1, 128, 4096]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_400: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_395, [1, 128, 16, 256]);  view_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_401: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_397, [1, 128, 16, 256]);  view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_402: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_399, [1, 128, 16, 256]);  view_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_157: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_28: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(primals_328, [1, 1, 1]);  primals_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_14: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_28, 1, repeat_1);  repeat_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_14 = torch.ops.aten.split_with_sizes.default(gather_14, [32, 32], 2);  gather_14 = None
    getitem_58: "f32[1, 128, 32]" = split_with_sizes_14[0]
    getitem_59: "f32[1, 128, 32]" = split_with_sizes_14[1];  split_with_sizes_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_676: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_401, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_680: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_401, 3, 64, 9223372036854775807);  view_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_684: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_400, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_688: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_400, 3, 64, 9223372036854775807);  view_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    slice_689: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_58, 0, 0, 9223372036854775807);  getitem_58 = None
    slice_690: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_689, 1, 0, 9223372036854775807);  slice_689 = None
    unsqueeze_184: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_690, 2);  slice_690 = None
    slice_691: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_184, 3, 0, 9223372036854775807);  unsqueeze_184 = None
    unsqueeze_185: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_691, 4);  slice_691 = None
    expand_112: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_185, [1, 128, 1, 32, 2])
    clone_113: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_112, memory_format = torch.contiguous_format);  expand_112 = None
    view_403: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_113, [1, 128, 1, 64]);  clone_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    slice_692: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_59, 0, 0, 9223372036854775807);  getitem_59 = None
    slice_693: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_692, 1, 0, 9223372036854775807);  slice_692 = None
    unsqueeze_186: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_693, 2);  slice_693 = None
    slice_694: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_186, 3, 0, 9223372036854775807);  unsqueeze_186 = None
    unsqueeze_187: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_694, 4);  slice_694 = None
    expand_113: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_187, [1, 128, 1, 32, 2])
    clone_114: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_113, memory_format = torch.contiguous_format);  expand_113 = None
    view_404: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_114, [1, 128, 1, 64]);  clone_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_142: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_676, view_404)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_698: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_676, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_702: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_676, 3, 1, 9223372036854775807, 2);  slice_676 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_28: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_702);  slice_702 = None
    unsqueeze_188: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_28, 4);  neg_28 = None
    unsqueeze_189: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_698, 4);  slice_698 = None
    cat_56: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_188, unsqueeze_189], 4);  unsqueeze_188 = unsqueeze_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_405: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_56, [1, 128, 16, 64]);  cat_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_143: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_405, view_403);  view_405 = None
    add_114: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_142, mul_143);  mul_142 = mul_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_144: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_684, view_404);  view_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_712: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_684, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_716: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_684, 3, 1, 9223372036854775807, 2);  slice_684 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_29: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_716);  slice_716 = None
    unsqueeze_194: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_29, 4);  neg_29 = None
    unsqueeze_195: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_712, 4);  slice_712 = None
    cat_57: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_194, unsqueeze_195], 4);  unsqueeze_194 = unsqueeze_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_408: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_57, [1, 128, 16, 64]);  cat_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_145: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_408, view_403);  view_408 = view_403 = None
    add_115: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_58: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_114, slice_680], 3);  add_114 = slice_680 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_59: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_115, slice_688], 3);  add_115 = slice_688 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_158: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_58, [0, 2, 1, 3]);  cat_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_159: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_59, [0, 2, 1, 3]);  cat_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_717: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_329, 0, 0, 9223372036854775807);  primals_329 = None
    slice_718: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_717, 1, 0, 9223372036854775807);  slice_717 = None
    slice_719: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_718, 2, 0, 128);  slice_718 = None
    slice_720: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_719, 3, 0, 128);  slice_719 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_160: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_158, [0, 1, 3, 2]);  permute_158 = None
    expand_116: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_159, [1, 16, 128, 256]);  permute_159 = None
    view_409: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_116, [16, 128, 256]);  expand_116 = None
    expand_117: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_160, [1, 16, 256, 128]);  permute_160 = None
    view_410: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_117, [16, 256, 128]);  expand_117 = None
    bmm_28: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_409, view_410)
    view_411: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_28, [1, 16, 128, 128]);  bmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_14: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_720, view_411, full_default);  view_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_28: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_14, primals_330);  where_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_14: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_28, [-1], True)
    sub_29: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_28, amax_14);  div_28 = amax_14 = None
    exp_14: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_29);  sub_29 = None
    sum_15: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_29: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
    alias_28: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    clone_117: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_29);  div_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_118: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_117, [1, 16, 128, 128]);  clone_117 = None
    view_412: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_118, [16, 128, 128]);  expand_118 = None
    expand_119: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_157, [1, 16, 128, 256]);  permute_157 = None
    view_413: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_119, [16, 128, 256]);  expand_119 = None
    bmm_29: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_412, view_413)
    view_414: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_29, [1, 16, 128, 256]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_161: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_414, [0, 2, 1, 3]);  view_414 = None
    clone_118: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_415: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_118, [1, 128, 4096]);  clone_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_162: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
    view_416: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_415, [128, 4096]);  view_415 = None
    mm_59: "f32[128, 4096]" = torch.ops.aten.mm.default(view_416, permute_162)
    view_417: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_59, [1, 128, 4096]);  mm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_163: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_148, [1, 0]);  primals_148 = None
    addmm_28: "f32[128, 16384]" = torch.ops.aten.addmm.default(primals_149, view_394, permute_163);  primals_149 = None
    view_419: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_28, [1, 128, 16384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_146: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_419, 0.5)
    pow_15: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_419, 3.0)
    mul_147: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
    add_116: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_419, mul_147);  view_419 = mul_147 = None
    mul_148: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_116, 0.7978845608028654);  add_116 = None
    tanh_14: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_148);  mul_148 = None
    add_117: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_14, 1.0)
    mul_149: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_146, add_117);  mul_146 = add_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_420: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_149, [128, 16384]);  mul_149 = None
    permute_164: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
    addmm_29: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_151, view_420, permute_164);  primals_151 = None
    view_421: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_29, [1, 128, 4096]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_118: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_417, view_421);  view_417 = view_421 = None
    add_119: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_118, add_111);  add_118 = add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_15 = torch.ops.aten.var_mean.correction(add_119, [2], correction = 0, keepdim = True)
    getitem_60: "f32[1, 128, 1]" = var_mean_15[0]
    getitem_61: "f32[1, 128, 1]" = var_mean_15[1];  var_mean_15 = None
    add_120: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
    rsqrt_15: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
    sub_30: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_119, getitem_61);  getitem_61 = None
    mul_150: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_15);  sub_30 = None
    mul_151: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_150, primals_152)
    add_121: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_151, primals_153);  mul_151 = primals_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_165: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
    view_422: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_121, [128, 4096]);  add_121 = None
    mm_60: "f32[128, 4096]" = torch.ops.aten.mm.default(view_422, permute_165)
    view_423: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_60, [1, 128, 4096]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_166: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_155, [1, 0]);  primals_155 = None
    mm_61: "f32[128, 4096]" = torch.ops.aten.mm.default(view_422, permute_166)
    view_425: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_61, [1, 128, 4096]);  mm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_167: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
    mm_62: "f32[128, 4096]" = torch.ops.aten.mm.default(view_422, permute_167)
    view_427: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_62, [1, 128, 4096]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_428: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_423, [1, 128, 16, 256]);  view_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_429: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_425, [1, 128, 16, 256]);  view_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_430: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_427, [1, 128, 16, 256]);  view_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_168: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_430, [0, 2, 1, 3]);  view_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_30: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(primals_331, [1, 1, 1]);  primals_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_15: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_30, 1, repeat_1);  repeat_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_15 = torch.ops.aten.split_with_sizes.default(gather_15, [32, 32], 2);  gather_15 = None
    getitem_62: "f32[1, 128, 32]" = split_with_sizes_15[0]
    getitem_63: "f32[1, 128, 32]" = split_with_sizes_15[1];  split_with_sizes_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_724: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_429, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_728: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_429, 3, 64, 9223372036854775807);  view_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_732: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_428, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_736: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_428, 3, 64, 9223372036854775807);  view_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    slice_737: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_62, 0, 0, 9223372036854775807);  getitem_62 = None
    slice_738: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_737, 1, 0, 9223372036854775807);  slice_737 = None
    unsqueeze_197: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_738, 2);  slice_738 = None
    slice_739: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_197, 3, 0, 9223372036854775807);  unsqueeze_197 = None
    unsqueeze_198: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_739, 4);  slice_739 = None
    expand_120: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_198, [1, 128, 1, 32, 2])
    clone_121: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_120, memory_format = torch.contiguous_format);  expand_120 = None
    view_431: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_121, [1, 128, 1, 64]);  clone_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    slice_740: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_63, 0, 0, 9223372036854775807);  getitem_63 = None
    slice_741: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_740, 1, 0, 9223372036854775807);  slice_740 = None
    unsqueeze_199: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_741, 2);  slice_741 = None
    slice_742: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_199, 3, 0, 9223372036854775807);  unsqueeze_199 = None
    unsqueeze_200: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_742, 4);  slice_742 = None
    expand_121: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_200, [1, 128, 1, 32, 2])
    clone_122: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_121, memory_format = torch.contiguous_format);  expand_121 = None
    view_432: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_122, [1, 128, 1, 64]);  clone_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_152: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_724, view_432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_746: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_724, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_750: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_724, 3, 1, 9223372036854775807, 2);  slice_724 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_30: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_750);  slice_750 = None
    unsqueeze_201: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_30, 4);  neg_30 = None
    unsqueeze_202: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_746, 4);  slice_746 = None
    cat_60: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_201, unsqueeze_202], 4);  unsqueeze_201 = unsqueeze_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_433: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_60, [1, 128, 16, 64]);  cat_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_153: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_433, view_431);  view_433 = None
    add_122: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_152, mul_153);  mul_152 = mul_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_154: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_732, view_432);  view_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_760: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_732, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_764: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_732, 3, 1, 9223372036854775807, 2);  slice_732 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_31: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_764);  slice_764 = None
    unsqueeze_207: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_31, 4);  neg_31 = None
    unsqueeze_208: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_760, 4);  slice_760 = None
    cat_61: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_207, unsqueeze_208], 4);  unsqueeze_207 = unsqueeze_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_436: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_61, [1, 128, 16, 64]);  cat_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_155: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_436, view_431);  view_436 = view_431 = None
    add_123: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_154, mul_155);  mul_154 = mul_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_62: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_122, slice_728], 3);  add_122 = slice_728 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_63: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_123, slice_736], 3);  add_123 = slice_736 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_169: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_62, [0, 2, 1, 3]);  cat_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_170: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_63, [0, 2, 1, 3]);  cat_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_765: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_332, 0, 0, 9223372036854775807);  primals_332 = None
    slice_766: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_765, 1, 0, 9223372036854775807);  slice_765 = None
    slice_767: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_766, 2, 0, 128);  slice_766 = None
    slice_768: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_767, 3, 0, 128);  slice_767 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_171: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_169, [0, 1, 3, 2]);  permute_169 = None
    expand_124: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_170, [1, 16, 128, 256]);  permute_170 = None
    view_437: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_124, [16, 128, 256]);  expand_124 = None
    expand_125: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_171, [1, 16, 256, 128]);  permute_171 = None
    view_438: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_125, [16, 256, 128]);  expand_125 = None
    bmm_30: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_437, view_438)
    view_439: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_30, [1, 16, 128, 128]);  bmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_15: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_768, view_439, full_default);  view_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_30: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_15, primals_333);  where_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_15: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_30, [-1], True)
    sub_31: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_30, amax_15);  div_30 = amax_15 = None
    exp_15: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_16: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
    div_31: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
    alias_30: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    clone_125: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_31);  div_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_126: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_125, [1, 16, 128, 128]);  clone_125 = None
    view_440: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_126, [16, 128, 128]);  expand_126 = None
    expand_127: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_168, [1, 16, 128, 256]);  permute_168 = None
    view_441: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_127, [16, 128, 256]);  expand_127 = None
    bmm_31: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_440, view_441)
    view_442: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_31, [1, 16, 128, 256]);  bmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_172: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_442, [0, 2, 1, 3]);  view_442 = None
    clone_126: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_443: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_126, [1, 128, 4096]);  clone_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_173: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_157, [1, 0]);  primals_157 = None
    view_444: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_443, [128, 4096]);  view_443 = None
    mm_63: "f32[128, 4096]" = torch.ops.aten.mm.default(view_444, permute_173)
    view_445: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_63, [1, 128, 4096]);  mm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_174: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_158, [1, 0]);  primals_158 = None
    addmm_30: "f32[128, 16384]" = torch.ops.aten.addmm.default(primals_159, view_422, permute_174);  primals_159 = None
    view_447: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_30, [1, 128, 16384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_156: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_447, 0.5)
    pow_16: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_447, 3.0)
    mul_157: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_16, 0.044715);  pow_16 = None
    add_124: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_447, mul_157);  view_447 = mul_157 = None
    mul_158: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_124, 0.7978845608028654);  add_124 = None
    tanh_15: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_158);  mul_158 = None
    add_125: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_15, 1.0)
    mul_159: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_156, add_125);  mul_156 = add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_448: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_159, [128, 16384]);  mul_159 = None
    permute_175: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None
    addmm_31: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_161, view_448, permute_175);  primals_161 = None
    view_449: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_31, [1, 128, 4096]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_126: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_445, view_449);  view_445 = view_449 = None
    add_127: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_126, add_119);  add_126 = add_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_16 = torch.ops.aten.var_mean.correction(add_127, [2], correction = 0, keepdim = True)
    getitem_64: "f32[1, 128, 1]" = var_mean_16[0]
    getitem_65: "f32[1, 128, 1]" = var_mean_16[1];  var_mean_16 = None
    add_128: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
    rsqrt_16: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
    sub_32: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_127, getitem_65);  getitem_65 = None
    mul_160: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_16);  sub_32 = None
    mul_161: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_160, primals_162)
    add_129: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_161, primals_163);  mul_161 = primals_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_176: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_164, [1, 0]);  primals_164 = None
    view_450: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_129, [128, 4096]);  add_129 = None
    mm_64: "f32[128, 4096]" = torch.ops.aten.mm.default(view_450, permute_176)
    view_451: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_64, [1, 128, 4096]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_177: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_165, [1, 0]);  primals_165 = None
    mm_65: "f32[128, 4096]" = torch.ops.aten.mm.default(view_450, permute_177)
    view_453: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_65, [1, 128, 4096]);  mm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_178: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_166, [1, 0]);  primals_166 = None
    mm_66: "f32[128, 4096]" = torch.ops.aten.mm.default(view_450, permute_178)
    view_455: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_66, [1, 128, 4096]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_456: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_451, [1, 128, 16, 256]);  view_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_457: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_453, [1, 128, 16, 256]);  view_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_458: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_455, [1, 128, 16, 256]);  view_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_179: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_458, [0, 2, 1, 3]);  view_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_32: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(primals_334, [1, 1, 1]);  primals_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_16: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_32, 1, repeat_1);  repeat_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_16 = torch.ops.aten.split_with_sizes.default(gather_16, [32, 32], 2);  gather_16 = None
    getitem_66: "f32[1, 128, 32]" = split_with_sizes_16[0]
    getitem_67: "f32[1, 128, 32]" = split_with_sizes_16[1];  split_with_sizes_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_772: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_457, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_776: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_457, 3, 64, 9223372036854775807);  view_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_780: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_456, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_784: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_456, 3, 64, 9223372036854775807);  view_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    slice_785: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_66, 0, 0, 9223372036854775807);  getitem_66 = None
    slice_786: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_785, 1, 0, 9223372036854775807);  slice_785 = None
    unsqueeze_210: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_786, 2);  slice_786 = None
    slice_787: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_210, 3, 0, 9223372036854775807);  unsqueeze_210 = None
    unsqueeze_211: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_787, 4);  slice_787 = None
    expand_128: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_211, [1, 128, 1, 32, 2])
    clone_129: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_128, memory_format = torch.contiguous_format);  expand_128 = None
    view_459: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_129, [1, 128, 1, 64]);  clone_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    slice_788: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_67, 0, 0, 9223372036854775807);  getitem_67 = None
    slice_789: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_788, 1, 0, 9223372036854775807);  slice_788 = None
    unsqueeze_212: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_789, 2);  slice_789 = None
    slice_790: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_212, 3, 0, 9223372036854775807);  unsqueeze_212 = None
    unsqueeze_213: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_790, 4);  slice_790 = None
    expand_129: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_213, [1, 128, 1, 32, 2])
    clone_130: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_129, memory_format = torch.contiguous_format);  expand_129 = None
    view_460: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_130, [1, 128, 1, 64]);  clone_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_162: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_772, view_460)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_794: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_772, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_798: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_772, 3, 1, 9223372036854775807, 2);  slice_772 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_32: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_798);  slice_798 = None
    unsqueeze_214: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_32, 4);  neg_32 = None
    unsqueeze_215: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_794, 4);  slice_794 = None
    cat_64: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_214, unsqueeze_215], 4);  unsqueeze_214 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_461: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_64, [1, 128, 16, 64]);  cat_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_163: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_461, view_459);  view_461 = None
    add_130: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_164: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_780, view_460);  view_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_808: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_780, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_812: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_780, 3, 1, 9223372036854775807, 2);  slice_780 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_33: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_812);  slice_812 = None
    unsqueeze_220: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_33, 4);  neg_33 = None
    unsqueeze_221: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_808, 4);  slice_808 = None
    cat_65: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_220, unsqueeze_221], 4);  unsqueeze_220 = unsqueeze_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_464: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_65, [1, 128, 16, 64]);  cat_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_165: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_464, view_459);  view_464 = view_459 = None
    add_131: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_164, mul_165);  mul_164 = mul_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_66: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_130, slice_776], 3);  add_130 = slice_776 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_67: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_131, slice_784], 3);  add_131 = slice_784 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_180: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_66, [0, 2, 1, 3]);  cat_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_181: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_67, [0, 2, 1, 3]);  cat_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_813: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_335, 0, 0, 9223372036854775807);  primals_335 = None
    slice_814: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_813, 1, 0, 9223372036854775807);  slice_813 = None
    slice_815: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_814, 2, 0, 128);  slice_814 = None
    slice_816: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_815, 3, 0, 128);  slice_815 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_182: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_180, [0, 1, 3, 2]);  permute_180 = None
    expand_132: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_181, [1, 16, 128, 256]);  permute_181 = None
    view_465: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_132, [16, 128, 256]);  expand_132 = None
    expand_133: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_182, [1, 16, 256, 128]);  permute_182 = None
    view_466: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_133, [16, 256, 128]);  expand_133 = None
    bmm_32: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_465, view_466)
    view_467: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_32, [1, 16, 128, 128]);  bmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_16: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_816, view_467, full_default);  view_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_32: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_16, primals_336);  where_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_16: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_32, [-1], True)
    sub_33: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_32, amax_16);  div_32 = amax_16 = None
    exp_16: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_33);  sub_33 = None
    sum_17: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_33: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
    alias_32: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_33)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    clone_133: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_33);  div_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_134: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_133, [1, 16, 128, 128]);  clone_133 = None
    view_468: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_134, [16, 128, 128]);  expand_134 = None
    expand_135: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_179, [1, 16, 128, 256]);  permute_179 = None
    view_469: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_135, [16, 128, 256]);  expand_135 = None
    bmm_33: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_468, view_469)
    view_470: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_33, [1, 16, 128, 256]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_183: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_470, [0, 2, 1, 3]);  view_470 = None
    clone_134: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_471: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_134, [1, 128, 4096]);  clone_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_184: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_167, [1, 0]);  primals_167 = None
    view_472: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_471, [128, 4096]);  view_471 = None
    mm_67: "f32[128, 4096]" = torch.ops.aten.mm.default(view_472, permute_184)
    view_473: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_67, [1, 128, 4096]);  mm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_185: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_168, [1, 0]);  primals_168 = None
    addmm_32: "f32[128, 16384]" = torch.ops.aten.addmm.default(primals_169, view_450, permute_185);  primals_169 = None
    view_475: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_32, [1, 128, 16384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_166: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_475, 0.5)
    pow_17: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_475, 3.0)
    mul_167: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_17, 0.044715);  pow_17 = None
    add_132: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_475, mul_167);  view_475 = mul_167 = None
    mul_168: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_132, 0.7978845608028654);  add_132 = None
    tanh_16: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_168);  mul_168 = None
    add_133: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_16, 1.0)
    mul_169: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_166, add_133);  mul_166 = add_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_476: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_169, [128, 16384]);  mul_169 = None
    permute_186: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
    addmm_33: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_171, view_476, permute_186);  primals_171 = None
    view_477: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_33, [1, 128, 4096]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_134: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_473, view_477);  view_473 = view_477 = None
    add_135: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_134, add_127);  add_134 = add_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_17 = torch.ops.aten.var_mean.correction(add_135, [2], correction = 0, keepdim = True)
    getitem_68: "f32[1, 128, 1]" = var_mean_17[0]
    getitem_69: "f32[1, 128, 1]" = var_mean_17[1];  var_mean_17 = None
    add_136: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
    rsqrt_17: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
    sub_34: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_135, getitem_69);  getitem_69 = None
    mul_170: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_17);  sub_34 = None
    mul_171: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_170, primals_172)
    add_137: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_171, primals_173);  mul_171 = primals_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_187: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_174, [1, 0]);  primals_174 = None
    view_478: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_137, [128, 4096]);  add_137 = None
    mm_68: "f32[128, 4096]" = torch.ops.aten.mm.default(view_478, permute_187)
    view_479: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_68, [1, 128, 4096]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_188: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_175, [1, 0]);  primals_175 = None
    mm_69: "f32[128, 4096]" = torch.ops.aten.mm.default(view_478, permute_188)
    view_481: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_69, [1, 128, 4096]);  mm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_189: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_176, [1, 0]);  primals_176 = None
    mm_70: "f32[128, 4096]" = torch.ops.aten.mm.default(view_478, permute_189)
    view_483: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_70, [1, 128, 4096]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_484: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_479, [1, 128, 16, 256]);  view_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_485: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_481, [1, 128, 16, 256]);  view_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_486: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_483, [1, 128, 16, 256]);  view_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_190: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_486, [0, 2, 1, 3]);  view_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_34: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(primals_337, [1, 1, 1]);  primals_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_17: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_34, 1, repeat_1);  repeat_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_17 = torch.ops.aten.split_with_sizes.default(gather_17, [32, 32], 2);  gather_17 = None
    getitem_70: "f32[1, 128, 32]" = split_with_sizes_17[0]
    getitem_71: "f32[1, 128, 32]" = split_with_sizes_17[1];  split_with_sizes_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_820: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_485, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_824: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_485, 3, 64, 9223372036854775807);  view_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_828: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_484, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_832: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_484, 3, 64, 9223372036854775807);  view_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    slice_833: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_70, 0, 0, 9223372036854775807);  getitem_70 = None
    slice_834: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_833, 1, 0, 9223372036854775807);  slice_833 = None
    unsqueeze_223: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_834, 2);  slice_834 = None
    slice_835: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_223, 3, 0, 9223372036854775807);  unsqueeze_223 = None
    unsqueeze_224: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_835, 4);  slice_835 = None
    expand_136: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_224, [1, 128, 1, 32, 2])
    clone_137: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_136, memory_format = torch.contiguous_format);  expand_136 = None
    view_487: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_137, [1, 128, 1, 64]);  clone_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    slice_836: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_71, 0, 0, 9223372036854775807);  getitem_71 = None
    slice_837: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_836, 1, 0, 9223372036854775807);  slice_836 = None
    unsqueeze_225: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_837, 2);  slice_837 = None
    slice_838: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_225, 3, 0, 9223372036854775807);  unsqueeze_225 = None
    unsqueeze_226: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_838, 4);  slice_838 = None
    expand_137: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_226, [1, 128, 1, 32, 2])
    clone_138: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_137, memory_format = torch.contiguous_format);  expand_137 = None
    view_488: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_138, [1, 128, 1, 64]);  clone_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_172: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_820, view_488)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_842: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_820, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_846: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_820, 3, 1, 9223372036854775807, 2);  slice_820 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_34: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_846);  slice_846 = None
    unsqueeze_227: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_34, 4);  neg_34 = None
    unsqueeze_228: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_842, 4);  slice_842 = None
    cat_68: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_227, unsqueeze_228], 4);  unsqueeze_227 = unsqueeze_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_489: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_68, [1, 128, 16, 64]);  cat_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_173: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_489, view_487);  view_489 = None
    add_138: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_174: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_828, view_488);  view_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_856: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_828, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_860: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_828, 3, 1, 9223372036854775807, 2);  slice_828 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_35: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_860);  slice_860 = None
    unsqueeze_233: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_35, 4);  neg_35 = None
    unsqueeze_234: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_856, 4);  slice_856 = None
    cat_69: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_233, unsqueeze_234], 4);  unsqueeze_233 = unsqueeze_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_492: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_69, [1, 128, 16, 64]);  cat_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_175: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_492, view_487);  view_492 = view_487 = None
    add_139: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_174, mul_175);  mul_174 = mul_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_70: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_138, slice_824], 3);  add_138 = slice_824 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_71: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_139, slice_832], 3);  add_139 = slice_832 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_191: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_70, [0, 2, 1, 3]);  cat_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_192: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_71, [0, 2, 1, 3]);  cat_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_861: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_338, 0, 0, 9223372036854775807);  primals_338 = None
    slice_862: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_861, 1, 0, 9223372036854775807);  slice_861 = None
    slice_863: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_862, 2, 0, 128);  slice_862 = None
    slice_864: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_863, 3, 0, 128);  slice_863 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_193: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_191, [0, 1, 3, 2]);  permute_191 = None
    expand_140: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_192, [1, 16, 128, 256]);  permute_192 = None
    view_493: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_140, [16, 128, 256]);  expand_140 = None
    expand_141: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_193, [1, 16, 256, 128]);  permute_193 = None
    view_494: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_141, [16, 256, 128]);  expand_141 = None
    bmm_34: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_493, view_494)
    view_495: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_34, [1, 16, 128, 128]);  bmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_17: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_864, view_495, full_default);  view_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_34: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_17, primals_339);  where_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_17: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_34, [-1], True)
    sub_35: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_34, amax_17);  div_34 = amax_17 = None
    exp_17: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_35);  sub_35 = None
    sum_18: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
    div_35: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
    alias_34: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    clone_141: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_35);  div_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_142: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_141, [1, 16, 128, 128]);  clone_141 = None
    view_496: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_142, [16, 128, 128]);  expand_142 = None
    expand_143: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_190, [1, 16, 128, 256]);  permute_190 = None
    view_497: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_143, [16, 128, 256]);  expand_143 = None
    bmm_35: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_496, view_497)
    view_498: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_35, [1, 16, 128, 256]);  bmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_194: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_498, [0, 2, 1, 3]);  view_498 = None
    clone_142: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_499: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_142, [1, 128, 4096]);  clone_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_195: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_177, [1, 0]);  primals_177 = None
    view_500: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_499, [128, 4096]);  view_499 = None
    mm_71: "f32[128, 4096]" = torch.ops.aten.mm.default(view_500, permute_195)
    view_501: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_71, [1, 128, 4096]);  mm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_196: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_178, [1, 0]);  primals_178 = None
    addmm_34: "f32[128, 16384]" = torch.ops.aten.addmm.default(primals_179, view_478, permute_196);  primals_179 = None
    view_503: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_34, [1, 128, 16384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_176: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_503, 0.5)
    pow_18: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_503, 3.0)
    mul_177: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_18, 0.044715);  pow_18 = None
    add_140: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_503, mul_177);  view_503 = mul_177 = None
    mul_178: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_140, 0.7978845608028654);  add_140 = None
    tanh_17: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_178);  mul_178 = None
    add_141: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_17, 1.0)
    mul_179: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_176, add_141);  mul_176 = add_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_504: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_179, [128, 16384]);  mul_179 = None
    permute_197: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_180, [1, 0]);  primals_180 = None
    addmm_35: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_181, view_504, permute_197);  primals_181 = None
    view_505: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_35, [1, 128, 4096]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_142: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_501, view_505);  view_501 = view_505 = None
    add_143: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_142, add_135);  add_142 = add_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_18 = torch.ops.aten.var_mean.correction(add_143, [2], correction = 0, keepdim = True)
    getitem_72: "f32[1, 128, 1]" = var_mean_18[0]
    getitem_73: "f32[1, 128, 1]" = var_mean_18[1];  var_mean_18 = None
    add_144: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
    rsqrt_18: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    sub_36: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_143, getitem_73);  getitem_73 = None
    mul_180: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_18);  sub_36 = None
    mul_181: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_180, primals_182)
    add_145: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_181, primals_183);  mul_181 = primals_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_198: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None
    view_506: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_145, [128, 4096]);  add_145 = None
    mm_72: "f32[128, 4096]" = torch.ops.aten.mm.default(view_506, permute_198)
    view_507: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_72, [1, 128, 4096]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_199: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_185, [1, 0]);  primals_185 = None
    mm_73: "f32[128, 4096]" = torch.ops.aten.mm.default(view_506, permute_199)
    view_509: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_73, [1, 128, 4096]);  mm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_200: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_186, [1, 0]);  primals_186 = None
    mm_74: "f32[128, 4096]" = torch.ops.aten.mm.default(view_506, permute_200)
    view_511: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_74, [1, 128, 4096]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_512: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_507, [1, 128, 16, 256]);  view_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_513: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_509, [1, 128, 16, 256]);  view_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_514: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_511, [1, 128, 16, 256]);  view_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_201: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_514, [0, 2, 1, 3]);  view_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_36: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(primals_340, [1, 1, 1]);  primals_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_18: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_36, 1, repeat_1);  repeat_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_18 = torch.ops.aten.split_with_sizes.default(gather_18, [32, 32], 2);  gather_18 = None
    getitem_74: "f32[1, 128, 32]" = split_with_sizes_18[0]
    getitem_75: "f32[1, 128, 32]" = split_with_sizes_18[1];  split_with_sizes_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_868: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_513, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_872: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_513, 3, 64, 9223372036854775807);  view_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_876: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_512, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_880: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_512, 3, 64, 9223372036854775807);  view_512 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    slice_881: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_74, 0, 0, 9223372036854775807);  getitem_74 = None
    slice_882: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_881, 1, 0, 9223372036854775807);  slice_881 = None
    unsqueeze_236: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_882, 2);  slice_882 = None
    slice_883: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_236, 3, 0, 9223372036854775807);  unsqueeze_236 = None
    unsqueeze_237: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_883, 4);  slice_883 = None
    expand_144: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_237, [1, 128, 1, 32, 2])
    clone_145: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_144, memory_format = torch.contiguous_format);  expand_144 = None
    view_515: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_145, [1, 128, 1, 64]);  clone_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    slice_884: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_75, 0, 0, 9223372036854775807);  getitem_75 = None
    slice_885: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_884, 1, 0, 9223372036854775807);  slice_884 = None
    unsqueeze_238: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_885, 2);  slice_885 = None
    slice_886: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_238, 3, 0, 9223372036854775807);  unsqueeze_238 = None
    unsqueeze_239: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_886, 4);  slice_886 = None
    expand_145: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_239, [1, 128, 1, 32, 2])
    clone_146: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_145, memory_format = torch.contiguous_format);  expand_145 = None
    view_516: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_146, [1, 128, 1, 64]);  clone_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_182: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_868, view_516)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_890: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_868, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_894: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_868, 3, 1, 9223372036854775807, 2);  slice_868 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_36: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_894);  slice_894 = None
    unsqueeze_240: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_36, 4);  neg_36 = None
    unsqueeze_241: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_890, 4);  slice_890 = None
    cat_72: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_240, unsqueeze_241], 4);  unsqueeze_240 = unsqueeze_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_517: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_72, [1, 128, 16, 64]);  cat_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_183: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_517, view_515);  view_517 = None
    add_146: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_182, mul_183);  mul_182 = mul_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_184: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_876, view_516);  view_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_904: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_876, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_908: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_876, 3, 1, 9223372036854775807, 2);  slice_876 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_37: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_908);  slice_908 = None
    unsqueeze_246: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_37, 4);  neg_37 = None
    unsqueeze_247: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_904, 4);  slice_904 = None
    cat_73: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_246, unsqueeze_247], 4);  unsqueeze_246 = unsqueeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_520: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_73, [1, 128, 16, 64]);  cat_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_185: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_520, view_515);  view_520 = view_515 = None
    add_147: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_184, mul_185);  mul_184 = mul_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_74: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_146, slice_872], 3);  add_146 = slice_872 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_75: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_147, slice_880], 3);  add_147 = slice_880 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_202: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_74, [0, 2, 1, 3]);  cat_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_203: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_75, [0, 2, 1, 3]);  cat_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_909: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_341, 0, 0, 9223372036854775807);  primals_341 = None
    slice_910: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_909, 1, 0, 9223372036854775807);  slice_909 = None
    slice_911: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_910, 2, 0, 128);  slice_910 = None
    slice_912: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_911, 3, 0, 128);  slice_911 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_204: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_202, [0, 1, 3, 2]);  permute_202 = None
    expand_148: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_203, [1, 16, 128, 256]);  permute_203 = None
    view_521: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_148, [16, 128, 256]);  expand_148 = None
    expand_149: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_204, [1, 16, 256, 128]);  permute_204 = None
    view_522: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_149, [16, 256, 128]);  expand_149 = None
    bmm_36: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_521, view_522)
    view_523: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_36, [1, 16, 128, 128]);  bmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_18: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_912, view_523, full_default);  view_523 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_36: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_18, primals_342);  where_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_18: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_36, [-1], True)
    sub_37: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_36, amax_18);  div_36 = amax_18 = None
    exp_18: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
    sum_19: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
    div_37: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
    alias_36: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_37)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    clone_149: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_37);  div_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_150: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_149, [1, 16, 128, 128]);  clone_149 = None
    view_524: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_150, [16, 128, 128]);  expand_150 = None
    expand_151: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_201, [1, 16, 128, 256]);  permute_201 = None
    view_525: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_151, [16, 128, 256]);  expand_151 = None
    bmm_37: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_524, view_525)
    view_526: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_37, [1, 16, 128, 256]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_205: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_526, [0, 2, 1, 3]);  view_526 = None
    clone_150: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_527: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_150, [1, 128, 4096]);  clone_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_206: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_187, [1, 0]);  primals_187 = None
    view_528: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_527, [128, 4096]);  view_527 = None
    mm_75: "f32[128, 4096]" = torch.ops.aten.mm.default(view_528, permute_206)
    view_529: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_75, [1, 128, 4096]);  mm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_207: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
    addmm_36: "f32[128, 16384]" = torch.ops.aten.addmm.default(primals_189, view_506, permute_207);  primals_189 = None
    view_531: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_36, [1, 128, 16384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_186: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_531, 0.5)
    pow_19: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_531, 3.0)
    mul_187: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_19, 0.044715);  pow_19 = None
    add_148: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_531, mul_187);  view_531 = mul_187 = None
    mul_188: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_148, 0.7978845608028654);  add_148 = None
    tanh_18: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_188);  mul_188 = None
    add_149: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_18, 1.0)
    mul_189: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_186, add_149);  mul_186 = add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_532: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_189, [128, 16384]);  mul_189 = None
    permute_208: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_190, [1, 0]);  primals_190 = None
    addmm_37: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_191, view_532, permute_208);  primals_191 = None
    view_533: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_37, [1, 128, 4096]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_150: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_529, view_533);  view_529 = view_533 = None
    add_151: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_150, add_143);  add_150 = add_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_19 = torch.ops.aten.var_mean.correction(add_151, [2], correction = 0, keepdim = True)
    getitem_76: "f32[1, 128, 1]" = var_mean_19[0]
    getitem_77: "f32[1, 128, 1]" = var_mean_19[1];  var_mean_19 = None
    add_152: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
    rsqrt_19: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
    sub_38: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_151, getitem_77);  getitem_77 = None
    mul_190: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_19);  sub_38 = None
    mul_191: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_190, primals_192)
    add_153: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_191, primals_193);  mul_191 = primals_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_209: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_194, [1, 0]);  primals_194 = None
    view_534: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_153, [128, 4096]);  add_153 = None
    mm_76: "f32[128, 4096]" = torch.ops.aten.mm.default(view_534, permute_209)
    view_535: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_76, [1, 128, 4096]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_210: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_195, [1, 0]);  primals_195 = None
    mm_77: "f32[128, 4096]" = torch.ops.aten.mm.default(view_534, permute_210)
    view_537: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_77, [1, 128, 4096]);  mm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_211: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_196, [1, 0]);  primals_196 = None
    mm_78: "f32[128, 4096]" = torch.ops.aten.mm.default(view_534, permute_211)
    view_539: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_78, [1, 128, 4096]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_540: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_535, [1, 128, 16, 256]);  view_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_541: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_537, [1, 128, 16, 256]);  view_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_542: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_539, [1, 128, 16, 256]);  view_539 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_212: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_542, [0, 2, 1, 3]);  view_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_38: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(primals_343, [1, 1, 1]);  primals_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_19: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_38, 1, repeat_1);  repeat_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_19 = torch.ops.aten.split_with_sizes.default(gather_19, [32, 32], 2);  gather_19 = None
    getitem_78: "f32[1, 128, 32]" = split_with_sizes_19[0]
    getitem_79: "f32[1, 128, 32]" = split_with_sizes_19[1];  split_with_sizes_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_916: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_541, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_920: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_541, 3, 64, 9223372036854775807);  view_541 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_924: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_540, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_928: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_540, 3, 64, 9223372036854775807);  view_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    slice_929: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_78, 0, 0, 9223372036854775807);  getitem_78 = None
    slice_930: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_929, 1, 0, 9223372036854775807);  slice_929 = None
    unsqueeze_249: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_930, 2);  slice_930 = None
    slice_931: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_249, 3, 0, 9223372036854775807);  unsqueeze_249 = None
    unsqueeze_250: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_931, 4);  slice_931 = None
    expand_152: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_250, [1, 128, 1, 32, 2])
    clone_153: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_152, memory_format = torch.contiguous_format);  expand_152 = None
    view_543: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_153, [1, 128, 1, 64]);  clone_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    slice_932: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_79, 0, 0, 9223372036854775807);  getitem_79 = None
    slice_933: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_932, 1, 0, 9223372036854775807);  slice_932 = None
    unsqueeze_251: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_933, 2);  slice_933 = None
    slice_934: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_251, 3, 0, 9223372036854775807);  unsqueeze_251 = None
    unsqueeze_252: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_934, 4);  slice_934 = None
    expand_153: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_252, [1, 128, 1, 32, 2])
    clone_154: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_153, memory_format = torch.contiguous_format);  expand_153 = None
    view_544: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_154, [1, 128, 1, 64]);  clone_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_192: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_916, view_544)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_938: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_916, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_942: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_916, 3, 1, 9223372036854775807, 2);  slice_916 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_38: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_942);  slice_942 = None
    unsqueeze_253: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_38, 4);  neg_38 = None
    unsqueeze_254: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_938, 4);  slice_938 = None
    cat_76: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_253, unsqueeze_254], 4);  unsqueeze_253 = unsqueeze_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_545: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_76, [1, 128, 16, 64]);  cat_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_193: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_545, view_543);  view_545 = None
    add_154: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_192, mul_193);  mul_192 = mul_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_194: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_924, view_544);  view_544 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_952: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_924, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_956: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_924, 3, 1, 9223372036854775807, 2);  slice_924 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_39: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_956);  slice_956 = None
    unsqueeze_259: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_39, 4);  neg_39 = None
    unsqueeze_260: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_952, 4);  slice_952 = None
    cat_77: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_259, unsqueeze_260], 4);  unsqueeze_259 = unsqueeze_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_548: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_77, [1, 128, 16, 64]);  cat_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_195: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_548, view_543);  view_548 = view_543 = None
    add_155: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_194, mul_195);  mul_194 = mul_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_78: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_154, slice_920], 3);  add_154 = slice_920 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_79: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_155, slice_928], 3);  add_155 = slice_928 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_213: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_78, [0, 2, 1, 3]);  cat_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_214: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_79, [0, 2, 1, 3]);  cat_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_957: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_344, 0, 0, 9223372036854775807);  primals_344 = None
    slice_958: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_957, 1, 0, 9223372036854775807);  slice_957 = None
    slice_959: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_958, 2, 0, 128);  slice_958 = None
    slice_960: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_959, 3, 0, 128);  slice_959 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_215: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_213, [0, 1, 3, 2]);  permute_213 = None
    expand_156: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_214, [1, 16, 128, 256]);  permute_214 = None
    view_549: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_156, [16, 128, 256]);  expand_156 = None
    expand_157: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_215, [1, 16, 256, 128]);  permute_215 = None
    view_550: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_157, [16, 256, 128]);  expand_157 = None
    bmm_38: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_549, view_550)
    view_551: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_38, [1, 16, 128, 128]);  bmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_19: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_960, view_551, full_default);  view_551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_38: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_19, primals_345);  where_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_19: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_38, [-1], True)
    sub_39: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_38, amax_19);  div_38 = amax_19 = None
    exp_19: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_39);  sub_39 = None
    sum_20: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
    div_39: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
    alias_38: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_39)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    clone_157: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_39);  div_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_158: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_157, [1, 16, 128, 128]);  clone_157 = None
    view_552: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_158, [16, 128, 128]);  expand_158 = None
    expand_159: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_212, [1, 16, 128, 256]);  permute_212 = None
    view_553: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_159, [16, 128, 256]);  expand_159 = None
    bmm_39: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_552, view_553)
    view_554: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_39, [1, 16, 128, 256]);  bmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_216: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_554, [0, 2, 1, 3]);  view_554 = None
    clone_158: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_555: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_158, [1, 128, 4096]);  clone_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_217: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_197, [1, 0]);  primals_197 = None
    view_556: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_555, [128, 4096]);  view_555 = None
    mm_79: "f32[128, 4096]" = torch.ops.aten.mm.default(view_556, permute_217)
    view_557: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_79, [1, 128, 4096]);  mm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_218: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_198, [1, 0]);  primals_198 = None
    addmm_38: "f32[128, 16384]" = torch.ops.aten.addmm.default(primals_199, view_534, permute_218);  primals_199 = None
    view_559: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_38, [1, 128, 16384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_196: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_559, 0.5)
    pow_20: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_559, 3.0)
    mul_197: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_20, 0.044715);  pow_20 = None
    add_156: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_559, mul_197);  view_559 = mul_197 = None
    mul_198: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_156, 0.7978845608028654);  add_156 = None
    tanh_19: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_198);  mul_198 = None
    add_157: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_19, 1.0)
    mul_199: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_196, add_157);  mul_196 = add_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_560: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_199, [128, 16384]);  mul_199 = None
    permute_219: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_200, [1, 0]);  primals_200 = None
    addmm_39: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_201, view_560, permute_219);  primals_201 = None
    view_561: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_39, [1, 128, 4096]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_158: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_557, view_561);  view_557 = view_561 = None
    add_159: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_158, add_151);  add_158 = add_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_159, [2], correction = 0, keepdim = True)
    getitem_80: "f32[1, 128, 1]" = var_mean_20[0]
    getitem_81: "f32[1, 128, 1]" = var_mean_20[1];  var_mean_20 = None
    add_160: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
    rsqrt_20: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_160);  add_160 = None
    sub_40: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_159, getitem_81);  getitem_81 = None
    mul_200: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_20);  sub_40 = None
    mul_201: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_200, primals_202)
    add_161: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_201, primals_203);  mul_201 = primals_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_220: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_204, [1, 0]);  primals_204 = None
    view_562: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_161, [128, 4096]);  add_161 = None
    mm_80: "f32[128, 4096]" = torch.ops.aten.mm.default(view_562, permute_220)
    view_563: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_80, [1, 128, 4096]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_221: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_205, [1, 0]);  primals_205 = None
    mm_81: "f32[128, 4096]" = torch.ops.aten.mm.default(view_562, permute_221)
    view_565: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_81, [1, 128, 4096]);  mm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_222: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_206, [1, 0]);  primals_206 = None
    mm_82: "f32[128, 4096]" = torch.ops.aten.mm.default(view_562, permute_222)
    view_567: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_82, [1, 128, 4096]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_568: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_563, [1, 128, 16, 256]);  view_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_569: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_565, [1, 128, 16, 256]);  view_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_570: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_567, [1, 128, 16, 256]);  view_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_223: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_570, [0, 2, 1, 3]);  view_570 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_40: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(primals_346, [1, 1, 1]);  primals_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_20: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_40, 1, repeat_1);  repeat_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_20 = torch.ops.aten.split_with_sizes.default(gather_20, [32, 32], 2);  gather_20 = None
    getitem_82: "f32[1, 128, 32]" = split_with_sizes_20[0]
    getitem_83: "f32[1, 128, 32]" = split_with_sizes_20[1];  split_with_sizes_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_964: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_569, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_968: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_569, 3, 64, 9223372036854775807);  view_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_972: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_568, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_976: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_568, 3, 64, 9223372036854775807);  view_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    slice_977: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_82, 0, 0, 9223372036854775807);  getitem_82 = None
    slice_978: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_977, 1, 0, 9223372036854775807);  slice_977 = None
    unsqueeze_262: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_978, 2);  slice_978 = None
    slice_979: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_262, 3, 0, 9223372036854775807);  unsqueeze_262 = None
    unsqueeze_263: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_979, 4);  slice_979 = None
    expand_160: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_263, [1, 128, 1, 32, 2])
    clone_161: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_160, memory_format = torch.contiguous_format);  expand_160 = None
    view_571: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_161, [1, 128, 1, 64]);  clone_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    slice_980: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_83, 0, 0, 9223372036854775807);  getitem_83 = None
    slice_981: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_980, 1, 0, 9223372036854775807);  slice_980 = None
    unsqueeze_264: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_981, 2);  slice_981 = None
    slice_982: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_264, 3, 0, 9223372036854775807);  unsqueeze_264 = None
    unsqueeze_265: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_982, 4);  slice_982 = None
    expand_161: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_265, [1, 128, 1, 32, 2])
    clone_162: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_161, memory_format = torch.contiguous_format);  expand_161 = None
    view_572: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_162, [1, 128, 1, 64]);  clone_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_202: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_964, view_572)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_986: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_964, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_990: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_964, 3, 1, 9223372036854775807, 2);  slice_964 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_40: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_990);  slice_990 = None
    unsqueeze_266: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_40, 4);  neg_40 = None
    unsqueeze_267: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_986, 4);  slice_986 = None
    cat_80: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_266, unsqueeze_267], 4);  unsqueeze_266 = unsqueeze_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_573: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_80, [1, 128, 16, 64]);  cat_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_203: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_573, view_571);  view_573 = None
    add_162: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_202, mul_203);  mul_202 = mul_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_204: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_972, view_572);  view_572 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_1000: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_972, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_1004: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_972, 3, 1, 9223372036854775807, 2);  slice_972 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_41: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_1004);  slice_1004 = None
    unsqueeze_272: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_41, 4);  neg_41 = None
    unsqueeze_273: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1000, 4);  slice_1000 = None
    cat_81: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_272, unsqueeze_273], 4);  unsqueeze_272 = unsqueeze_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_576: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_81, [1, 128, 16, 64]);  cat_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_205: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_576, view_571);  view_576 = view_571 = None
    add_163: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_82: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_162, slice_968], 3);  add_162 = slice_968 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_83: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_163, slice_976], 3);  add_163 = slice_976 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_224: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_82, [0, 2, 1, 3]);  cat_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_225: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_83, [0, 2, 1, 3]);  cat_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_1005: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_347, 0, 0, 9223372036854775807);  primals_347 = None
    slice_1006: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_1005, 1, 0, 9223372036854775807);  slice_1005 = None
    slice_1007: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_1006, 2, 0, 128);  slice_1006 = None
    slice_1008: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_1007, 3, 0, 128);  slice_1007 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_226: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_224, [0, 1, 3, 2]);  permute_224 = None
    expand_164: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_225, [1, 16, 128, 256]);  permute_225 = None
    view_577: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_164, [16, 128, 256]);  expand_164 = None
    expand_165: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_226, [1, 16, 256, 128]);  permute_226 = None
    view_578: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_165, [16, 256, 128]);  expand_165 = None
    bmm_40: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_577, view_578)
    view_579: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_40, [1, 16, 128, 128]);  bmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_20: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_1008, view_579, full_default);  view_579 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_40: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_20, primals_348);  where_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_20: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_40, [-1], True)
    sub_41: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_40, amax_20);  div_40 = amax_20 = None
    exp_20: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_41);  sub_41 = None
    sum_21: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
    div_41: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
    alias_40: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_41)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    clone_165: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_41);  div_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_166: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_165, [1, 16, 128, 128]);  clone_165 = None
    view_580: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_166, [16, 128, 128]);  expand_166 = None
    expand_167: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_223, [1, 16, 128, 256]);  permute_223 = None
    view_581: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_167, [16, 128, 256]);  expand_167 = None
    bmm_41: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_580, view_581)
    view_582: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_41, [1, 16, 128, 256]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_227: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_582, [0, 2, 1, 3]);  view_582 = None
    clone_166: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_583: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_166, [1, 128, 4096]);  clone_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_228: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_207, [1, 0]);  primals_207 = None
    view_584: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_583, [128, 4096]);  view_583 = None
    mm_83: "f32[128, 4096]" = torch.ops.aten.mm.default(view_584, permute_228)
    view_585: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_83, [1, 128, 4096]);  mm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_229: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_208, [1, 0]);  primals_208 = None
    addmm_40: "f32[128, 16384]" = torch.ops.aten.addmm.default(primals_209, view_562, permute_229);  primals_209 = None
    view_587: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_40, [1, 128, 16384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_206: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_587, 0.5)
    pow_21: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_587, 3.0)
    mul_207: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
    add_164: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_587, mul_207);  view_587 = mul_207 = None
    mul_208: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_164, 0.7978845608028654);  add_164 = None
    tanh_20: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_208);  mul_208 = None
    add_165: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_20, 1.0)
    mul_209: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_206, add_165);  mul_206 = add_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_588: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_209, [128, 16384]);  mul_209 = None
    permute_230: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_210, [1, 0]);  primals_210 = None
    addmm_41: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_211, view_588, permute_230);  primals_211 = None
    view_589: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_41, [1, 128, 4096]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_166: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_585, view_589);  view_585 = view_589 = None
    add_167: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_166, add_159);  add_166 = add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_167, [2], correction = 0, keepdim = True)
    getitem_84: "f32[1, 128, 1]" = var_mean_21[0]
    getitem_85: "f32[1, 128, 1]" = var_mean_21[1];  var_mean_21 = None
    add_168: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05);  getitem_84 = None
    rsqrt_21: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
    sub_42: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_167, getitem_85);  getitem_85 = None
    mul_210: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_21);  sub_42 = None
    mul_211: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_210, primals_212)
    add_169: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_211, primals_213);  mul_211 = primals_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_231: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_214, [1, 0]);  primals_214 = None
    view_590: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_169, [128, 4096]);  add_169 = None
    mm_84: "f32[128, 4096]" = torch.ops.aten.mm.default(view_590, permute_231)
    view_591: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_84, [1, 128, 4096]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_232: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_215, [1, 0]);  primals_215 = None
    mm_85: "f32[128, 4096]" = torch.ops.aten.mm.default(view_590, permute_232)
    view_593: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_85, [1, 128, 4096]);  mm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_233: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_216, [1, 0]);  primals_216 = None
    mm_86: "f32[128, 4096]" = torch.ops.aten.mm.default(view_590, permute_233)
    view_595: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_86, [1, 128, 4096]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_596: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_591, [1, 128, 16, 256]);  view_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_597: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_593, [1, 128, 16, 256]);  view_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_598: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_595, [1, 128, 16, 256]);  view_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_234: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_598, [0, 2, 1, 3]);  view_598 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_42: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(primals_349, [1, 1, 1]);  primals_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_21: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_42, 1, repeat_1);  repeat_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_21 = torch.ops.aten.split_with_sizes.default(gather_21, [32, 32], 2);  gather_21 = None
    getitem_86: "f32[1, 128, 32]" = split_with_sizes_21[0]
    getitem_87: "f32[1, 128, 32]" = split_with_sizes_21[1];  split_with_sizes_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_1012: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_597, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_1016: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_597, 3, 64, 9223372036854775807);  view_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_1020: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_596, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_1024: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_596, 3, 64, 9223372036854775807);  view_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    slice_1025: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_86, 0, 0, 9223372036854775807);  getitem_86 = None
    slice_1026: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_1025, 1, 0, 9223372036854775807);  slice_1025 = None
    unsqueeze_275: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_1026, 2);  slice_1026 = None
    slice_1027: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_275, 3, 0, 9223372036854775807);  unsqueeze_275 = None
    unsqueeze_276: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1027, 4);  slice_1027 = None
    expand_168: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_276, [1, 128, 1, 32, 2])
    clone_169: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_168, memory_format = torch.contiguous_format);  expand_168 = None
    view_599: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_169, [1, 128, 1, 64]);  clone_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    slice_1028: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_87, 0, 0, 9223372036854775807);  getitem_87 = None
    slice_1029: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_1028, 1, 0, 9223372036854775807);  slice_1028 = None
    unsqueeze_277: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_1029, 2);  slice_1029 = None
    slice_1030: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_277, 3, 0, 9223372036854775807);  unsqueeze_277 = None
    unsqueeze_278: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1030, 4);  slice_1030 = None
    expand_169: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_278, [1, 128, 1, 32, 2])
    clone_170: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_169, memory_format = torch.contiguous_format);  expand_169 = None
    view_600: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_170, [1, 128, 1, 64]);  clone_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_212: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1012, view_600)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_1034: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1012, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_1038: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1012, 3, 1, 9223372036854775807, 2);  slice_1012 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_42: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_1038);  slice_1038 = None
    unsqueeze_279: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_42, 4);  neg_42 = None
    unsqueeze_280: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1034, 4);  slice_1034 = None
    cat_84: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_279, unsqueeze_280], 4);  unsqueeze_279 = unsqueeze_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_601: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_84, [1, 128, 16, 64]);  cat_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_213: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_601, view_599);  view_601 = None
    add_170: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_212, mul_213);  mul_212 = mul_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_214: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1020, view_600);  view_600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_1048: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1020, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_1052: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1020, 3, 1, 9223372036854775807, 2);  slice_1020 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_43: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_1052);  slice_1052 = None
    unsqueeze_285: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_43, 4);  neg_43 = None
    unsqueeze_286: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1048, 4);  slice_1048 = None
    cat_85: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_285, unsqueeze_286], 4);  unsqueeze_285 = unsqueeze_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_604: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_85, [1, 128, 16, 64]);  cat_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_215: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_604, view_599);  view_604 = view_599 = None
    add_171: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_86: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_170, slice_1016], 3);  add_170 = slice_1016 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_87: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_171, slice_1024], 3);  add_171 = slice_1024 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_235: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_86, [0, 2, 1, 3]);  cat_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_236: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_87, [0, 2, 1, 3]);  cat_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_1053: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_350, 0, 0, 9223372036854775807);  primals_350 = None
    slice_1054: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_1053, 1, 0, 9223372036854775807);  slice_1053 = None
    slice_1055: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_1054, 2, 0, 128);  slice_1054 = None
    slice_1056: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_1055, 3, 0, 128);  slice_1055 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_237: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_235, [0, 1, 3, 2]);  permute_235 = None
    expand_172: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_236, [1, 16, 128, 256]);  permute_236 = None
    view_605: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_172, [16, 128, 256]);  expand_172 = None
    expand_173: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_237, [1, 16, 256, 128]);  permute_237 = None
    view_606: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_173, [16, 256, 128]);  expand_173 = None
    bmm_42: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_605, view_606)
    view_607: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_42, [1, 16, 128, 128]);  bmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_21: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_1056, view_607, full_default);  view_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_42: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_21, primals_351);  where_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_21: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_42, [-1], True)
    sub_43: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_42, amax_21);  div_42 = amax_21 = None
    exp_21: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
    sum_22: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
    div_43: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
    alias_42: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_43)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    clone_173: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_43);  div_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_174: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_173, [1, 16, 128, 128]);  clone_173 = None
    view_608: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_174, [16, 128, 128]);  expand_174 = None
    expand_175: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_234, [1, 16, 128, 256]);  permute_234 = None
    view_609: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_175, [16, 128, 256]);  expand_175 = None
    bmm_43: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_608, view_609)
    view_610: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_43, [1, 16, 128, 256]);  bmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_238: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_610, [0, 2, 1, 3]);  view_610 = None
    clone_174: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_611: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_174, [1, 128, 4096]);  clone_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_239: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_217, [1, 0]);  primals_217 = None
    view_612: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_611, [128, 4096]);  view_611 = None
    mm_87: "f32[128, 4096]" = torch.ops.aten.mm.default(view_612, permute_239)
    view_613: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_87, [1, 128, 4096]);  mm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_240: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_218, [1, 0]);  primals_218 = None
    addmm_42: "f32[128, 16384]" = torch.ops.aten.addmm.default(primals_219, view_590, permute_240);  primals_219 = None
    view_615: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_42, [1, 128, 16384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_216: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_615, 0.5)
    pow_22: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_615, 3.0)
    mul_217: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_22, 0.044715);  pow_22 = None
    add_172: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_615, mul_217);  view_615 = mul_217 = None
    mul_218: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_172, 0.7978845608028654);  add_172 = None
    tanh_21: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_218);  mul_218 = None
    add_173: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_21, 1.0)
    mul_219: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_216, add_173);  mul_216 = add_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_616: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_219, [128, 16384]);  mul_219 = None
    permute_241: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_220, [1, 0]);  primals_220 = None
    addmm_43: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_221, view_616, permute_241);  primals_221 = None
    view_617: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_43, [1, 128, 4096]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_174: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_613, view_617);  view_613 = view_617 = None
    add_175: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_174, add_167);  add_174 = add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_175, [2], correction = 0, keepdim = True)
    getitem_88: "f32[1, 128, 1]" = var_mean_22[0]
    getitem_89: "f32[1, 128, 1]" = var_mean_22[1];  var_mean_22 = None
    add_176: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
    rsqrt_22: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_176);  add_176 = None
    sub_44: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_175, getitem_89);  getitem_89 = None
    mul_220: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_22);  sub_44 = None
    mul_221: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_220, primals_222)
    add_177: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_221, primals_223);  mul_221 = primals_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_242: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_224, [1, 0]);  primals_224 = None
    view_618: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_177, [128, 4096]);  add_177 = None
    mm_88: "f32[128, 4096]" = torch.ops.aten.mm.default(view_618, permute_242)
    view_619: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_88, [1, 128, 4096]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_243: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_225, [1, 0]);  primals_225 = None
    mm_89: "f32[128, 4096]" = torch.ops.aten.mm.default(view_618, permute_243)
    view_621: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_89, [1, 128, 4096]);  mm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_244: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_226, [1, 0]);  primals_226 = None
    mm_90: "f32[128, 4096]" = torch.ops.aten.mm.default(view_618, permute_244)
    view_623: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_90, [1, 128, 4096]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_624: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_619, [1, 128, 16, 256]);  view_619 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_625: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_621, [1, 128, 16, 256]);  view_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_626: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_623, [1, 128, 16, 256]);  view_623 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_245: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_626, [0, 2, 1, 3]);  view_626 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_44: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(primals_352, [1, 1, 1]);  primals_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_22: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_44, 1, repeat_1);  repeat_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_22 = torch.ops.aten.split_with_sizes.default(gather_22, [32, 32], 2);  gather_22 = None
    getitem_90: "f32[1, 128, 32]" = split_with_sizes_22[0]
    getitem_91: "f32[1, 128, 32]" = split_with_sizes_22[1];  split_with_sizes_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_1060: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_625, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_1064: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_625, 3, 64, 9223372036854775807);  view_625 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_1068: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_624, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_1072: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_624, 3, 64, 9223372036854775807);  view_624 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    slice_1073: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_90, 0, 0, 9223372036854775807);  getitem_90 = None
    slice_1074: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_1073, 1, 0, 9223372036854775807);  slice_1073 = None
    unsqueeze_288: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_1074, 2);  slice_1074 = None
    slice_1075: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_288, 3, 0, 9223372036854775807);  unsqueeze_288 = None
    unsqueeze_289: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1075, 4);  slice_1075 = None
    expand_176: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_289, [1, 128, 1, 32, 2])
    clone_177: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_176, memory_format = torch.contiguous_format);  expand_176 = None
    view_627: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_177, [1, 128, 1, 64]);  clone_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    slice_1076: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_91, 0, 0, 9223372036854775807);  getitem_91 = None
    slice_1077: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_1076, 1, 0, 9223372036854775807);  slice_1076 = None
    unsqueeze_290: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_1077, 2);  slice_1077 = None
    slice_1078: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_290, 3, 0, 9223372036854775807);  unsqueeze_290 = None
    unsqueeze_291: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1078, 4);  slice_1078 = None
    expand_177: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_291, [1, 128, 1, 32, 2])
    clone_178: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_177, memory_format = torch.contiguous_format);  expand_177 = None
    view_628: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_178, [1, 128, 1, 64]);  clone_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_222: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1060, view_628)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_1082: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1060, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_1086: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1060, 3, 1, 9223372036854775807, 2);  slice_1060 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_44: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_1086);  slice_1086 = None
    unsqueeze_292: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_44, 4);  neg_44 = None
    unsqueeze_293: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1082, 4);  slice_1082 = None
    cat_88: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_292, unsqueeze_293], 4);  unsqueeze_292 = unsqueeze_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_629: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_88, [1, 128, 16, 64]);  cat_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_223: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_629, view_627);  view_629 = None
    add_178: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_222, mul_223);  mul_222 = mul_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_224: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1068, view_628);  view_628 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_1096: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1068, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_1100: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1068, 3, 1, 9223372036854775807, 2);  slice_1068 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_45: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_1100);  slice_1100 = None
    unsqueeze_298: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_45, 4);  neg_45 = None
    unsqueeze_299: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1096, 4);  slice_1096 = None
    cat_89: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_298, unsqueeze_299], 4);  unsqueeze_298 = unsqueeze_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_632: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_89, [1, 128, 16, 64]);  cat_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_225: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_632, view_627);  view_632 = view_627 = None
    add_179: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_224, mul_225);  mul_224 = mul_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_90: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_178, slice_1064], 3);  add_178 = slice_1064 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_91: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_179, slice_1072], 3);  add_179 = slice_1072 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_246: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_90, [0, 2, 1, 3]);  cat_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_247: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_91, [0, 2, 1, 3]);  cat_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_1101: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_353, 0, 0, 9223372036854775807);  primals_353 = None
    slice_1102: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_1101, 1, 0, 9223372036854775807);  slice_1101 = None
    slice_1103: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_1102, 2, 0, 128);  slice_1102 = None
    slice_1104: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_1103, 3, 0, 128);  slice_1103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_248: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_246, [0, 1, 3, 2]);  permute_246 = None
    expand_180: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_247, [1, 16, 128, 256]);  permute_247 = None
    view_633: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_180, [16, 128, 256]);  expand_180 = None
    expand_181: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_248, [1, 16, 256, 128]);  permute_248 = None
    view_634: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_181, [16, 256, 128]);  expand_181 = None
    bmm_44: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_633, view_634)
    view_635: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_44, [1, 16, 128, 128]);  bmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_22: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_1104, view_635, full_default);  view_635 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_44: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_22, primals_354);  where_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_22: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_44, [-1], True)
    sub_45: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_44, amax_22);  div_44 = amax_22 = None
    exp_22: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_45);  sub_45 = None
    sum_23: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
    div_45: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
    alias_44: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_45)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    clone_181: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_45);  div_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_182: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_181, [1, 16, 128, 128]);  clone_181 = None
    view_636: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_182, [16, 128, 128]);  expand_182 = None
    expand_183: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_245, [1, 16, 128, 256]);  permute_245 = None
    view_637: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_183, [16, 128, 256]);  expand_183 = None
    bmm_45: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_636, view_637)
    view_638: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_45, [1, 16, 128, 256]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_249: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_638, [0, 2, 1, 3]);  view_638 = None
    clone_182: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_639: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_182, [1, 128, 4096]);  clone_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_250: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_227, [1, 0]);  primals_227 = None
    view_640: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_639, [128, 4096]);  view_639 = None
    mm_91: "f32[128, 4096]" = torch.ops.aten.mm.default(view_640, permute_250)
    view_641: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_91, [1, 128, 4096]);  mm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_251: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_228, [1, 0]);  primals_228 = None
    addmm_44: "f32[128, 16384]" = torch.ops.aten.addmm.default(primals_229, view_618, permute_251);  primals_229 = None
    view_643: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_44, [1, 128, 16384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_226: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_643, 0.5)
    pow_23: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_643, 3.0)
    mul_227: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_23, 0.044715);  pow_23 = None
    add_180: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_643, mul_227);  view_643 = mul_227 = None
    mul_228: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_180, 0.7978845608028654);  add_180 = None
    tanh_22: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_228);  mul_228 = None
    add_181: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_22, 1.0)
    mul_229: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_226, add_181);  mul_226 = add_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_644: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_229, [128, 16384]);  mul_229 = None
    permute_252: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_230, [1, 0]);  primals_230 = None
    addmm_45: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_231, view_644, permute_252);  primals_231 = None
    view_645: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_45, [1, 128, 4096]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_182: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_641, view_645);  view_641 = view_645 = None
    add_183: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_182, add_175);  add_182 = add_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_183, [2], correction = 0, keepdim = True)
    getitem_92: "f32[1, 128, 1]" = var_mean_23[0]
    getitem_93: "f32[1, 128, 1]" = var_mean_23[1];  var_mean_23 = None
    add_184: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
    rsqrt_23: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_184);  add_184 = None
    sub_46: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_183, getitem_93);  getitem_93 = None
    mul_230: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_23);  sub_46 = None
    mul_231: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_230, primals_232)
    add_185: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_231, primals_233);  mul_231 = primals_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_253: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_234, [1, 0]);  primals_234 = None
    view_646: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_185, [128, 4096]);  add_185 = None
    mm_92: "f32[128, 4096]" = torch.ops.aten.mm.default(view_646, permute_253)
    view_647: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_92, [1, 128, 4096]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_254: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_235, [1, 0]);  primals_235 = None
    mm_93: "f32[128, 4096]" = torch.ops.aten.mm.default(view_646, permute_254)
    view_649: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_93, [1, 128, 4096]);  mm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_255: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_236, [1, 0]);  primals_236 = None
    mm_94: "f32[128, 4096]" = torch.ops.aten.mm.default(view_646, permute_255)
    view_651: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_94, [1, 128, 4096]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_652: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_647, [1, 128, 16, 256]);  view_647 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_653: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_649, [1, 128, 16, 256]);  view_649 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_654: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_651, [1, 128, 16, 256]);  view_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_256: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_654, [0, 2, 1, 3]);  view_654 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_46: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(primals_355, [1, 1, 1]);  primals_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_23: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_46, 1, repeat_1);  repeat_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_23 = torch.ops.aten.split_with_sizes.default(gather_23, [32, 32], 2);  gather_23 = None
    getitem_94: "f32[1, 128, 32]" = split_with_sizes_23[0]
    getitem_95: "f32[1, 128, 32]" = split_with_sizes_23[1];  split_with_sizes_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_1108: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_653, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_1112: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_653, 3, 64, 9223372036854775807);  view_653 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_1116: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_652, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_1120: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_652, 3, 64, 9223372036854775807);  view_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    slice_1121: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_94, 0, 0, 9223372036854775807);  getitem_94 = None
    slice_1122: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_1121, 1, 0, 9223372036854775807);  slice_1121 = None
    unsqueeze_301: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_1122, 2);  slice_1122 = None
    slice_1123: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_301, 3, 0, 9223372036854775807);  unsqueeze_301 = None
    unsqueeze_302: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1123, 4);  slice_1123 = None
    expand_184: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_302, [1, 128, 1, 32, 2])
    clone_185: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_184, memory_format = torch.contiguous_format);  expand_184 = None
    view_655: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_185, [1, 128, 1, 64]);  clone_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    slice_1124: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_95, 0, 0, 9223372036854775807);  getitem_95 = None
    slice_1125: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_1124, 1, 0, 9223372036854775807);  slice_1124 = None
    unsqueeze_303: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_1125, 2);  slice_1125 = None
    slice_1126: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_303, 3, 0, 9223372036854775807);  unsqueeze_303 = None
    unsqueeze_304: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1126, 4);  slice_1126 = None
    expand_185: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_304, [1, 128, 1, 32, 2])
    clone_186: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_185, memory_format = torch.contiguous_format);  expand_185 = None
    view_656: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_186, [1, 128, 1, 64]);  clone_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_232: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1108, view_656)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_1130: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1108, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_1134: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1108, 3, 1, 9223372036854775807, 2);  slice_1108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_46: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_1134);  slice_1134 = None
    unsqueeze_305: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_46, 4);  neg_46 = None
    unsqueeze_306: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1130, 4);  slice_1130 = None
    cat_92: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_305, unsqueeze_306], 4);  unsqueeze_305 = unsqueeze_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_657: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_92, [1, 128, 16, 64]);  cat_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_233: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_657, view_655);  view_657 = None
    add_186: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_234: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1116, view_656);  view_656 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_1144: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1116, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_1148: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1116, 3, 1, 9223372036854775807, 2);  slice_1116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_47: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_1148);  slice_1148 = None
    unsqueeze_311: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_47, 4);  neg_47 = None
    unsqueeze_312: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1144, 4);  slice_1144 = None
    cat_93: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_311, unsqueeze_312], 4);  unsqueeze_311 = unsqueeze_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_660: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_93, [1, 128, 16, 64]);  cat_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_235: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_660, view_655);  view_660 = view_655 = None
    add_187: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_234, mul_235);  mul_234 = mul_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_94: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_186, slice_1112], 3);  add_186 = slice_1112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_95: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_187, slice_1120], 3);  add_187 = slice_1120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_257: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_94, [0, 2, 1, 3]);  cat_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_258: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_95, [0, 2, 1, 3]);  cat_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_1149: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_356, 0, 0, 9223372036854775807);  primals_356 = None
    slice_1150: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_1149, 1, 0, 9223372036854775807);  slice_1149 = None
    slice_1151: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_1150, 2, 0, 128);  slice_1150 = None
    slice_1152: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_1151, 3, 0, 128);  slice_1151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_259: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_257, [0, 1, 3, 2]);  permute_257 = None
    expand_188: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_258, [1, 16, 128, 256]);  permute_258 = None
    view_661: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_188, [16, 128, 256]);  expand_188 = None
    expand_189: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_259, [1, 16, 256, 128]);  permute_259 = None
    view_662: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_189, [16, 256, 128]);  expand_189 = None
    bmm_46: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_661, view_662)
    view_663: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_46, [1, 16, 128, 128]);  bmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_23: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_1152, view_663, full_default);  view_663 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_46: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_23, primals_357);  where_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_23: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_46, [-1], True)
    sub_47: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_46, amax_23);  div_46 = amax_23 = None
    exp_23: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_47);  sub_47 = None
    sum_24: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
    div_47: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
    alias_46: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_47)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    clone_189: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_47);  div_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_190: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_189, [1, 16, 128, 128]);  clone_189 = None
    view_664: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_190, [16, 128, 128]);  expand_190 = None
    expand_191: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_256, [1, 16, 128, 256]);  permute_256 = None
    view_665: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_191, [16, 128, 256]);  expand_191 = None
    bmm_47: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_664, view_665)
    view_666: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_47, [1, 16, 128, 256]);  bmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_260: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_666, [0, 2, 1, 3]);  view_666 = None
    clone_190: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_667: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_190, [1, 128, 4096]);  clone_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_261: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_237, [1, 0]);  primals_237 = None
    view_668: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_667, [128, 4096]);  view_667 = None
    mm_95: "f32[128, 4096]" = torch.ops.aten.mm.default(view_668, permute_261)
    view_669: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_95, [1, 128, 4096]);  mm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_262: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_238, [1, 0]);  primals_238 = None
    addmm_46: "f32[128, 16384]" = torch.ops.aten.addmm.default(primals_239, view_646, permute_262);  primals_239 = None
    view_671: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_46, [1, 128, 16384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_236: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_671, 0.5)
    pow_24: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_671, 3.0)
    mul_237: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_24, 0.044715);  pow_24 = None
    add_188: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_671, mul_237);  view_671 = mul_237 = None
    mul_238: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_188, 0.7978845608028654);  add_188 = None
    tanh_23: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_238);  mul_238 = None
    add_189: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_23, 1.0)
    mul_239: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_236, add_189);  mul_236 = add_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_672: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_239, [128, 16384]);  mul_239 = None
    permute_263: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_240, [1, 0]);  primals_240 = None
    addmm_47: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_241, view_672, permute_263);  primals_241 = None
    view_673: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_47, [1, 128, 4096]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_190: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_669, view_673);  view_669 = view_673 = None
    add_191: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_190, add_183);  add_190 = add_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_191, [2], correction = 0, keepdim = True)
    getitem_96: "f32[1, 128, 1]" = var_mean_24[0]
    getitem_97: "f32[1, 128, 1]" = var_mean_24[1];  var_mean_24 = None
    add_192: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
    rsqrt_24: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
    sub_48: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_191, getitem_97);  getitem_97 = None
    mul_240: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_24);  sub_48 = None
    mul_241: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_240, primals_242)
    add_193: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_241, primals_243);  mul_241 = primals_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_264: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_244, [1, 0]);  primals_244 = None
    view_674: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_193, [128, 4096]);  add_193 = None
    mm_96: "f32[128, 4096]" = torch.ops.aten.mm.default(view_674, permute_264)
    view_675: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_96, [1, 128, 4096]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_265: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_245, [1, 0]);  primals_245 = None
    mm_97: "f32[128, 4096]" = torch.ops.aten.mm.default(view_674, permute_265)
    view_677: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_97, [1, 128, 4096]);  mm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_266: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_246, [1, 0]);  primals_246 = None
    mm_98: "f32[128, 4096]" = torch.ops.aten.mm.default(view_674, permute_266)
    view_679: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_98, [1, 128, 4096]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_680: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_675, [1, 128, 16, 256]);  view_675 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_681: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_677, [1, 128, 16, 256]);  view_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_682: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_679, [1, 128, 16, 256]);  view_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_267: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_682, [0, 2, 1, 3]);  view_682 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_48: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(primals_358, [1, 1, 1]);  primals_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_24: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_48, 1, repeat_1);  repeat_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_24 = torch.ops.aten.split_with_sizes.default(gather_24, [32, 32], 2);  gather_24 = None
    getitem_98: "f32[1, 128, 32]" = split_with_sizes_24[0]
    getitem_99: "f32[1, 128, 32]" = split_with_sizes_24[1];  split_with_sizes_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_1156: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_681, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_1160: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_681, 3, 64, 9223372036854775807);  view_681 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_1164: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_680, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_1168: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_680, 3, 64, 9223372036854775807);  view_680 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    slice_1169: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_98, 0, 0, 9223372036854775807);  getitem_98 = None
    slice_1170: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_1169, 1, 0, 9223372036854775807);  slice_1169 = None
    unsqueeze_314: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_1170, 2);  slice_1170 = None
    slice_1171: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_314, 3, 0, 9223372036854775807);  unsqueeze_314 = None
    unsqueeze_315: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1171, 4);  slice_1171 = None
    expand_192: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_315, [1, 128, 1, 32, 2])
    clone_193: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_192, memory_format = torch.contiguous_format);  expand_192 = None
    view_683: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_193, [1, 128, 1, 64]);  clone_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    slice_1172: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_99, 0, 0, 9223372036854775807);  getitem_99 = None
    slice_1173: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_1172, 1, 0, 9223372036854775807);  slice_1172 = None
    unsqueeze_316: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_1173, 2);  slice_1173 = None
    slice_1174: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_316, 3, 0, 9223372036854775807);  unsqueeze_316 = None
    unsqueeze_317: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1174, 4);  slice_1174 = None
    expand_193: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_317, [1, 128, 1, 32, 2])
    clone_194: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_193, memory_format = torch.contiguous_format);  expand_193 = None
    view_684: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_194, [1, 128, 1, 64]);  clone_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_242: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1156, view_684)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_1178: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1156, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_1182: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1156, 3, 1, 9223372036854775807, 2);  slice_1156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_48: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_1182);  slice_1182 = None
    unsqueeze_318: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_48, 4);  neg_48 = None
    unsqueeze_319: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1178, 4);  slice_1178 = None
    cat_96: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_318, unsqueeze_319], 4);  unsqueeze_318 = unsqueeze_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_685: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_96, [1, 128, 16, 64]);  cat_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_243: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_685, view_683);  view_685 = None
    add_194: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_244: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1164, view_684);  view_684 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_1192: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1164, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_1196: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1164, 3, 1, 9223372036854775807, 2);  slice_1164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_49: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_1196);  slice_1196 = None
    unsqueeze_324: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_49, 4);  neg_49 = None
    unsqueeze_325: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1192, 4);  slice_1192 = None
    cat_97: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_324, unsqueeze_325], 4);  unsqueeze_324 = unsqueeze_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_688: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_97, [1, 128, 16, 64]);  cat_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_245: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_688, view_683);  view_688 = view_683 = None
    add_195: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_244, mul_245);  mul_244 = mul_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_98: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_194, slice_1160], 3);  add_194 = slice_1160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_99: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_195, slice_1168], 3);  add_195 = slice_1168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_268: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_98, [0, 2, 1, 3]);  cat_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_269: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_99, [0, 2, 1, 3]);  cat_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_1197: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_359, 0, 0, 9223372036854775807);  primals_359 = None
    slice_1198: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_1197, 1, 0, 9223372036854775807);  slice_1197 = None
    slice_1199: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_1198, 2, 0, 128);  slice_1198 = None
    slice_1200: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_1199, 3, 0, 128);  slice_1199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_270: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_268, [0, 1, 3, 2]);  permute_268 = None
    expand_196: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_269, [1, 16, 128, 256]);  permute_269 = None
    view_689: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_196, [16, 128, 256]);  expand_196 = None
    expand_197: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_270, [1, 16, 256, 128]);  permute_270 = None
    view_690: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_197, [16, 256, 128]);  expand_197 = None
    bmm_48: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_689, view_690)
    view_691: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_48, [1, 16, 128, 128]);  bmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_24: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_1200, view_691, full_default);  view_691 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_48: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_24, primals_360);  where_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_24: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_48, [-1], True)
    sub_49: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_48, amax_24);  div_48 = amax_24 = None
    exp_24: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
    sum_25: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [-1], True)
    div_49: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_24, sum_25);  exp_24 = sum_25 = None
    alias_48: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_49)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    clone_197: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_49);  div_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_198: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_197, [1, 16, 128, 128]);  clone_197 = None
    view_692: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_198, [16, 128, 128]);  expand_198 = None
    expand_199: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_267, [1, 16, 128, 256]);  permute_267 = None
    view_693: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_199, [16, 128, 256]);  expand_199 = None
    bmm_49: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_692, view_693)
    view_694: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_49, [1, 16, 128, 256]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_271: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_694, [0, 2, 1, 3]);  view_694 = None
    clone_198: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_271, memory_format = torch.contiguous_format);  permute_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_695: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_198, [1, 128, 4096]);  clone_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_272: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_247, [1, 0]);  primals_247 = None
    view_696: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_695, [128, 4096]);  view_695 = None
    mm_99: "f32[128, 4096]" = torch.ops.aten.mm.default(view_696, permute_272)
    view_697: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_99, [1, 128, 4096]);  mm_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_273: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_248, [1, 0]);  primals_248 = None
    addmm_48: "f32[128, 16384]" = torch.ops.aten.addmm.default(primals_249, view_674, permute_273);  primals_249 = None
    view_699: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_48, [1, 128, 16384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_246: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_699, 0.5)
    pow_25: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_699, 3.0)
    mul_247: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_25, 0.044715);  pow_25 = None
    add_196: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_699, mul_247);  view_699 = mul_247 = None
    mul_248: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_196, 0.7978845608028654);  add_196 = None
    tanh_24: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_248);  mul_248 = None
    add_197: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_24, 1.0)
    mul_249: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_246, add_197);  mul_246 = add_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_700: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_249, [128, 16384]);  mul_249 = None
    permute_274: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_250, [1, 0]);  primals_250 = None
    addmm_49: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_251, view_700, permute_274);  primals_251 = None
    view_701: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_49, [1, 128, 4096]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_198: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_697, view_701);  view_697 = view_701 = None
    add_199: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_198, add_191);  add_198 = add_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_25 = torch.ops.aten.var_mean.correction(add_199, [2], correction = 0, keepdim = True)
    getitem_100: "f32[1, 128, 1]" = var_mean_25[0]
    getitem_101: "f32[1, 128, 1]" = var_mean_25[1];  var_mean_25 = None
    add_200: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05);  getitem_100 = None
    rsqrt_25: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_200);  add_200 = None
    sub_50: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_199, getitem_101);  getitem_101 = None
    mul_250: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_25);  sub_50 = None
    mul_251: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_250, primals_252)
    add_201: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_251, primals_253);  mul_251 = primals_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_275: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_254, [1, 0]);  primals_254 = None
    view_702: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_201, [128, 4096]);  add_201 = None
    mm_100: "f32[128, 4096]" = torch.ops.aten.mm.default(view_702, permute_275)
    view_703: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_100, [1, 128, 4096]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_276: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_255, [1, 0]);  primals_255 = None
    mm_101: "f32[128, 4096]" = torch.ops.aten.mm.default(view_702, permute_276)
    view_705: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_101, [1, 128, 4096]);  mm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_277: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_256, [1, 0]);  primals_256 = None
    mm_102: "f32[128, 4096]" = torch.ops.aten.mm.default(view_702, permute_277)
    view_707: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_102, [1, 128, 4096]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_708: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_703, [1, 128, 16, 256]);  view_703 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_709: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_705, [1, 128, 16, 256]);  view_705 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_710: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_707, [1, 128, 16, 256]);  view_707 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_278: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_710, [0, 2, 1, 3]);  view_710 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_50: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(primals_361, [1, 1, 1]);  primals_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_25: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_50, 1, repeat_1);  repeat_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_25 = torch.ops.aten.split_with_sizes.default(gather_25, [32, 32], 2);  gather_25 = None
    getitem_102: "f32[1, 128, 32]" = split_with_sizes_25[0]
    getitem_103: "f32[1, 128, 32]" = split_with_sizes_25[1];  split_with_sizes_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_1204: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_709, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_1208: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_709, 3, 64, 9223372036854775807);  view_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_1212: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_708, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_1216: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_708, 3, 64, 9223372036854775807);  view_708 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    slice_1217: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_102, 0, 0, 9223372036854775807);  getitem_102 = None
    slice_1218: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_1217, 1, 0, 9223372036854775807);  slice_1217 = None
    unsqueeze_327: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_1218, 2);  slice_1218 = None
    slice_1219: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_327, 3, 0, 9223372036854775807);  unsqueeze_327 = None
    unsqueeze_328: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1219, 4);  slice_1219 = None
    expand_200: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_328, [1, 128, 1, 32, 2])
    clone_201: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_200, memory_format = torch.contiguous_format);  expand_200 = None
    view_711: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_201, [1, 128, 1, 64]);  clone_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    slice_1220: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_103, 0, 0, 9223372036854775807);  getitem_103 = None
    slice_1221: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_1220, 1, 0, 9223372036854775807);  slice_1220 = None
    unsqueeze_329: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_1221, 2);  slice_1221 = None
    slice_1222: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_329, 3, 0, 9223372036854775807);  unsqueeze_329 = None
    unsqueeze_330: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1222, 4);  slice_1222 = None
    expand_201: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_330, [1, 128, 1, 32, 2])
    clone_202: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_201, memory_format = torch.contiguous_format);  expand_201 = None
    view_712: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_202, [1, 128, 1, 64]);  clone_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_252: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1204, view_712)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_1226: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1204, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_1230: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1204, 3, 1, 9223372036854775807, 2);  slice_1204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_50: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_1230);  slice_1230 = None
    unsqueeze_331: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_50, 4);  neg_50 = None
    unsqueeze_332: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1226, 4);  slice_1226 = None
    cat_100: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_331, unsqueeze_332], 4);  unsqueeze_331 = unsqueeze_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_713: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_100, [1, 128, 16, 64]);  cat_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_253: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_713, view_711);  view_713 = None
    add_202: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_252, mul_253);  mul_252 = mul_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_254: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1212, view_712);  view_712 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_1240: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1212, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_1244: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1212, 3, 1, 9223372036854775807, 2);  slice_1212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_51: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_1244);  slice_1244 = None
    unsqueeze_337: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_51, 4);  neg_51 = None
    unsqueeze_338: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1240, 4);  slice_1240 = None
    cat_101: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_337, unsqueeze_338], 4);  unsqueeze_337 = unsqueeze_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_716: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_101, [1, 128, 16, 64]);  cat_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_255: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_716, view_711);  view_716 = view_711 = None
    add_203: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_254, mul_255);  mul_254 = mul_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_102: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_202, slice_1208], 3);  add_202 = slice_1208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_103: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_203, slice_1216], 3);  add_203 = slice_1216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_279: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_102, [0, 2, 1, 3]);  cat_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_280: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_103, [0, 2, 1, 3]);  cat_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_1245: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_362, 0, 0, 9223372036854775807);  primals_362 = None
    slice_1246: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_1245, 1, 0, 9223372036854775807);  slice_1245 = None
    slice_1247: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_1246, 2, 0, 128);  slice_1246 = None
    slice_1248: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_1247, 3, 0, 128);  slice_1247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_281: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_279, [0, 1, 3, 2]);  permute_279 = None
    expand_204: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_280, [1, 16, 128, 256]);  permute_280 = None
    view_717: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_204, [16, 128, 256]);  expand_204 = None
    expand_205: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_281, [1, 16, 256, 128]);  permute_281 = None
    view_718: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_205, [16, 256, 128]);  expand_205 = None
    bmm_50: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_717, view_718)
    view_719: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_50, [1, 16, 128, 128]);  bmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_25: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_1248, view_719, full_default);  view_719 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_50: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_25, primals_363);  where_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_25: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_50, [-1], True)
    sub_51: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_50, amax_25);  div_50 = amax_25 = None
    exp_25: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_51);  sub_51 = None
    sum_26: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_25, [-1], True)
    div_51: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_25, sum_26);  exp_25 = sum_26 = None
    alias_50: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_51)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    clone_205: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_51);  div_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_206: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_205, [1, 16, 128, 128]);  clone_205 = None
    view_720: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_206, [16, 128, 128]);  expand_206 = None
    expand_207: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_278, [1, 16, 128, 256]);  permute_278 = None
    view_721: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_207, [16, 128, 256]);  expand_207 = None
    bmm_51: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_720, view_721)
    view_722: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_51, [1, 16, 128, 256]);  bmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_282: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_722, [0, 2, 1, 3]);  view_722 = None
    clone_206: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_282, memory_format = torch.contiguous_format);  permute_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_723: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_206, [1, 128, 4096]);  clone_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_283: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_257, [1, 0]);  primals_257 = None
    view_724: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_723, [128, 4096]);  view_723 = None
    mm_103: "f32[128, 4096]" = torch.ops.aten.mm.default(view_724, permute_283)
    view_725: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_103, [1, 128, 4096]);  mm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_284: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_258, [1, 0]);  primals_258 = None
    addmm_50: "f32[128, 16384]" = torch.ops.aten.addmm.default(primals_259, view_702, permute_284);  primals_259 = None
    view_727: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_50, [1, 128, 16384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_256: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_727, 0.5)
    pow_26: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_727, 3.0)
    mul_257: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_26, 0.044715);  pow_26 = None
    add_204: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_727, mul_257);  view_727 = mul_257 = None
    mul_258: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_204, 0.7978845608028654);  add_204 = None
    tanh_25: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_258);  mul_258 = None
    add_205: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_25, 1.0)
    mul_259: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_256, add_205);  mul_256 = add_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_728: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_259, [128, 16384]);  mul_259 = None
    permute_285: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_260, [1, 0]);  primals_260 = None
    addmm_51: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_261, view_728, permute_285);  primals_261 = None
    view_729: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_51, [1, 128, 4096]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_206: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_725, view_729);  view_725 = view_729 = None
    add_207: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_206, add_199);  add_206 = add_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_26 = torch.ops.aten.var_mean.correction(add_207, [2], correction = 0, keepdim = True)
    getitem_104: "f32[1, 128, 1]" = var_mean_26[0]
    getitem_105: "f32[1, 128, 1]" = var_mean_26[1];  var_mean_26 = None
    add_208: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
    rsqrt_26: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_208);  add_208 = None
    sub_52: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_207, getitem_105);  getitem_105 = None
    mul_260: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_26);  sub_52 = None
    mul_261: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_260, primals_262)
    add_209: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_261, primals_263);  mul_261 = primals_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_286: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_264, [1, 0]);  primals_264 = None
    view_730: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_209, [128, 4096]);  add_209 = None
    mm_104: "f32[128, 4096]" = torch.ops.aten.mm.default(view_730, permute_286)
    view_731: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_104, [1, 128, 4096]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_287: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_265, [1, 0]);  primals_265 = None
    mm_105: "f32[128, 4096]" = torch.ops.aten.mm.default(view_730, permute_287)
    view_733: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_105, [1, 128, 4096]);  mm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_288: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_266, [1, 0]);  primals_266 = None
    mm_106: "f32[128, 4096]" = torch.ops.aten.mm.default(view_730, permute_288)
    view_735: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_106, [1, 128, 4096]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_736: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_731, [1, 128, 16, 256]);  view_731 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_737: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_733, [1, 128, 16, 256]);  view_733 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_738: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_735, [1, 128, 16, 256]);  view_735 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_289: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_738, [0, 2, 1, 3]);  view_738 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_52: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(primals_364, [1, 1, 1]);  primals_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_26: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_52, 1, repeat_1);  repeat_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_26 = torch.ops.aten.split_with_sizes.default(gather_26, [32, 32], 2);  gather_26 = None
    getitem_106: "f32[1, 128, 32]" = split_with_sizes_26[0]
    getitem_107: "f32[1, 128, 32]" = split_with_sizes_26[1];  split_with_sizes_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_1252: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_737, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_1256: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_737, 3, 64, 9223372036854775807);  view_737 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_1260: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_736, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_1264: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_736, 3, 64, 9223372036854775807);  view_736 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    slice_1265: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_106, 0, 0, 9223372036854775807);  getitem_106 = None
    slice_1266: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_1265, 1, 0, 9223372036854775807);  slice_1265 = None
    unsqueeze_340: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_1266, 2);  slice_1266 = None
    slice_1267: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_340, 3, 0, 9223372036854775807);  unsqueeze_340 = None
    unsqueeze_341: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1267, 4);  slice_1267 = None
    expand_208: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_341, [1, 128, 1, 32, 2])
    clone_209: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_208, memory_format = torch.contiguous_format);  expand_208 = None
    view_739: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_209, [1, 128, 1, 64]);  clone_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    slice_1268: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_107, 0, 0, 9223372036854775807);  getitem_107 = None
    slice_1269: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_1268, 1, 0, 9223372036854775807);  slice_1268 = None
    unsqueeze_342: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_1269, 2);  slice_1269 = None
    slice_1270: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_342, 3, 0, 9223372036854775807);  unsqueeze_342 = None
    unsqueeze_343: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1270, 4);  slice_1270 = None
    expand_209: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_343, [1, 128, 1, 32, 2])
    clone_210: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_209, memory_format = torch.contiguous_format);  expand_209 = None
    view_740: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_210, [1, 128, 1, 64]);  clone_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_262: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1252, view_740)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_1274: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1252, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_1278: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1252, 3, 1, 9223372036854775807, 2);  slice_1252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_52: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_1278);  slice_1278 = None
    unsqueeze_344: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_52, 4);  neg_52 = None
    unsqueeze_345: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1274, 4);  slice_1274 = None
    cat_104: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_344, unsqueeze_345], 4);  unsqueeze_344 = unsqueeze_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_741: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_104, [1, 128, 16, 64]);  cat_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_263: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_741, view_739);  view_741 = None
    add_210: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_262, mul_263);  mul_262 = mul_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_264: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1260, view_740);  view_740 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_1288: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1260, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_1292: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1260, 3, 1, 9223372036854775807, 2);  slice_1260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_53: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_1292);  slice_1292 = None
    unsqueeze_350: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_53, 4);  neg_53 = None
    unsqueeze_351: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1288, 4);  slice_1288 = None
    cat_105: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_350, unsqueeze_351], 4);  unsqueeze_350 = unsqueeze_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_744: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_105, [1, 128, 16, 64]);  cat_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_265: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_744, view_739);  view_744 = view_739 = None
    add_211: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_264, mul_265);  mul_264 = mul_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_106: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_210, slice_1256], 3);  add_210 = slice_1256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_107: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_211, slice_1264], 3);  add_211 = slice_1264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_290: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_106, [0, 2, 1, 3]);  cat_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_291: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_107, [0, 2, 1, 3]);  cat_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_1293: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_365, 0, 0, 9223372036854775807);  primals_365 = None
    slice_1294: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_1293, 1, 0, 9223372036854775807);  slice_1293 = None
    slice_1295: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_1294, 2, 0, 128);  slice_1294 = None
    slice_1296: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_1295, 3, 0, 128);  slice_1295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_292: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_290, [0, 1, 3, 2]);  permute_290 = None
    expand_212: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_291, [1, 16, 128, 256]);  permute_291 = None
    view_745: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_212, [16, 128, 256]);  expand_212 = None
    expand_213: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_292, [1, 16, 256, 128]);  permute_292 = None
    view_746: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_213, [16, 256, 128]);  expand_213 = None
    bmm_52: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_745, view_746)
    view_747: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_52, [1, 16, 128, 128]);  bmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_26: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_1296, view_747, full_default);  view_747 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_52: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_26, primals_366);  where_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_26: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_52, [-1], True)
    sub_53: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_52, amax_26);  div_52 = amax_26 = None
    exp_26: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_53);  sub_53 = None
    sum_27: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_26, [-1], True)
    div_53: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_26, sum_27);  exp_26 = sum_27 = None
    alias_52: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_53)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    clone_213: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_53);  div_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_214: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_213, [1, 16, 128, 128]);  clone_213 = None
    view_748: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_214, [16, 128, 128]);  expand_214 = None
    expand_215: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_289, [1, 16, 128, 256]);  permute_289 = None
    view_749: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_215, [16, 128, 256]);  expand_215 = None
    bmm_53: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_748, view_749)
    view_750: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_53, [1, 16, 128, 256]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_293: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_750, [0, 2, 1, 3]);  view_750 = None
    clone_214: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_751: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_214, [1, 128, 4096]);  clone_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_294: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_267, [1, 0]);  primals_267 = None
    view_752: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_751, [128, 4096]);  view_751 = None
    mm_107: "f32[128, 4096]" = torch.ops.aten.mm.default(view_752, permute_294)
    view_753: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_107, [1, 128, 4096]);  mm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_295: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_268, [1, 0]);  primals_268 = None
    addmm_52: "f32[128, 16384]" = torch.ops.aten.addmm.default(primals_269, view_730, permute_295);  primals_269 = None
    view_755: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_52, [1, 128, 16384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_266: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_755, 0.5)
    pow_27: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_755, 3.0)
    mul_267: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_27, 0.044715);  pow_27 = None
    add_212: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_755, mul_267);  view_755 = mul_267 = None
    mul_268: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_212, 0.7978845608028654);  add_212 = None
    tanh_26: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_268);  mul_268 = None
    add_213: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_26, 1.0)
    mul_269: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_266, add_213);  mul_266 = add_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_756: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_269, [128, 16384]);  mul_269 = None
    permute_296: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_270, [1, 0]);  primals_270 = None
    addmm_53: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_271, view_756, permute_296);  primals_271 = None
    view_757: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_53, [1, 128, 4096]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_214: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_753, view_757);  view_753 = view_757 = None
    add_215: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_214, add_207);  add_214 = add_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_27 = torch.ops.aten.var_mean.correction(add_215, [2], correction = 0, keepdim = True)
    getitem_108: "f32[1, 128, 1]" = var_mean_27[0]
    getitem_109: "f32[1, 128, 1]" = var_mean_27[1];  var_mean_27 = None
    add_216: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
    rsqrt_27: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_216);  add_216 = None
    sub_54: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_215, getitem_109);  getitem_109 = None
    mul_270: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_27);  sub_54 = None
    mul_271: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_270, primals_272)
    add_217: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_271, primals_273);  mul_271 = primals_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_297: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_274, [1, 0]);  primals_274 = None
    view_758: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_217, [128, 4096]);  add_217 = None
    mm_108: "f32[128, 4096]" = torch.ops.aten.mm.default(view_758, permute_297)
    view_759: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_108, [1, 128, 4096]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_298: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_275, [1, 0]);  primals_275 = None
    mm_109: "f32[128, 4096]" = torch.ops.aten.mm.default(view_758, permute_298)
    view_761: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_109, [1, 128, 4096]);  mm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_299: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_276, [1, 0]);  primals_276 = None
    mm_110: "f32[128, 4096]" = torch.ops.aten.mm.default(view_758, permute_299)
    view_763: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_110, [1, 128, 4096]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_764: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_759, [1, 128, 16, 256]);  view_759 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_765: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_761, [1, 128, 16, 256]);  view_761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_766: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_763, [1, 128, 16, 256]);  view_763 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_300: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_766, [0, 2, 1, 3]);  view_766 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_54: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(primals_367, [1, 1, 1]);  primals_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_27: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_54, 1, repeat_1);  repeat_54 = repeat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_27 = torch.ops.aten.split_with_sizes.default(gather_27, [32, 32], 2);  gather_27 = None
    getitem_110: "f32[1, 128, 32]" = split_with_sizes_27[0]
    getitem_111: "f32[1, 128, 32]" = split_with_sizes_27[1];  split_with_sizes_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_1300: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_765, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_1304: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_765, 3, 64, 9223372036854775807);  view_765 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_1308: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_764, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_1312: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_764, 3, 64, 9223372036854775807);  view_764 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    slice_1313: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_110, 0, 0, 9223372036854775807);  getitem_110 = None
    slice_1314: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_1313, 1, 0, 9223372036854775807);  slice_1313 = None
    unsqueeze_353: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_1314, 2);  slice_1314 = None
    slice_1315: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_353, 3, 0, 9223372036854775807);  unsqueeze_353 = None
    unsqueeze_354: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1315, 4);  slice_1315 = None
    expand_216: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_354, [1, 128, 1, 32, 2])
    clone_217: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_216, memory_format = torch.contiguous_format);  expand_216 = None
    view_767: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_217, [1, 128, 1, 64]);  clone_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    slice_1316: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(getitem_111, 0, 0, 9223372036854775807);  getitem_111 = None
    slice_1317: "f32[1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_1316, 1, 0, 9223372036854775807);  slice_1316 = None
    unsqueeze_355: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_1317, 2);  slice_1317 = None
    slice_1318: "f32[1, 128, 1, 32]" = torch.ops.aten.slice.Tensor(unsqueeze_355, 3, 0, 9223372036854775807);  unsqueeze_355 = None
    unsqueeze_356: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1318, 4);  slice_1318 = None
    expand_217: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_356, [1, 128, 1, 32, 2])
    clone_218: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_217, memory_format = torch.contiguous_format);  expand_217 = None
    view_768: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_218, [1, 128, 1, 64]);  clone_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_272: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1300, view_768)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_1322: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1300, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_1326: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1300, 3, 1, 9223372036854775807, 2);  slice_1300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_54: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_1326);  slice_1326 = None
    unsqueeze_357: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_54, 4);  neg_54 = None
    unsqueeze_358: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1322, 4);  slice_1322 = None
    cat_108: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_357, unsqueeze_358], 4);  unsqueeze_357 = unsqueeze_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_769: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_108, [1, 128, 16, 64]);  cat_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_273: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_769, view_767);  view_769 = None
    add_218: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_272, mul_273);  mul_272 = mul_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_274: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1308, view_768);  view_768 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_1336: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1308, 3, 0, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_1340: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1308, 3, 1, 9223372036854775807, 2);  slice_1308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_55: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_1340);  slice_1340 = None
    unsqueeze_363: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_55, 4);  neg_55 = None
    unsqueeze_364: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1336, 4);  slice_1336 = None
    cat_109: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_363, unsqueeze_364], 4);  unsqueeze_363 = unsqueeze_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_772: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_109, [1, 128, 16, 64]);  cat_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_275: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_772, view_767);  view_772 = view_767 = None
    add_219: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_274, mul_275);  mul_274 = mul_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_110: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_218, slice_1304], 3);  add_218 = slice_1304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_111: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_219, slice_1312], 3);  add_219 = slice_1312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_301: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_110, [0, 2, 1, 3]);  cat_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_302: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_111, [0, 2, 1, 3]);  cat_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_1341: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_368, 0, 0, 9223372036854775807);  primals_368 = None
    slice_1342: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_1341, 1, 0, 9223372036854775807);  slice_1341 = None
    slice_1343: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_1342, 2, 0, 128);  slice_1342 = None
    slice_1344: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_1343, 3, 0, 128);  slice_1343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_303: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_301, [0, 1, 3, 2]);  permute_301 = None
    expand_220: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_302, [1, 16, 128, 256]);  permute_302 = None
    view_773: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_220, [16, 128, 256]);  expand_220 = None
    expand_221: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_303, [1, 16, 256, 128]);  permute_303 = None
    view_774: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_221, [16, 256, 128]);  expand_221 = None
    bmm_54: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_773, view_774)
    view_775: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_54, [1, 16, 128, 128]);  bmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_27: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_1344, view_775, full_default);  view_775 = full_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_54: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_27, primals_369);  where_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_27: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_54, [-1], True)
    sub_55: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_54, amax_27);  div_54 = amax_27 = None
    exp_27: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
    sum_28: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_27, [-1], True)
    div_55: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_27, sum_28);  exp_27 = sum_28 = None
    alias_54: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_55)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:177, code: attn_weights = self.attn_dropout(attn_weights)
    clone_221: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_55);  div_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_222: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_221, [1, 16, 128, 128]);  clone_221 = None
    view_776: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_222, [16, 128, 128]);  expand_222 = None
    expand_223: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_300, [1, 16, 128, 256]);  permute_300 = None
    view_777: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_223, [16, 128, 256]);  expand_223 = None
    bmm_55: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_776, view_777)
    view_778: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_55, [1, 16, 128, 256]);  bmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_304: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_778, [0, 2, 1, 3]);  view_778 = None
    clone_222: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_304, memory_format = torch.contiguous_format);  permute_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_779: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_222, [1, 128, 4096]);  clone_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_305: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_277, [1, 0]);  primals_277 = None
    view_780: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_779, [128, 4096]);  view_779 = None
    mm_111: "f32[128, 4096]" = torch.ops.aten.mm.default(view_780, permute_305)
    view_781: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_111, [1, 128, 4096]);  mm_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_306: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_278, [1, 0]);  primals_278 = None
    addmm_54: "f32[128, 16384]" = torch.ops.aten.addmm.default(primals_279, view_758, permute_306);  primals_279 = None
    view_783: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_54, [1, 128, 16384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_276: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_783, 0.5)
    pow_28: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_783, 3.0)
    mul_277: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_28, 0.044715);  pow_28 = None
    add_220: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_783, mul_277);  view_783 = mul_277 = None
    mul_278: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_220, 0.7978845608028654);  add_220 = None
    tanh_27: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_278);  mul_278 = None
    add_221: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_27, 1.0)
    mul_279: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_276, add_221);  mul_276 = add_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_784: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_279, [128, 16384]);  mul_279 = None
    permute_307: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_280, [1, 0]);  primals_280 = None
    addmm_55: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_281, view_784, permute_307);  primals_281 = None
    view_785: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_55, [1, 128, 4096]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_222: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_781, view_785);  view_781 = view_785 = None
    add_223: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_222, add_215);  add_222 = add_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:713, code: hidden_states = self.ln_f(hidden_states)
    var_mean_28 = torch.ops.aten.var_mean.correction(add_223, [2], correction = 0, keepdim = True)
    getitem_112: "f32[1, 128, 1]" = var_mean_28[0]
    getitem_113: "f32[1, 128, 1]" = var_mean_28[1];  var_mean_28 = None
    add_224: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
    rsqrt_28: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_224);  add_224 = None
    sub_56: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_223, getitem_113);  add_223 = getitem_113 = None
    mul_280: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_28);  sub_56 = None
    mul_281: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_280, primals_282)
    add_225: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_281, primals_283);  mul_281 = primals_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:715, code: hidden_states = hidden_states.view(output_shape)
    view_786: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_225, [-1, 128, 4096]);  add_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:1122, code: logits = self.qa_outputs(sequence_output)
    view_787: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_786, [128, 4096]);  view_786 = None
    permute_308: "f32[4096, 2]" = torch.ops.aten.permute.default(primals_284, [1, 0]);  primals_284 = None
    addmm_56: "f32[128, 2]" = torch.ops.aten.addmm.default(primals_285, view_787, permute_308);  primals_285 = None
    view_788: "f32[1, 128, 2]" = torch.ops.aten.reshape.default(addmm_56, [1, 128, 2]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:1123, code: start_logits, end_logits = logits.split(1, dim=-1)
    split_with_sizes_28 = torch.ops.aten.split_with_sizes.default(view_788, [1, 1], 2);  view_788 = None
    getitem_114: "f32[1, 128, 1]" = split_with_sizes_28[0]
    getitem_115: "f32[1, 128, 1]" = split_with_sizes_28[1];  split_with_sizes_28 = None
    
    # No stacktrace found for following nodes
    squeeze: "f32[1, 128]" = torch.ops.aten.squeeze.dim(getitem_114, -1);  getitem_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:1124, code: start_logits = start_logits.squeeze(-1).contiguous()
    clone_225: "f32[1, 128]" = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
    
    # No stacktrace found for following nodes
    squeeze_1: "f32[1, 128]" = torch.ops.aten.squeeze.dim(getitem_115, -1);  getitem_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:1125, code: end_logits = end_logits.squeeze(-1).contiguous()
    clone_226: "f32[1, 128]" = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:1136, code: start_positions = start_positions.clamp(0, ignored_index)
    clamp_min: "i64[1]" = torch.ops.aten.clamp_min.default(primals_371, 0);  primals_371 = None
    clamp_max: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min, 128);  clamp_min = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:1137, code: end_positions = end_positions.clamp(0, ignored_index)
    clamp_min_1: "i64[1]" = torch.ops.aten.clamp_min.default(primals_372, 0);  primals_372 = None
    clamp_max_1: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min_1, 128);  clamp_min_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:1140, code: start_loss = loss_fct(start_logits, start_positions)
    amax_28: "f32[1, 1]" = torch.ops.aten.amax.default(clone_225, [1], True)
    sub_57: "f32[1, 128]" = torch.ops.aten.sub.Tensor(clone_225, amax_28);  amax_28 = None
    exp_28: "f32[1, 128]" = torch.ops.aten.exp.default(sub_57)
    sum_29: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp_28, [1], True);  exp_28 = None
    log: "f32[1, 1]" = torch.ops.aten.log.default(sum_29);  sum_29 = None
    sub_58: "f32[1, 128]" = torch.ops.aten.sub.Tensor(sub_57, log);  sub_57 = log = None
    ne: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max, 128)
    full_default_28: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_28: "i64[1]" = torch.ops.aten.where.self(ne, clamp_max, full_default_28)
    unsqueeze_365: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where_28, 1);  where_28 = None
    gather_28: "f32[1, 1]" = torch.ops.aten.gather.default(sub_58, 1, unsqueeze_365);  unsqueeze_365 = None
    squeeze_2: "f32[1]" = torch.ops.aten.squeeze.dim(gather_28, 1);  gather_28 = None
    neg_56: "f32[1]" = torch.ops.aten.neg.default(squeeze_2);  squeeze_2 = None
    full_default_29: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_29: "f32[1]" = torch.ops.aten.where.self(ne, neg_56, full_default_29);  neg_56 = None
    sum_30: "i64[]" = torch.ops.aten.sum.default(ne)
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_30, torch.float32);  sum_30 = None
    sum_31: "f32[]" = torch.ops.aten.sum.default(where_29);  where_29 = None
    div_56: "f32[]" = torch.ops.aten.div.Tensor(sum_31, convert_element_type);  sum_31 = convert_element_type = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:1141, code: end_loss = loss_fct(end_logits, end_positions)
    amax_29: "f32[1, 1]" = torch.ops.aten.amax.default(clone_226, [1], True)
    sub_59: "f32[1, 128]" = torch.ops.aten.sub.Tensor(clone_226, amax_29);  amax_29 = None
    exp_29: "f32[1, 128]" = torch.ops.aten.exp.default(sub_59)
    sum_32: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp_29, [1], True);  exp_29 = None
    log_1: "f32[1, 1]" = torch.ops.aten.log.default(sum_32);  sum_32 = None
    sub_60: "f32[1, 128]" = torch.ops.aten.sub.Tensor(sub_59, log_1);  sub_59 = log_1 = None
    ne_3: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_1, 128)
    where_30: "i64[1]" = torch.ops.aten.where.self(ne_3, clamp_max_1, full_default_28)
    unsqueeze_366: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where_30, 1);  where_30 = None
    gather_29: "f32[1, 1]" = torch.ops.aten.gather.default(sub_60, 1, unsqueeze_366);  unsqueeze_366 = None
    squeeze_3: "f32[1]" = torch.ops.aten.squeeze.dim(gather_29, 1);  gather_29 = None
    neg_57: "f32[1]" = torch.ops.aten.neg.default(squeeze_3);  squeeze_3 = None
    where_31: "f32[1]" = torch.ops.aten.where.self(ne_3, neg_57, full_default_29);  neg_57 = full_default_29 = None
    sum_33: "i64[]" = torch.ops.aten.sum.default(ne_3)
    convert_element_type_1: "f32[]" = torch.ops.prims.convert_element_type.default(sum_33, torch.float32);  sum_33 = None
    sum_34: "f32[]" = torch.ops.aten.sum.default(where_31);  where_31 = None
    div_57: "f32[]" = torch.ops.aten.div.Tensor(sum_34, convert_element_type_1);  sum_34 = convert_element_type_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:1142, code: total_loss = (start_loss + end_loss) / 2
    add_226: "f32[]" = torch.ops.aten.add.Tensor(div_56, div_57);  div_56 = div_57 = None
    div_58: "f32[]" = torch.ops.aten.div.Tensor(add_226, 2);  add_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:1141, code: end_loss = loss_fct(end_logits, end_positions)
    unsqueeze_367: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(clamp_max_1, 1);  clamp_max_1 = None
    ne_6: "b8[1, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_367, 128)
    where_32: "i64[1, 1]" = torch.ops.aten.where.self(ne_6, unsqueeze_367, full_default_28);  unsqueeze_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:1140, code: start_loss = loss_fct(start_logits, start_positions)
    unsqueeze_368: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(clamp_max, 1);  clamp_max = None
    ne_8: "b8[1, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_368, 128)
    where_34: "i64[1, 1]" = torch.ops.aten.where.self(ne_8, unsqueeze_368, full_default_28);  unsqueeze_368 = full_default_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:1122, code: logits = self.qa_outputs(sequence_output)
    permute_309: "f32[2, 4096]" = torch.ops.aten.permute.default(permute_308, [1, 0]);  permute_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:713, code: hidden_states = self.ln_f(hidden_states)
    div_62: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 4096);  rsqrt_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    permute_313: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_307, [1, 0]);  permute_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_317: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_306, [1, 0]);  permute_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_323: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_305, [1, 0]);  permute_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    permute_326: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_776, [0, 2, 1]);  view_776 = None
    permute_327: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_777, [0, 2, 1]);  view_777 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_61: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_54);  alias_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_328: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_773, [0, 2, 1]);  view_773 = None
    permute_329: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_774, [0, 2, 1]);  view_774 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_336: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_299, [1, 0]);  permute_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_340: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_298, [1, 0]);  permute_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_344: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_297, [1, 0]);  permute_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    div_64: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 4096);  rsqrt_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    permute_346: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_296, [1, 0]);  permute_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_350: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_295, [1, 0]);  permute_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_356: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_294, [1, 0]);  permute_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    permute_359: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_748, [0, 2, 1]);  view_748 = None
    permute_360: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_749, [0, 2, 1]);  view_749 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_63: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_52);  alias_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_361: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_745, [0, 2, 1]);  view_745 = None
    permute_362: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_746, [0, 2, 1]);  view_746 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_369: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_288, [1, 0]);  permute_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_373: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_287, [1, 0]);  permute_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_377: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_286, [1, 0]);  permute_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    div_66: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 4096);  rsqrt_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    permute_379: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_285, [1, 0]);  permute_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_383: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_284, [1, 0]);  permute_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_389: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_283, [1, 0]);  permute_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    permute_392: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_720, [0, 2, 1]);  view_720 = None
    permute_393: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_721, [0, 2, 1]);  view_721 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_65: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_50);  alias_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_394: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_717, [0, 2, 1]);  view_717 = None
    permute_395: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_718, [0, 2, 1]);  view_718 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_402: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_277, [1, 0]);  permute_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_406: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_276, [1, 0]);  permute_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_410: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_275, [1, 0]);  permute_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    div_68: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 4096);  rsqrt_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    permute_412: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_274, [1, 0]);  permute_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_416: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_273, [1, 0]);  permute_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_422: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_272, [1, 0]);  permute_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    permute_425: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_692, [0, 2, 1]);  view_692 = None
    permute_426: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_693, [0, 2, 1]);  view_693 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_67: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_48);  alias_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_427: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_689, [0, 2, 1]);  view_689 = None
    permute_428: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_690, [0, 2, 1]);  view_690 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_435: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_266, [1, 0]);  permute_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_439: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_265, [1, 0]);  permute_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_443: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    div_70: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 4096);  rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    permute_445: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_449: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_455: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_261, [1, 0]);  permute_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    permute_458: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_664, [0, 2, 1]);  view_664 = None
    permute_459: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_665, [0, 2, 1]);  view_665 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_69: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_46);  alias_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_460: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_661, [0, 2, 1]);  view_661 = None
    permute_461: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_662, [0, 2, 1]);  view_662 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_468: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_255, [1, 0]);  permute_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_472: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_254, [1, 0]);  permute_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_476: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_253, [1, 0]);  permute_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    div_72: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 4096);  rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    permute_478: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_482: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_488: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_250, [1, 0]);  permute_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    permute_491: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_636, [0, 2, 1]);  view_636 = None
    permute_492: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_637, [0, 2, 1]);  view_637 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_71: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_44);  alias_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_493: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_633, [0, 2, 1]);  view_633 = None
    permute_494: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_634, [0, 2, 1]);  view_634 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_501: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_505: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_509: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    div_74: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 4096);  rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    permute_511: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_241, [1, 0]);  permute_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_515: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_521: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    permute_524: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_608, [0, 2, 1]);  view_608 = None
    permute_525: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_609, [0, 2, 1]);  view_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_73: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_42);  alias_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_526: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_605, [0, 2, 1]);  view_605 = None
    permute_527: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_606, [0, 2, 1]);  view_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_534: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_538: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_542: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_231, [1, 0]);  permute_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    div_76: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 4096);  rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    permute_544: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_230, [1, 0]);  permute_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_548: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_554: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    permute_557: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_580, [0, 2, 1]);  view_580 = None
    permute_558: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_581, [0, 2, 1]);  view_581 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_75: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_40);  alias_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_559: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_577, [0, 2, 1]);  view_577 = None
    permute_560: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_578, [0, 2, 1]);  view_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_567: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_571: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_221, [1, 0]);  permute_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_575: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    div_78: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 4096);  rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    permute_577: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_219, [1, 0]);  permute_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_581: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_587: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_217, [1, 0]);  permute_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    permute_590: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_552, [0, 2, 1]);  view_552 = None
    permute_591: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_553, [0, 2, 1]);  view_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_77: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_38);  alias_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_592: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_549, [0, 2, 1]);  view_549 = None
    permute_593: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_550, [0, 2, 1]);  view_550 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_600: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_604: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_608: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_209, [1, 0]);  permute_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    div_80: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 4096);  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    permute_610: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_208, [1, 0]);  permute_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_614: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_620: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    permute_623: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_524, [0, 2, 1]);  view_524 = None
    permute_624: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_525, [0, 2, 1]);  view_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_79: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_36);  alias_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_625: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_521, [0, 2, 1]);  view_521 = None
    permute_626: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_522, [0, 2, 1]);  view_522 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_633: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_637: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_199, [1, 0]);  permute_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_641: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_198, [1, 0]);  permute_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    div_82: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 4096);  rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    permute_643: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_647: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_653: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_195, [1, 0]);  permute_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    permute_656: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_496, [0, 2, 1]);  view_496 = None
    permute_657: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_497, [0, 2, 1]);  view_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_81: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_34);  alias_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_658: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_493, [0, 2, 1]);  view_493 = None
    permute_659: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_494, [0, 2, 1]);  view_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_666: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_189, [1, 0]);  permute_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_670: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_188, [1, 0]);  permute_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_674: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    div_84: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 4096);  rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    permute_676: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_680: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_686: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    permute_689: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_468, [0, 2, 1]);  view_468 = None
    permute_690: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_469, [0, 2, 1]);  view_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_83: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_32);  alias_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_691: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_465, [0, 2, 1]);  view_465 = None
    permute_692: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_466, [0, 2, 1]);  view_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_699: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_703: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_707: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    div_86: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 4096);  rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    permute_709: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_175, [1, 0]);  permute_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_713: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_719: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    permute_722: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_440, [0, 2, 1]);  view_440 = None
    permute_723: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_441, [0, 2, 1]);  view_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_85: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_30);  alias_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_724: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_437, [0, 2, 1]);  view_437 = None
    permute_725: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_438, [0, 2, 1]);  view_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_732: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_736: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_166, [1, 0]);  permute_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_740: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    div_88: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 4096);  rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    permute_742: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_746: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_752: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_162, [1, 0]);  permute_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    permute_755: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_412, [0, 2, 1]);  view_412 = None
    permute_756: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_413, [0, 2, 1]);  view_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_87: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_28);  alias_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_757: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_409, [0, 2, 1]);  view_409 = None
    permute_758: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_410, [0, 2, 1]);  view_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_765: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_769: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_155, [1, 0]);  permute_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_773: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    div_90: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 4096);  rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    permute_775: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_779: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_785: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    permute_788: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_384, [0, 2, 1]);  view_384 = None
    permute_789: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_385, [0, 2, 1]);  view_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_89: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_26);  alias_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_790: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_381, [0, 2, 1]);  view_381 = None
    permute_791: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_382, [0, 2, 1]);  view_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_798: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_802: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_806: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    div_92: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 4096);  rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    permute_808: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_812: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_818: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    permute_821: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_356, [0, 2, 1]);  view_356 = None
    permute_822: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_357, [0, 2, 1]);  view_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_91: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_24);  alias_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_823: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_353, [0, 2, 1]);  view_353 = None
    permute_824: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_354, [0, 2, 1]);  view_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_831: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_835: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_839: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    div_94: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 4096);  rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    permute_841: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_845: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_851: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    permute_854: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_328, [0, 2, 1]);  view_328 = None
    permute_855: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_329, [0, 2, 1]);  view_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_93: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_22);  alias_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_856: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_325, [0, 2, 1]);  view_325 = None
    permute_857: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_326, [0, 2, 1]);  view_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_864: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_868: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_872: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    div_96: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 4096);  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    permute_874: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_878: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_884: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    permute_887: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_300, [0, 2, 1]);  view_300 = None
    permute_888: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_301, [0, 2, 1]);  view_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_95: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_20);  alias_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_889: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_297, [0, 2, 1]);  view_297 = None
    permute_890: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_298, [0, 2, 1]);  view_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_897: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_901: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_905: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    div_98: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 4096);  rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    permute_907: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_911: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_917: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    permute_920: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_272, [0, 2, 1]);  view_272 = None
    permute_921: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_273, [0, 2, 1]);  view_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_97: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_18);  alias_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_922: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_269, [0, 2, 1]);  view_269 = None
    permute_923: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_270, [0, 2, 1]);  view_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_930: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_934: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_938: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    div_100: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 4096);  rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    permute_940: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_944: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_950: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    permute_953: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_244, [0, 2, 1]);  view_244 = None
    permute_954: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_245, [0, 2, 1]);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_99: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_16);  alias_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_955: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_241, [0, 2, 1]);  view_241 = None
    permute_956: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_242, [0, 2, 1]);  view_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_963: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_967: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_971: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    div_102: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 4096);  rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    permute_973: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_977: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_983: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    permute_986: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_216, [0, 2, 1]);  view_216 = None
    permute_987: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_217, [0, 2, 1]);  view_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_101: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_988: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_213, [0, 2, 1]);  view_213 = None
    permute_989: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_214, [0, 2, 1]);  view_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_996: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_1000: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_1004: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    div_104: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 4096);  rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    permute_1006: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_1010: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_1016: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    permute_1019: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
    permute_1020: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_189, [0, 2, 1]);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_103: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_1021: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_185, [0, 2, 1]);  view_185 = None
    permute_1022: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_186, [0, 2, 1]);  view_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_1029: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_1033: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_1037: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    div_106: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 4096);  rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    permute_1039: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_1043: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_1049: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    permute_1052: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_160, [0, 2, 1]);  view_160 = None
    permute_1053: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_161, [0, 2, 1]);  view_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_105: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_1054: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_157, [0, 2, 1]);  view_157 = None
    permute_1055: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_158, [0, 2, 1]);  view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_1062: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_1066: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_1070: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    div_108: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 4096);  rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    permute_1072: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_1076: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_1082: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    permute_1085: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_132, [0, 2, 1]);  view_132 = None
    permute_1086: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_133, [0, 2, 1]);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_107: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_1087: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_129, [0, 2, 1]);  view_129 = None
    permute_1088: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_130, [0, 2, 1]);  view_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_1095: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_1099: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_1103: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    div_110: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 4096);  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    permute_1105: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_1109: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_1115: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    permute_1118: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_104, [0, 2, 1]);  view_104 = None
    permute_1119: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_105, [0, 2, 1]);  view_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_109: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_1120: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_101, [0, 2, 1]);  view_101 = None
    permute_1121: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_102, [0, 2, 1]);  view_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_1128: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_1132: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_1136: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    div_112: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 4096);  rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    permute_1138: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_1142: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_1148: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    permute_1151: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_76, [0, 2, 1]);  view_76 = None
    permute_1152: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_77, [0, 2, 1]);  view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_111: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_1153: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_73, [0, 2, 1]);  view_73 = None
    permute_1154: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_74, [0, 2, 1]);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_1161: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_1165: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_1169: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    div_114: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 4096);  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    permute_1171: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_1175: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_1181: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    permute_1184: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_48, [0, 2, 1]);  view_48 = None
    permute_1185: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_49, [0, 2, 1]);  view_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_113: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_1186: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_45, [0, 2, 1]);  view_45 = None
    permute_1187: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_46, [0, 2, 1]);  view_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_1194: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_1198: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_1202: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    div_116: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 4096);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    permute_1204: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    permute_1208: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    permute_1214: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    permute_1217: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_20, [0, 2, 1]);  view_20 = None
    permute_1218: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_21, [0, 2, 1]);  view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_115: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias);  alias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_1219: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_17, [0, 2, 1]);  view_17 = None
    permute_1220: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_18, [0, 2, 1]);  view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_1227: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_1231: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_1235: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    return [div_58, clone_225, clone_226, primals_2, primals_12, primals_22, primals_32, primals_42, primals_52, primals_62, primals_72, primals_82, primals_92, primals_102, primals_112, primals_122, primals_132, primals_142, primals_152, primals_162, primals_172, primals_182, primals_192, primals_202, primals_212, primals_222, primals_232, primals_242, primals_252, primals_262, primals_272, primals_282, primals_288, primals_291, primals_294, primals_297, primals_300, primals_303, primals_306, primals_309, primals_312, primals_315, primals_318, primals_321, primals_324, primals_327, primals_330, primals_333, primals_336, primals_339, primals_342, primals_345, primals_348, primals_351, primals_354, primals_357, primals_360, primals_363, primals_366, primals_369, view, embedding, getitem_1, rsqrt, view_2, unsqueeze_3, unsqueeze_5, slice_48, view_24, addmm, tanh, view_28, mul_10, view_30, unsqueeze_16, unsqueeze_18, slice_96, view_52, addmm_2, tanh_1, view_56, mul_20, view_58, unsqueeze_29, unsqueeze_31, slice_144, view_80, addmm_4, tanh_2, view_84, mul_30, view_86, unsqueeze_42, unsqueeze_44, slice_192, view_108, addmm_6, tanh_3, view_112, mul_40, view_114, unsqueeze_55, unsqueeze_57, slice_240, view_136, addmm_8, tanh_4, view_140, mul_50, view_142, unsqueeze_68, unsqueeze_70, slice_288, view_164, addmm_10, tanh_5, view_168, mul_60, view_170, unsqueeze_81, unsqueeze_83, slice_336, view_192, addmm_12, tanh_6, view_196, mul_70, view_198, unsqueeze_94, unsqueeze_96, slice_384, view_220, addmm_14, tanh_7, view_224, mul_80, view_226, unsqueeze_107, unsqueeze_109, slice_432, view_248, addmm_16, tanh_8, view_252, mul_90, view_254, unsqueeze_120, unsqueeze_122, slice_480, view_276, addmm_18, tanh_9, view_280, mul_100, view_282, unsqueeze_133, unsqueeze_135, slice_528, view_304, addmm_20, tanh_10, view_308, mul_110, view_310, unsqueeze_146, unsqueeze_148, slice_576, view_332, addmm_22, tanh_11, view_336, mul_120, view_338, unsqueeze_159, unsqueeze_161, slice_624, view_360, addmm_24, tanh_12, view_364, mul_130, view_366, unsqueeze_172, unsqueeze_174, slice_672, view_388, addmm_26, tanh_13, view_392, mul_140, view_394, unsqueeze_185, unsqueeze_187, slice_720, view_416, addmm_28, tanh_14, view_420, mul_150, view_422, unsqueeze_198, unsqueeze_200, slice_768, view_444, addmm_30, tanh_15, view_448, mul_160, view_450, unsqueeze_211, unsqueeze_213, slice_816, view_472, addmm_32, tanh_16, view_476, mul_170, view_478, unsqueeze_224, unsqueeze_226, slice_864, view_500, addmm_34, tanh_17, view_504, mul_180, view_506, unsqueeze_237, unsqueeze_239, slice_912, view_528, addmm_36, tanh_18, view_532, mul_190, view_534, unsqueeze_250, unsqueeze_252, slice_960, view_556, addmm_38, tanh_19, view_560, mul_200, view_562, unsqueeze_263, unsqueeze_265, slice_1008, view_584, addmm_40, tanh_20, view_588, mul_210, view_590, unsqueeze_276, unsqueeze_278, slice_1056, view_612, addmm_42, tanh_21, view_616, mul_220, view_618, unsqueeze_289, unsqueeze_291, slice_1104, view_640, addmm_44, tanh_22, view_644, mul_230, view_646, unsqueeze_302, unsqueeze_304, slice_1152, view_668, addmm_46, tanh_23, view_672, mul_240, view_674, unsqueeze_315, unsqueeze_317, slice_1200, view_696, addmm_48, tanh_24, view_700, mul_250, view_702, unsqueeze_328, unsqueeze_330, slice_1248, view_724, addmm_50, tanh_25, view_728, mul_260, view_730, unsqueeze_341, unsqueeze_343, slice_1296, view_752, addmm_52, tanh_26, view_756, mul_270, view_758, unsqueeze_354, unsqueeze_356, slice_1344, view_780, addmm_54, tanh_27, view_784, mul_280, view_787, sub_58, ne, sub_60, ne_3, ne_6, where_32, ne_8, where_34, permute_309, div_62, permute_313, permute_317, permute_323, permute_326, permute_327, alias_61, permute_328, permute_329, permute_336, permute_340, permute_344, div_64, permute_346, permute_350, permute_356, permute_359, permute_360, alias_63, permute_361, permute_362, permute_369, permute_373, permute_377, div_66, permute_379, permute_383, permute_389, permute_392, permute_393, alias_65, permute_394, permute_395, permute_402, permute_406, permute_410, div_68, permute_412, permute_416, permute_422, permute_425, permute_426, alias_67, permute_427, permute_428, permute_435, permute_439, permute_443, div_70, permute_445, permute_449, permute_455, permute_458, permute_459, alias_69, permute_460, permute_461, permute_468, permute_472, permute_476, div_72, permute_478, permute_482, permute_488, permute_491, permute_492, alias_71, permute_493, permute_494, permute_501, permute_505, permute_509, div_74, permute_511, permute_515, permute_521, permute_524, permute_525, alias_73, permute_526, permute_527, permute_534, permute_538, permute_542, div_76, permute_544, permute_548, permute_554, permute_557, permute_558, alias_75, permute_559, permute_560, permute_567, permute_571, permute_575, div_78, permute_577, permute_581, permute_587, permute_590, permute_591, alias_77, permute_592, permute_593, permute_600, permute_604, permute_608, div_80, permute_610, permute_614, permute_620, permute_623, permute_624, alias_79, permute_625, permute_626, permute_633, permute_637, permute_641, div_82, permute_643, permute_647, permute_653, permute_656, permute_657, alias_81, permute_658, permute_659, permute_666, permute_670, permute_674, div_84, permute_676, permute_680, permute_686, permute_689, permute_690, alias_83, permute_691, permute_692, permute_699, permute_703, permute_707, div_86, permute_709, permute_713, permute_719, permute_722, permute_723, alias_85, permute_724, permute_725, permute_732, permute_736, permute_740, div_88, permute_742, permute_746, permute_752, permute_755, permute_756, alias_87, permute_757, permute_758, permute_765, permute_769, permute_773, div_90, permute_775, permute_779, permute_785, permute_788, permute_789, alias_89, permute_790, permute_791, permute_798, permute_802, permute_806, div_92, permute_808, permute_812, permute_818, permute_821, permute_822, alias_91, permute_823, permute_824, permute_831, permute_835, permute_839, div_94, permute_841, permute_845, permute_851, permute_854, permute_855, alias_93, permute_856, permute_857, permute_864, permute_868, permute_872, div_96, permute_874, permute_878, permute_884, permute_887, permute_888, alias_95, permute_889, permute_890, permute_897, permute_901, permute_905, div_98, permute_907, permute_911, permute_917, permute_920, permute_921, alias_97, permute_922, permute_923, permute_930, permute_934, permute_938, div_100, permute_940, permute_944, permute_950, permute_953, permute_954, alias_99, permute_955, permute_956, permute_963, permute_967, permute_971, div_102, permute_973, permute_977, permute_983, permute_986, permute_987, alias_101, permute_988, permute_989, permute_996, permute_1000, permute_1004, div_104, permute_1006, permute_1010, permute_1016, permute_1019, permute_1020, alias_103, permute_1021, permute_1022, permute_1029, permute_1033, permute_1037, div_106, permute_1039, permute_1043, permute_1049, permute_1052, permute_1053, alias_105, permute_1054, permute_1055, permute_1062, permute_1066, permute_1070, div_108, permute_1072, permute_1076, permute_1082, permute_1085, permute_1086, alias_107, permute_1087, permute_1088, permute_1095, permute_1099, permute_1103, div_110, permute_1105, permute_1109, permute_1115, permute_1118, permute_1119, alias_109, permute_1120, permute_1121, permute_1128, permute_1132, permute_1136, div_112, permute_1138, permute_1142, permute_1148, permute_1151, permute_1152, alias_111, permute_1153, permute_1154, permute_1161, permute_1165, permute_1169, div_114, permute_1171, permute_1175, permute_1181, permute_1184, permute_1185, alias_113, permute_1186, permute_1187, permute_1194, permute_1198, permute_1202, div_116, permute_1204, permute_1208, permute_1214, permute_1217, permute_1218, alias_115, permute_1219, permute_1220, permute_1227, permute_1231, permute_1235]
    