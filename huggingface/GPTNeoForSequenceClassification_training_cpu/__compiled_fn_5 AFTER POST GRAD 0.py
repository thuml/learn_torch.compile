from __future__ import annotations



def forward(self, primals_1: "f32[50257, 2048]", primals_2: "f32[2048, 2048]", primals_3: "f32[2048]", primals_4: "f32[2048]", primals_5: "f32[2048, 2048]", primals_6: "f32[2048, 2048]", primals_7: "f32[2048, 2048]", primals_8: "f32[2048, 2048]", primals_9: "f32[2048]", primals_10: "f32[2048]", primals_11: "f32[2048]", primals_12: "f32[8192, 2048]", primals_13: "f32[8192]", primals_14: "f32[2048, 8192]", primals_15: "f32[2048]", primals_16: "f32[2048]", primals_17: "f32[2048]", primals_18: "f32[2048, 2048]", primals_19: "f32[2048, 2048]", primals_20: "f32[2048, 2048]", primals_21: "f32[2048, 2048]", primals_22: "f32[2048]", primals_23: "f32[2048]", primals_24: "f32[2048]", primals_25: "f32[8192, 2048]", primals_26: "f32[8192]", primals_27: "f32[2048, 8192]", primals_28: "f32[2048]", primals_29: "f32[2048]", primals_30: "f32[2048]", primals_31: "f32[2048, 2048]", primals_32: "f32[2048, 2048]", primals_33: "f32[2048, 2048]", primals_34: "f32[2048, 2048]", primals_35: "f32[2048]", primals_36: "f32[2048]", primals_37: "f32[2048]", primals_38: "f32[8192, 2048]", primals_39: "f32[8192]", primals_40: "f32[2048, 8192]", primals_41: "f32[2048]", primals_42: "f32[2048]", primals_43: "f32[2048]", primals_44: "f32[2048, 2048]", primals_45: "f32[2048, 2048]", primals_46: "f32[2048, 2048]", primals_47: "f32[2048, 2048]", primals_48: "f32[2048]", primals_49: "f32[2048]", primals_50: "f32[2048]", primals_51: "f32[8192, 2048]", primals_52: "f32[8192]", primals_53: "f32[2048, 8192]", primals_54: "f32[2048]", primals_55: "f32[2048]", primals_56: "f32[2048]", primals_57: "f32[2048, 2048]", primals_58: "f32[2048, 2048]", primals_59: "f32[2048, 2048]", primals_60: "f32[2048, 2048]", primals_61: "f32[2048]", primals_62: "f32[2048]", primals_63: "f32[2048]", primals_64: "f32[8192, 2048]", primals_65: "f32[8192]", primals_66: "f32[2048, 8192]", primals_67: "f32[2048]", primals_68: "f32[2048]", primals_69: "f32[2048]", primals_70: "f32[2048, 2048]", primals_71: "f32[2048, 2048]", primals_72: "f32[2048, 2048]", primals_73: "f32[2048, 2048]", primals_74: "f32[2048]", primals_75: "f32[2048]", primals_76: "f32[2048]", primals_77: "f32[8192, 2048]", primals_78: "f32[8192]", primals_79: "f32[2048, 8192]", primals_80: "f32[2048]", primals_81: "f32[2048]", primals_82: "f32[2048]", primals_83: "f32[2048, 2048]", primals_84: "f32[2048, 2048]", primals_85: "f32[2048, 2048]", primals_86: "f32[2048, 2048]", primals_87: "f32[2048]", primals_88: "f32[2048]", primals_89: "f32[2048]", primals_90: "f32[8192, 2048]", primals_91: "f32[8192]", primals_92: "f32[2048, 8192]", primals_93: "f32[2048]", primals_94: "f32[2048]", primals_95: "f32[2048]", primals_96: "f32[2048, 2048]", primals_97: "f32[2048, 2048]", primals_98: "f32[2048, 2048]", primals_99: "f32[2048, 2048]", primals_100: "f32[2048]", primals_101: "f32[2048]", primals_102: "f32[2048]", primals_103: "f32[8192, 2048]", primals_104: "f32[8192]", primals_105: "f32[2048, 8192]", primals_106: "f32[2048]", primals_107: "f32[2048]", primals_108: "f32[2048]", primals_109: "f32[2048, 2048]", primals_110: "f32[2048, 2048]", primals_111: "f32[2048, 2048]", primals_112: "f32[2048, 2048]", primals_113: "f32[2048]", primals_114: "f32[2048]", primals_115: "f32[2048]", primals_116: "f32[8192, 2048]", primals_117: "f32[8192]", primals_118: "f32[2048, 8192]", primals_119: "f32[2048]", primals_120: "f32[2048]", primals_121: "f32[2048]", primals_122: "f32[2048, 2048]", primals_123: "f32[2048, 2048]", primals_124: "f32[2048, 2048]", primals_125: "f32[2048, 2048]", primals_126: "f32[2048]", primals_127: "f32[2048]", primals_128: "f32[2048]", primals_129: "f32[8192, 2048]", primals_130: "f32[8192]", primals_131: "f32[2048, 8192]", primals_132: "f32[2048]", primals_133: "f32[2048]", primals_134: "f32[2048]", primals_135: "f32[2048, 2048]", primals_136: "f32[2048, 2048]", primals_137: "f32[2048, 2048]", primals_138: "f32[2048, 2048]", primals_139: "f32[2048]", primals_140: "f32[2048]", primals_141: "f32[2048]", primals_142: "f32[8192, 2048]", primals_143: "f32[8192]", primals_144: "f32[2048, 8192]", primals_145: "f32[2048]", primals_146: "f32[2048]", primals_147: "f32[2048]", primals_148: "f32[2048, 2048]", primals_149: "f32[2048, 2048]", primals_150: "f32[2048, 2048]", primals_151: "f32[2048, 2048]", primals_152: "f32[2048]", primals_153: "f32[2048]", primals_154: "f32[2048]", primals_155: "f32[8192, 2048]", primals_156: "f32[8192]", primals_157: "f32[2048, 8192]", primals_158: "f32[2048]", primals_159: "f32[2048]", primals_160: "f32[2048]", primals_161: "f32[2048, 2048]", primals_162: "f32[2048, 2048]", primals_163: "f32[2048, 2048]", primals_164: "f32[2048, 2048]", primals_165: "f32[2048]", primals_166: "f32[2048]", primals_167: "f32[2048]", primals_168: "f32[8192, 2048]", primals_169: "f32[8192]", primals_170: "f32[2048, 8192]", primals_171: "f32[2048]", primals_172: "f32[2048]", primals_173: "f32[2048]", primals_174: "f32[2048, 2048]", primals_175: "f32[2048, 2048]", primals_176: "f32[2048, 2048]", primals_177: "f32[2048, 2048]", primals_178: "f32[2048]", primals_179: "f32[2048]", primals_180: "f32[2048]", primals_181: "f32[8192, 2048]", primals_182: "f32[8192]", primals_183: "f32[2048, 8192]", primals_184: "f32[2048]", primals_185: "f32[2048]", primals_186: "f32[2048]", primals_187: "f32[2048, 2048]", primals_188: "f32[2048, 2048]", primals_189: "f32[2048, 2048]", primals_190: "f32[2048, 2048]", primals_191: "f32[2048]", primals_192: "f32[2048]", primals_193: "f32[2048]", primals_194: "f32[8192, 2048]", primals_195: "f32[8192]", primals_196: "f32[2048, 8192]", primals_197: "f32[2048]", primals_198: "f32[2048]", primals_199: "f32[2048]", primals_200: "f32[2048, 2048]", primals_201: "f32[2048, 2048]", primals_202: "f32[2048, 2048]", primals_203: "f32[2048, 2048]", primals_204: "f32[2048]", primals_205: "f32[2048]", primals_206: "f32[2048]", primals_207: "f32[8192, 2048]", primals_208: "f32[8192]", primals_209: "f32[2048, 8192]", primals_210: "f32[2048]", primals_211: "f32[2048]", primals_212: "f32[2048]", primals_213: "f32[2048, 2048]", primals_214: "f32[2048, 2048]", primals_215: "f32[2048, 2048]", primals_216: "f32[2048, 2048]", primals_217: "f32[2048]", primals_218: "f32[2048]", primals_219: "f32[2048]", primals_220: "f32[8192, 2048]", primals_221: "f32[8192]", primals_222: "f32[2048, 8192]", primals_223: "f32[2048]", primals_224: "f32[2048]", primals_225: "f32[2048]", primals_226: "f32[2048, 2048]", primals_227: "f32[2048, 2048]", primals_228: "f32[2048, 2048]", primals_229: "f32[2048, 2048]", primals_230: "f32[2048]", primals_231: "f32[2048]", primals_232: "f32[2048]", primals_233: "f32[8192, 2048]", primals_234: "f32[8192]", primals_235: "f32[2048, 8192]", primals_236: "f32[2048]", primals_237: "f32[2048]", primals_238: "f32[2048]", primals_239: "f32[2048, 2048]", primals_240: "f32[2048, 2048]", primals_241: "f32[2048, 2048]", primals_242: "f32[2048, 2048]", primals_243: "f32[2048]", primals_244: "f32[2048]", primals_245: "f32[2048]", primals_246: "f32[8192, 2048]", primals_247: "f32[8192]", primals_248: "f32[2048, 8192]", primals_249: "f32[2048]", primals_250: "f32[2048]", primals_251: "f32[2048]", primals_252: "f32[2048, 2048]", primals_253: "f32[2048, 2048]", primals_254: "f32[2048, 2048]", primals_255: "f32[2048, 2048]", primals_256: "f32[2048]", primals_257: "f32[2048]", primals_258: "f32[2048]", primals_259: "f32[8192, 2048]", primals_260: "f32[8192]", primals_261: "f32[2048, 8192]", primals_262: "f32[2048]", primals_263: "f32[2048]", primals_264: "f32[2048]", primals_265: "f32[2048, 2048]", primals_266: "f32[2048, 2048]", primals_267: "f32[2048, 2048]", primals_268: "f32[2048, 2048]", primals_269: "f32[2048]", primals_270: "f32[2048]", primals_271: "f32[2048]", primals_272: "f32[8192, 2048]", primals_273: "f32[8192]", primals_274: "f32[2048, 8192]", primals_275: "f32[2048]", primals_276: "f32[2048]", primals_277: "f32[2048]", primals_278: "f32[2048, 2048]", primals_279: "f32[2048, 2048]", primals_280: "f32[2048, 2048]", primals_281: "f32[2048, 2048]", primals_282: "f32[2048]", primals_283: "f32[2048]", primals_284: "f32[2048]", primals_285: "f32[8192, 2048]", primals_286: "f32[8192]", primals_287: "f32[2048, 8192]", primals_288: "f32[2048]", primals_289: "f32[2048]", primals_290: "f32[2048]", primals_291: "f32[2048, 2048]", primals_292: "f32[2048, 2048]", primals_293: "f32[2048, 2048]", primals_294: "f32[2048, 2048]", primals_295: "f32[2048]", primals_296: "f32[2048]", primals_297: "f32[2048]", primals_298: "f32[8192, 2048]", primals_299: "f32[8192]", primals_300: "f32[2048, 8192]", primals_301: "f32[2048]", primals_302: "f32[2048]", primals_303: "f32[2048]", primals_304: "f32[2048, 2048]", primals_305: "f32[2048, 2048]", primals_306: "f32[2048, 2048]", primals_307: "f32[2048, 2048]", primals_308: "f32[2048]", primals_309: "f32[2048]", primals_310: "f32[2048]", primals_311: "f32[8192, 2048]", primals_312: "f32[8192]", primals_313: "f32[2048, 8192]", primals_314: "f32[2048]", primals_315: "f32[2048]", primals_316: "f32[2048]", primals_317: "f32[2, 2048]", primals_318: "b8[1, 1, 2048, 2048]", primals_319: "b8[1, 1, 2048, 2048]", primals_320: "b8[1, 1, 2048, 2048]", primals_321: "b8[1, 1, 2048, 2048]", primals_322: "b8[1, 1, 2048, 2048]", primals_323: "b8[1, 1, 2048, 2048]", primals_324: "b8[1, 1, 2048, 2048]", primals_325: "b8[1, 1, 2048, 2048]", primals_326: "b8[1, 1, 2048, 2048]", primals_327: "b8[1, 1, 2048, 2048]", primals_328: "b8[1, 1, 2048, 2048]", primals_329: "b8[1, 1, 2048, 2048]", primals_330: "b8[1, 1, 2048, 2048]", primals_331: "b8[1, 1, 2048, 2048]", primals_332: "b8[1, 1, 2048, 2048]", primals_333: "b8[1, 1, 2048, 2048]", primals_334: "b8[1, 1, 2048, 2048]", primals_335: "b8[1, 1, 2048, 2048]", primals_336: "b8[1, 1, 2048, 2048]", primals_337: "b8[1, 1, 2048, 2048]", primals_338: "b8[1, 1, 2048, 2048]", primals_339: "b8[1, 1, 2048, 2048]", primals_340: "b8[1, 1, 2048, 2048]", primals_341: "b8[1, 1, 2048, 2048]", primals_342: "i64[1, 128]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:530, code: input_ids = input_ids.view(-1, input_shape[-1])
    view: "i64[1, 128]" = torch.ops.aten.reshape.default(primals_342, [-1, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:552, code: position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
    iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:553, code: position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
    unsqueeze: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
    view_1: "i64[1, 128]" = torch.ops.aten.reshape.default(unsqueeze, [-1, 128]);  unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:582, code: inputs_embeds = self.wte(input_ids)
    embedding: "f32[1, 128, 2048]" = torch.ops.aten.embedding.default(primals_1, view);  primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:583, code: position_embeds = self.wpe(position_ids)
    embedding_1: "f32[1, 128, 2048]" = torch.ops.aten.embedding.default(primals_2, view_1);  primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:584, code: hidden_states = inputs_embeds + position_embeds
    add: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 128, 1]" = var_mean[0]
    getitem_1: "f32[1, 128, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add, getitem_1);  getitem_1 = None
    mul: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul, primals_3)
    add_2: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_1, primals_4);  mul_1 = primals_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
    view_2: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_2, [128, 2048]);  add_2 = None
    mm: "f32[128, 2048]" = torch.ops.aten.mm.default(view_2, permute)
    view_3: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm, [1, 128, 2048]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_1: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
    mm_1: "f32[128, 2048]" = torch.ops.aten.mm.default(view_2, permute_1)
    view_5: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_1, [1, 128, 2048]);  mm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_2: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
    mm_2: "f32[128, 2048]" = torch.ops.aten.mm.default(view_2, permute_2)
    view_7: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_2, [1, 128, 2048]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_8: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_3, [1, 128, 16, 128]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_3: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_9: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_5, [1, 128, 16, 128]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_4: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_10: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_7, [1, 128, 16, 128]);  view_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_5: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_6: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_4, [0, 1, 3, 2])
    expand: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_3, [1, 16, 128, 128]);  permute_3 = None
    view_11: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand, [16, 128, 128]);  expand = None
    expand_1: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_6, [1, 16, 128, 128]);  permute_6 = None
    view_12: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_1, [16, 128, 128]);  expand_1 = None
    bmm: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_11, view_12)
    view_13: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm, [1, 16, 128, 128]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_1: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_318, 0, 0, 9223372036854775807);  primals_318 = None
    slice_2: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 9223372036854775807);  slice_1 = None
    slice_3: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_2, 2, 0, 128);  slice_2 = None
    slice_4: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_3, 3, 0, 128);  slice_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_4, view_13, full_default);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where, [-1], True)
    sub_1: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where, amax);  where = amax = None
    exp: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_1: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_1: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_2: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_1, [1, 16, 128, 128]);  clone_1 = None
    view_14: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_2, [16, 128, 128]);  expand_2 = None
    expand_3: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_5, [1, 16, 128, 128])
    view_15: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_3, [16, 128, 128]);  expand_3 = None
    bmm_1: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_14, view_15)
    view_16: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_1, [1, 16, 128, 128]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_7: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
    clone_2: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_17: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_2, [1, 128, 2048]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_18: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_17, [128, 2048]);  view_17 = None
    permute_8: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
    addmm: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_9, view_18, permute_8);  primals_9 = None
    view_19: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm, [1, 128, 2048]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_3: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_19, add);  view_19 = add = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
    getitem_2: "f32[1, 128, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 128, 1]" = var_mean_1[1];  var_mean_1 = None
    add_4: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
    rsqrt_1: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_2: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_3, getitem_3);  getitem_3 = None
    mul_2: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
    mul_3: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_2, primals_10)
    add_5: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_3, primals_11);  mul_3 = primals_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_20: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_5, [128, 2048]);  add_5 = None
    permute_9: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
    addmm_1: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_13, view_20, permute_9);  primals_13 = None
    view_21: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_1, [1, 128, 8192])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_4: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_21, 0.5)
    pow_1: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_21, 3.0)
    mul_5: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
    add_6: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_21, mul_5);  view_21 = mul_5 = None
    mul_6: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_6, 0.7978845608028654);  add_6 = None
    tanh: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
    add_7: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh, 1.0)
    mul_7: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_4, add_7);  mul_4 = add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_22: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_7, [128, 8192]);  mul_7 = None
    permute_10: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
    addmm_2: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_15, view_22, permute_10);  primals_15 = None
    view_23: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_2, [1, 128, 2048]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_8: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_3, view_23);  add_3 = view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 128, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 128, 1]" = var_mean_2[1];  var_mean_2 = None
    add_9: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_2: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
    sub_3: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_8, getitem_5);  getitem_5 = None
    mul_8: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
    mul_9: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_8, primals_16)
    add_10: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_9, primals_17);  mul_9 = primals_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_11: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
    view_24: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_10, [128, 2048]);  add_10 = None
    mm_3: "f32[128, 2048]" = torch.ops.aten.mm.default(view_24, permute_11)
    view_25: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_3, [1, 128, 2048]);  mm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_12: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
    mm_4: "f32[128, 2048]" = torch.ops.aten.mm.default(view_24, permute_12)
    view_27: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_4, [1, 128, 2048]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_13: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
    mm_5: "f32[128, 2048]" = torch.ops.aten.mm.default(view_24, permute_13)
    view_29: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_5, [1, 128, 2048]);  mm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_30: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_25, [1, 128, 16, 128]);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_14: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_31: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_27, [1, 128, 16, 128]);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_15: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_32: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_29, [1, 128, 16, 128]);  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_16: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_17: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_15, [0, 1, 3, 2])
    expand_4: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_14, [1, 16, 128, 128]);  permute_14 = None
    view_33: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_4, [16, 128, 128]);  expand_4 = None
    expand_5: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_17, [1, 16, 128, 128]);  permute_17 = None
    view_34: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_5, [16, 128, 128]);  expand_5 = None
    bmm_2: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_33, view_34)
    view_35: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_2, [1, 16, 128, 128]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_5: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_319, 0, 0, 9223372036854775807);  primals_319 = None
    slice_6: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_5, 1, 0, 9223372036854775807);  slice_5 = None
    slice_7: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_6, 2, 0, 128);  slice_6 = None
    slice_8: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_7, 3, 0, 128);  slice_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_1: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_8, view_35, full_default);  view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_1: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_1, [-1], True)
    sub_4: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_1, amax_1);  where_1 = amax_1 = None
    exp_1: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_2: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_2: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_5: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_1);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_6: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_5, [1, 16, 128, 128]);  clone_5 = None
    view_36: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_6, [16, 128, 128]);  expand_6 = None
    expand_7: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_16, [1, 16, 128, 128])
    view_37: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_7, [16, 128, 128]);  expand_7 = None
    bmm_3: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_36, view_37)
    view_38: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_3, [1, 16, 128, 128]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_18: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
    clone_6: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_39: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_6, [1, 128, 2048]);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_40: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_39, [128, 2048]);  view_39 = None
    permute_19: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
    addmm_3: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_22, view_40, permute_19);  primals_22 = None
    view_41: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_3, [1, 128, 2048]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_11: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_41, add_8);  view_41 = add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
    getitem_6: "f32[1, 128, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 128, 1]" = var_mean_3[1];  var_mean_3 = None
    add_12: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
    rsqrt_3: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    sub_5: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_11, getitem_7);  getitem_7 = None
    mul_10: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
    mul_11: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_10, primals_23)
    add_13: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_11, primals_24);  mul_11 = primals_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_42: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_13, [128, 2048]);  add_13 = None
    permute_20: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
    addmm_4: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_26, view_42, permute_20);  primals_26 = None
    view_43: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_4, [1, 128, 8192])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_12: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
    pow_2: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_43, 3.0)
    mul_13: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
    add_14: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_43, mul_13);  view_43 = mul_13 = None
    mul_14: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_14, 0.7978845608028654);  add_14 = None
    tanh_1: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
    add_15: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_1, 1.0)
    mul_15: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_12, add_15);  mul_12 = add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_44: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_15, [128, 8192]);  mul_15 = None
    permute_21: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
    addmm_5: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_28, view_44, permute_21);  primals_28 = None
    view_45: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_5, [1, 128, 2048]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_16: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_11, view_45);  add_11 = view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_16, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 128, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 128, 1]" = var_mean_4[1];  var_mean_4 = None
    add_17: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_4: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
    sub_6: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_16, getitem_9);  getitem_9 = None
    mul_16: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
    mul_17: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_16, primals_29)
    add_18: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_17, primals_30);  mul_17 = primals_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_22: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_31, [1, 0]);  primals_31 = None
    view_46: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_18, [128, 2048]);  add_18 = None
    mm_6: "f32[128, 2048]" = torch.ops.aten.mm.default(view_46, permute_22)
    view_47: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_6, [1, 128, 2048]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_23: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
    mm_7: "f32[128, 2048]" = torch.ops.aten.mm.default(view_46, permute_23)
    view_49: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_7, [1, 128, 2048]);  mm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_24: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_33, [1, 0]);  primals_33 = None
    mm_8: "f32[128, 2048]" = torch.ops.aten.mm.default(view_46, permute_24)
    view_51: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_8, [1, 128, 2048]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_52: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_47, [1, 128, 16, 128]);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_25: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_53: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_49, [1, 128, 16, 128]);  view_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_26: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_54: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_51, [1, 128, 16, 128]);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_27: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_28: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_26, [0, 1, 3, 2])
    expand_8: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_25, [1, 16, 128, 128]);  permute_25 = None
    view_55: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_8, [16, 128, 128]);  expand_8 = None
    expand_9: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_28, [1, 16, 128, 128]);  permute_28 = None
    view_56: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_9, [16, 128, 128]);  expand_9 = None
    bmm_4: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_55, view_56)
    view_57: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_4, [1, 16, 128, 128]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_9: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_320, 0, 0, 9223372036854775807);  primals_320 = None
    slice_10: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_9, 1, 0, 9223372036854775807);  slice_9 = None
    slice_11: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_10, 2, 0, 128);  slice_10 = None
    slice_12: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_11, 3, 0, 128);  slice_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_2: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_12, view_57, full_default);  view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_2: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_2, [-1], True)
    sub_7: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_2, amax_2);  where_2 = amax_2 = None
    exp_2: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_3: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_2: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_4: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_9: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_2);  div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_10: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_9, [1, 16, 128, 128]);  clone_9 = None
    view_58: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_10, [16, 128, 128]);  expand_10 = None
    expand_11: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_27, [1, 16, 128, 128])
    view_59: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_11, [16, 128, 128]);  expand_11 = None
    bmm_5: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_58, view_59)
    view_60: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_5, [1, 16, 128, 128]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_29: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
    clone_10: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_61: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_10, [1, 128, 2048]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_62: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_61, [128, 2048]);  view_61 = None
    permute_30: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
    addmm_6: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_35, view_62, permute_30);  primals_35 = None
    view_63: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_6, [1, 128, 2048]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_19: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_63, add_16);  view_63 = add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
    getitem_10: "f32[1, 128, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 128, 1]" = var_mean_5[1];  var_mean_5 = None
    add_20: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
    rsqrt_5: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
    sub_8: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_19, getitem_11);  getitem_11 = None
    mul_18: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = None
    mul_19: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_18, primals_36)
    add_21: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_19, primals_37);  mul_19 = primals_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_64: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_21, [128, 2048]);  add_21 = None
    permute_31: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
    addmm_7: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_39, view_64, permute_31);  primals_39 = None
    view_65: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_7, [1, 128, 8192])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_20: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_65, 0.5)
    pow_3: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_65, 3.0)
    mul_21: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
    add_22: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_65, mul_21);  view_65 = mul_21 = None
    mul_22: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_22, 0.7978845608028654);  add_22 = None
    tanh_2: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
    add_23: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_2, 1.0)
    mul_23: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_20, add_23);  mul_20 = add_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_66: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_23, [128, 8192]);  mul_23 = None
    permute_32: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
    addmm_8: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_41, view_66, permute_32);  primals_41 = None
    view_67: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_8, [1, 128, 2048]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_24: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_19, view_67);  add_19 = view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 128, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 128, 1]" = var_mean_6[1];  var_mean_6 = None
    add_25: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_6: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    sub_9: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_24, getitem_13);  getitem_13 = None
    mul_24: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = None
    mul_25: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_24, primals_42)
    add_26: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_25, primals_43);  mul_25 = primals_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_33: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
    view_68: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_26, [128, 2048]);  add_26 = None
    mm_9: "f32[128, 2048]" = torch.ops.aten.mm.default(view_68, permute_33)
    view_69: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_9, [1, 128, 2048]);  mm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_34: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
    mm_10: "f32[128, 2048]" = torch.ops.aten.mm.default(view_68, permute_34)
    view_71: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_10, [1, 128, 2048]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_35: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_46, [1, 0]);  primals_46 = None
    mm_11: "f32[128, 2048]" = torch.ops.aten.mm.default(view_68, permute_35)
    view_73: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_11, [1, 128, 2048]);  mm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_74: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_69, [1, 128, 16, 128]);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_36: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_75: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_71, [1, 128, 16, 128]);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_37: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_76: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_73, [1, 128, 16, 128]);  view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_38: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_39: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_37, [0, 1, 3, 2])
    expand_12: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_36, [1, 16, 128, 128]);  permute_36 = None
    view_77: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_12, [16, 128, 128]);  expand_12 = None
    expand_13: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_39, [1, 16, 128, 128]);  permute_39 = None
    view_78: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_13, [16, 128, 128]);  expand_13 = None
    bmm_6: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_77, view_78)
    view_79: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_6, [1, 16, 128, 128]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_13: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_321, 0, 0, 9223372036854775807);  primals_321 = None
    slice_14: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_13, 1, 0, 9223372036854775807);  slice_13 = None
    slice_15: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_14, 2, 0, 128);  slice_14 = None
    slice_16: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_15, 3, 0, 128);  slice_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_3: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_16, view_79, full_default);  view_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_3: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_3, [-1], True)
    sub_10: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_3, amax_3);  where_3 = amax_3 = None
    exp_3: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_4: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_3: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_6: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_13: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_3);  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_14: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_13, [1, 16, 128, 128]);  clone_13 = None
    view_80: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_14, [16, 128, 128]);  expand_14 = None
    expand_15: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_38, [1, 16, 128, 128])
    view_81: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_15, [16, 128, 128]);  expand_15 = None
    bmm_7: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_80, view_81)
    view_82: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_7, [1, 16, 128, 128]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_40: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
    clone_14: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_83: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_14, [1, 128, 2048]);  clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_84: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_83, [128, 2048]);  view_83 = None
    permute_41: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_47, [1, 0]);  primals_47 = None
    addmm_9: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_48, view_84, permute_41);  primals_48 = None
    view_85: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_9, [1, 128, 2048]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_27: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_85, add_24);  view_85 = add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
    getitem_14: "f32[1, 128, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 128, 1]" = var_mean_7[1];  var_mean_7 = None
    add_28: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
    rsqrt_7: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_11: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_27, getitem_15);  getitem_15 = None
    mul_26: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = None
    mul_27: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_26, primals_49)
    add_29: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_27, primals_50);  mul_27 = primals_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_86: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_29, [128, 2048]);  add_29 = None
    permute_42: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
    addmm_10: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_52, view_86, permute_42);  primals_52 = None
    view_87: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_10, [1, 128, 8192])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_28: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_87, 0.5)
    pow_4: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_87, 3.0)
    mul_29: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
    add_30: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_87, mul_29);  view_87 = mul_29 = None
    mul_30: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_30, 0.7978845608028654);  add_30 = None
    tanh_3: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
    add_31: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_3, 1.0)
    mul_31: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_28, add_31);  mul_28 = add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_88: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_31, [128, 8192]);  mul_31 = None
    permute_43: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
    addmm_11: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_54, view_88, permute_43);  primals_54 = None
    view_89: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_11, [1, 128, 2048]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_32: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_27, view_89);  add_27 = view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
    getitem_16: "f32[1, 128, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 128, 1]" = var_mean_8[1];  var_mean_8 = None
    add_33: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_8: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
    sub_12: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_32, getitem_17);  getitem_17 = None
    mul_32: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = None
    mul_33: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_32, primals_55)
    add_34: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_33, primals_56);  mul_33 = primals_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_44: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
    view_90: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_34, [128, 2048]);  add_34 = None
    mm_12: "f32[128, 2048]" = torch.ops.aten.mm.default(view_90, permute_44)
    view_91: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_12, [1, 128, 2048]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_45: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
    mm_13: "f32[128, 2048]" = torch.ops.aten.mm.default(view_90, permute_45)
    view_93: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_13, [1, 128, 2048]);  mm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_46: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
    mm_14: "f32[128, 2048]" = torch.ops.aten.mm.default(view_90, permute_46)
    view_95: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_14, [1, 128, 2048]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_96: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_91, [1, 128, 16, 128]);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_47: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_97: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_93, [1, 128, 16, 128]);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_48: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_98: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_95, [1, 128, 16, 128]);  view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_49: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_50: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_48, [0, 1, 3, 2])
    expand_16: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_47, [1, 16, 128, 128]);  permute_47 = None
    view_99: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_16, [16, 128, 128]);  expand_16 = None
    expand_17: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_50, [1, 16, 128, 128]);  permute_50 = None
    view_100: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_17, [16, 128, 128]);  expand_17 = None
    bmm_8: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_99, view_100)
    view_101: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_8, [1, 16, 128, 128]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_17: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_322, 0, 0, 9223372036854775807);  primals_322 = None
    slice_18: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_17, 1, 0, 9223372036854775807);  slice_17 = None
    slice_19: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_18, 2, 0, 128);  slice_18 = None
    slice_20: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_19, 3, 0, 128);  slice_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_4: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_20, view_101, full_default);  view_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_4: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_4, [-1], True)
    sub_13: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_4, amax_4);  where_4 = amax_4 = None
    exp_4: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_5: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_4: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    alias_8: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_17: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_4);  div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_18: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_17, [1, 16, 128, 128]);  clone_17 = None
    view_102: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_18, [16, 128, 128]);  expand_18 = None
    expand_19: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_49, [1, 16, 128, 128])
    view_103: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_19, [16, 128, 128]);  expand_19 = None
    bmm_9: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_102, view_103)
    view_104: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_9, [1, 16, 128, 128]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_51: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
    clone_18: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_105: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_18, [1, 128, 2048]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_106: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_105, [128, 2048]);  view_105 = None
    permute_52: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
    addmm_12: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_61, view_106, permute_52);  primals_61 = None
    view_107: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_12, [1, 128, 2048]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_35: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_107, add_32);  view_107 = add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_9 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 128, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 128, 1]" = var_mean_9[1];  var_mean_9 = None
    add_36: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
    rsqrt_9: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_14: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_35, getitem_19);  getitem_19 = None
    mul_34: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = None
    mul_35: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_34, primals_62)
    add_37: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_35, primals_63);  mul_35 = primals_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_108: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_37, [128, 2048]);  add_37 = None
    permute_53: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
    addmm_13: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_65, view_108, permute_53);  primals_65 = None
    view_109: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_13, [1, 128, 8192])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_36: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
    pow_5: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_109, 3.0)
    mul_37: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
    add_38: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_109, mul_37);  view_109 = mul_37 = None
    mul_38: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_38, 0.7978845608028654);  add_38 = None
    tanh_4: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
    add_39: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_4, 1.0)
    mul_39: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_36, add_39);  mul_36 = add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_110: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_39, [128, 8192]);  mul_39 = None
    permute_54: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
    addmm_14: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_67, view_110, permute_54);  primals_67 = None
    view_111: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_14, [1, 128, 2048]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_40: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_35, view_111);  add_35 = view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_40, [2], correction = 0, keepdim = True)
    getitem_20: "f32[1, 128, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 128, 1]" = var_mean_10[1];  var_mean_10 = None
    add_41: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
    rsqrt_10: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
    sub_15: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_40, getitem_21);  getitem_21 = None
    mul_40: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = None
    mul_41: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_40, primals_68)
    add_42: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_41, primals_69);  mul_41 = primals_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_55: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
    view_112: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_42, [128, 2048]);  add_42 = None
    mm_15: "f32[128, 2048]" = torch.ops.aten.mm.default(view_112, permute_55)
    view_113: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_15, [1, 128, 2048]);  mm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_56: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
    mm_16: "f32[128, 2048]" = torch.ops.aten.mm.default(view_112, permute_56)
    view_115: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_16, [1, 128, 2048]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_57: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
    mm_17: "f32[128, 2048]" = torch.ops.aten.mm.default(view_112, permute_57)
    view_117: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_17, [1, 128, 2048]);  mm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_118: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_113, [1, 128, 16, 128]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_58: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_119: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_115, [1, 128, 16, 128]);  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_59: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_120: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_117, [1, 128, 16, 128]);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_60: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_61: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_59, [0, 1, 3, 2])
    expand_20: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_58, [1, 16, 128, 128]);  permute_58 = None
    view_121: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_20, [16, 128, 128]);  expand_20 = None
    expand_21: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_61, [1, 16, 128, 128]);  permute_61 = None
    view_122: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_21, [16, 128, 128]);  expand_21 = None
    bmm_10: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_121, view_122)
    view_123: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_10, [1, 16, 128, 128]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_21: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_323, 0, 0, 9223372036854775807);  primals_323 = None
    slice_22: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_21, 1, 0, 9223372036854775807);  slice_21 = None
    slice_23: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_22, 2, 0, 128);  slice_22 = None
    slice_24: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_23, 3, 0, 128);  slice_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_5: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_24, view_123, full_default);  view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_5: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_5, [-1], True)
    sub_16: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_5, amax_5);  where_5 = amax_5 = None
    exp_5: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_6: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_5: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    alias_10: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_21: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_5);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_22: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_21, [1, 16, 128, 128]);  clone_21 = None
    view_124: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_22, [16, 128, 128]);  expand_22 = None
    expand_23: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_60, [1, 16, 128, 128])
    view_125: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_23, [16, 128, 128]);  expand_23 = None
    bmm_11: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_124, view_125)
    view_126: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_11, [1, 16, 128, 128]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_62: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
    clone_22: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_127: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_22, [1, 128, 2048]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_128: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_127, [128, 2048]);  view_127 = None
    permute_63: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
    addmm_15: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_74, view_128, permute_63);  primals_74 = None
    view_129: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_15, [1, 128, 2048]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_43: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_129, add_40);  view_129 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 128, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 128, 1]" = var_mean_11[1];  var_mean_11 = None
    add_44: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_11: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
    sub_17: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_43, getitem_23);  getitem_23 = None
    mul_42: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = None
    mul_43: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_42, primals_75)
    add_45: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_43, primals_76);  mul_43 = primals_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_130: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_45, [128, 2048]);  add_45 = None
    permute_64: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
    addmm_16: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_78, view_130, permute_64);  primals_78 = None
    view_131: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_16, [1, 128, 8192])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_44: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_131, 0.5)
    pow_6: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_131, 3.0)
    mul_45: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
    add_46: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_131, mul_45);  view_131 = mul_45 = None
    mul_46: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_46, 0.7978845608028654);  add_46 = None
    tanh_5: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
    add_47: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_5, 1.0)
    mul_47: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_44, add_47);  mul_44 = add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_132: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_47, [128, 8192]);  mul_47 = None
    permute_65: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
    addmm_17: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_80, view_132, permute_65);  primals_80 = None
    view_133: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_17, [1, 128, 2048]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_48: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_43, view_133);  add_43 = view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
    getitem_24: "f32[1, 128, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 128, 1]" = var_mean_12[1];  var_mean_12 = None
    add_49: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
    rsqrt_12: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    sub_18: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_48, getitem_25);  getitem_25 = None
    mul_48: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = None
    mul_49: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_48, primals_81)
    add_50: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_49, primals_82);  mul_49 = primals_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_66: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_83, [1, 0]);  primals_83 = None
    view_134: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_50, [128, 2048]);  add_50 = None
    mm_18: "f32[128, 2048]" = torch.ops.aten.mm.default(view_134, permute_66)
    view_135: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_18, [1, 128, 2048]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_67: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_84, [1, 0]);  primals_84 = None
    mm_19: "f32[128, 2048]" = torch.ops.aten.mm.default(view_134, permute_67)
    view_137: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_19, [1, 128, 2048]);  mm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_68: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
    mm_20: "f32[128, 2048]" = torch.ops.aten.mm.default(view_134, permute_68)
    view_139: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_20, [1, 128, 2048]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_140: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_135, [1, 128, 16, 128]);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_69: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_141: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_137, [1, 128, 16, 128]);  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_70: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_141, [0, 2, 1, 3]);  view_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_142: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_139, [1, 128, 16, 128]);  view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_71: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_72: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_70, [0, 1, 3, 2])
    expand_24: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_69, [1, 16, 128, 128]);  permute_69 = None
    view_143: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_24, [16, 128, 128]);  expand_24 = None
    expand_25: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_72, [1, 16, 128, 128]);  permute_72 = None
    view_144: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_25, [16, 128, 128]);  expand_25 = None
    bmm_12: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_143, view_144)
    view_145: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_12, [1, 16, 128, 128]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_25: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_324, 0, 0, 9223372036854775807);  primals_324 = None
    slice_26: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_25, 1, 0, 9223372036854775807);  slice_25 = None
    slice_27: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_26, 2, 0, 128);  slice_26 = None
    slice_28: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_27, 3, 0, 128);  slice_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_6: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_28, view_145, full_default);  view_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_6: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_6, [-1], True)
    sub_19: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_6, amax_6);  where_6 = amax_6 = None
    exp_6: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_7: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_6: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    alias_12: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_25: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_6);  div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_26: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_25, [1, 16, 128, 128]);  clone_25 = None
    view_146: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_26, [16, 128, 128]);  expand_26 = None
    expand_27: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_71, [1, 16, 128, 128])
    view_147: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_27, [16, 128, 128]);  expand_27 = None
    bmm_13: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_146, view_147)
    view_148: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_13, [1, 16, 128, 128]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_73: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
    clone_26: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_149: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_26, [1, 128, 2048]);  clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_150: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_149, [128, 2048]);  view_149 = None
    permute_74: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
    addmm_18: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_87, view_150, permute_74);  primals_87 = None
    view_151: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_18, [1, 128, 2048]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_51: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_151, add_48);  view_151 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_13 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
    getitem_26: "f32[1, 128, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 128, 1]" = var_mean_13[1];  var_mean_13 = None
    add_52: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
    rsqrt_13: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_20: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_51, getitem_27);  getitem_27 = None
    mul_50: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = None
    mul_51: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_50, primals_88)
    add_53: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_51, primals_89);  mul_51 = primals_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_152: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_53, [128, 2048]);  add_53 = None
    permute_75: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
    addmm_19: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_91, view_152, permute_75);  primals_91 = None
    view_153: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_19, [1, 128, 8192])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_52: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_153, 0.5)
    pow_7: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_153, 3.0)
    mul_53: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
    add_54: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_153, mul_53);  view_153 = mul_53 = None
    mul_54: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_54, 0.7978845608028654);  add_54 = None
    tanh_6: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_54);  mul_54 = None
    add_55: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_6, 1.0)
    mul_55: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_52, add_55);  mul_52 = add_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_154: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_55, [128, 8192]);  mul_55 = None
    permute_76: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
    addmm_20: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_93, view_154, permute_76);  primals_93 = None
    view_155: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_20, [1, 128, 2048]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_56: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_51, view_155);  add_51 = view_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_56, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 128, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 128, 1]" = var_mean_14[1];  var_mean_14 = None
    add_57: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_14: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
    sub_21: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_56, getitem_29);  getitem_29 = None
    mul_56: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = None
    mul_57: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_56, primals_94)
    add_58: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_57, primals_95);  mul_57 = primals_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_77: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
    view_156: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_58, [128, 2048]);  add_58 = None
    mm_21: "f32[128, 2048]" = torch.ops.aten.mm.default(view_156, permute_77)
    view_157: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_21, [1, 128, 2048]);  mm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_78: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
    mm_22: "f32[128, 2048]" = torch.ops.aten.mm.default(view_156, permute_78)
    view_159: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_22, [1, 128, 2048]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_79: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
    mm_23: "f32[128, 2048]" = torch.ops.aten.mm.default(view_156, permute_79)
    view_161: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_23, [1, 128, 2048]);  mm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_162: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_157, [1, 128, 16, 128]);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_80: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_163: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_159, [1, 128, 16, 128]);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_81: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_163, [0, 2, 1, 3]);  view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_164: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_161, [1, 128, 16, 128]);  view_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_82: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_83: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_81, [0, 1, 3, 2])
    expand_28: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_80, [1, 16, 128, 128]);  permute_80 = None
    view_165: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_28, [16, 128, 128]);  expand_28 = None
    expand_29: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_83, [1, 16, 128, 128]);  permute_83 = None
    view_166: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_29, [16, 128, 128]);  expand_29 = None
    bmm_14: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_165, view_166)
    view_167: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_14, [1, 16, 128, 128]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_29: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_325, 0, 0, 9223372036854775807);  primals_325 = None
    slice_30: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_29, 1, 0, 9223372036854775807);  slice_29 = None
    slice_31: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_30, 2, 0, 128);  slice_30 = None
    slice_32: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_31, 3, 0, 128);  slice_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_7: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_32, view_167, full_default);  view_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_7: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_7, [-1], True)
    sub_22: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_7, amax_7);  where_7 = amax_7 = None
    exp_7: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_8: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_7: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    alias_14: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_29: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_7);  div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_30: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_29, [1, 16, 128, 128]);  clone_29 = None
    view_168: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_30, [16, 128, 128]);  expand_30 = None
    expand_31: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_82, [1, 16, 128, 128])
    view_169: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_31, [16, 128, 128]);  expand_31 = None
    bmm_15: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_168, view_169)
    view_170: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_15, [1, 16, 128, 128]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_84: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
    clone_30: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_171: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_30, [1, 128, 2048]);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_172: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_171, [128, 2048]);  view_171 = None
    permute_85: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
    addmm_21: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_100, view_172, permute_85);  primals_100 = None
    view_173: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_21, [1, 128, 2048]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_59: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_173, add_56);  view_173 = add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_15 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
    getitem_30: "f32[1, 128, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 128, 1]" = var_mean_15[1];  var_mean_15 = None
    add_60: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
    rsqrt_15: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_23: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_59, getitem_31);  getitem_31 = None
    mul_58: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = None
    mul_59: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_58, primals_101)
    add_61: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_59, primals_102);  mul_59 = primals_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_174: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_61, [128, 2048]);  add_61 = None
    permute_86: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
    addmm_22: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_104, view_174, permute_86);  primals_104 = None
    view_175: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_22, [1, 128, 8192])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_60: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_175, 0.5)
    pow_8: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_175, 3.0)
    mul_61: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
    add_62: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_175, mul_61);  view_175 = mul_61 = None
    mul_62: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_62, 0.7978845608028654);  add_62 = None
    tanh_7: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_62);  mul_62 = None
    add_63: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_7, 1.0)
    mul_63: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_60, add_63);  mul_60 = add_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_176: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_63, [128, 8192]);  mul_63 = None
    permute_87: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
    addmm_23: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_106, view_176, permute_87);  primals_106 = None
    view_177: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_23, [1, 128, 2048]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_64: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_59, view_177);  add_59 = view_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_16 = torch.ops.aten.var_mean.correction(add_64, [2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 128, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 128, 1]" = var_mean_16[1];  var_mean_16 = None
    add_65: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
    rsqrt_16: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
    sub_24: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_64, getitem_33);  getitem_33 = None
    mul_64: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = None
    mul_65: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_64, primals_107)
    add_66: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_65, primals_108);  mul_65 = primals_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_88: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
    view_178: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_66, [128, 2048]);  add_66 = None
    mm_24: "f32[128, 2048]" = torch.ops.aten.mm.default(view_178, permute_88)
    view_179: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_24, [1, 128, 2048]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_89: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_110, [1, 0]);  primals_110 = None
    mm_25: "f32[128, 2048]" = torch.ops.aten.mm.default(view_178, permute_89)
    view_181: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_25, [1, 128, 2048]);  mm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_90: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
    mm_26: "f32[128, 2048]" = torch.ops.aten.mm.default(view_178, permute_90)
    view_183: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_26, [1, 128, 2048]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_184: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_179, [1, 128, 16, 128]);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_91: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_185: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_181, [1, 128, 16, 128]);  view_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_92: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_186: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_183, [1, 128, 16, 128]);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_93: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_94: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_92, [0, 1, 3, 2])
    expand_32: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_91, [1, 16, 128, 128]);  permute_91 = None
    view_187: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_32, [16, 128, 128]);  expand_32 = None
    expand_33: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_94, [1, 16, 128, 128]);  permute_94 = None
    view_188: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_33, [16, 128, 128]);  expand_33 = None
    bmm_16: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_187, view_188)
    view_189: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_16, [1, 16, 128, 128]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_33: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_326, 0, 0, 9223372036854775807);  primals_326 = None
    slice_34: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_33, 1, 0, 9223372036854775807);  slice_33 = None
    slice_35: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_34, 2, 0, 128);  slice_34 = None
    slice_36: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_35, 3, 0, 128);  slice_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_8: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_36, view_189, full_default);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_8: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_8, [-1], True)
    sub_25: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_8, amax_8);  where_8 = amax_8 = None
    exp_8: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
    sum_9: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_8: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    alias_16: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_33: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_8);  div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_34: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_33, [1, 16, 128, 128]);  clone_33 = None
    view_190: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_34, [16, 128, 128]);  expand_34 = None
    expand_35: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_93, [1, 16, 128, 128])
    view_191: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_35, [16, 128, 128]);  expand_35 = None
    bmm_17: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_190, view_191)
    view_192: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_17, [1, 16, 128, 128]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_95: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
    clone_34: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_193: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_34, [1, 128, 2048]);  clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_194: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_193, [128, 2048]);  view_193 = None
    permute_96: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_112, [1, 0]);  primals_112 = None
    addmm_24: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_113, view_194, permute_96);  primals_113 = None
    view_195: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_24, [1, 128, 2048]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_67: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_195, add_64);  view_195 = add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_17 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
    getitem_34: "f32[1, 128, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 128, 1]" = var_mean_17[1];  var_mean_17 = None
    add_68: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
    rsqrt_17: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    sub_26: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_67, getitem_35);  getitem_35 = None
    mul_66: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = None
    mul_67: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_66, primals_114)
    add_69: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_67, primals_115);  mul_67 = primals_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_196: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_69, [128, 2048]);  add_69 = None
    permute_97: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_116, [1, 0]);  primals_116 = None
    addmm_25: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_117, view_196, permute_97);  primals_117 = None
    view_197: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_25, [1, 128, 8192])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_68: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_197, 0.5)
    pow_9: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_197, 3.0)
    mul_69: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
    add_70: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_197, mul_69);  view_197 = mul_69 = None
    mul_70: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_70, 0.7978845608028654);  add_70 = None
    tanh_8: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
    add_71: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_8, 1.0)
    mul_71: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_68, add_71);  mul_68 = add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_198: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_71, [128, 8192]);  mul_71 = None
    permute_98: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
    addmm_26: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_119, view_198, permute_98);  primals_119 = None
    view_199: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_26, [1, 128, 2048]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_72: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_67, view_199);  add_67 = view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_18 = torch.ops.aten.var_mean.correction(add_72, [2], correction = 0, keepdim = True)
    getitem_36: "f32[1, 128, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 128, 1]" = var_mean_18[1];  var_mean_18 = None
    add_73: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
    rsqrt_18: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    sub_27: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_72, getitem_37);  getitem_37 = None
    mul_72: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = None
    mul_73: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_72, primals_120)
    add_74: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_73, primals_121);  mul_73 = primals_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_99: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
    view_200: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_74, [128, 2048]);  add_74 = None
    mm_27: "f32[128, 2048]" = torch.ops.aten.mm.default(view_200, permute_99)
    view_201: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_27, [1, 128, 2048]);  mm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_100: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_123, [1, 0]);  primals_123 = None
    mm_28: "f32[128, 2048]" = torch.ops.aten.mm.default(view_200, permute_100)
    view_203: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_28, [1, 128, 2048]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_101: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
    mm_29: "f32[128, 2048]" = torch.ops.aten.mm.default(view_200, permute_101)
    view_205: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_29, [1, 128, 2048]);  mm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_206: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_201, [1, 128, 16, 128]);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_102: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_207: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_203, [1, 128, 16, 128]);  view_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_103: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_208: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_205, [1, 128, 16, 128]);  view_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_104: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_105: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_103, [0, 1, 3, 2])
    expand_36: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_102, [1, 16, 128, 128]);  permute_102 = None
    view_209: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_36, [16, 128, 128]);  expand_36 = None
    expand_37: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_105, [1, 16, 128, 128]);  permute_105 = None
    view_210: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_37, [16, 128, 128]);  expand_37 = None
    bmm_18: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_209, view_210)
    view_211: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_18, [1, 16, 128, 128]);  bmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_37: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_327, 0, 0, 9223372036854775807);  primals_327 = None
    slice_38: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_37, 1, 0, 9223372036854775807);  slice_37 = None
    slice_39: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_38, 2, 0, 128);  slice_38 = None
    slice_40: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_39, 3, 0, 128);  slice_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_9: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_40, view_211, full_default);  view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_9: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_9, [-1], True)
    sub_28: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_9, amax_9);  where_9 = amax_9 = None
    exp_9: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
    sum_10: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_9: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    alias_18: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_37: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_9);  div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_38: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_37, [1, 16, 128, 128]);  clone_37 = None
    view_212: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_38, [16, 128, 128]);  expand_38 = None
    expand_39: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_104, [1, 16, 128, 128])
    view_213: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_39, [16, 128, 128]);  expand_39 = None
    bmm_19: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_212, view_213)
    view_214: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_19, [1, 16, 128, 128]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_106: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_214, [0, 2, 1, 3]);  view_214 = None
    clone_38: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_215: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_38, [1, 128, 2048]);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_216: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_215, [128, 2048]);  view_215 = None
    permute_107: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
    addmm_27: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_126, view_216, permute_107);  primals_126 = None
    view_217: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_27, [1, 128, 2048]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_75: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_217, add_72);  view_217 = add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_19 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 128, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 128, 1]" = var_mean_19[1];  var_mean_19 = None
    add_76: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
    rsqrt_19: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
    sub_29: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_75, getitem_39);  getitem_39 = None
    mul_74: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = None
    mul_75: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_74, primals_127)
    add_77: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_75, primals_128);  mul_75 = primals_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_218: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_77, [128, 2048]);  add_77 = None
    permute_108: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
    addmm_28: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_130, view_218, permute_108);  primals_130 = None
    view_219: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_28, [1, 128, 8192])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_76: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_219, 0.5)
    pow_10: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_219, 3.0)
    mul_77: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
    add_78: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_219, mul_77);  view_219 = mul_77 = None
    mul_78: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_78, 0.7978845608028654);  add_78 = None
    tanh_9: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
    add_79: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_9, 1.0)
    mul_79: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_76, add_79);  mul_76 = add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_220: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_79, [128, 8192]);  mul_79 = None
    permute_109: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
    addmm_29: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_132, view_220, permute_109);  primals_132 = None
    view_221: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_29, [1, 128, 2048]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_80: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_75, view_221);  add_75 = view_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_80, [2], correction = 0, keepdim = True)
    getitem_40: "f32[1, 128, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 128, 1]" = var_mean_20[1];  var_mean_20 = None
    add_81: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
    rsqrt_20: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    sub_30: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_80, getitem_41);  getitem_41 = None
    mul_80: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = None
    mul_81: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_80, primals_133)
    add_82: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_81, primals_134);  mul_81 = primals_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_110: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_135, [1, 0]);  primals_135 = None
    view_222: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_82, [128, 2048]);  add_82 = None
    mm_30: "f32[128, 2048]" = torch.ops.aten.mm.default(view_222, permute_110)
    view_223: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_30, [1, 128, 2048]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_111: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
    mm_31: "f32[128, 2048]" = torch.ops.aten.mm.default(view_222, permute_111)
    view_225: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_31, [1, 128, 2048]);  mm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_112: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_137, [1, 0]);  primals_137 = None
    mm_32: "f32[128, 2048]" = torch.ops.aten.mm.default(view_222, permute_112)
    view_227: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_32, [1, 128, 2048]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_228: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_223, [1, 128, 16, 128]);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_113: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_229: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_225, [1, 128, 16, 128]);  view_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_114: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_230: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_227, [1, 128, 16, 128]);  view_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_115: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_230, [0, 2, 1, 3]);  view_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_116: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_114, [0, 1, 3, 2])
    expand_40: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_113, [1, 16, 128, 128]);  permute_113 = None
    view_231: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_40, [16, 128, 128]);  expand_40 = None
    expand_41: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_116, [1, 16, 128, 128]);  permute_116 = None
    view_232: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_41, [16, 128, 128]);  expand_41 = None
    bmm_20: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_231, view_232)
    view_233: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_20, [1, 16, 128, 128]);  bmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_41: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_328, 0, 0, 9223372036854775807);  primals_328 = None
    slice_42: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_41, 1, 0, 9223372036854775807);  slice_41 = None
    slice_43: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_42, 2, 0, 128);  slice_42 = None
    slice_44: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_43, 3, 0, 128);  slice_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_10: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_44, view_233, full_default);  view_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_10: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_10, [-1], True)
    sub_31: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_10, amax_10);  where_10 = amax_10 = None
    exp_10: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_11: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_10: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    alias_20: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_41: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_10);  div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_42: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_41, [1, 16, 128, 128]);  clone_41 = None
    view_234: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_42, [16, 128, 128]);  expand_42 = None
    expand_43: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_115, [1, 16, 128, 128])
    view_235: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_43, [16, 128, 128]);  expand_43 = None
    bmm_21: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_234, view_235)
    view_236: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_21, [1, 16, 128, 128]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_117: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
    clone_42: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_237: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_42, [1, 128, 2048]);  clone_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_238: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_237, [128, 2048]);  view_237 = None
    permute_118: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
    addmm_30: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_139, view_238, permute_118);  primals_139 = None
    view_239: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_30, [1, 128, 2048]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_83: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_239, add_80);  view_239 = add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 128, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 128, 1]" = var_mean_21[1];  var_mean_21 = None
    add_84: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
    rsqrt_21: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_32: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_83, getitem_43);  getitem_43 = None
    mul_82: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = None
    mul_83: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_82, primals_140)
    add_85: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_83, primals_141);  mul_83 = primals_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_240: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_85, [128, 2048]);  add_85 = None
    permute_119: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_142, [1, 0]);  primals_142 = None
    addmm_31: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_143, view_240, permute_119);  primals_143 = None
    view_241: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_31, [1, 128, 8192])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_84: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_241, 0.5)
    pow_11: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_241, 3.0)
    mul_85: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
    add_86: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_241, mul_85);  view_241 = mul_85 = None
    mul_86: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_86, 0.7978845608028654);  add_86 = None
    tanh_10: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_86);  mul_86 = None
    add_87: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_10, 1.0)
    mul_87: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_84, add_87);  mul_84 = add_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_242: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_87, [128, 8192]);  mul_87 = None
    permute_120: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
    addmm_32: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_145, view_242, permute_120);  primals_145 = None
    view_243: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_32, [1, 128, 2048]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_88: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_83, view_243);  add_83 = view_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_88, [2], correction = 0, keepdim = True)
    getitem_44: "f32[1, 128, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 128, 1]" = var_mean_22[1];  var_mean_22 = None
    add_89: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
    rsqrt_22: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
    sub_33: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_88, getitem_45);  getitem_45 = None
    mul_88: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = None
    mul_89: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_88, primals_146)
    add_90: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_89, primals_147);  mul_89 = primals_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_121: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_148, [1, 0]);  primals_148 = None
    view_244: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_90, [128, 2048]);  add_90 = None
    mm_33: "f32[128, 2048]" = torch.ops.aten.mm.default(view_244, permute_121)
    view_245: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_33, [1, 128, 2048]);  mm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_122: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_149, [1, 0]);  primals_149 = None
    mm_34: "f32[128, 2048]" = torch.ops.aten.mm.default(view_244, permute_122)
    view_247: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_34, [1, 128, 2048]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_123: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
    mm_35: "f32[128, 2048]" = torch.ops.aten.mm.default(view_244, permute_123)
    view_249: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_35, [1, 128, 2048]);  mm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_250: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_245, [1, 128, 16, 128]);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_124: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_251: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_247, [1, 128, 16, 128]);  view_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_125: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_251, [0, 2, 1, 3]);  view_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_252: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_249, [1, 128, 16, 128]);  view_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_126: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_127: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_125, [0, 1, 3, 2])
    expand_44: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_124, [1, 16, 128, 128]);  permute_124 = None
    view_253: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_44, [16, 128, 128]);  expand_44 = None
    expand_45: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_127, [1, 16, 128, 128]);  permute_127 = None
    view_254: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_45, [16, 128, 128]);  expand_45 = None
    bmm_22: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_253, view_254)
    view_255: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_22, [1, 16, 128, 128]);  bmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_45: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_329, 0, 0, 9223372036854775807);  primals_329 = None
    slice_46: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_45, 1, 0, 9223372036854775807);  slice_45 = None
    slice_47: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_46, 2, 0, 128);  slice_46 = None
    slice_48: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_47, 3, 0, 128);  slice_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_11: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_48, view_255, full_default);  view_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_11: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_11, [-1], True)
    sub_34: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_11, amax_11);  where_11 = amax_11 = None
    exp_11: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
    sum_12: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_11: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    alias_22: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_45: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_11);  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_46: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_45, [1, 16, 128, 128]);  clone_45 = None
    view_256: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_46, [16, 128, 128]);  expand_46 = None
    expand_47: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_126, [1, 16, 128, 128])
    view_257: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_47, [16, 128, 128]);  expand_47 = None
    bmm_23: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_256, view_257)
    view_258: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_23, [1, 16, 128, 128]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_128: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
    clone_46: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_259: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_46, [1, 128, 2048]);  clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_260: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_259, [128, 2048]);  view_259 = None
    permute_129: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_151, [1, 0]);  primals_151 = None
    addmm_33: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_152, view_260, permute_129);  primals_152 = None
    view_261: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_33, [1, 128, 2048]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_91: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_261, add_88);  view_261 = add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
    getitem_46: "f32[1, 128, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 128, 1]" = var_mean_23[1];  var_mean_23 = None
    add_92: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
    rsqrt_23: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
    sub_35: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_91, getitem_47);  getitem_47 = None
    mul_90: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = None
    mul_91: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_90, primals_153)
    add_93: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_91, primals_154);  mul_91 = primals_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_262: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_93, [128, 2048]);  add_93 = None
    permute_130: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_155, [1, 0]);  primals_155 = None
    addmm_34: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_156, view_262, permute_130);  primals_156 = None
    view_263: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_34, [1, 128, 8192])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_92: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_263, 0.5)
    pow_12: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_263, 3.0)
    mul_93: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
    add_94: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_263, mul_93);  view_263 = mul_93 = None
    mul_94: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_94, 0.7978845608028654);  add_94 = None
    tanh_11: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
    add_95: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_11, 1.0)
    mul_95: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_92, add_95);  mul_92 = add_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_264: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_95, [128, 8192]);  mul_95 = None
    permute_131: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_157, [1, 0]);  primals_157 = None
    addmm_35: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_158, view_264, permute_131);  primals_158 = None
    view_265: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_35, [1, 128, 2048]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_96: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_91, view_265);  add_91 = view_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_96, [2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 128, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 128, 1]" = var_mean_24[1];  var_mean_24 = None
    add_97: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
    rsqrt_24: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
    sub_36: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_96, getitem_49);  getitem_49 = None
    mul_96: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = None
    mul_97: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_96, primals_159)
    add_98: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_97, primals_160);  mul_97 = primals_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_132: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_161, [1, 0]);  primals_161 = None
    view_266: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_98, [128, 2048]);  add_98 = None
    mm_36: "f32[128, 2048]" = torch.ops.aten.mm.default(view_266, permute_132)
    view_267: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_36, [1, 128, 2048]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_133: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
    mm_37: "f32[128, 2048]" = torch.ops.aten.mm.default(view_266, permute_133)
    view_269: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_37, [1, 128, 2048]);  mm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_134: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_163, [1, 0]);  primals_163 = None
    mm_38: "f32[128, 2048]" = torch.ops.aten.mm.default(view_266, permute_134)
    view_271: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_38, [1, 128, 2048]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_272: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_267, [1, 128, 16, 128]);  view_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_135: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_272, [0, 2, 1, 3]);  view_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_273: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_269, [1, 128, 16, 128]);  view_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_136: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_273, [0, 2, 1, 3]);  view_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_274: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_271, [1, 128, 16, 128]);  view_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_137: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_274, [0, 2, 1, 3]);  view_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_138: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_136, [0, 1, 3, 2])
    expand_48: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_135, [1, 16, 128, 128]);  permute_135 = None
    view_275: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_48, [16, 128, 128]);  expand_48 = None
    expand_49: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_138, [1, 16, 128, 128]);  permute_138 = None
    view_276: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_49, [16, 128, 128]);  expand_49 = None
    bmm_24: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_275, view_276)
    view_277: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_24, [1, 16, 128, 128]);  bmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_49: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_330, 0, 0, 9223372036854775807);  primals_330 = None
    slice_50: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_49, 1, 0, 9223372036854775807);  slice_49 = None
    slice_51: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_50, 2, 0, 128);  slice_50 = None
    slice_52: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_51, 3, 0, 128);  slice_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_12: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_52, view_277, full_default);  view_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_12: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_12, [-1], True)
    sub_37: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_12, amax_12);  where_12 = amax_12 = None
    exp_12: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
    sum_13: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_12: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    alias_24: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_49: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_12);  div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_50: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_49, [1, 16, 128, 128]);  clone_49 = None
    view_278: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_50, [16, 128, 128]);  expand_50 = None
    expand_51: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_137, [1, 16, 128, 128])
    view_279: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_51, [16, 128, 128]);  expand_51 = None
    bmm_25: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_278, view_279)
    view_280: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_25, [1, 16, 128, 128]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_139: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_280, [0, 2, 1, 3]);  view_280 = None
    clone_50: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_281: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_50, [1, 128, 2048]);  clone_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_282: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_281, [128, 2048]);  view_281 = None
    permute_140: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_164, [1, 0]);  primals_164 = None
    addmm_36: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_165, view_282, permute_140);  primals_165 = None
    view_283: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_36, [1, 128, 2048]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_99: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_283, add_96);  view_283 = add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_25 = torch.ops.aten.var_mean.correction(add_99, [2], correction = 0, keepdim = True)
    getitem_50: "f32[1, 128, 1]" = var_mean_25[0]
    getitem_51: "f32[1, 128, 1]" = var_mean_25[1];  var_mean_25 = None
    add_100: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
    rsqrt_25: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
    sub_38: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_99, getitem_51);  getitem_51 = None
    mul_98: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_25);  sub_38 = None
    mul_99: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_98, primals_166)
    add_101: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_99, primals_167);  mul_99 = primals_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_284: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_101, [128, 2048]);  add_101 = None
    permute_141: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_168, [1, 0]);  primals_168 = None
    addmm_37: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_169, view_284, permute_141);  primals_169 = None
    view_285: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_37, [1, 128, 8192])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_100: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_285, 0.5)
    pow_13: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_285, 3.0)
    mul_101: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
    add_102: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_285, mul_101);  view_285 = mul_101 = None
    mul_102: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_102, 0.7978845608028654);  add_102 = None
    tanh_12: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_102);  mul_102 = None
    add_103: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_12, 1.0)
    mul_103: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_100, add_103);  mul_100 = add_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_286: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_103, [128, 8192]);  mul_103 = None
    permute_142: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
    addmm_38: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_171, view_286, permute_142);  primals_171 = None
    view_287: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_38, [1, 128, 2048]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_104: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_99, view_287);  add_99 = view_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_26 = torch.ops.aten.var_mean.correction(add_104, [2], correction = 0, keepdim = True)
    getitem_52: "f32[1, 128, 1]" = var_mean_26[0]
    getitem_53: "f32[1, 128, 1]" = var_mean_26[1];  var_mean_26 = None
    add_105: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
    rsqrt_26: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
    sub_39: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_104, getitem_53);  getitem_53 = None
    mul_104: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_26);  sub_39 = None
    mul_105: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_104, primals_172)
    add_106: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_105, primals_173);  mul_105 = primals_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_143: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_174, [1, 0]);  primals_174 = None
    view_288: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_106, [128, 2048]);  add_106 = None
    mm_39: "f32[128, 2048]" = torch.ops.aten.mm.default(view_288, permute_143)
    view_289: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_39, [1, 128, 2048]);  mm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_144: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_175, [1, 0]);  primals_175 = None
    mm_40: "f32[128, 2048]" = torch.ops.aten.mm.default(view_288, permute_144)
    view_291: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_40, [1, 128, 2048]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_145: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_176, [1, 0]);  primals_176 = None
    mm_41: "f32[128, 2048]" = torch.ops.aten.mm.default(view_288, permute_145)
    view_293: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_41, [1, 128, 2048]);  mm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_294: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_289, [1, 128, 16, 128]);  view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_146: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_295: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_291, [1, 128, 16, 128]);  view_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_147: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_295, [0, 2, 1, 3]);  view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_296: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_293, [1, 128, 16, 128]);  view_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_148: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_296, [0, 2, 1, 3]);  view_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_149: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_147, [0, 1, 3, 2])
    expand_52: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_146, [1, 16, 128, 128]);  permute_146 = None
    view_297: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_52, [16, 128, 128]);  expand_52 = None
    expand_53: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_149, [1, 16, 128, 128]);  permute_149 = None
    view_298: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_53, [16, 128, 128]);  expand_53 = None
    bmm_26: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_297, view_298)
    view_299: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_26, [1, 16, 128, 128]);  bmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_53: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_331, 0, 0, 9223372036854775807);  primals_331 = None
    slice_54: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_53, 1, 0, 9223372036854775807);  slice_53 = None
    slice_55: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_54, 2, 0, 128);  slice_54 = None
    slice_56: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_55, 3, 0, 128);  slice_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_13: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_56, view_299, full_default);  view_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_13: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_13, [-1], True)
    sub_40: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_13, amax_13);  where_13 = amax_13 = None
    exp_13: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_40);  sub_40 = None
    sum_14: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_13: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
    alias_26: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_53: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_13);  div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_54: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_53, [1, 16, 128, 128]);  clone_53 = None
    view_300: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_54, [16, 128, 128]);  expand_54 = None
    expand_55: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_148, [1, 16, 128, 128])
    view_301: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_55, [16, 128, 128]);  expand_55 = None
    bmm_27: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_300, view_301)
    view_302: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_27, [1, 16, 128, 128]);  bmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_150: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_302, [0, 2, 1, 3]);  view_302 = None
    clone_54: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_303: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_54, [1, 128, 2048]);  clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_304: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_303, [128, 2048]);  view_303 = None
    permute_151: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_177, [1, 0]);  primals_177 = None
    addmm_39: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_178, view_304, permute_151);  primals_178 = None
    view_305: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_39, [1, 128, 2048]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_107: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_305, add_104);  view_305 = add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_27 = torch.ops.aten.var_mean.correction(add_107, [2], correction = 0, keepdim = True)
    getitem_54: "f32[1, 128, 1]" = var_mean_27[0]
    getitem_55: "f32[1, 128, 1]" = var_mean_27[1];  var_mean_27 = None
    add_108: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
    rsqrt_27: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    sub_41: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_107, getitem_55);  getitem_55 = None
    mul_106: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_27);  sub_41 = None
    mul_107: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_106, primals_179)
    add_109: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_107, primals_180);  mul_107 = primals_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_306: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_109, [128, 2048]);  add_109 = None
    permute_152: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_181, [1, 0]);  primals_181 = None
    addmm_40: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_182, view_306, permute_152);  primals_182 = None
    view_307: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_40, [1, 128, 8192])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_108: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_307, 0.5)
    pow_14: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_307, 3.0)
    mul_109: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_14, 0.044715);  pow_14 = None
    add_110: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_307, mul_109);  view_307 = mul_109 = None
    mul_110: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_110, 0.7978845608028654);  add_110 = None
    tanh_13: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_110);  mul_110 = None
    add_111: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_13, 1.0)
    mul_111: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_108, add_111);  mul_108 = add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_308: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_111, [128, 8192]);  mul_111 = None
    permute_153: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_183, [1, 0]);  primals_183 = None
    addmm_41: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_184, view_308, permute_153);  primals_184 = None
    view_309: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_41, [1, 128, 2048]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_112: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_107, view_309);  add_107 = view_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_28 = torch.ops.aten.var_mean.correction(add_112, [2], correction = 0, keepdim = True)
    getitem_56: "f32[1, 128, 1]" = var_mean_28[0]
    getitem_57: "f32[1, 128, 1]" = var_mean_28[1];  var_mean_28 = None
    add_113: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
    rsqrt_28: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
    sub_42: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_112, getitem_57);  getitem_57 = None
    mul_112: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_28);  sub_42 = None
    mul_113: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_112, primals_185)
    add_114: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_113, primals_186);  mul_113 = primals_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_154: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_187, [1, 0]);  primals_187 = None
    view_310: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_114, [128, 2048]);  add_114 = None
    mm_42: "f32[128, 2048]" = torch.ops.aten.mm.default(view_310, permute_154)
    view_311: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_42, [1, 128, 2048]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_155: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
    mm_43: "f32[128, 2048]" = torch.ops.aten.mm.default(view_310, permute_155)
    view_313: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_43, [1, 128, 2048]);  mm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_156: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_189, [1, 0]);  primals_189 = None
    mm_44: "f32[128, 2048]" = torch.ops.aten.mm.default(view_310, permute_156)
    view_315: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_44, [1, 128, 2048]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_316: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_311, [1, 128, 16, 128]);  view_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_157: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_317: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_313, [1, 128, 16, 128]);  view_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_158: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_317, [0, 2, 1, 3]);  view_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_318: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_315, [1, 128, 16, 128]);  view_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_159: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_318, [0, 2, 1, 3]);  view_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_160: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_158, [0, 1, 3, 2])
    expand_56: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_157, [1, 16, 128, 128]);  permute_157 = None
    view_319: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_56, [16, 128, 128]);  expand_56 = None
    expand_57: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_160, [1, 16, 128, 128]);  permute_160 = None
    view_320: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_57, [16, 128, 128]);  expand_57 = None
    bmm_28: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_319, view_320)
    view_321: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_28, [1, 16, 128, 128]);  bmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_57: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_332, 0, 0, 9223372036854775807);  primals_332 = None
    slice_58: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_57, 1, 0, 9223372036854775807);  slice_57 = None
    slice_59: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_58, 2, 0, 128);  slice_58 = None
    slice_60: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_59, 3, 0, 128);  slice_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_14: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_60, view_321, full_default);  view_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_14: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_14, [-1], True)
    sub_43: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_14, amax_14);  where_14 = amax_14 = None
    exp_14: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
    sum_15: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_14: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
    alias_28: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_57: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_14);  div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_58: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_57, [1, 16, 128, 128]);  clone_57 = None
    view_322: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_58, [16, 128, 128]);  expand_58 = None
    expand_59: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_159, [1, 16, 128, 128])
    view_323: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_59, [16, 128, 128]);  expand_59 = None
    bmm_29: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_322, view_323)
    view_324: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_29, [1, 16, 128, 128]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_161: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_324, [0, 2, 1, 3]);  view_324 = None
    clone_58: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_325: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_58, [1, 128, 2048]);  clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_326: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_325, [128, 2048]);  view_325 = None
    permute_162: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_190, [1, 0]);  primals_190 = None
    addmm_42: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_191, view_326, permute_162);  primals_191 = None
    view_327: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_42, [1, 128, 2048]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_115: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_327, add_112);  view_327 = add_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_29 = torch.ops.aten.var_mean.correction(add_115, [2], correction = 0, keepdim = True)
    getitem_58: "f32[1, 128, 1]" = var_mean_29[0]
    getitem_59: "f32[1, 128, 1]" = var_mean_29[1];  var_mean_29 = None
    add_116: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
    rsqrt_29: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
    sub_44: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_115, getitem_59);  getitem_59 = None
    mul_114: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_29);  sub_44 = None
    mul_115: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_114, primals_192)
    add_117: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_115, primals_193);  mul_115 = primals_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_328: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_117, [128, 2048]);  add_117 = None
    permute_163: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_194, [1, 0]);  primals_194 = None
    addmm_43: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_195, view_328, permute_163);  primals_195 = None
    view_329: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_43, [1, 128, 8192])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_116: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_329, 0.5)
    pow_15: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_329, 3.0)
    mul_117: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
    add_118: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_329, mul_117);  view_329 = mul_117 = None
    mul_118: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_118, 0.7978845608028654);  add_118 = None
    tanh_14: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_118);  mul_118 = None
    add_119: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_14, 1.0)
    mul_119: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_116, add_119);  mul_116 = add_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_330: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_119, [128, 8192]);  mul_119 = None
    permute_164: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_196, [1, 0]);  primals_196 = None
    addmm_44: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_197, view_330, permute_164);  primals_197 = None
    view_331: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_44, [1, 128, 2048]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_120: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_115, view_331);  add_115 = view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_30 = torch.ops.aten.var_mean.correction(add_120, [2], correction = 0, keepdim = True)
    getitem_60: "f32[1, 128, 1]" = var_mean_30[0]
    getitem_61: "f32[1, 128, 1]" = var_mean_30[1];  var_mean_30 = None
    add_121: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
    rsqrt_30: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_121);  add_121 = None
    sub_45: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_120, getitem_61);  getitem_61 = None
    mul_120: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_30);  sub_45 = None
    mul_121: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_120, primals_198)
    add_122: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_121, primals_199);  mul_121 = primals_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_165: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_200, [1, 0]);  primals_200 = None
    view_332: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_122, [128, 2048]);  add_122 = None
    mm_45: "f32[128, 2048]" = torch.ops.aten.mm.default(view_332, permute_165)
    view_333: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_45, [1, 128, 2048]);  mm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_166: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_201, [1, 0]);  primals_201 = None
    mm_46: "f32[128, 2048]" = torch.ops.aten.mm.default(view_332, permute_166)
    view_335: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_46, [1, 128, 2048]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_167: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_202, [1, 0]);  primals_202 = None
    mm_47: "f32[128, 2048]" = torch.ops.aten.mm.default(view_332, permute_167)
    view_337: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_47, [1, 128, 2048]);  mm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_338: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_333, [1, 128, 16, 128]);  view_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_168: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_339: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_335, [1, 128, 16, 128]);  view_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_169: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_339, [0, 2, 1, 3]);  view_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_340: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_337, [1, 128, 16, 128]);  view_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_170: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_340, [0, 2, 1, 3]);  view_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_171: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_169, [0, 1, 3, 2])
    expand_60: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_168, [1, 16, 128, 128]);  permute_168 = None
    view_341: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_60, [16, 128, 128]);  expand_60 = None
    expand_61: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_171, [1, 16, 128, 128]);  permute_171 = None
    view_342: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_61, [16, 128, 128]);  expand_61 = None
    bmm_30: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_341, view_342)
    view_343: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_30, [1, 16, 128, 128]);  bmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_61: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_333, 0, 0, 9223372036854775807);  primals_333 = None
    slice_62: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_61, 1, 0, 9223372036854775807);  slice_61 = None
    slice_63: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_62, 2, 0, 128);  slice_62 = None
    slice_64: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_63, 3, 0, 128);  slice_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_15: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_64, view_343, full_default);  view_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_15: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_15, [-1], True)
    sub_46: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_15, amax_15);  where_15 = amax_15 = None
    exp_15: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
    sum_16: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
    div_15: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
    alias_30: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_61: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_15);  div_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_62: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_61, [1, 16, 128, 128]);  clone_61 = None
    view_344: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_62, [16, 128, 128]);  expand_62 = None
    expand_63: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_170, [1, 16, 128, 128])
    view_345: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_63, [16, 128, 128]);  expand_63 = None
    bmm_31: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_344, view_345)
    view_346: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_31, [1, 16, 128, 128]);  bmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_172: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_346, [0, 2, 1, 3]);  view_346 = None
    clone_62: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_347: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_62, [1, 128, 2048]);  clone_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_348: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_347, [128, 2048]);  view_347 = None
    permute_173: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_203, [1, 0]);  primals_203 = None
    addmm_45: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_204, view_348, permute_173);  primals_204 = None
    view_349: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_45, [1, 128, 2048]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_123: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_349, add_120);  view_349 = add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_31 = torch.ops.aten.var_mean.correction(add_123, [2], correction = 0, keepdim = True)
    getitem_62: "f32[1, 128, 1]" = var_mean_31[0]
    getitem_63: "f32[1, 128, 1]" = var_mean_31[1];  var_mean_31 = None
    add_124: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
    rsqrt_31: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
    sub_47: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_123, getitem_63);  getitem_63 = None
    mul_122: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_31);  sub_47 = None
    mul_123: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_122, primals_205)
    add_125: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_123, primals_206);  mul_123 = primals_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_350: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_125, [128, 2048]);  add_125 = None
    permute_174: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_207, [1, 0]);  primals_207 = None
    addmm_46: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_208, view_350, permute_174);  primals_208 = None
    view_351: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_46, [1, 128, 8192])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_124: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_351, 0.5)
    pow_16: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_351, 3.0)
    mul_125: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_16, 0.044715);  pow_16 = None
    add_126: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_351, mul_125);  view_351 = mul_125 = None
    mul_126: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_126, 0.7978845608028654);  add_126 = None
    tanh_15: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_126);  mul_126 = None
    add_127: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_15, 1.0)
    mul_127: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_124, add_127);  mul_124 = add_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_352: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_127, [128, 8192]);  mul_127 = None
    permute_175: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_209, [1, 0]);  primals_209 = None
    addmm_47: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_210, view_352, permute_175);  primals_210 = None
    view_353: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_47, [1, 128, 2048]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_128: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_123, view_353);  add_123 = view_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_32 = torch.ops.aten.var_mean.correction(add_128, [2], correction = 0, keepdim = True)
    getitem_64: "f32[1, 128, 1]" = var_mean_32[0]
    getitem_65: "f32[1, 128, 1]" = var_mean_32[1];  var_mean_32 = None
    add_129: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
    rsqrt_32: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
    sub_48: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_128, getitem_65);  getitem_65 = None
    mul_128: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_32);  sub_48 = None
    mul_129: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_128, primals_211)
    add_130: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_129, primals_212);  mul_129 = primals_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_176: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_213, [1, 0]);  primals_213 = None
    view_354: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_130, [128, 2048]);  add_130 = None
    mm_48: "f32[128, 2048]" = torch.ops.aten.mm.default(view_354, permute_176)
    view_355: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_48, [1, 128, 2048]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_177: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_214, [1, 0]);  primals_214 = None
    mm_49: "f32[128, 2048]" = torch.ops.aten.mm.default(view_354, permute_177)
    view_357: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_49, [1, 128, 2048]);  mm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_178: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_215, [1, 0]);  primals_215 = None
    mm_50: "f32[128, 2048]" = torch.ops.aten.mm.default(view_354, permute_178)
    view_359: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_50, [1, 128, 2048]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_360: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_355, [1, 128, 16, 128]);  view_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_179: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_360, [0, 2, 1, 3]);  view_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_361: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_357, [1, 128, 16, 128]);  view_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_180: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_361, [0, 2, 1, 3]);  view_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_362: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_359, [1, 128, 16, 128]);  view_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_181: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_362, [0, 2, 1, 3]);  view_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_182: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_180, [0, 1, 3, 2])
    expand_64: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_179, [1, 16, 128, 128]);  permute_179 = None
    view_363: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_64, [16, 128, 128]);  expand_64 = None
    expand_65: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_182, [1, 16, 128, 128]);  permute_182 = None
    view_364: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_65, [16, 128, 128]);  expand_65 = None
    bmm_32: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_363, view_364)
    view_365: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_32, [1, 16, 128, 128]);  bmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_65: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_334, 0, 0, 9223372036854775807);  primals_334 = None
    slice_66: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_65, 1, 0, 9223372036854775807);  slice_65 = None
    slice_67: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_66, 2, 0, 128);  slice_66 = None
    slice_68: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_67, 3, 0, 128);  slice_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_16: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_68, view_365, full_default);  view_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_16: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_16, [-1], True)
    sub_49: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_16, amax_16);  where_16 = amax_16 = None
    exp_16: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
    sum_17: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_16: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
    alias_32: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_65: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_16);  div_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_66: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_65, [1, 16, 128, 128]);  clone_65 = None
    view_366: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_66, [16, 128, 128]);  expand_66 = None
    expand_67: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_181, [1, 16, 128, 128])
    view_367: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_67, [16, 128, 128]);  expand_67 = None
    bmm_33: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_366, view_367)
    view_368: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_33, [1, 16, 128, 128]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_183: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_368, [0, 2, 1, 3]);  view_368 = None
    clone_66: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_369: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_66, [1, 128, 2048]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_370: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_369, [128, 2048]);  view_369 = None
    permute_184: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_216, [1, 0]);  primals_216 = None
    addmm_48: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_217, view_370, permute_184);  primals_217 = None
    view_371: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_48, [1, 128, 2048]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_131: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_371, add_128);  view_371 = add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_33 = torch.ops.aten.var_mean.correction(add_131, [2], correction = 0, keepdim = True)
    getitem_66: "f32[1, 128, 1]" = var_mean_33[0]
    getitem_67: "f32[1, 128, 1]" = var_mean_33[1];  var_mean_33 = None
    add_132: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
    rsqrt_33: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
    sub_50: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_131, getitem_67);  getitem_67 = None
    mul_130: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_33);  sub_50 = None
    mul_131: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_130, primals_218)
    add_133: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_131, primals_219);  mul_131 = primals_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_372: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_133, [128, 2048]);  add_133 = None
    permute_185: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_220, [1, 0]);  primals_220 = None
    addmm_49: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_221, view_372, permute_185);  primals_221 = None
    view_373: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_49, [1, 128, 8192])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_132: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_373, 0.5)
    pow_17: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_373, 3.0)
    mul_133: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_17, 0.044715);  pow_17 = None
    add_134: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_373, mul_133);  view_373 = mul_133 = None
    mul_134: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_134, 0.7978845608028654);  add_134 = None
    tanh_16: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_134);  mul_134 = None
    add_135: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_16, 1.0)
    mul_135: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_132, add_135);  mul_132 = add_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_374: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_135, [128, 8192]);  mul_135 = None
    permute_186: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_222, [1, 0]);  primals_222 = None
    addmm_50: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_223, view_374, permute_186);  primals_223 = None
    view_375: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_50, [1, 128, 2048]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_136: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_131, view_375);  add_131 = view_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_34 = torch.ops.aten.var_mean.correction(add_136, [2], correction = 0, keepdim = True)
    getitem_68: "f32[1, 128, 1]" = var_mean_34[0]
    getitem_69: "f32[1, 128, 1]" = var_mean_34[1];  var_mean_34 = None
    add_137: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
    rsqrt_34: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
    sub_51: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_136, getitem_69);  getitem_69 = None
    mul_136: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_34);  sub_51 = None
    mul_137: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_136, primals_224)
    add_138: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_137, primals_225);  mul_137 = primals_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_187: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_226, [1, 0]);  primals_226 = None
    view_376: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_138, [128, 2048]);  add_138 = None
    mm_51: "f32[128, 2048]" = torch.ops.aten.mm.default(view_376, permute_187)
    view_377: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_51, [1, 128, 2048]);  mm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_188: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_227, [1, 0]);  primals_227 = None
    mm_52: "f32[128, 2048]" = torch.ops.aten.mm.default(view_376, permute_188)
    view_379: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_52, [1, 128, 2048]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_189: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_228, [1, 0]);  primals_228 = None
    mm_53: "f32[128, 2048]" = torch.ops.aten.mm.default(view_376, permute_189)
    view_381: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_53, [1, 128, 2048]);  mm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_382: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_377, [1, 128, 16, 128]);  view_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_190: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_382, [0, 2, 1, 3]);  view_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_383: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_379, [1, 128, 16, 128]);  view_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_191: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_383, [0, 2, 1, 3]);  view_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_384: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_381, [1, 128, 16, 128]);  view_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_192: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_384, [0, 2, 1, 3]);  view_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_193: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_191, [0, 1, 3, 2])
    expand_68: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_190, [1, 16, 128, 128]);  permute_190 = None
    view_385: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_68, [16, 128, 128]);  expand_68 = None
    expand_69: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_193, [1, 16, 128, 128]);  permute_193 = None
    view_386: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_69, [16, 128, 128]);  expand_69 = None
    bmm_34: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_385, view_386)
    view_387: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_34, [1, 16, 128, 128]);  bmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_69: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_335, 0, 0, 9223372036854775807);  primals_335 = None
    slice_70: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_69, 1, 0, 9223372036854775807);  slice_69 = None
    slice_71: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_70, 2, 0, 128);  slice_70 = None
    slice_72: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_71, 3, 0, 128);  slice_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_17: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_72, view_387, full_default);  view_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_17: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_17, [-1], True)
    sub_52: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_17, amax_17);  where_17 = amax_17 = None
    exp_17: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
    sum_18: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
    div_17: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
    alias_34: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_69: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_17);  div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_70: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_69, [1, 16, 128, 128]);  clone_69 = None
    view_388: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_70, [16, 128, 128]);  expand_70 = None
    expand_71: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_192, [1, 16, 128, 128])
    view_389: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_71, [16, 128, 128]);  expand_71 = None
    bmm_35: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_388, view_389)
    view_390: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_35, [1, 16, 128, 128]);  bmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_194: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_390, [0, 2, 1, 3]);  view_390 = None
    clone_70: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_391: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_70, [1, 128, 2048]);  clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_392: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_391, [128, 2048]);  view_391 = None
    permute_195: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_229, [1, 0]);  primals_229 = None
    addmm_51: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_230, view_392, permute_195);  primals_230 = None
    view_393: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_51, [1, 128, 2048]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_139: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_393, add_136);  view_393 = add_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_35 = torch.ops.aten.var_mean.correction(add_139, [2], correction = 0, keepdim = True)
    getitem_70: "f32[1, 128, 1]" = var_mean_35[0]
    getitem_71: "f32[1, 128, 1]" = var_mean_35[1];  var_mean_35 = None
    add_140: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
    rsqrt_35: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    sub_53: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_139, getitem_71);  getitem_71 = None
    mul_138: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_35);  sub_53 = None
    mul_139: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_138, primals_231)
    add_141: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_139, primals_232);  mul_139 = primals_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_394: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_141, [128, 2048]);  add_141 = None
    permute_196: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_233, [1, 0]);  primals_233 = None
    addmm_52: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_234, view_394, permute_196);  primals_234 = None
    view_395: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_52, [1, 128, 8192])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_140: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_395, 0.5)
    pow_18: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_395, 3.0)
    mul_141: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_18, 0.044715);  pow_18 = None
    add_142: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_395, mul_141);  view_395 = mul_141 = None
    mul_142: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_142, 0.7978845608028654);  add_142 = None
    tanh_17: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_142);  mul_142 = None
    add_143: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_17, 1.0)
    mul_143: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_140, add_143);  mul_140 = add_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_396: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_143, [128, 8192]);  mul_143 = None
    permute_197: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_235, [1, 0]);  primals_235 = None
    addmm_53: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_236, view_396, permute_197);  primals_236 = None
    view_397: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_53, [1, 128, 2048]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_144: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_139, view_397);  add_139 = view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_36 = torch.ops.aten.var_mean.correction(add_144, [2], correction = 0, keepdim = True)
    getitem_72: "f32[1, 128, 1]" = var_mean_36[0]
    getitem_73: "f32[1, 128, 1]" = var_mean_36[1];  var_mean_36 = None
    add_145: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
    rsqrt_36: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
    sub_54: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_144, getitem_73);  getitem_73 = None
    mul_144: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_36);  sub_54 = None
    mul_145: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_144, primals_237)
    add_146: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_145, primals_238);  mul_145 = primals_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_198: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_239, [1, 0]);  primals_239 = None
    view_398: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_146, [128, 2048]);  add_146 = None
    mm_54: "f32[128, 2048]" = torch.ops.aten.mm.default(view_398, permute_198)
    view_399: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_54, [1, 128, 2048]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_199: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_240, [1, 0]);  primals_240 = None
    mm_55: "f32[128, 2048]" = torch.ops.aten.mm.default(view_398, permute_199)
    view_401: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_55, [1, 128, 2048]);  mm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_200: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_241, [1, 0]);  primals_241 = None
    mm_56: "f32[128, 2048]" = torch.ops.aten.mm.default(view_398, permute_200)
    view_403: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_56, [1, 128, 2048]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_404: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_399, [1, 128, 16, 128]);  view_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_201: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_404, [0, 2, 1, 3]);  view_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_405: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_401, [1, 128, 16, 128]);  view_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_202: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_405, [0, 2, 1, 3]);  view_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_406: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_403, [1, 128, 16, 128]);  view_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_203: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_406, [0, 2, 1, 3]);  view_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_204: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_202, [0, 1, 3, 2])
    expand_72: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_201, [1, 16, 128, 128]);  permute_201 = None
    view_407: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_72, [16, 128, 128]);  expand_72 = None
    expand_73: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_204, [1, 16, 128, 128]);  permute_204 = None
    view_408: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_73, [16, 128, 128]);  expand_73 = None
    bmm_36: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_407, view_408)
    view_409: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_36, [1, 16, 128, 128]);  bmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_73: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_336, 0, 0, 9223372036854775807);  primals_336 = None
    slice_74: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_73, 1, 0, 9223372036854775807);  slice_73 = None
    slice_75: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_74, 2, 0, 128);  slice_74 = None
    slice_76: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_75, 3, 0, 128);  slice_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_18: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_76, view_409, full_default);  view_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_18: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_18, [-1], True)
    sub_55: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_18, amax_18);  where_18 = amax_18 = None
    exp_18: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
    sum_19: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
    div_18: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
    alias_36: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_73: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_18);  div_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_74: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_73, [1, 16, 128, 128]);  clone_73 = None
    view_410: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_74, [16, 128, 128]);  expand_74 = None
    expand_75: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_203, [1, 16, 128, 128])
    view_411: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_75, [16, 128, 128]);  expand_75 = None
    bmm_37: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_410, view_411)
    view_412: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_37, [1, 16, 128, 128]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_205: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_412, [0, 2, 1, 3]);  view_412 = None
    clone_74: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_413: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_74, [1, 128, 2048]);  clone_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_414: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_413, [128, 2048]);  view_413 = None
    permute_206: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_242, [1, 0]);  primals_242 = None
    addmm_54: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_243, view_414, permute_206);  primals_243 = None
    view_415: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_54, [1, 128, 2048]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_147: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_415, add_144);  view_415 = add_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_37 = torch.ops.aten.var_mean.correction(add_147, [2], correction = 0, keepdim = True)
    getitem_74: "f32[1, 128, 1]" = var_mean_37[0]
    getitem_75: "f32[1, 128, 1]" = var_mean_37[1];  var_mean_37 = None
    add_148: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
    rsqrt_37: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
    sub_56: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_147, getitem_75);  getitem_75 = None
    mul_146: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_37);  sub_56 = None
    mul_147: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_146, primals_244)
    add_149: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_147, primals_245);  mul_147 = primals_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_416: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_149, [128, 2048]);  add_149 = None
    permute_207: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_246, [1, 0]);  primals_246 = None
    addmm_55: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_247, view_416, permute_207);  primals_247 = None
    view_417: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_55, [1, 128, 8192])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_148: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_417, 0.5)
    pow_19: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_417, 3.0)
    mul_149: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_19, 0.044715);  pow_19 = None
    add_150: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_417, mul_149);  view_417 = mul_149 = None
    mul_150: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_150, 0.7978845608028654);  add_150 = None
    tanh_18: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_150);  mul_150 = None
    add_151: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_18, 1.0)
    mul_151: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_148, add_151);  mul_148 = add_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_418: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_151, [128, 8192]);  mul_151 = None
    permute_208: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_248, [1, 0]);  primals_248 = None
    addmm_56: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_249, view_418, permute_208);  primals_249 = None
    view_419: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_56, [1, 128, 2048]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_152: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_147, view_419);  add_147 = view_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_38 = torch.ops.aten.var_mean.correction(add_152, [2], correction = 0, keepdim = True)
    getitem_76: "f32[1, 128, 1]" = var_mean_38[0]
    getitem_77: "f32[1, 128, 1]" = var_mean_38[1];  var_mean_38 = None
    add_153: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
    rsqrt_38: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
    sub_57: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_152, getitem_77);  getitem_77 = None
    mul_152: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_38);  sub_57 = None
    mul_153: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_152, primals_250)
    add_154: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_153, primals_251);  mul_153 = primals_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_209: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_252, [1, 0]);  primals_252 = None
    view_420: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_154, [128, 2048]);  add_154 = None
    mm_57: "f32[128, 2048]" = torch.ops.aten.mm.default(view_420, permute_209)
    view_421: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_57, [1, 128, 2048]);  mm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_210: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_253, [1, 0]);  primals_253 = None
    mm_58: "f32[128, 2048]" = torch.ops.aten.mm.default(view_420, permute_210)
    view_423: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_58, [1, 128, 2048]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_211: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_254, [1, 0]);  primals_254 = None
    mm_59: "f32[128, 2048]" = torch.ops.aten.mm.default(view_420, permute_211)
    view_425: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_59, [1, 128, 2048]);  mm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_426: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_421, [1, 128, 16, 128]);  view_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_212: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_426, [0, 2, 1, 3]);  view_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_427: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_423, [1, 128, 16, 128]);  view_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_213: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_427, [0, 2, 1, 3]);  view_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_428: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_425, [1, 128, 16, 128]);  view_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_214: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_428, [0, 2, 1, 3]);  view_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_215: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_213, [0, 1, 3, 2])
    expand_76: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_212, [1, 16, 128, 128]);  permute_212 = None
    view_429: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_76, [16, 128, 128]);  expand_76 = None
    expand_77: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_215, [1, 16, 128, 128]);  permute_215 = None
    view_430: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_77, [16, 128, 128]);  expand_77 = None
    bmm_38: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_429, view_430)
    view_431: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_38, [1, 16, 128, 128]);  bmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_77: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_337, 0, 0, 9223372036854775807);  primals_337 = None
    slice_78: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_77, 1, 0, 9223372036854775807);  slice_77 = None
    slice_79: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_78, 2, 0, 128);  slice_78 = None
    slice_80: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_79, 3, 0, 128);  slice_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_19: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_80, view_431, full_default);  view_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_19: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_19, [-1], True)
    sub_58: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_19, amax_19);  where_19 = amax_19 = None
    exp_19: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_58);  sub_58 = None
    sum_20: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
    div_19: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
    alias_38: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_77: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_19);  div_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_78: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_77, [1, 16, 128, 128]);  clone_77 = None
    view_432: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_78, [16, 128, 128]);  expand_78 = None
    expand_79: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_214, [1, 16, 128, 128])
    view_433: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_79, [16, 128, 128]);  expand_79 = None
    bmm_39: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_432, view_433)
    view_434: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_39, [1, 16, 128, 128]);  bmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_216: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_434, [0, 2, 1, 3]);  view_434 = None
    clone_78: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_435: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_78, [1, 128, 2048]);  clone_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_436: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_435, [128, 2048]);  view_435 = None
    permute_217: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_255, [1, 0]);  primals_255 = None
    addmm_57: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_256, view_436, permute_217);  primals_256 = None
    view_437: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_57, [1, 128, 2048]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_155: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_437, add_152);  view_437 = add_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_39 = torch.ops.aten.var_mean.correction(add_155, [2], correction = 0, keepdim = True)
    getitem_78: "f32[1, 128, 1]" = var_mean_39[0]
    getitem_79: "f32[1, 128, 1]" = var_mean_39[1];  var_mean_39 = None
    add_156: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
    rsqrt_39: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
    sub_59: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_155, getitem_79);  getitem_79 = None
    mul_154: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_39);  sub_59 = None
    mul_155: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_154, primals_257)
    add_157: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_155, primals_258);  mul_155 = primals_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_438: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_157, [128, 2048]);  add_157 = None
    permute_218: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_259, [1, 0]);  primals_259 = None
    addmm_58: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_260, view_438, permute_218);  primals_260 = None
    view_439: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_58, [1, 128, 8192])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_156: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_439, 0.5)
    pow_20: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_439, 3.0)
    mul_157: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_20, 0.044715);  pow_20 = None
    add_158: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_439, mul_157);  view_439 = mul_157 = None
    mul_158: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_158, 0.7978845608028654);  add_158 = None
    tanh_19: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_158);  mul_158 = None
    add_159: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_19, 1.0)
    mul_159: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_156, add_159);  mul_156 = add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_440: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_159, [128, 8192]);  mul_159 = None
    permute_219: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_261, [1, 0]);  primals_261 = None
    addmm_59: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_262, view_440, permute_219);  primals_262 = None
    view_441: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_59, [1, 128, 2048]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_160: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_155, view_441);  add_155 = view_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_40 = torch.ops.aten.var_mean.correction(add_160, [2], correction = 0, keepdim = True)
    getitem_80: "f32[1, 128, 1]" = var_mean_40[0]
    getitem_81: "f32[1, 128, 1]" = var_mean_40[1];  var_mean_40 = None
    add_161: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
    rsqrt_40: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    sub_60: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_160, getitem_81);  getitem_81 = None
    mul_160: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_40);  sub_60 = None
    mul_161: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_160, primals_263)
    add_162: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_161, primals_264);  mul_161 = primals_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_220: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_265, [1, 0]);  primals_265 = None
    view_442: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_162, [128, 2048]);  add_162 = None
    mm_60: "f32[128, 2048]" = torch.ops.aten.mm.default(view_442, permute_220)
    view_443: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_60, [1, 128, 2048]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_221: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_266, [1, 0]);  primals_266 = None
    mm_61: "f32[128, 2048]" = torch.ops.aten.mm.default(view_442, permute_221)
    view_445: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_61, [1, 128, 2048]);  mm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_222: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_267, [1, 0]);  primals_267 = None
    mm_62: "f32[128, 2048]" = torch.ops.aten.mm.default(view_442, permute_222)
    view_447: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_62, [1, 128, 2048]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_448: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_443, [1, 128, 16, 128]);  view_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_223: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_449: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_445, [1, 128, 16, 128]);  view_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_224: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_449, [0, 2, 1, 3]);  view_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_450: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_447, [1, 128, 16, 128]);  view_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_225: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_450, [0, 2, 1, 3]);  view_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_226: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_224, [0, 1, 3, 2])
    expand_80: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_223, [1, 16, 128, 128]);  permute_223 = None
    view_451: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_80, [16, 128, 128]);  expand_80 = None
    expand_81: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_226, [1, 16, 128, 128]);  permute_226 = None
    view_452: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_81, [16, 128, 128]);  expand_81 = None
    bmm_40: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_451, view_452)
    view_453: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_40, [1, 16, 128, 128]);  bmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_81: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_338, 0, 0, 9223372036854775807);  primals_338 = None
    slice_82: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_81, 1, 0, 9223372036854775807);  slice_81 = None
    slice_83: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_82, 2, 0, 128);  slice_82 = None
    slice_84: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_83, 3, 0, 128);  slice_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_20: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_84, view_453, full_default);  view_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_20: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_20, [-1], True)
    sub_61: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_20, amax_20);  where_20 = amax_20 = None
    exp_20: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_61);  sub_61 = None
    sum_21: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
    div_20: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
    alias_40: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_81: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_20);  div_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_82: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_81, [1, 16, 128, 128]);  clone_81 = None
    view_454: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_82, [16, 128, 128]);  expand_82 = None
    expand_83: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_225, [1, 16, 128, 128])
    view_455: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_83, [16, 128, 128]);  expand_83 = None
    bmm_41: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_454, view_455)
    view_456: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_41, [1, 16, 128, 128]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_227: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_456, [0, 2, 1, 3]);  view_456 = None
    clone_82: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_457: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_82, [1, 128, 2048]);  clone_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_458: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_457, [128, 2048]);  view_457 = None
    permute_228: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_268, [1, 0]);  primals_268 = None
    addmm_60: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_269, view_458, permute_228);  primals_269 = None
    view_459: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_60, [1, 128, 2048]);  addmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_163: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_459, add_160);  view_459 = add_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_41 = torch.ops.aten.var_mean.correction(add_163, [2], correction = 0, keepdim = True)
    getitem_82: "f32[1, 128, 1]" = var_mean_41[0]
    getitem_83: "f32[1, 128, 1]" = var_mean_41[1];  var_mean_41 = None
    add_164: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
    rsqrt_41: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
    sub_62: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_163, getitem_83);  getitem_83 = None
    mul_162: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_41);  sub_62 = None
    mul_163: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_162, primals_270)
    add_165: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_163, primals_271);  mul_163 = primals_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_460: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_165, [128, 2048]);  add_165 = None
    permute_229: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_272, [1, 0]);  primals_272 = None
    addmm_61: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_273, view_460, permute_229);  primals_273 = None
    view_461: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_61, [1, 128, 8192])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_164: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_461, 0.5)
    pow_21: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_461, 3.0)
    mul_165: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
    add_166: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_461, mul_165);  view_461 = mul_165 = None
    mul_166: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_166, 0.7978845608028654);  add_166 = None
    tanh_20: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_166);  mul_166 = None
    add_167: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_20, 1.0)
    mul_167: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_164, add_167);  mul_164 = add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_462: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_167, [128, 8192]);  mul_167 = None
    permute_230: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_274, [1, 0]);  primals_274 = None
    addmm_62: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_275, view_462, permute_230);  primals_275 = None
    view_463: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_62, [1, 128, 2048]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_168: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_163, view_463);  add_163 = view_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_42 = torch.ops.aten.var_mean.correction(add_168, [2], correction = 0, keepdim = True)
    getitem_84: "f32[1, 128, 1]" = var_mean_42[0]
    getitem_85: "f32[1, 128, 1]" = var_mean_42[1];  var_mean_42 = None
    add_169: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05);  getitem_84 = None
    rsqrt_42: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_169);  add_169 = None
    sub_63: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_168, getitem_85);  getitem_85 = None
    mul_168: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_42);  sub_63 = None
    mul_169: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_168, primals_276)
    add_170: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_169, primals_277);  mul_169 = primals_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_231: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_278, [1, 0]);  primals_278 = None
    view_464: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_170, [128, 2048]);  add_170 = None
    mm_63: "f32[128, 2048]" = torch.ops.aten.mm.default(view_464, permute_231)
    view_465: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_63, [1, 128, 2048]);  mm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_232: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_279, [1, 0]);  primals_279 = None
    mm_64: "f32[128, 2048]" = torch.ops.aten.mm.default(view_464, permute_232)
    view_467: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_64, [1, 128, 2048]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_233: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_280, [1, 0]);  primals_280 = None
    mm_65: "f32[128, 2048]" = torch.ops.aten.mm.default(view_464, permute_233)
    view_469: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_65, [1, 128, 2048]);  mm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_470: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_465, [1, 128, 16, 128]);  view_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_234: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_470, [0, 2, 1, 3]);  view_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_471: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_467, [1, 128, 16, 128]);  view_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_235: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_471, [0, 2, 1, 3]);  view_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_472: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_469, [1, 128, 16, 128]);  view_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_236: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_472, [0, 2, 1, 3]);  view_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_237: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_235, [0, 1, 3, 2])
    expand_84: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_234, [1, 16, 128, 128]);  permute_234 = None
    view_473: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_84, [16, 128, 128]);  expand_84 = None
    expand_85: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_237, [1, 16, 128, 128]);  permute_237 = None
    view_474: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_85, [16, 128, 128]);  expand_85 = None
    bmm_42: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_473, view_474)
    view_475: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_42, [1, 16, 128, 128]);  bmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_85: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_339, 0, 0, 9223372036854775807);  primals_339 = None
    slice_86: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_85, 1, 0, 9223372036854775807);  slice_85 = None
    slice_87: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_86, 2, 0, 128);  slice_86 = None
    slice_88: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_87, 3, 0, 128);  slice_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_21: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_88, view_475, full_default);  view_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_21: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_21, [-1], True)
    sub_64: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_21, amax_21);  where_21 = amax_21 = None
    exp_21: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_64);  sub_64 = None
    sum_22: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
    div_21: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
    alias_42: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_85: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_21);  div_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_86: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_85, [1, 16, 128, 128]);  clone_85 = None
    view_476: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_86, [16, 128, 128]);  expand_86 = None
    expand_87: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_236, [1, 16, 128, 128])
    view_477: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_87, [16, 128, 128]);  expand_87 = None
    bmm_43: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_476, view_477)
    view_478: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_43, [1, 16, 128, 128]);  bmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_238: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_478, [0, 2, 1, 3]);  view_478 = None
    clone_86: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_479: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_86, [1, 128, 2048]);  clone_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_480: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_479, [128, 2048]);  view_479 = None
    permute_239: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_281, [1, 0]);  primals_281 = None
    addmm_63: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_282, view_480, permute_239);  primals_282 = None
    view_481: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_63, [1, 128, 2048]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_171: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_481, add_168);  view_481 = add_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_43 = torch.ops.aten.var_mean.correction(add_171, [2], correction = 0, keepdim = True)
    getitem_86: "f32[1, 128, 1]" = var_mean_43[0]
    getitem_87: "f32[1, 128, 1]" = var_mean_43[1];  var_mean_43 = None
    add_172: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
    rsqrt_43: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
    sub_65: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_171, getitem_87);  getitem_87 = None
    mul_170: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_43);  sub_65 = None
    mul_171: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_170, primals_283)
    add_173: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_171, primals_284);  mul_171 = primals_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_482: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_173, [128, 2048]);  add_173 = None
    permute_240: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_285, [1, 0]);  primals_285 = None
    addmm_64: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_286, view_482, permute_240);  primals_286 = None
    view_483: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_64, [1, 128, 8192])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_172: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_483, 0.5)
    pow_22: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_483, 3.0)
    mul_173: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_22, 0.044715);  pow_22 = None
    add_174: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_483, mul_173);  view_483 = mul_173 = None
    mul_174: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_174, 0.7978845608028654);  add_174 = None
    tanh_21: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_174);  mul_174 = None
    add_175: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_21, 1.0)
    mul_175: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_172, add_175);  mul_172 = add_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_484: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_175, [128, 8192]);  mul_175 = None
    permute_241: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_287, [1, 0]);  primals_287 = None
    addmm_65: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_288, view_484, permute_241);  primals_288 = None
    view_485: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_65, [1, 128, 2048]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_176: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_171, view_485);  add_171 = view_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_44 = torch.ops.aten.var_mean.correction(add_176, [2], correction = 0, keepdim = True)
    getitem_88: "f32[1, 128, 1]" = var_mean_44[0]
    getitem_89: "f32[1, 128, 1]" = var_mean_44[1];  var_mean_44 = None
    add_177: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
    rsqrt_44: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_177);  add_177 = None
    sub_66: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_176, getitem_89);  getitem_89 = None
    mul_176: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_44);  sub_66 = None
    mul_177: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_176, primals_289)
    add_178: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_177, primals_290);  mul_177 = primals_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_242: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_291, [1, 0]);  primals_291 = None
    view_486: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_178, [128, 2048]);  add_178 = None
    mm_66: "f32[128, 2048]" = torch.ops.aten.mm.default(view_486, permute_242)
    view_487: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_66, [1, 128, 2048]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_243: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_292, [1, 0]);  primals_292 = None
    mm_67: "f32[128, 2048]" = torch.ops.aten.mm.default(view_486, permute_243)
    view_489: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_67, [1, 128, 2048]);  mm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_244: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_293, [1, 0]);  primals_293 = None
    mm_68: "f32[128, 2048]" = torch.ops.aten.mm.default(view_486, permute_244)
    view_491: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_68, [1, 128, 2048]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_492: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_487, [1, 128, 16, 128]);  view_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_245: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_492, [0, 2, 1, 3]);  view_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_493: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_489, [1, 128, 16, 128]);  view_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_246: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_493, [0, 2, 1, 3]);  view_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_494: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_491, [1, 128, 16, 128]);  view_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_247: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_494, [0, 2, 1, 3]);  view_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_248: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_246, [0, 1, 3, 2])
    expand_88: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_245, [1, 16, 128, 128]);  permute_245 = None
    view_495: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_88, [16, 128, 128]);  expand_88 = None
    expand_89: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_248, [1, 16, 128, 128]);  permute_248 = None
    view_496: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_89, [16, 128, 128]);  expand_89 = None
    bmm_44: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_495, view_496)
    view_497: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_44, [1, 16, 128, 128]);  bmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_89: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_340, 0, 0, 9223372036854775807);  primals_340 = None
    slice_90: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_89, 1, 0, 9223372036854775807);  slice_89 = None
    slice_91: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_90, 2, 0, 128);  slice_90 = None
    slice_92: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_91, 3, 0, 128);  slice_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_22: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_92, view_497, full_default);  view_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_22: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_22, [-1], True)
    sub_67: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_22, amax_22);  where_22 = amax_22 = None
    exp_22: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_67);  sub_67 = None
    sum_23: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
    div_22: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
    alias_44: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_89: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_22);  div_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_90: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_89, [1, 16, 128, 128]);  clone_89 = None
    view_498: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_90, [16, 128, 128]);  expand_90 = None
    expand_91: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_247, [1, 16, 128, 128])
    view_499: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_91, [16, 128, 128]);  expand_91 = None
    bmm_45: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_498, view_499)
    view_500: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_45, [1, 16, 128, 128]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_249: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_500, [0, 2, 1, 3]);  view_500 = None
    clone_90: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_501: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_90, [1, 128, 2048]);  clone_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_502: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_501, [128, 2048]);  view_501 = None
    permute_250: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_294, [1, 0]);  primals_294 = None
    addmm_66: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_295, view_502, permute_250);  primals_295 = None
    view_503: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_66, [1, 128, 2048]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_179: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_503, add_176);  view_503 = add_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_45 = torch.ops.aten.var_mean.correction(add_179, [2], correction = 0, keepdim = True)
    getitem_90: "f32[1, 128, 1]" = var_mean_45[0]
    getitem_91: "f32[1, 128, 1]" = var_mean_45[1];  var_mean_45 = None
    add_180: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
    rsqrt_45: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_180);  add_180 = None
    sub_68: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_179, getitem_91);  getitem_91 = None
    mul_178: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_45);  sub_68 = None
    mul_179: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_178, primals_296)
    add_181: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_179, primals_297);  mul_179 = primals_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_504: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_181, [128, 2048]);  add_181 = None
    permute_251: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_298, [1, 0]);  primals_298 = None
    addmm_67: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_299, view_504, permute_251);  primals_299 = None
    view_505: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_67, [1, 128, 8192])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_180: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_505, 0.5)
    pow_23: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_505, 3.0)
    mul_181: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_23, 0.044715);  pow_23 = None
    add_182: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_505, mul_181);  view_505 = mul_181 = None
    mul_182: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_182, 0.7978845608028654);  add_182 = None
    tanh_22: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_182);  mul_182 = None
    add_183: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_22, 1.0)
    mul_183: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_180, add_183);  mul_180 = add_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_506: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_183, [128, 8192]);  mul_183 = None
    permute_252: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_300, [1, 0]);  primals_300 = None
    addmm_68: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_301, view_506, permute_252);  primals_301 = None
    view_507: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_68, [1, 128, 2048]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_184: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_179, view_507);  add_179 = view_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_46 = torch.ops.aten.var_mean.correction(add_184, [2], correction = 0, keepdim = True)
    getitem_92: "f32[1, 128, 1]" = var_mean_46[0]
    getitem_93: "f32[1, 128, 1]" = var_mean_46[1];  var_mean_46 = None
    add_185: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
    rsqrt_46: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
    sub_69: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_184, getitem_93);  getitem_93 = None
    mul_184: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_46);  sub_69 = None
    mul_185: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_184, primals_302)
    add_186: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_185, primals_303);  mul_185 = primals_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_253: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_304, [1, 0]);  primals_304 = None
    view_508: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_186, [128, 2048]);  add_186 = None
    mm_69: "f32[128, 2048]" = torch.ops.aten.mm.default(view_508, permute_253)
    view_509: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_69, [1, 128, 2048]);  mm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_254: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_305, [1, 0]);  primals_305 = None
    mm_70: "f32[128, 2048]" = torch.ops.aten.mm.default(view_508, permute_254)
    view_511: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_70, [1, 128, 2048]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_255: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_306, [1, 0]);  primals_306 = None
    mm_71: "f32[128, 2048]" = torch.ops.aten.mm.default(view_508, permute_255)
    view_513: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_71, [1, 128, 2048]);  mm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_514: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_509, [1, 128, 16, 128]);  view_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_256: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_514, [0, 2, 1, 3]);  view_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_515: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_511, [1, 128, 16, 128]);  view_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_257: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_515, [0, 2, 1, 3]);  view_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_516: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_513, [1, 128, 16, 128]);  view_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_258: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_516, [0, 2, 1, 3]);  view_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_259: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_257, [0, 1, 3, 2])
    expand_92: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_256, [1, 16, 128, 128]);  permute_256 = None
    view_517: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_92, [16, 128, 128]);  expand_92 = None
    expand_93: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_259, [1, 16, 128, 128]);  permute_259 = None
    view_518: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_93, [16, 128, 128]);  expand_93 = None
    bmm_46: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_517, view_518)
    view_519: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_46, [1, 16, 128, 128]);  bmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_93: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_341, 0, 0, 9223372036854775807);  primals_341 = None
    slice_94: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_93, 1, 0, 9223372036854775807);  slice_93 = None
    slice_95: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_94, 2, 0, 128);  slice_94 = None
    slice_96: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_95, 3, 0, 128);  slice_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_23: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_96, view_519, full_default);  view_519 = full_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_23: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_23, [-1], True)
    sub_70: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_23, amax_23);  where_23 = amax_23 = None
    exp_23: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_70);  sub_70 = None
    sum_24: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
    div_23: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
    alias_46: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_93: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_23);  div_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_94: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_93, [1, 16, 128, 128]);  clone_93 = None
    view_520: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_94, [16, 128, 128]);  expand_94 = None
    expand_95: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_258, [1, 16, 128, 128])
    view_521: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_95, [16, 128, 128]);  expand_95 = None
    bmm_47: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_520, view_521)
    view_522: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_47, [1, 16, 128, 128]);  bmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_260: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_522, [0, 2, 1, 3]);  view_522 = None
    clone_94: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_523: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_94, [1, 128, 2048]);  clone_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_524: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_523, [128, 2048]);  view_523 = None
    permute_261: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_307, [1, 0]);  primals_307 = None
    addmm_69: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_308, view_524, permute_261);  primals_308 = None
    view_525: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_69, [1, 128, 2048]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_187: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_525, add_184);  view_525 = add_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_47 = torch.ops.aten.var_mean.correction(add_187, [2], correction = 0, keepdim = True)
    getitem_94: "f32[1, 128, 1]" = var_mean_47[0]
    getitem_95: "f32[1, 128, 1]" = var_mean_47[1];  var_mean_47 = None
    add_188: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
    rsqrt_47: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
    sub_71: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_187, getitem_95);  getitem_95 = None
    mul_186: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_47);  sub_71 = None
    mul_187: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_186, primals_309)
    add_189: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_187, primals_310);  mul_187 = primals_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_526: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_189, [128, 2048]);  add_189 = None
    permute_262: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_311, [1, 0]);  primals_311 = None
    addmm_70: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_312, view_526, permute_262);  primals_312 = None
    view_527: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_70, [1, 128, 8192])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_188: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_527, 0.5)
    pow_24: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_527, 3.0)
    mul_189: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_24, 0.044715);  pow_24 = None
    add_190: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_527, mul_189);  view_527 = mul_189 = None
    mul_190: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_190, 0.7978845608028654);  add_190 = None
    tanh_23: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_190);  mul_190 = None
    add_191: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_23, 1.0)
    mul_191: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_188, add_191);  mul_188 = add_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_528: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_191, [128, 8192]);  mul_191 = None
    permute_263: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_313, [1, 0]);  primals_313 = None
    addmm_71: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_314, view_528, permute_263);  primals_314 = None
    view_529: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_71, [1, 128, 2048]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_192: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_187, view_529);  add_187 = view_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:641, code: hidden_states = self.ln_f(hidden_states)
    var_mean_48 = torch.ops.aten.var_mean.correction(add_192, [2], correction = 0, keepdim = True)
    getitem_96: "f32[1, 128, 1]" = var_mean_48[0]
    getitem_97: "f32[1, 128, 1]" = var_mean_48[1];  var_mean_48 = None
    add_193: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
    rsqrt_48: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_193);  add_193 = None
    sub_72: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_192, getitem_97);  add_192 = getitem_97 = None
    mul_192: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_48);  sub_72 = None
    mul_193: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_192, primals_315)
    add_194: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_193, primals_316);  mul_193 = primals_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:643, code: hidden_states = hidden_states.view(output_shape)
    view_530: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(add_194, [-1, 128, 2048]);  add_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:878, code: logits = self.score(hidden_states)
    permute_264: "f32[2048, 2]" = torch.ops.aten.permute.default(primals_317, [1, 0]);  primals_317 = None
    view_531: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_530, [128, 2048])
    mm_72: "f32[128, 2]" = torch.ops.aten.mm.default(view_531, permute_264)
    view_532: "f32[1, 128, 2]" = torch.ops.aten.reshape.default(mm_72, [1, 128, 2]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:891, code: sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(
    eq: "b8[1, 128]" = torch.ops.aten.eq.Scalar(primals_342, 0);  primals_342 = None
    convert_element_type: "i64[1, 128]" = torch.ops.prims.convert_element_type.default(eq, torch.int64);  eq = None
    argmax: "i64[1]" = torch.ops.aten.argmax.default(convert_element_type, -1);  convert_element_type = None
    sub_73: "i64[1]" = torch.ops.aten.sub.Tensor(argmax, 1);  argmax = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:901, code: pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
    full_default_24: "i64[1]" = torch.ops.aten.full.default([1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    index: "f32[1, 2]" = torch.ops.aten.index.Tensor(view_532, [full_default_24, sub_73]);  view_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:878, code: logits = self.score(hidden_states)
    permute_267: "f32[2, 2048]" = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:641, code: hidden_states = self.ln_f(hidden_states)
    div_24: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_48, 2048);  rsqrt_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    permute_269: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    permute_273: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    div_25: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_47, 2048);  rsqrt_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    permute_277: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_261, [1, 0]);  permute_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    permute_282: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_520, [0, 2, 1]);  view_520 = None
    permute_283: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_521, [0, 2, 1]);  view_521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_49: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_46);  alias_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_284: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_517, [0, 2, 1]);  view_517 = None
    permute_285: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_518, [0, 2, 1]);  view_518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_292: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_255, [1, 0]);  permute_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_296: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_254, [1, 0]);  permute_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_300: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_253, [1, 0]);  permute_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    div_26: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_46, 2048);  rsqrt_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    permute_302: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    permute_306: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    div_27: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_45, 2048);  rsqrt_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    permute_310: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_250, [1, 0]);  permute_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    permute_315: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_498, [0, 2, 1]);  view_498 = None
    permute_316: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_499, [0, 2, 1]);  view_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_51: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_44);  alias_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_317: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_495, [0, 2, 1]);  view_495 = None
    permute_318: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_496, [0, 2, 1]);  view_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_325: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_329: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_333: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    div_28: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 2048);  rsqrt_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    permute_335: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_241, [1, 0]);  permute_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    permute_339: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    div_29: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 2048);  rsqrt_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    permute_343: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    permute_348: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_476, [0, 2, 1]);  view_476 = None
    permute_349: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_477, [0, 2, 1]);  view_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_53: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_42);  alias_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_350: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_473, [0, 2, 1]);  view_473 = None
    permute_351: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_474, [0, 2, 1]);  view_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_358: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_362: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_366: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_231, [1, 0]);  permute_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    div_30: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_42, 2048);  rsqrt_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    permute_368: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_230, [1, 0]);  permute_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    permute_372: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    div_31: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_41, 2048);  rsqrt_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    permute_376: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    permute_381: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_454, [0, 2, 1]);  view_454 = None
    permute_382: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_455, [0, 2, 1]);  view_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_55: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_40);  alias_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_383: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_451, [0, 2, 1]);  view_451 = None
    permute_384: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_452, [0, 2, 1]);  view_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_391: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_395: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_221, [1, 0]);  permute_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_399: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    div_32: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 2048);  rsqrt_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    permute_401: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_219, [1, 0]);  permute_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    permute_405: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    div_33: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 2048);  rsqrt_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    permute_409: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_217, [1, 0]);  permute_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    permute_414: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_432, [0, 2, 1]);  view_432 = None
    permute_415: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_433, [0, 2, 1]);  view_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_57: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_38);  alias_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_416: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_429, [0, 2, 1]);  view_429 = None
    permute_417: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_430, [0, 2, 1]);  view_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_424: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_428: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_432: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_209, [1, 0]);  permute_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    div_34: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 2048);  rsqrt_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    permute_434: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_208, [1, 0]);  permute_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    permute_438: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    div_35: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 2048);  rsqrt_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    permute_442: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    permute_447: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_410, [0, 2, 1]);  view_410 = None
    permute_448: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_411, [0, 2, 1]);  view_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_59: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_36);  alias_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_449: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_407, [0, 2, 1]);  view_407 = None
    permute_450: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_408, [0, 2, 1]);  view_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_457: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_461: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_199, [1, 0]);  permute_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_465: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_198, [1, 0]);  permute_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    div_36: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 2048);  rsqrt_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    permute_467: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    permute_471: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    div_37: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 2048);  rsqrt_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    permute_475: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_195, [1, 0]);  permute_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    permute_480: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_388, [0, 2, 1]);  view_388 = None
    permute_481: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_389, [0, 2, 1]);  view_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_61: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_34);  alias_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_482: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_385, [0, 2, 1]);  view_385 = None
    permute_483: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_386, [0, 2, 1]);  view_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_490: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_189, [1, 0]);  permute_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_494: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_188, [1, 0]);  permute_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_498: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    div_38: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 2048);  rsqrt_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    permute_500: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    permute_504: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    div_39: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 2048);  rsqrt_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    permute_508: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    permute_513: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_366, [0, 2, 1]);  view_366 = None
    permute_514: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_367, [0, 2, 1]);  view_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_63: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_32);  alias_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_515: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_363, [0, 2, 1]);  view_363 = None
    permute_516: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_364, [0, 2, 1]);  view_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_523: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_527: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_531: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    div_40: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 2048);  rsqrt_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    permute_533: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_175, [1, 0]);  permute_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    permute_537: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    div_41: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 2048);  rsqrt_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    permute_541: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    permute_546: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_344, [0, 2, 1]);  view_344 = None
    permute_547: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_345, [0, 2, 1]);  view_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_65: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_30);  alias_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_548: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_341, [0, 2, 1]);  view_341 = None
    permute_549: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_342, [0, 2, 1]);  view_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_556: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_560: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_166, [1, 0]);  permute_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_564: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    div_42: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 2048);  rsqrt_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    permute_566: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    permute_570: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    div_43: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 2048);  rsqrt_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    permute_574: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_162, [1, 0]);  permute_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    permute_579: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_322, [0, 2, 1]);  view_322 = None
    permute_580: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_323, [0, 2, 1]);  view_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_67: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_28);  alias_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_581: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_319, [0, 2, 1]);  view_319 = None
    permute_582: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_320, [0, 2, 1]);  view_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_589: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_593: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_155, [1, 0]);  permute_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_597: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    div_44: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 2048);  rsqrt_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    permute_599: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    permute_603: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    div_45: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 2048);  rsqrt_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    permute_607: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    permute_612: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_300, [0, 2, 1]);  view_300 = None
    permute_613: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_301, [0, 2, 1]);  view_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_69: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_26);  alias_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_614: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_297, [0, 2, 1]);  view_297 = None
    permute_615: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_298, [0, 2, 1]);  view_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_622: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_626: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_630: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    div_46: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 2048);  rsqrt_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    permute_632: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    permute_636: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    div_47: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 2048);  rsqrt_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    permute_640: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    permute_645: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_278, [0, 2, 1]);  view_278 = None
    permute_646: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_279, [0, 2, 1]);  view_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_71: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_24);  alias_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_647: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_275, [0, 2, 1]);  view_275 = None
    permute_648: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_276, [0, 2, 1]);  view_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_655: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_659: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_663: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    div_48: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 2048);  rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    permute_665: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    permute_669: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    div_49: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 2048);  rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    permute_673: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    permute_678: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_256, [0, 2, 1]);  view_256 = None
    permute_679: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_257, [0, 2, 1]);  view_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_73: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_22);  alias_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_680: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_253, [0, 2, 1]);  view_253 = None
    permute_681: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_688: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_692: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_696: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    div_50: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 2048);  rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    permute_698: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    permute_702: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    div_51: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 2048);  rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    permute_706: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    permute_711: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_234, [0, 2, 1]);  view_234 = None
    permute_712: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_235, [0, 2, 1]);  view_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_75: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_20);  alias_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_713: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_231, [0, 2, 1]);  view_231 = None
    permute_714: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_721: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_725: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_729: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    div_52: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 2048);  rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    permute_731: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    permute_735: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    div_53: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 2048);  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    permute_739: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    permute_744: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_212, [0, 2, 1]);  view_212 = None
    permute_745: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_213, [0, 2, 1]);  view_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_77: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_18);  alias_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_746: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_209, [0, 2, 1]);  view_209 = None
    permute_747: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_754: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_758: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_762: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    div_54: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 2048);  rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    permute_764: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    permute_768: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    div_55: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 2048);  rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    permute_772: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    permute_777: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_190, [0, 2, 1]);  view_190 = None
    permute_778: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_191, [0, 2, 1]);  view_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_79: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_16);  alias_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_779: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_187, [0, 2, 1]);  view_187 = None
    permute_780: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_787: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_791: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_795: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    div_56: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 2048);  rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    permute_797: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    permute_801: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    div_57: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 2048);  rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    permute_805: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    permute_810: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_168, [0, 2, 1]);  view_168 = None
    permute_811: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_169, [0, 2, 1]);  view_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_81: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_812: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_165, [0, 2, 1]);  view_165 = None
    permute_813: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_820: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_824: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_828: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    div_58: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 2048);  rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    permute_830: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    permute_834: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    div_59: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 2048);  rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    permute_838: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    permute_843: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_146, [0, 2, 1]);  view_146 = None
    permute_844: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_147, [0, 2, 1]);  view_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_83: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_845: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_143, [0, 2, 1]);  view_143 = None
    permute_846: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_853: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_857: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_861: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    div_60: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 2048);  rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    permute_863: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    permute_867: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    div_61: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 2048);  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    permute_871: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    permute_876: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_124, [0, 2, 1]);  view_124 = None
    permute_877: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_125, [0, 2, 1]);  view_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_85: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_878: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_121, [0, 2, 1]);  view_121 = None
    permute_879: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_886: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_890: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_894: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    div_62: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 2048);  rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    permute_896: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    permute_900: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    div_63: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 2048);  rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    permute_904: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    permute_909: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_102, [0, 2, 1]);  view_102 = None
    permute_910: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_103, [0, 2, 1]);  view_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_87: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_911: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_99, [0, 2, 1]);  view_99 = None
    permute_912: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_919: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_923: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_927: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    div_64: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 2048);  rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    permute_929: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    permute_933: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    div_65: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 2048);  rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    permute_937: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    permute_942: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_80, [0, 2, 1]);  view_80 = None
    permute_943: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_81, [0, 2, 1]);  view_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_89: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_944: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_77, [0, 2, 1]);  view_77 = None
    permute_945: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_952: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_956: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_960: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    div_66: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 2048);  rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    permute_962: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    permute_966: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    div_67: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 2048);  rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    permute_970: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    permute_975: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_58, [0, 2, 1]);  view_58 = None
    permute_976: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_59, [0, 2, 1]);  view_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_91: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_977: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_55, [0, 2, 1]);  view_55 = None
    permute_978: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_985: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_989: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_993: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    div_68: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 2048);  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    permute_995: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    permute_999: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    div_69: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 2048);  rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    permute_1003: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    permute_1008: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_36, [0, 2, 1]);  view_36 = None
    permute_1009: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_37, [0, 2, 1]);  view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_93: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_1010: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None
    permute_1011: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_1018: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_1022: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_1026: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    div_70: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 2048);  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    permute_1028: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    permute_1032: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    div_71: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 2048);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    permute_1036: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    permute_1041: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_14, [0, 2, 1]);  view_14 = None
    permute_1042: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_15, [0, 2, 1]);  view_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_95: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias);  alias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_1043: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
    permute_1044: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_1051: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_1055: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_1059: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    div_72: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt, 2048);  rsqrt = None
    return [view_530, permute_4, permute_5, permute_15, permute_16, permute_26, permute_27, permute_37, permute_38, permute_48, permute_49, permute_59, permute_60, permute_70, permute_71, permute_81, permute_82, permute_92, permute_93, permute_103, permute_104, permute_114, permute_115, permute_125, permute_126, permute_136, permute_137, permute_147, permute_148, permute_158, permute_159, permute_169, permute_170, permute_180, permute_181, permute_191, permute_192, permute_202, permute_203, permute_213, permute_214, permute_224, permute_225, permute_235, permute_236, permute_246, permute_247, permute_257, permute_258, index, primals_3, primals_10, primals_16, primals_23, primals_29, primals_36, primals_42, primals_49, primals_55, primals_62, primals_68, primals_75, primals_81, primals_88, primals_94, primals_101, primals_107, primals_114, primals_120, primals_127, primals_133, primals_140, primals_146, primals_153, primals_159, primals_166, primals_172, primals_179, primals_185, primals_192, primals_198, primals_205, primals_211, primals_218, primals_224, primals_231, primals_237, primals_244, primals_250, primals_257, primals_263, primals_270, primals_276, primals_283, primals_289, primals_296, primals_302, primals_309, primals_315, view, view_1, mul, view_2, slice_4, view_18, mul_2, view_20, addmm_1, tanh, view_22, mul_8, view_24, slice_8, view_40, mul_10, view_42, addmm_4, tanh_1, view_44, mul_16, view_46, slice_12, view_62, mul_18, view_64, addmm_7, tanh_2, view_66, mul_24, view_68, slice_16, view_84, mul_26, view_86, addmm_10, tanh_3, view_88, mul_32, view_90, slice_20, view_106, mul_34, view_108, addmm_13, tanh_4, view_110, mul_40, view_112, slice_24, view_128, mul_42, view_130, addmm_16, tanh_5, view_132, mul_48, view_134, slice_28, view_150, mul_50, view_152, addmm_19, tanh_6, view_154, mul_56, view_156, slice_32, view_172, mul_58, view_174, addmm_22, tanh_7, view_176, mul_64, view_178, slice_36, view_194, mul_66, view_196, addmm_25, tanh_8, view_198, mul_72, view_200, slice_40, view_216, mul_74, view_218, addmm_28, tanh_9, view_220, mul_80, view_222, slice_44, view_238, mul_82, view_240, addmm_31, tanh_10, view_242, mul_88, view_244, slice_48, view_260, mul_90, view_262, addmm_34, tanh_11, view_264, mul_96, view_266, slice_52, view_282, mul_98, view_284, addmm_37, tanh_12, view_286, mul_104, view_288, slice_56, view_304, mul_106, view_306, addmm_40, tanh_13, view_308, mul_112, view_310, slice_60, view_326, mul_114, view_328, addmm_43, tanh_14, view_330, mul_120, view_332, slice_64, view_348, mul_122, view_350, addmm_46, tanh_15, view_352, mul_128, view_354, slice_68, view_370, mul_130, view_372, addmm_49, tanh_16, view_374, mul_136, view_376, slice_72, view_392, mul_138, view_394, addmm_52, tanh_17, view_396, mul_144, view_398, slice_76, view_414, mul_146, view_416, addmm_55, tanh_18, view_418, mul_152, view_420, slice_80, view_436, mul_154, view_438, addmm_58, tanh_19, view_440, mul_160, view_442, slice_84, view_458, mul_162, view_460, addmm_61, tanh_20, view_462, mul_168, view_464, slice_88, view_480, mul_170, view_482, addmm_64, tanh_21, view_484, mul_176, view_486, slice_92, view_502, mul_178, view_504, addmm_67, tanh_22, view_506, mul_184, view_508, slice_96, view_524, mul_186, view_526, addmm_70, tanh_23, view_528, mul_192, view_531, sub_73, full_default_24, permute_267, div_24, permute_269, permute_273, div_25, permute_277, permute_282, permute_283, alias_49, permute_284, permute_285, permute_292, permute_296, permute_300, div_26, permute_302, permute_306, div_27, permute_310, permute_315, permute_316, alias_51, permute_317, permute_318, permute_325, permute_329, permute_333, div_28, permute_335, permute_339, div_29, permute_343, permute_348, permute_349, alias_53, permute_350, permute_351, permute_358, permute_362, permute_366, div_30, permute_368, permute_372, div_31, permute_376, permute_381, permute_382, alias_55, permute_383, permute_384, permute_391, permute_395, permute_399, div_32, permute_401, permute_405, div_33, permute_409, permute_414, permute_415, alias_57, permute_416, permute_417, permute_424, permute_428, permute_432, div_34, permute_434, permute_438, div_35, permute_442, permute_447, permute_448, alias_59, permute_449, permute_450, permute_457, permute_461, permute_465, div_36, permute_467, permute_471, div_37, permute_475, permute_480, permute_481, alias_61, permute_482, permute_483, permute_490, permute_494, permute_498, div_38, permute_500, permute_504, div_39, permute_508, permute_513, permute_514, alias_63, permute_515, permute_516, permute_523, permute_527, permute_531, div_40, permute_533, permute_537, div_41, permute_541, permute_546, permute_547, alias_65, permute_548, permute_549, permute_556, permute_560, permute_564, div_42, permute_566, permute_570, div_43, permute_574, permute_579, permute_580, alias_67, permute_581, permute_582, permute_589, permute_593, permute_597, div_44, permute_599, permute_603, div_45, permute_607, permute_612, permute_613, alias_69, permute_614, permute_615, permute_622, permute_626, permute_630, div_46, permute_632, permute_636, div_47, permute_640, permute_645, permute_646, alias_71, permute_647, permute_648, permute_655, permute_659, permute_663, div_48, permute_665, permute_669, div_49, permute_673, permute_678, permute_679, alias_73, permute_680, permute_681, permute_688, permute_692, permute_696, div_50, permute_698, permute_702, div_51, permute_706, permute_711, permute_712, alias_75, permute_713, permute_714, permute_721, permute_725, permute_729, div_52, permute_731, permute_735, div_53, permute_739, permute_744, permute_745, alias_77, permute_746, permute_747, permute_754, permute_758, permute_762, div_54, permute_764, permute_768, div_55, permute_772, permute_777, permute_778, alias_79, permute_779, permute_780, permute_787, permute_791, permute_795, div_56, permute_797, permute_801, div_57, permute_805, permute_810, permute_811, alias_81, permute_812, permute_813, permute_820, permute_824, permute_828, div_58, permute_830, permute_834, div_59, permute_838, permute_843, permute_844, alias_83, permute_845, permute_846, permute_853, permute_857, permute_861, div_60, permute_863, permute_867, div_61, permute_871, permute_876, permute_877, alias_85, permute_878, permute_879, permute_886, permute_890, permute_894, div_62, permute_896, permute_900, div_63, permute_904, permute_909, permute_910, alias_87, permute_911, permute_912, permute_919, permute_923, permute_927, div_64, permute_929, permute_933, div_65, permute_937, permute_942, permute_943, alias_89, permute_944, permute_945, permute_952, permute_956, permute_960, div_66, permute_962, permute_966, div_67, permute_970, permute_975, permute_976, alias_91, permute_977, permute_978, permute_985, permute_989, permute_993, div_68, permute_995, permute_999, div_69, permute_1003, permute_1008, permute_1009, alias_93, permute_1010, permute_1011, permute_1018, permute_1022, permute_1026, div_70, permute_1028, permute_1032, div_71, permute_1036, permute_1041, permute_1042, alias_95, permute_1043, permute_1044, permute_1051, permute_1055, permute_1059, div_72]
    