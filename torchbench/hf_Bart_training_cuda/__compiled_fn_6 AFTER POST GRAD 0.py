from __future__ import annotations



def forward(self, primals_4: "f32[768]", primals_14: "f32[768]", primals_20: "f32[768]", primals_30: "f32[768]", primals_36: "f32[768]", primals_46: "f32[768]", primals_52: "f32[768]", primals_62: "f32[768]", primals_68: "f32[768]", primals_78: "f32[768]", primals_84: "f32[768]", primals_94: "f32[768]", primals_100: "f32[768]", primals_103: "f32[768]", primals_113: "f32[768]", primals_123: "f32[768]", primals_129: "f32[768]", primals_139: "f32[768]", primals_149: "f32[768]", primals_155: "f32[768]", primals_165: "f32[768]", primals_175: "f32[768]", primals_181: "f32[768]", primals_191: "f32[768]", primals_201: "f32[768]", primals_207: "f32[768]", primals_217: "f32[768]", primals_227: "f32[768]", primals_233: "f32[768]", primals_243: "f32[768]", primals_253: "f32[768]", primals_259: "f32[768]", primals_264: "i64[4, 512]", view: "i64[4, 512]", add: "i64[4, 512]", mul_1: "f32[4, 512, 768]", view_1: "f32[2048, 768]", bmm: "f32[48, 512, 512]", amax: "f32[48, 512, 1]", sum_1: "f32[48, 512, 1]", view_15: "f32[2048, 768]", mul_4: "f32[4, 512, 768]", view_17: "f32[2048, 768]", addmm_4: "f32[2048, 3072]", view_19: "f32[2048, 3072]", mul_9: "f32[4, 512, 768]", view_21: "f32[2048, 768]", bmm_2: "f32[48, 512, 512]", amax_1: "f32[48, 512, 1]", sum_2: "f32[48, 512, 1]", view_35: "f32[2048, 768]", mul_12: "f32[4, 512, 768]", view_37: "f32[2048, 768]", addmm_10: "f32[2048, 3072]", view_39: "f32[2048, 3072]", mul_17: "f32[4, 512, 768]", view_41: "f32[2048, 768]", bmm_4: "f32[48, 512, 512]", amax_2: "f32[48, 512, 1]", sum_3: "f32[48, 512, 1]", view_55: "f32[2048, 768]", mul_20: "f32[4, 512, 768]", view_57: "f32[2048, 768]", addmm_16: "f32[2048, 3072]", view_59: "f32[2048, 3072]", mul_25: "f32[4, 512, 768]", view_61: "f32[2048, 768]", bmm_6: "f32[48, 512, 512]", amax_3: "f32[48, 512, 1]", sum_4: "f32[48, 512, 1]", view_75: "f32[2048, 768]", mul_28: "f32[4, 512, 768]", view_77: "f32[2048, 768]", addmm_22: "f32[2048, 3072]", view_79: "f32[2048, 3072]", mul_33: "f32[4, 512, 768]", view_81: "f32[2048, 768]", bmm_8: "f32[48, 512, 512]", amax_4: "f32[48, 512, 1]", sum_5: "f32[48, 512, 1]", view_95: "f32[2048, 768]", mul_36: "f32[4, 512, 768]", view_97: "f32[2048, 768]", addmm_28: "f32[2048, 3072]", view_99: "f32[2048, 3072]", mul_41: "f32[4, 512, 768]", view_101: "f32[2048, 768]", bmm_10: "f32[48, 512, 512]", amax_5: "f32[48, 512, 1]", sum_6: "f32[48, 512, 1]", view_115: "f32[2048, 768]", mul_44: "f32[4, 512, 768]", view_117: "f32[2048, 768]", addmm_34: "f32[2048, 3072]", view_119: "f32[2048, 3072]", mul_49: "f32[4, 512, 768]", mul_52: "f32[4, 512, 768]", view_123: "f32[2048, 768]", view_139: "f32[2048, 768]", mul_55: "f32[4, 512, 768]", view_141: "f32[2048, 768]", view_143: "f32[2048, 768]", bmm_14: "f32[48, 512, 512]", amax_7: "f32[48, 512, 1]", sum_8: "f32[48, 512, 1]", view_155: "f32[2048, 768]", mul_58: "f32[4, 512, 768]", view_157: "f32[2048, 768]", addmm_44: "f32[2048, 3072]", view_159: "f32[2048, 3072]", mul_63: "f32[4, 512, 768]", view_161: "f32[2048, 768]", view_177: "f32[2048, 768]", mul_66: "f32[4, 512, 768]", view_179: "f32[2048, 768]", bmm_18: "f32[48, 512, 512]", amax_9: "f32[48, 512, 1]", sum_10: "f32[48, 512, 1]", view_193: "f32[2048, 768]", mul_69: "f32[4, 512, 768]", view_195: "f32[2048, 768]", addmm_54: "f32[2048, 3072]", view_197: "f32[2048, 3072]", mul_74: "f32[4, 512, 768]", view_199: "f32[2048, 768]", view_215: "f32[2048, 768]", mul_77: "f32[4, 512, 768]", view_217: "f32[2048, 768]", bmm_22: "f32[48, 512, 512]", amax_11: "f32[48, 512, 1]", sum_12: "f32[48, 512, 1]", view_231: "f32[2048, 768]", mul_80: "f32[4, 512, 768]", view_233: "f32[2048, 768]", addmm_64: "f32[2048, 3072]", view_235: "f32[2048, 3072]", mul_85: "f32[4, 512, 768]", view_237: "f32[2048, 768]", view_253: "f32[2048, 768]", mul_88: "f32[4, 512, 768]", view_255: "f32[2048, 768]", bmm_26: "f32[48, 512, 512]", amax_13: "f32[48, 512, 1]", sum_14: "f32[48, 512, 1]", view_269: "f32[2048, 768]", mul_91: "f32[4, 512, 768]", view_271: "f32[2048, 768]", addmm_74: "f32[2048, 3072]", view_273: "f32[2048, 3072]", mul_96: "f32[4, 512, 768]", view_275: "f32[2048, 768]", view_291: "f32[2048, 768]", mul_99: "f32[4, 512, 768]", view_293: "f32[2048, 768]", bmm_30: "f32[48, 512, 512]", amax_15: "f32[48, 512, 1]", sum_16: "f32[48, 512, 1]", view_307: "f32[2048, 768]", mul_102: "f32[4, 512, 768]", view_309: "f32[2048, 768]", addmm_84: "f32[2048, 3072]", view_311: "f32[2048, 3072]", mul_107: "f32[4, 512, 768]", view_313: "f32[2048, 768]", view_329: "f32[2048, 768]", mul_110: "f32[4, 512, 768]", view_331: "f32[2048, 768]", bmm_34: "f32[48, 512, 512]", amax_17: "f32[48, 512, 1]", sum_18: "f32[48, 512, 1]", view_345: "f32[2048, 768]", mul_113: "f32[4, 512, 768]", view_347: "f32[2048, 768]", addmm_94: "f32[2048, 3072]", view_349: "f32[2048, 3072]", mul_118: "f32[4, 512, 768]", view_351: "f32[2048, 768]", permute_189: "f32[50265, 768]", div_18: "f32[4, 512, 1]", permute_191: "f32[768, 3072]", permute_195: "f32[3072, 768]", div_19: "f32[4, 512, 1]", permute_199: "f32[768, 768]", permute_204: "f32[48, 512, 512]", permute_205: "f32[48, 64, 512]", permute_206: "f32[48, 64, 512]", permute_207: "f32[48, 512, 64]", permute_211: "f32[768, 768]", permute_216: "f32[768, 768]", permute_220: "f32[768, 768]", div_20: "f32[4, 512, 1]", permute_224: "f32[768, 768]", permute_229: "f32[48, 512, 512]", permute_230: "f32[48, 64, 512]", alias_19: "f32[48, 512, 512]", permute_231: "f32[48, 64, 512]", permute_232: "f32[48, 512, 64]", permute_236: "f32[768, 768]", permute_241: "f32[768, 768]", permute_245: "f32[768, 768]", div_21: "f32[4, 512, 1]", permute_249: "f32[768, 3072]", permute_253: "f32[3072, 768]", div_22: "f32[4, 512, 1]", permute_257: "f32[768, 768]", permute_262: "f32[48, 512, 512]", permute_263: "f32[48, 64, 512]", permute_264: "f32[48, 64, 512]", permute_265: "f32[48, 512, 64]", permute_269: "f32[768, 768]", permute_274: "f32[768, 768]", permute_278: "f32[768, 768]", div_23: "f32[4, 512, 1]", permute_282: "f32[768, 768]", permute_287: "f32[48, 512, 512]", permute_288: "f32[48, 64, 512]", alias_21: "f32[48, 512, 512]", permute_289: "f32[48, 64, 512]", permute_290: "f32[48, 512, 64]", permute_294: "f32[768, 768]", permute_299: "f32[768, 768]", permute_303: "f32[768, 768]", div_24: "f32[4, 512, 1]", permute_307: "f32[768, 3072]", permute_311: "f32[3072, 768]", div_25: "f32[4, 512, 1]", permute_315: "f32[768, 768]", permute_320: "f32[48, 512, 512]", permute_321: "f32[48, 64, 512]", permute_322: "f32[48, 64, 512]", permute_323: "f32[48, 512, 64]", permute_327: "f32[768, 768]", permute_332: "f32[768, 768]", permute_336: "f32[768, 768]", div_26: "f32[4, 512, 1]", permute_340: "f32[768, 768]", permute_345: "f32[48, 512, 512]", permute_346: "f32[48, 64, 512]", alias_23: "f32[48, 512, 512]", permute_347: "f32[48, 64, 512]", permute_348: "f32[48, 512, 64]", permute_352: "f32[768, 768]", permute_357: "f32[768, 768]", permute_361: "f32[768, 768]", div_27: "f32[4, 512, 1]", permute_365: "f32[768, 3072]", permute_369: "f32[3072, 768]", div_28: "f32[4, 512, 1]", permute_373: "f32[768, 768]", permute_378: "f32[48, 512, 512]", permute_379: "f32[48, 64, 512]", permute_380: "f32[48, 64, 512]", permute_381: "f32[48, 512, 64]", permute_385: "f32[768, 768]", permute_390: "f32[768, 768]", permute_394: "f32[768, 768]", div_29: "f32[4, 512, 1]", permute_398: "f32[768, 768]", permute_403: "f32[48, 512, 512]", permute_404: "f32[48, 64, 512]", alias_25: "f32[48, 512, 512]", permute_405: "f32[48, 64, 512]", permute_406: "f32[48, 512, 64]", permute_410: "f32[768, 768]", permute_415: "f32[768, 768]", permute_419: "f32[768, 768]", div_30: "f32[4, 512, 1]", permute_423: "f32[768, 3072]", permute_427: "f32[3072, 768]", div_31: "f32[4, 512, 1]", permute_431: "f32[768, 768]", permute_436: "f32[48, 512, 512]", permute_437: "f32[48, 64, 512]", permute_438: "f32[48, 64, 512]", permute_439: "f32[48, 512, 64]", permute_443: "f32[768, 768]", permute_448: "f32[768, 768]", permute_452: "f32[768, 768]", div_32: "f32[4, 512, 1]", permute_456: "f32[768, 768]", permute_461: "f32[48, 512, 512]", permute_462: "f32[48, 64, 512]", alias_27: "f32[48, 512, 512]", permute_463: "f32[48, 64, 512]", permute_464: "f32[48, 512, 64]", permute_468: "f32[768, 768]", permute_473: "f32[768, 768]", permute_477: "f32[768, 768]", div_33: "f32[4, 512, 1]", permute_481: "f32[768, 3072]", permute_485: "f32[3072, 768]", div_34: "f32[4, 512, 1]", permute_489: "f32[768, 768]", permute_494: "f32[48, 512, 512]", permute_495: "f32[48, 64, 512]", permute_496: "f32[48, 64, 512]", permute_497: "f32[48, 512, 64]", permute_501: "f32[768, 768]", permute_506: "f32[768, 768]", permute_510: "f32[768, 768]", div_35: "f32[4, 512, 1]", permute_514: "f32[768, 768]", permute_519: "f32[48, 512, 512]", permute_520: "f32[48, 64, 512]", alias_29: "f32[48, 512, 512]", permute_521: "f32[48, 64, 512]", permute_522: "f32[48, 512, 64]", permute_526: "f32[768, 768]", permute_531: "f32[768, 768]", permute_535: "f32[768, 768]", div_36: "f32[4, 512, 1]", div_37: "f32[4, 512, 1]", permute_539: "f32[768, 3072]", permute_543: "f32[3072, 768]", div_38: "f32[4, 512, 1]", permute_547: "f32[768, 768]", permute_552: "f32[48, 512, 512]", permute_553: "f32[48, 64, 512]", permute_554: "f32[48, 64, 512]", permute_555: "f32[48, 512, 64]", permute_559: "f32[768, 768]", permute_564: "f32[768, 768]", permute_568: "f32[768, 768]", div_39: "f32[4, 512, 1]", permute_572: "f32[768, 3072]", permute_576: "f32[3072, 768]", div_40: "f32[4, 512, 1]", permute_580: "f32[768, 768]", permute_585: "f32[48, 512, 512]", permute_586: "f32[48, 64, 512]", permute_587: "f32[48, 64, 512]", permute_588: "f32[48, 512, 64]", permute_592: "f32[768, 768]", permute_597: "f32[768, 768]", permute_601: "f32[768, 768]", div_41: "f32[4, 512, 1]", permute_605: "f32[768, 3072]", permute_609: "f32[3072, 768]", div_42: "f32[4, 512, 1]", permute_613: "f32[768, 768]", permute_618: "f32[48, 512, 512]", permute_619: "f32[48, 64, 512]", permute_620: "f32[48, 64, 512]", permute_621: "f32[48, 512, 64]", permute_625: "f32[768, 768]", permute_630: "f32[768, 768]", permute_634: "f32[768, 768]", div_43: "f32[4, 512, 1]", permute_638: "f32[768, 3072]", permute_642: "f32[3072, 768]", div_44: "f32[4, 512, 1]", permute_646: "f32[768, 768]", permute_651: "f32[48, 512, 512]", permute_652: "f32[48, 64, 512]", permute_653: "f32[48, 64, 512]", permute_654: "f32[48, 512, 64]", permute_658: "f32[768, 768]", permute_663: "f32[768, 768]", permute_667: "f32[768, 768]", div_45: "f32[4, 512, 1]", permute_671: "f32[768, 3072]", permute_675: "f32[3072, 768]", div_46: "f32[4, 512, 1]", permute_679: "f32[768, 768]", permute_684: "f32[48, 512, 512]", permute_685: "f32[48, 64, 512]", permute_686: "f32[48, 64, 512]", permute_687: "f32[48, 512, 64]", permute_691: "f32[768, 768]", permute_696: "f32[768, 768]", permute_700: "f32[768, 768]", div_47: "f32[4, 512, 1]", permute_704: "f32[768, 3072]", permute_708: "f32[3072, 768]", div_48: "f32[4, 512, 1]", permute_712: "f32[768, 768]", permute_717: "f32[48, 512, 512]", permute_718: "f32[48, 64, 512]", permute_719: "f32[48, 64, 512]", permute_720: "f32[48, 512, 64]", permute_724: "f32[768, 768]", permute_729: "f32[768, 768]", permute_733: "f32[768, 768]", div_49: "f32[4, 512, 1]", tangents_1: "f32[4, 512, 50265]", tangents_2: "f32[4, 12, 512, 64]", tangents_3: "f32[4, 12, 512, 64]", tangents_4: "f32[4, 12, 512, 64]", tangents_5: "f32[4, 12, 512, 64]", tangents_6: "f32[4, 12, 512, 64]", tangents_7: "f32[4, 12, 512, 64]", tangents_8: "f32[4, 12, 512, 64]", tangents_9: "f32[4, 12, 512, 64]", tangents_10: "f32[4, 12, 512, 64]", tangents_11: "f32[4, 12, 512, 64]", tangents_12: "f32[4, 12, 512, 64]", tangents_13: "f32[4, 12, 512, 64]", tangents_14: "f32[4, 12, 512, 64]", tangents_15: "f32[4, 12, 512, 64]", tangents_16: "f32[4, 12, 512, 64]", tangents_17: "f32[4, 12, 512, 64]", tangents_18: "f32[4, 12, 512, 64]", tangents_19: "f32[4, 12, 512, 64]", tangents_20: "f32[4, 12, 512, 64]", tangents_21: "f32[4, 12, 512, 64]", tangents_22: "f32[4, 12, 512, 64]", tangents_23: "f32[4, 12, 512, 64]", tangents_24: "f32[4, 12, 512, 64]", tangents_25: "f32[4, 12, 512, 64]", tangents_26: "f32[4, 512, 768]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    sub_1: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(bmm, amax);  bmm = amax = None
    exp: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    div: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_18: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_4, [4, 512, 3072]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_7: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_18, 0.7071067811865476)
    erf: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
    add_7: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    sub_4: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(bmm_2, amax_1);  bmm_2 = amax_1 = None
    exp_1: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    div_1: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_38: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_10, [4, 512, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_15: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476)
    erf_1: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
    add_14: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    sub_7: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(bmm_4, amax_2);  bmm_4 = amax_2 = None
    exp_2: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    div_2: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_58: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_16, [4, 512, 3072]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_23: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.7071067811865476)
    erf_2: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_23);  mul_23 = None
    add_21: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    sub_10: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(bmm_6, amax_3);  bmm_6 = amax_3 = None
    exp_3: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    div_3: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_78: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_22, [4, 512, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_31: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_78, 0.7071067811865476)
    erf_3: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_31);  mul_31 = None
    add_28: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    sub_13: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(bmm_8, amax_4);  bmm_8 = amax_4 = None
    exp_4: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    div_4: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_98: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_28, [4, 512, 3072]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_39: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_98, 0.7071067811865476)
    erf_4: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
    add_35: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    sub_16: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(bmm_10, amax_5);  bmm_10 = amax_5 = None
    exp_5: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    div_5: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_118: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_34, [4, 512, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_47: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_118, 0.7071067811865476)
    erf_5: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
    add_42: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:98, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    sub_22: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(bmm_14, amax_7);  bmm_14 = amax_7 = None
    exp_7: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    div_7: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_158: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_44, [4, 512, 3072]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_61: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_158, 0.7071067811865476)
    erf_6: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
    add_58: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    sub_27: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(bmm_18, amax_9);  bmm_18 = amax_9 = None
    exp_9: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_27);  sub_27 = None
    div_9: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_196: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_54, [4, 512, 3072]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_72: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_196, 0.7071067811865476)
    erf_7: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_72);  mul_72 = None
    add_69: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    sub_32: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(bmm_22, amax_11);  bmm_22 = amax_11 = None
    exp_11: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_32);  sub_32 = None
    div_11: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_234: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_64, [4, 512, 3072]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_83: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_234, 0.7071067811865476)
    erf_8: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
    add_80: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    sub_37: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(bmm_26, amax_13);  bmm_26 = amax_13 = None
    exp_13: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
    div_13: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_272: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_74, [4, 512, 3072]);  addmm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_94: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_272, 0.7071067811865476)
    erf_9: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_94);  mul_94 = None
    add_91: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    sub_42: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(bmm_30, amax_15);  bmm_30 = amax_15 = None
    exp_15: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_42);  sub_42 = None
    div_15: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_310: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_84, [4, 512, 3072]);  addmm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_105: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_310, 0.7071067811865476)
    erf_10: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_105);  mul_105 = None
    add_102: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    sub_47: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(bmm_34, amax_17);  bmm_34 = amax_17 = None
    exp_17: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_47);  sub_47 = None
    div_17: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_348: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_94, [4, 512, 3072]);  addmm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_116: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_348, 0.7071067811865476)
    erf_11: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_116);  mul_116 = None
    add_113: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:1406, code: lm_logits = self.lm_head(outputs[0])
    view_353: "f32[2048, 50265]" = torch.ops.aten.reshape.default(tangents_1, [2048, 50265]);  tangents_1 = None
    permute_187: "f32[50265, 2048]" = torch.ops.aten.permute.default(view_353, [1, 0])
    mm_1: "f32[50265, 768]" = torch.ops.aten.mm.default(permute_187, view_351);  permute_187 = view_351 = None
    permute_188: "f32[768, 50265]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    mm_2: "f32[2048, 768]" = torch.ops.aten.mm.default(view_353, permute_189);  view_353 = permute_189 = None
    view_354: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_2, [4, 512, 768]);  mm_2 = None
    permute_190: "f32[50265, 768]" = torch.ops.aten.permute.default(permute_188, [1, 0]);  permute_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_121: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_354, primals_259);  primals_259 = None
    mul_122: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_121, 768)
    sum_19: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_121, [2], True)
    mul_123: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_121, mul_118);  mul_121 = None
    sum_20: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_123, [2], True);  mul_123 = None
    mul_124: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_118, sum_20);  sum_20 = None
    sub_51: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_122, sum_19);  mul_122 = sum_19 = None
    sub_52: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_51, mul_124);  sub_51 = mul_124 = None
    mul_125: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_18, sub_52);  div_18 = sub_52 = None
    mul_126: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_354, mul_118);  mul_118 = None
    sum_21: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_126, [0, 1]);  mul_126 = None
    sum_22: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_354, [0, 1]);  view_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_355: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_125, [2048, 768])
    mm_3: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_355, permute_191);  permute_191 = None
    permute_192: "f32[768, 2048]" = torch.ops.aten.permute.default(view_355, [1, 0])
    mm_4: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_192, view_349);  permute_192 = view_349 = None
    permute_193: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_4, [1, 0]);  mm_4 = None
    sum_23: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_355, [0], True);  view_355 = None
    view_356: "f32[768]" = torch.ops.aten.reshape.default(sum_23, [768]);  sum_23 = None
    permute_194: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_193, [1, 0]);  permute_193 = None
    view_357: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_3, [4, 512, 3072]);  mm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_128: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_113, 0.5);  add_113 = None
    mul_129: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_348, view_348)
    mul_130: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_129, -0.5);  mul_129 = None
    exp_18: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_130);  mul_130 = None
    mul_131: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
    mul_132: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_348, mul_131);  view_348 = mul_131 = None
    add_119: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_128, mul_132);  mul_128 = mul_132 = None
    mul_133: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_357, add_119);  view_357 = add_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_358: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_133, [2048, 3072]);  mul_133 = None
    mm_5: "f32[2048, 768]" = torch.ops.aten.mm.default(view_358, permute_195);  permute_195 = None
    permute_196: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_358, [1, 0])
    mm_6: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_196, view_347);  permute_196 = view_347 = None
    permute_197: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_6, [1, 0]);  mm_6 = None
    sum_24: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_358, [0], True);  view_358 = None
    view_359: "f32[3072]" = torch.ops.aten.reshape.default(sum_24, [3072]);  sum_24 = None
    permute_198: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
    view_360: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_5, [4, 512, 768]);  mm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    add_120: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_125, view_360);  mul_125 = view_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:456, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    mul_135: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_120, primals_253);  primals_253 = None
    mul_136: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_135, 768)
    sum_25: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_135, [2], True)
    mul_137: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_135, mul_113);  mul_135 = None
    sum_26: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_137, [2], True);  mul_137 = None
    mul_138: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_113, sum_26);  sum_26 = None
    sub_54: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_136, sum_25);  mul_136 = sum_25 = None
    sub_55: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_54, mul_138);  sub_54 = mul_138 = None
    mul_139: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_19, sub_55);  div_19 = sub_55 = None
    mul_140: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_120, mul_113);  mul_113 = None
    sum_27: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_140, [0, 1]);  mul_140 = None
    sum_28: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_120, [0, 1]);  add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_361: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_139, [2048, 768])
    mm_7: "f32[2048, 768]" = torch.ops.aten.mm.default(view_361, permute_199);  permute_199 = None
    permute_200: "f32[768, 2048]" = torch.ops.aten.permute.default(view_361, [1, 0])
    mm_8: "f32[768, 768]" = torch.ops.aten.mm.default(permute_200, view_345);  permute_200 = view_345 = None
    permute_201: "f32[768, 768]" = torch.ops.aten.permute.default(mm_8, [1, 0]);  mm_8 = None
    sum_29: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_361, [0], True);  view_361 = None
    view_362: "f32[768]" = torch.ops.aten.reshape.default(sum_29, [768]);  sum_29 = None
    permute_202: "f32[768, 768]" = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None
    view_363: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_7, [4, 512, 768]);  mm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_364: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_363, [4, 512, 12, 64]);  view_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_203: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_364, [0, 2, 1, 3]);  view_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    clone_134: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_203, memory_format = torch.contiguous_format);  permute_203 = None
    view_365: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_134, [48, 512, 64]);  clone_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_36: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_204, view_365);  permute_204 = None
    bmm_37: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_365, permute_205);  view_365 = permute_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_141: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(bmm_37, div_17);  bmm_37 = None
    sum_30: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_141, [-1], True)
    mul_142: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(div_17, sum_30);  div_17 = sum_30 = None
    sub_56: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(mul_141, mul_142);  mul_141 = mul_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_38: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_206, sub_56);  permute_206 = None
    bmm_39: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(sub_56, permute_207);  sub_56 = permute_207 = None
    permute_208: "f32[48, 512, 64]" = torch.ops.aten.permute.default(bmm_38, [0, 2, 1]);  bmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_366: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_36, [4, 12, 512, 64]);  bmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    add_121: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_25, view_366);  tangents_25 = view_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_367: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(permute_208, [4, 12, 512, 64]);  permute_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    add_122: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_24, view_367);  tangents_24 = view_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_368: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_39, [4, 12, 512, 64]);  bmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_209: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_368, [0, 2, 1, 3]);  view_368 = None
    clone_135: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_209, memory_format = torch.contiguous_format);  permute_209 = None
    view_369: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_135, [4, 512, 768]);  clone_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_210: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_121, [0, 2, 1, 3]);  add_121 = None
    clone_136: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_210, memory_format = torch.contiguous_format);  permute_210 = None
    view_370: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_136, [4, 512, 768]);  clone_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_371: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_370, [2048, 768]);  view_370 = None
    mm_9: "f32[2048, 768]" = torch.ops.aten.mm.default(view_371, permute_211);  permute_211 = None
    permute_212: "f32[768, 2048]" = torch.ops.aten.permute.default(view_371, [1, 0])
    mm_10: "f32[768, 768]" = torch.ops.aten.mm.default(permute_212, view_143);  permute_212 = None
    permute_213: "f32[768, 768]" = torch.ops.aten.permute.default(mm_10, [1, 0]);  mm_10 = None
    sum_31: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_371, [0], True);  view_371 = None
    view_372: "f32[768]" = torch.ops.aten.reshape.default(sum_31, [768]);  sum_31 = None
    permute_214: "f32[768, 768]" = torch.ops.aten.permute.default(permute_213, [1, 0]);  permute_213 = None
    view_373: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_9, [4, 512, 768]);  mm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    add_123: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(tangents_26, view_373);  tangents_26 = view_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_215: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_122, [0, 2, 1, 3]);  add_122 = None
    clone_137: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_215, memory_format = torch.contiguous_format);  permute_215 = None
    view_374: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_137, [4, 512, 768]);  clone_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_375: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_374, [2048, 768]);  view_374 = None
    mm_11: "f32[2048, 768]" = torch.ops.aten.mm.default(view_375, permute_216);  permute_216 = None
    permute_217: "f32[768, 2048]" = torch.ops.aten.permute.default(view_375, [1, 0])
    mm_12: "f32[768, 768]" = torch.ops.aten.mm.default(permute_217, view_143);  permute_217 = None
    permute_218: "f32[768, 768]" = torch.ops.aten.permute.default(mm_12, [1, 0]);  mm_12 = None
    sum_32: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_375, [0], True);  view_375 = None
    view_376: "f32[768]" = torch.ops.aten.reshape.default(sum_32, [768]);  sum_32 = None
    permute_219: "f32[768, 768]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
    view_377: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_11, [4, 512, 768]);  mm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    add_124: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_123, view_377);  add_123 = view_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_143: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_369, 0.125);  view_369 = None
    view_378: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_143, [2048, 768]);  mul_143 = None
    mm_13: "f32[2048, 768]" = torch.ops.aten.mm.default(view_378, permute_220);  permute_220 = None
    permute_221: "f32[768, 2048]" = torch.ops.aten.permute.default(view_378, [1, 0])
    mm_14: "f32[768, 768]" = torch.ops.aten.mm.default(permute_221, view_331);  permute_221 = view_331 = None
    permute_222: "f32[768, 768]" = torch.ops.aten.permute.default(mm_14, [1, 0]);  mm_14 = None
    sum_33: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_378, [0], True);  view_378 = None
    view_379: "f32[768]" = torch.ops.aten.reshape.default(sum_33, [768]);  sum_33 = None
    permute_223: "f32[768, 768]" = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None
    view_380: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_13, [4, 512, 768]);  mm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_125: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_139, view_380);  mul_139 = view_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_145: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_125, primals_243);  primals_243 = None
    mul_146: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_145, 768)
    sum_34: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_145, [2], True)
    mul_147: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_145, mul_110);  mul_145 = None
    sum_35: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_147, [2], True);  mul_147 = None
    mul_148: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_110, sum_35);  sum_35 = None
    sub_58: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_146, sum_34);  mul_146 = sum_34 = None
    sub_59: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_58, mul_148);  sub_58 = mul_148 = None
    mul_149: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_20, sub_59);  div_20 = sub_59 = None
    mul_150: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_125, mul_110);  mul_110 = None
    sum_36: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_150, [0, 1]);  mul_150 = None
    sum_37: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_125, [0, 1]);  add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_381: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_149, [2048, 768])
    mm_15: "f32[2048, 768]" = torch.ops.aten.mm.default(view_381, permute_224);  permute_224 = None
    permute_225: "f32[768, 2048]" = torch.ops.aten.permute.default(view_381, [1, 0])
    mm_16: "f32[768, 768]" = torch.ops.aten.mm.default(permute_225, view_329);  permute_225 = view_329 = None
    permute_226: "f32[768, 768]" = torch.ops.aten.permute.default(mm_16, [1, 0]);  mm_16 = None
    sum_38: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_381, [0], True);  view_381 = None
    view_382: "f32[768]" = torch.ops.aten.reshape.default(sum_38, [768]);  sum_38 = None
    permute_227: "f32[768, 768]" = torch.ops.aten.permute.default(permute_226, [1, 0]);  permute_226 = None
    view_383: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_15, [4, 512, 768]);  mm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_384: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_383, [4, 512, 12, 64]);  view_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_228: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_384, [0, 2, 1, 3]);  view_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    clone_138: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_228, memory_format = torch.contiguous_format);  permute_228 = None
    view_385: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_138, [48, 512, 64]);  clone_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_40: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_229, view_385);  permute_229 = None
    bmm_41: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_385, permute_230);  view_385 = permute_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_151: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(bmm_41, alias_19);  bmm_41 = None
    sum_39: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_151, [-1], True)
    mul_152: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(alias_19, sum_39);  alias_19 = sum_39 = None
    sub_60: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_386: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(sub_60, [4, 12, 512, 512]);  sub_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_387: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(view_386, [48, 512, 512]);  view_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_42: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_231, view_387);  permute_231 = None
    bmm_43: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_387, permute_232);  view_387 = permute_232 = None
    permute_233: "f32[48, 512, 64]" = torch.ops.aten.permute.default(bmm_42, [0, 2, 1]);  bmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_388: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_40, [4, 12, 512, 64]);  bmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    add_126: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_23, view_388);  tangents_23 = view_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_389: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(permute_233, [4, 12, 512, 64]);  permute_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    add_127: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_22, view_389);  tangents_22 = view_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_390: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_43, [4, 12, 512, 64]);  bmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_234: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_390, [0, 2, 1, 3]);  view_390 = None
    clone_139: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_234, memory_format = torch.contiguous_format);  permute_234 = None
    view_391: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_139, [4, 512, 768]);  clone_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_235: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_126, [0, 2, 1, 3]);  add_126 = None
    clone_140: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_235, memory_format = torch.contiguous_format);  permute_235 = None
    view_392: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_140, [4, 512, 768]);  clone_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_393: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_392, [2048, 768]);  view_392 = None
    mm_17: "f32[2048, 768]" = torch.ops.aten.mm.default(view_393, permute_236);  permute_236 = None
    permute_237: "f32[768, 2048]" = torch.ops.aten.permute.default(view_393, [1, 0])
    mm_18: "f32[768, 768]" = torch.ops.aten.mm.default(permute_237, view_313);  permute_237 = None
    permute_238: "f32[768, 768]" = torch.ops.aten.permute.default(mm_18, [1, 0]);  mm_18 = None
    sum_40: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_393, [0], True);  view_393 = None
    view_394: "f32[768]" = torch.ops.aten.reshape.default(sum_40, [768]);  sum_40 = None
    permute_239: "f32[768, 768]" = torch.ops.aten.permute.default(permute_238, [1, 0]);  permute_238 = None
    view_395: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_17, [4, 512, 768]);  mm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    add_128: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_149, view_395);  mul_149 = view_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_240: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_127, [0, 2, 1, 3]);  add_127 = None
    clone_141: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_240, memory_format = torch.contiguous_format);  permute_240 = None
    view_396: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_141, [4, 512, 768]);  clone_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_397: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_396, [2048, 768]);  view_396 = None
    mm_19: "f32[2048, 768]" = torch.ops.aten.mm.default(view_397, permute_241);  permute_241 = None
    permute_242: "f32[768, 2048]" = torch.ops.aten.permute.default(view_397, [1, 0])
    mm_20: "f32[768, 768]" = torch.ops.aten.mm.default(permute_242, view_313);  permute_242 = None
    permute_243: "f32[768, 768]" = torch.ops.aten.permute.default(mm_20, [1, 0]);  mm_20 = None
    sum_41: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_397, [0], True);  view_397 = None
    view_398: "f32[768]" = torch.ops.aten.reshape.default(sum_41, [768]);  sum_41 = None
    permute_244: "f32[768, 768]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
    view_399: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_19, [4, 512, 768]);  mm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_129: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_128, view_399);  add_128 = view_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_153: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_391, 0.125);  view_391 = None
    view_400: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_153, [2048, 768]);  mul_153 = None
    mm_21: "f32[2048, 768]" = torch.ops.aten.mm.default(view_400, permute_245);  permute_245 = None
    permute_246: "f32[768, 2048]" = torch.ops.aten.permute.default(view_400, [1, 0])
    mm_22: "f32[768, 768]" = torch.ops.aten.mm.default(permute_246, view_313);  permute_246 = view_313 = None
    permute_247: "f32[768, 768]" = torch.ops.aten.permute.default(mm_22, [1, 0]);  mm_22 = None
    sum_42: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_400, [0], True);  view_400 = None
    view_401: "f32[768]" = torch.ops.aten.reshape.default(sum_42, [768]);  sum_42 = None
    permute_248: "f32[768, 768]" = torch.ops.aten.permute.default(permute_247, [1, 0]);  permute_247 = None
    view_402: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_21, [4, 512, 768]);  mm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_130: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_129, view_402);  add_129 = view_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_155: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_130, primals_233);  primals_233 = None
    mul_156: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_155, 768)
    sum_43: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_155, [2], True)
    mul_157: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_155, mul_107);  mul_155 = None
    sum_44: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_157, [2], True);  mul_157 = None
    mul_158: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_107, sum_44);  sum_44 = None
    sub_62: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_156, sum_43);  mul_156 = sum_43 = None
    sub_63: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_62, mul_158);  sub_62 = mul_158 = None
    mul_159: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_21, sub_63);  div_21 = sub_63 = None
    mul_160: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_130, mul_107);  mul_107 = None
    sum_45: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_160, [0, 1]);  mul_160 = None
    sum_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_130, [0, 1]);  add_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_403: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_159, [2048, 768])
    mm_23: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_403, permute_249);  permute_249 = None
    permute_250: "f32[768, 2048]" = torch.ops.aten.permute.default(view_403, [1, 0])
    mm_24: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_250, view_311);  permute_250 = view_311 = None
    permute_251: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_24, [1, 0]);  mm_24 = None
    sum_47: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_403, [0], True);  view_403 = None
    view_404: "f32[768]" = torch.ops.aten.reshape.default(sum_47, [768]);  sum_47 = None
    permute_252: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None
    view_405: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_23, [4, 512, 3072]);  mm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_162: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_102, 0.5);  add_102 = None
    mul_163: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_310, view_310)
    mul_164: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_163, -0.5);  mul_163 = None
    exp_19: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_164);  mul_164 = None
    mul_165: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
    mul_166: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_310, mul_165);  view_310 = mul_165 = None
    add_132: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_162, mul_166);  mul_162 = mul_166 = None
    mul_167: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_405, add_132);  view_405 = add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_406: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_167, [2048, 3072]);  mul_167 = None
    mm_25: "f32[2048, 768]" = torch.ops.aten.mm.default(view_406, permute_253);  permute_253 = None
    permute_254: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_406, [1, 0])
    mm_26: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_254, view_309);  permute_254 = view_309 = None
    permute_255: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_26, [1, 0]);  mm_26 = None
    sum_48: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_406, [0], True);  view_406 = None
    view_407: "f32[3072]" = torch.ops.aten.reshape.default(sum_48, [3072]);  sum_48 = None
    permute_256: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_255, [1, 0]);  permute_255 = None
    view_408: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_25, [4, 512, 768]);  mm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    add_133: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_159, view_408);  mul_159 = view_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:456, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    mul_169: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_133, primals_227);  primals_227 = None
    mul_170: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_169, 768)
    sum_49: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_169, [2], True)
    mul_171: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_169, mul_102);  mul_169 = None
    sum_50: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_171, [2], True);  mul_171 = None
    mul_172: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_102, sum_50);  sum_50 = None
    sub_65: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_170, sum_49);  mul_170 = sum_49 = None
    sub_66: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_65, mul_172);  sub_65 = mul_172 = None
    mul_173: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_22, sub_66);  div_22 = sub_66 = None
    mul_174: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_133, mul_102);  mul_102 = None
    sum_51: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_174, [0, 1]);  mul_174 = None
    sum_52: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_133, [0, 1]);  add_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_409: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_173, [2048, 768])
    mm_27: "f32[2048, 768]" = torch.ops.aten.mm.default(view_409, permute_257);  permute_257 = None
    permute_258: "f32[768, 2048]" = torch.ops.aten.permute.default(view_409, [1, 0])
    mm_28: "f32[768, 768]" = torch.ops.aten.mm.default(permute_258, view_307);  permute_258 = view_307 = None
    permute_259: "f32[768, 768]" = torch.ops.aten.permute.default(mm_28, [1, 0]);  mm_28 = None
    sum_53: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_409, [0], True);  view_409 = None
    view_410: "f32[768]" = torch.ops.aten.reshape.default(sum_53, [768]);  sum_53 = None
    permute_260: "f32[768, 768]" = torch.ops.aten.permute.default(permute_259, [1, 0]);  permute_259 = None
    view_411: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_27, [4, 512, 768]);  mm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_412: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_411, [4, 512, 12, 64]);  view_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_261: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_412, [0, 2, 1, 3]);  view_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    clone_142: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_261, memory_format = torch.contiguous_format);  permute_261 = None
    view_413: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_142, [48, 512, 64]);  clone_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_44: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_262, view_413);  permute_262 = None
    bmm_45: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_413, permute_263);  view_413 = permute_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_175: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(bmm_45, div_15);  bmm_45 = None
    sum_54: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_175, [-1], True)
    mul_176: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(div_15, sum_54);  div_15 = sum_54 = None
    sub_67: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(mul_175, mul_176);  mul_175 = mul_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_46: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_264, sub_67);  permute_264 = None
    bmm_47: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(sub_67, permute_265);  sub_67 = permute_265 = None
    permute_266: "f32[48, 512, 64]" = torch.ops.aten.permute.default(bmm_46, [0, 2, 1]);  bmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_414: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_44, [4, 12, 512, 64]);  bmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    add_134: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_21, view_414);  tangents_21 = view_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_415: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(permute_266, [4, 12, 512, 64]);  permute_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    add_135: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_20, view_415);  tangents_20 = view_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_416: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_47, [4, 12, 512, 64]);  bmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_267: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_416, [0, 2, 1, 3]);  view_416 = None
    clone_143: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_267, memory_format = torch.contiguous_format);  permute_267 = None
    view_417: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_143, [4, 512, 768]);  clone_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_268: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_134, [0, 2, 1, 3]);  add_134 = None
    clone_144: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_268, memory_format = torch.contiguous_format);  permute_268 = None
    view_418: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_144, [4, 512, 768]);  clone_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_419: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_418, [2048, 768]);  view_418 = None
    mm_29: "f32[2048, 768]" = torch.ops.aten.mm.default(view_419, permute_269);  permute_269 = None
    permute_270: "f32[768, 2048]" = torch.ops.aten.permute.default(view_419, [1, 0])
    mm_30: "f32[768, 768]" = torch.ops.aten.mm.default(permute_270, view_143);  permute_270 = None
    permute_271: "f32[768, 768]" = torch.ops.aten.permute.default(mm_30, [1, 0]);  mm_30 = None
    sum_55: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_419, [0], True);  view_419 = None
    view_420: "f32[768]" = torch.ops.aten.reshape.default(sum_55, [768]);  sum_55 = None
    permute_272: "f32[768, 768]" = torch.ops.aten.permute.default(permute_271, [1, 0]);  permute_271 = None
    view_421: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_29, [4, 512, 768]);  mm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    add_136: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_124, view_421);  add_124 = view_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_273: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_135, [0, 2, 1, 3]);  add_135 = None
    clone_145: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_273, memory_format = torch.contiguous_format);  permute_273 = None
    view_422: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_145, [4, 512, 768]);  clone_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_423: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_422, [2048, 768]);  view_422 = None
    mm_31: "f32[2048, 768]" = torch.ops.aten.mm.default(view_423, permute_274);  permute_274 = None
    permute_275: "f32[768, 2048]" = torch.ops.aten.permute.default(view_423, [1, 0])
    mm_32: "f32[768, 768]" = torch.ops.aten.mm.default(permute_275, view_143);  permute_275 = None
    permute_276: "f32[768, 768]" = torch.ops.aten.permute.default(mm_32, [1, 0]);  mm_32 = None
    sum_56: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_423, [0], True);  view_423 = None
    view_424: "f32[768]" = torch.ops.aten.reshape.default(sum_56, [768]);  sum_56 = None
    permute_277: "f32[768, 768]" = torch.ops.aten.permute.default(permute_276, [1, 0]);  permute_276 = None
    view_425: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_31, [4, 512, 768]);  mm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    add_137: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_136, view_425);  add_136 = view_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_177: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_417, 0.125);  view_417 = None
    view_426: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_177, [2048, 768]);  mul_177 = None
    mm_33: "f32[2048, 768]" = torch.ops.aten.mm.default(view_426, permute_278);  permute_278 = None
    permute_279: "f32[768, 2048]" = torch.ops.aten.permute.default(view_426, [1, 0])
    mm_34: "f32[768, 768]" = torch.ops.aten.mm.default(permute_279, view_293);  permute_279 = view_293 = None
    permute_280: "f32[768, 768]" = torch.ops.aten.permute.default(mm_34, [1, 0]);  mm_34 = None
    sum_57: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_426, [0], True);  view_426 = None
    view_427: "f32[768]" = torch.ops.aten.reshape.default(sum_57, [768]);  sum_57 = None
    permute_281: "f32[768, 768]" = torch.ops.aten.permute.default(permute_280, [1, 0]);  permute_280 = None
    view_428: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_33, [4, 512, 768]);  mm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_138: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_173, view_428);  mul_173 = view_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_179: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_138, primals_217);  primals_217 = None
    mul_180: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_179, 768)
    sum_58: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_179, [2], True)
    mul_181: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_179, mul_99);  mul_179 = None
    sum_59: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_181, [2], True);  mul_181 = None
    mul_182: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_99, sum_59);  sum_59 = None
    sub_69: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_180, sum_58);  mul_180 = sum_58 = None
    sub_70: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_69, mul_182);  sub_69 = mul_182 = None
    mul_183: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_23, sub_70);  div_23 = sub_70 = None
    mul_184: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_138, mul_99);  mul_99 = None
    sum_60: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_184, [0, 1]);  mul_184 = None
    sum_61: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_138, [0, 1]);  add_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_429: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_183, [2048, 768])
    mm_35: "f32[2048, 768]" = torch.ops.aten.mm.default(view_429, permute_282);  permute_282 = None
    permute_283: "f32[768, 2048]" = torch.ops.aten.permute.default(view_429, [1, 0])
    mm_36: "f32[768, 768]" = torch.ops.aten.mm.default(permute_283, view_291);  permute_283 = view_291 = None
    permute_284: "f32[768, 768]" = torch.ops.aten.permute.default(mm_36, [1, 0]);  mm_36 = None
    sum_62: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_429, [0], True);  view_429 = None
    view_430: "f32[768]" = torch.ops.aten.reshape.default(sum_62, [768]);  sum_62 = None
    permute_285: "f32[768, 768]" = torch.ops.aten.permute.default(permute_284, [1, 0]);  permute_284 = None
    view_431: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_35, [4, 512, 768]);  mm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_432: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_431, [4, 512, 12, 64]);  view_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_286: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_432, [0, 2, 1, 3]);  view_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    clone_146: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_286, memory_format = torch.contiguous_format);  permute_286 = None
    view_433: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_146, [48, 512, 64]);  clone_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_48: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_287, view_433);  permute_287 = None
    bmm_49: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_433, permute_288);  view_433 = permute_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_185: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(bmm_49, alias_21);  bmm_49 = None
    sum_63: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_185, [-1], True)
    mul_186: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(alias_21, sum_63);  alias_21 = sum_63 = None
    sub_71: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(mul_185, mul_186);  mul_185 = mul_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_434: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(sub_71, [4, 12, 512, 512]);  sub_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_435: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(view_434, [48, 512, 512]);  view_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_50: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_289, view_435);  permute_289 = None
    bmm_51: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_435, permute_290);  view_435 = permute_290 = None
    permute_291: "f32[48, 512, 64]" = torch.ops.aten.permute.default(bmm_50, [0, 2, 1]);  bmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_436: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_48, [4, 12, 512, 64]);  bmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    add_139: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_19, view_436);  tangents_19 = view_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_437: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(permute_291, [4, 12, 512, 64]);  permute_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    add_140: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_18, view_437);  tangents_18 = view_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_438: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_51, [4, 12, 512, 64]);  bmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_292: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_438, [0, 2, 1, 3]);  view_438 = None
    clone_147: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_292, memory_format = torch.contiguous_format);  permute_292 = None
    view_439: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_147, [4, 512, 768]);  clone_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_293: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_139, [0, 2, 1, 3]);  add_139 = None
    clone_148: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
    view_440: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_148, [4, 512, 768]);  clone_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_441: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_440, [2048, 768]);  view_440 = None
    mm_37: "f32[2048, 768]" = torch.ops.aten.mm.default(view_441, permute_294);  permute_294 = None
    permute_295: "f32[768, 2048]" = torch.ops.aten.permute.default(view_441, [1, 0])
    mm_38: "f32[768, 768]" = torch.ops.aten.mm.default(permute_295, view_275);  permute_295 = None
    permute_296: "f32[768, 768]" = torch.ops.aten.permute.default(mm_38, [1, 0]);  mm_38 = None
    sum_64: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_441, [0], True);  view_441 = None
    view_442: "f32[768]" = torch.ops.aten.reshape.default(sum_64, [768]);  sum_64 = None
    permute_297: "f32[768, 768]" = torch.ops.aten.permute.default(permute_296, [1, 0]);  permute_296 = None
    view_443: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_37, [4, 512, 768]);  mm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    add_141: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_183, view_443);  mul_183 = view_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_298: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_140, [0, 2, 1, 3]);  add_140 = None
    clone_149: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_298, memory_format = torch.contiguous_format);  permute_298 = None
    view_444: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_149, [4, 512, 768]);  clone_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_445: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_444, [2048, 768]);  view_444 = None
    mm_39: "f32[2048, 768]" = torch.ops.aten.mm.default(view_445, permute_299);  permute_299 = None
    permute_300: "f32[768, 2048]" = torch.ops.aten.permute.default(view_445, [1, 0])
    mm_40: "f32[768, 768]" = torch.ops.aten.mm.default(permute_300, view_275);  permute_300 = None
    permute_301: "f32[768, 768]" = torch.ops.aten.permute.default(mm_40, [1, 0]);  mm_40 = None
    sum_65: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_445, [0], True);  view_445 = None
    view_446: "f32[768]" = torch.ops.aten.reshape.default(sum_65, [768]);  sum_65 = None
    permute_302: "f32[768, 768]" = torch.ops.aten.permute.default(permute_301, [1, 0]);  permute_301 = None
    view_447: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_39, [4, 512, 768]);  mm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_142: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_141, view_447);  add_141 = view_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_187: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_439, 0.125);  view_439 = None
    view_448: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_187, [2048, 768]);  mul_187 = None
    mm_41: "f32[2048, 768]" = torch.ops.aten.mm.default(view_448, permute_303);  permute_303 = None
    permute_304: "f32[768, 2048]" = torch.ops.aten.permute.default(view_448, [1, 0])
    mm_42: "f32[768, 768]" = torch.ops.aten.mm.default(permute_304, view_275);  permute_304 = view_275 = None
    permute_305: "f32[768, 768]" = torch.ops.aten.permute.default(mm_42, [1, 0]);  mm_42 = None
    sum_66: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_448, [0], True);  view_448 = None
    view_449: "f32[768]" = torch.ops.aten.reshape.default(sum_66, [768]);  sum_66 = None
    permute_306: "f32[768, 768]" = torch.ops.aten.permute.default(permute_305, [1, 0]);  permute_305 = None
    view_450: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_41, [4, 512, 768]);  mm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_143: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_142, view_450);  add_142 = view_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_189: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_143, primals_207);  primals_207 = None
    mul_190: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_189, 768)
    sum_67: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_189, [2], True)
    mul_191: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_189, mul_96);  mul_189 = None
    sum_68: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_191, [2], True);  mul_191 = None
    mul_192: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_96, sum_68);  sum_68 = None
    sub_73: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_190, sum_67);  mul_190 = sum_67 = None
    sub_74: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_73, mul_192);  sub_73 = mul_192 = None
    mul_193: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_24, sub_74);  div_24 = sub_74 = None
    mul_194: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_143, mul_96);  mul_96 = None
    sum_69: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_194, [0, 1]);  mul_194 = None
    sum_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_143, [0, 1]);  add_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_451: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_193, [2048, 768])
    mm_43: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_451, permute_307);  permute_307 = None
    permute_308: "f32[768, 2048]" = torch.ops.aten.permute.default(view_451, [1, 0])
    mm_44: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_308, view_273);  permute_308 = view_273 = None
    permute_309: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_44, [1, 0]);  mm_44 = None
    sum_71: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_451, [0], True);  view_451 = None
    view_452: "f32[768]" = torch.ops.aten.reshape.default(sum_71, [768]);  sum_71 = None
    permute_310: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_309, [1, 0]);  permute_309 = None
    view_453: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_43, [4, 512, 3072]);  mm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_196: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_91, 0.5);  add_91 = None
    mul_197: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_272, view_272)
    mul_198: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_197, -0.5);  mul_197 = None
    exp_20: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_198);  mul_198 = None
    mul_199: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
    mul_200: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_272, mul_199);  view_272 = mul_199 = None
    add_145: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_196, mul_200);  mul_196 = mul_200 = None
    mul_201: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_453, add_145);  view_453 = add_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_454: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_201, [2048, 3072]);  mul_201 = None
    mm_45: "f32[2048, 768]" = torch.ops.aten.mm.default(view_454, permute_311);  permute_311 = None
    permute_312: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_454, [1, 0])
    mm_46: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_312, view_271);  permute_312 = view_271 = None
    permute_313: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_46, [1, 0]);  mm_46 = None
    sum_72: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_454, [0], True);  view_454 = None
    view_455: "f32[3072]" = torch.ops.aten.reshape.default(sum_72, [3072]);  sum_72 = None
    permute_314: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_313, [1, 0]);  permute_313 = None
    view_456: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_45, [4, 512, 768]);  mm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    add_146: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_193, view_456);  mul_193 = view_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:456, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    mul_203: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_146, primals_201);  primals_201 = None
    mul_204: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_203, 768)
    sum_73: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_203, [2], True)
    mul_205: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_203, mul_91);  mul_203 = None
    sum_74: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_205, [2], True);  mul_205 = None
    mul_206: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_91, sum_74);  sum_74 = None
    sub_76: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_204, sum_73);  mul_204 = sum_73 = None
    sub_77: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_76, mul_206);  sub_76 = mul_206 = None
    mul_207: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_25, sub_77);  div_25 = sub_77 = None
    mul_208: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_146, mul_91);  mul_91 = None
    sum_75: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_208, [0, 1]);  mul_208 = None
    sum_76: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_146, [0, 1]);  add_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_457: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_207, [2048, 768])
    mm_47: "f32[2048, 768]" = torch.ops.aten.mm.default(view_457, permute_315);  permute_315 = None
    permute_316: "f32[768, 2048]" = torch.ops.aten.permute.default(view_457, [1, 0])
    mm_48: "f32[768, 768]" = torch.ops.aten.mm.default(permute_316, view_269);  permute_316 = view_269 = None
    permute_317: "f32[768, 768]" = torch.ops.aten.permute.default(mm_48, [1, 0]);  mm_48 = None
    sum_77: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_457, [0], True);  view_457 = None
    view_458: "f32[768]" = torch.ops.aten.reshape.default(sum_77, [768]);  sum_77 = None
    permute_318: "f32[768, 768]" = torch.ops.aten.permute.default(permute_317, [1, 0]);  permute_317 = None
    view_459: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_47, [4, 512, 768]);  mm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_460: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_459, [4, 512, 12, 64]);  view_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_319: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_460, [0, 2, 1, 3]);  view_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    clone_150: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_319, memory_format = torch.contiguous_format);  permute_319 = None
    view_461: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_150, [48, 512, 64]);  clone_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_52: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_320, view_461);  permute_320 = None
    bmm_53: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_461, permute_321);  view_461 = permute_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_209: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(bmm_53, div_13);  bmm_53 = None
    sum_78: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_209, [-1], True)
    mul_210: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(div_13, sum_78);  div_13 = sum_78 = None
    sub_78: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(mul_209, mul_210);  mul_209 = mul_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_54: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_322, sub_78);  permute_322 = None
    bmm_55: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(sub_78, permute_323);  sub_78 = permute_323 = None
    permute_324: "f32[48, 512, 64]" = torch.ops.aten.permute.default(bmm_54, [0, 2, 1]);  bmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_462: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_52, [4, 12, 512, 64]);  bmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    add_147: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_17, view_462);  tangents_17 = view_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_463: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(permute_324, [4, 12, 512, 64]);  permute_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    add_148: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_16, view_463);  tangents_16 = view_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_464: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_55, [4, 12, 512, 64]);  bmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_325: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_464, [0, 2, 1, 3]);  view_464 = None
    clone_151: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_325, memory_format = torch.contiguous_format);  permute_325 = None
    view_465: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_151, [4, 512, 768]);  clone_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_326: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_147, [0, 2, 1, 3]);  add_147 = None
    clone_152: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_326, memory_format = torch.contiguous_format);  permute_326 = None
    view_466: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_152, [4, 512, 768]);  clone_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_467: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_466, [2048, 768]);  view_466 = None
    mm_49: "f32[2048, 768]" = torch.ops.aten.mm.default(view_467, permute_327);  permute_327 = None
    permute_328: "f32[768, 2048]" = torch.ops.aten.permute.default(view_467, [1, 0])
    mm_50: "f32[768, 768]" = torch.ops.aten.mm.default(permute_328, view_143);  permute_328 = None
    permute_329: "f32[768, 768]" = torch.ops.aten.permute.default(mm_50, [1, 0]);  mm_50 = None
    sum_79: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_467, [0], True);  view_467 = None
    view_468: "f32[768]" = torch.ops.aten.reshape.default(sum_79, [768]);  sum_79 = None
    permute_330: "f32[768, 768]" = torch.ops.aten.permute.default(permute_329, [1, 0]);  permute_329 = None
    view_469: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_49, [4, 512, 768]);  mm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    add_149: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_137, view_469);  add_137 = view_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_331: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_148, [0, 2, 1, 3]);  add_148 = None
    clone_153: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_331, memory_format = torch.contiguous_format);  permute_331 = None
    view_470: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_153, [4, 512, 768]);  clone_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_471: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_470, [2048, 768]);  view_470 = None
    mm_51: "f32[2048, 768]" = torch.ops.aten.mm.default(view_471, permute_332);  permute_332 = None
    permute_333: "f32[768, 2048]" = torch.ops.aten.permute.default(view_471, [1, 0])
    mm_52: "f32[768, 768]" = torch.ops.aten.mm.default(permute_333, view_143);  permute_333 = None
    permute_334: "f32[768, 768]" = torch.ops.aten.permute.default(mm_52, [1, 0]);  mm_52 = None
    sum_80: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_471, [0], True);  view_471 = None
    view_472: "f32[768]" = torch.ops.aten.reshape.default(sum_80, [768]);  sum_80 = None
    permute_335: "f32[768, 768]" = torch.ops.aten.permute.default(permute_334, [1, 0]);  permute_334 = None
    view_473: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_51, [4, 512, 768]);  mm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    add_150: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_149, view_473);  add_149 = view_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_211: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_465, 0.125);  view_465 = None
    view_474: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_211, [2048, 768]);  mul_211 = None
    mm_53: "f32[2048, 768]" = torch.ops.aten.mm.default(view_474, permute_336);  permute_336 = None
    permute_337: "f32[768, 2048]" = torch.ops.aten.permute.default(view_474, [1, 0])
    mm_54: "f32[768, 768]" = torch.ops.aten.mm.default(permute_337, view_255);  permute_337 = view_255 = None
    permute_338: "f32[768, 768]" = torch.ops.aten.permute.default(mm_54, [1, 0]);  mm_54 = None
    sum_81: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_474, [0], True);  view_474 = None
    view_475: "f32[768]" = torch.ops.aten.reshape.default(sum_81, [768]);  sum_81 = None
    permute_339: "f32[768, 768]" = torch.ops.aten.permute.default(permute_338, [1, 0]);  permute_338 = None
    view_476: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_53, [4, 512, 768]);  mm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_151: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_207, view_476);  mul_207 = view_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_213: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_151, primals_191);  primals_191 = None
    mul_214: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_213, 768)
    sum_82: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_213, [2], True)
    mul_215: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_213, mul_88);  mul_213 = None
    sum_83: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_215, [2], True);  mul_215 = None
    mul_216: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_88, sum_83);  sum_83 = None
    sub_80: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_214, sum_82);  mul_214 = sum_82 = None
    sub_81: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_80, mul_216);  sub_80 = mul_216 = None
    mul_217: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_26, sub_81);  div_26 = sub_81 = None
    mul_218: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_151, mul_88);  mul_88 = None
    sum_84: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_218, [0, 1]);  mul_218 = None
    sum_85: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_151, [0, 1]);  add_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_477: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_217, [2048, 768])
    mm_55: "f32[2048, 768]" = torch.ops.aten.mm.default(view_477, permute_340);  permute_340 = None
    permute_341: "f32[768, 2048]" = torch.ops.aten.permute.default(view_477, [1, 0])
    mm_56: "f32[768, 768]" = torch.ops.aten.mm.default(permute_341, view_253);  permute_341 = view_253 = None
    permute_342: "f32[768, 768]" = torch.ops.aten.permute.default(mm_56, [1, 0]);  mm_56 = None
    sum_86: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_477, [0], True);  view_477 = None
    view_478: "f32[768]" = torch.ops.aten.reshape.default(sum_86, [768]);  sum_86 = None
    permute_343: "f32[768, 768]" = torch.ops.aten.permute.default(permute_342, [1, 0]);  permute_342 = None
    view_479: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_55, [4, 512, 768]);  mm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_480: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_479, [4, 512, 12, 64]);  view_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_344: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_480, [0, 2, 1, 3]);  view_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    clone_154: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_344, memory_format = torch.contiguous_format);  permute_344 = None
    view_481: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_154, [48, 512, 64]);  clone_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_56: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_345, view_481);  permute_345 = None
    bmm_57: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_481, permute_346);  view_481 = permute_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_219: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(bmm_57, alias_23);  bmm_57 = None
    sum_87: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_219, [-1], True)
    mul_220: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(alias_23, sum_87);  alias_23 = sum_87 = None
    sub_82: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(mul_219, mul_220);  mul_219 = mul_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_482: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(sub_82, [4, 12, 512, 512]);  sub_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_483: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(view_482, [48, 512, 512]);  view_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_58: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_347, view_483);  permute_347 = None
    bmm_59: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_483, permute_348);  view_483 = permute_348 = None
    permute_349: "f32[48, 512, 64]" = torch.ops.aten.permute.default(bmm_58, [0, 2, 1]);  bmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_484: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_56, [4, 12, 512, 64]);  bmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    add_152: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_15, view_484);  tangents_15 = view_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_485: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(permute_349, [4, 12, 512, 64]);  permute_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    add_153: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_14, view_485);  tangents_14 = view_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_486: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_59, [4, 12, 512, 64]);  bmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_350: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_486, [0, 2, 1, 3]);  view_486 = None
    clone_155: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_350, memory_format = torch.contiguous_format);  permute_350 = None
    view_487: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_155, [4, 512, 768]);  clone_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_351: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_152, [0, 2, 1, 3]);  add_152 = None
    clone_156: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_351, memory_format = torch.contiguous_format);  permute_351 = None
    view_488: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_156, [4, 512, 768]);  clone_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_489: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_488, [2048, 768]);  view_488 = None
    mm_57: "f32[2048, 768]" = torch.ops.aten.mm.default(view_489, permute_352);  permute_352 = None
    permute_353: "f32[768, 2048]" = torch.ops.aten.permute.default(view_489, [1, 0])
    mm_58: "f32[768, 768]" = torch.ops.aten.mm.default(permute_353, view_237);  permute_353 = None
    permute_354: "f32[768, 768]" = torch.ops.aten.permute.default(mm_58, [1, 0]);  mm_58 = None
    sum_88: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_489, [0], True);  view_489 = None
    view_490: "f32[768]" = torch.ops.aten.reshape.default(sum_88, [768]);  sum_88 = None
    permute_355: "f32[768, 768]" = torch.ops.aten.permute.default(permute_354, [1, 0]);  permute_354 = None
    view_491: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_57, [4, 512, 768]);  mm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    add_154: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_217, view_491);  mul_217 = view_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_356: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_153, [0, 2, 1, 3]);  add_153 = None
    clone_157: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_356, memory_format = torch.contiguous_format);  permute_356 = None
    view_492: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_157, [4, 512, 768]);  clone_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_493: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_492, [2048, 768]);  view_492 = None
    mm_59: "f32[2048, 768]" = torch.ops.aten.mm.default(view_493, permute_357);  permute_357 = None
    permute_358: "f32[768, 2048]" = torch.ops.aten.permute.default(view_493, [1, 0])
    mm_60: "f32[768, 768]" = torch.ops.aten.mm.default(permute_358, view_237);  permute_358 = None
    permute_359: "f32[768, 768]" = torch.ops.aten.permute.default(mm_60, [1, 0]);  mm_60 = None
    sum_89: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_493, [0], True);  view_493 = None
    view_494: "f32[768]" = torch.ops.aten.reshape.default(sum_89, [768]);  sum_89 = None
    permute_360: "f32[768, 768]" = torch.ops.aten.permute.default(permute_359, [1, 0]);  permute_359 = None
    view_495: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_59, [4, 512, 768]);  mm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_155: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_154, view_495);  add_154 = view_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_221: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_487, 0.125);  view_487 = None
    view_496: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_221, [2048, 768]);  mul_221 = None
    mm_61: "f32[2048, 768]" = torch.ops.aten.mm.default(view_496, permute_361);  permute_361 = None
    permute_362: "f32[768, 2048]" = torch.ops.aten.permute.default(view_496, [1, 0])
    mm_62: "f32[768, 768]" = torch.ops.aten.mm.default(permute_362, view_237);  permute_362 = view_237 = None
    permute_363: "f32[768, 768]" = torch.ops.aten.permute.default(mm_62, [1, 0]);  mm_62 = None
    sum_90: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_496, [0], True);  view_496 = None
    view_497: "f32[768]" = torch.ops.aten.reshape.default(sum_90, [768]);  sum_90 = None
    permute_364: "f32[768, 768]" = torch.ops.aten.permute.default(permute_363, [1, 0]);  permute_363 = None
    view_498: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_61, [4, 512, 768]);  mm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_156: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_155, view_498);  add_155 = view_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_223: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_156, primals_181);  primals_181 = None
    mul_224: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_223, 768)
    sum_91: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_223, [2], True)
    mul_225: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_223, mul_85);  mul_223 = None
    sum_92: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_225, [2], True);  mul_225 = None
    mul_226: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_85, sum_92);  sum_92 = None
    sub_84: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_224, sum_91);  mul_224 = sum_91 = None
    sub_85: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_84, mul_226);  sub_84 = mul_226 = None
    mul_227: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_27, sub_85);  div_27 = sub_85 = None
    mul_228: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_156, mul_85);  mul_85 = None
    sum_93: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_228, [0, 1]);  mul_228 = None
    sum_94: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_156, [0, 1]);  add_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_499: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_227, [2048, 768])
    mm_63: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_499, permute_365);  permute_365 = None
    permute_366: "f32[768, 2048]" = torch.ops.aten.permute.default(view_499, [1, 0])
    mm_64: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_366, view_235);  permute_366 = view_235 = None
    permute_367: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_64, [1, 0]);  mm_64 = None
    sum_95: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_499, [0], True);  view_499 = None
    view_500: "f32[768]" = torch.ops.aten.reshape.default(sum_95, [768]);  sum_95 = None
    permute_368: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_367, [1, 0]);  permute_367 = None
    view_501: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_63, [4, 512, 3072]);  mm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_230: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_80, 0.5);  add_80 = None
    mul_231: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_234, view_234)
    mul_232: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_231, -0.5);  mul_231 = None
    exp_21: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_232);  mul_232 = None
    mul_233: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
    mul_234: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_234, mul_233);  view_234 = mul_233 = None
    add_158: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_230, mul_234);  mul_230 = mul_234 = None
    mul_235: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_501, add_158);  view_501 = add_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_502: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_235, [2048, 3072]);  mul_235 = None
    mm_65: "f32[2048, 768]" = torch.ops.aten.mm.default(view_502, permute_369);  permute_369 = None
    permute_370: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_502, [1, 0])
    mm_66: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_370, view_233);  permute_370 = view_233 = None
    permute_371: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_66, [1, 0]);  mm_66 = None
    sum_96: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_502, [0], True);  view_502 = None
    view_503: "f32[3072]" = torch.ops.aten.reshape.default(sum_96, [3072]);  sum_96 = None
    permute_372: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_371, [1, 0]);  permute_371 = None
    view_504: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_65, [4, 512, 768]);  mm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    add_159: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_227, view_504);  mul_227 = view_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:456, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    mul_237: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_159, primals_175);  primals_175 = None
    mul_238: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_237, 768)
    sum_97: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_237, [2], True)
    mul_239: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_237, mul_80);  mul_237 = None
    sum_98: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_239, [2], True);  mul_239 = None
    mul_240: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_80, sum_98);  sum_98 = None
    sub_87: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_238, sum_97);  mul_238 = sum_97 = None
    sub_88: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_87, mul_240);  sub_87 = mul_240 = None
    mul_241: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_28, sub_88);  div_28 = sub_88 = None
    mul_242: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_159, mul_80);  mul_80 = None
    sum_99: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_242, [0, 1]);  mul_242 = None
    sum_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_159, [0, 1]);  add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_505: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_241, [2048, 768])
    mm_67: "f32[2048, 768]" = torch.ops.aten.mm.default(view_505, permute_373);  permute_373 = None
    permute_374: "f32[768, 2048]" = torch.ops.aten.permute.default(view_505, [1, 0])
    mm_68: "f32[768, 768]" = torch.ops.aten.mm.default(permute_374, view_231);  permute_374 = view_231 = None
    permute_375: "f32[768, 768]" = torch.ops.aten.permute.default(mm_68, [1, 0]);  mm_68 = None
    sum_101: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_505, [0], True);  view_505 = None
    view_506: "f32[768]" = torch.ops.aten.reshape.default(sum_101, [768]);  sum_101 = None
    permute_376: "f32[768, 768]" = torch.ops.aten.permute.default(permute_375, [1, 0]);  permute_375 = None
    view_507: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_67, [4, 512, 768]);  mm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_508: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_507, [4, 512, 12, 64]);  view_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_377: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_508, [0, 2, 1, 3]);  view_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    clone_158: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_377, memory_format = torch.contiguous_format);  permute_377 = None
    view_509: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_158, [48, 512, 64]);  clone_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_60: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_378, view_509);  permute_378 = None
    bmm_61: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_509, permute_379);  view_509 = permute_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_243: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(bmm_61, div_11);  bmm_61 = None
    sum_102: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_243, [-1], True)
    mul_244: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(div_11, sum_102);  div_11 = sum_102 = None
    sub_89: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(mul_243, mul_244);  mul_243 = mul_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_62: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_380, sub_89);  permute_380 = None
    bmm_63: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(sub_89, permute_381);  sub_89 = permute_381 = None
    permute_382: "f32[48, 512, 64]" = torch.ops.aten.permute.default(bmm_62, [0, 2, 1]);  bmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_510: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_60, [4, 12, 512, 64]);  bmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    add_160: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_13, view_510);  tangents_13 = view_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_511: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(permute_382, [4, 12, 512, 64]);  permute_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    add_161: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_12, view_511);  tangents_12 = view_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_512: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_63, [4, 12, 512, 64]);  bmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_383: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_512, [0, 2, 1, 3]);  view_512 = None
    clone_159: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_383, memory_format = torch.contiguous_format);  permute_383 = None
    view_513: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_159, [4, 512, 768]);  clone_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_384: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_160, [0, 2, 1, 3]);  add_160 = None
    clone_160: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_384, memory_format = torch.contiguous_format);  permute_384 = None
    view_514: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_160, [4, 512, 768]);  clone_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_515: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_514, [2048, 768]);  view_514 = None
    mm_69: "f32[2048, 768]" = torch.ops.aten.mm.default(view_515, permute_385);  permute_385 = None
    permute_386: "f32[768, 2048]" = torch.ops.aten.permute.default(view_515, [1, 0])
    mm_70: "f32[768, 768]" = torch.ops.aten.mm.default(permute_386, view_143);  permute_386 = None
    permute_387: "f32[768, 768]" = torch.ops.aten.permute.default(mm_70, [1, 0]);  mm_70 = None
    sum_103: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_515, [0], True);  view_515 = None
    view_516: "f32[768]" = torch.ops.aten.reshape.default(sum_103, [768]);  sum_103 = None
    permute_388: "f32[768, 768]" = torch.ops.aten.permute.default(permute_387, [1, 0]);  permute_387 = None
    view_517: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_69, [4, 512, 768]);  mm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    add_162: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_150, view_517);  add_150 = view_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_389: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_161, [0, 2, 1, 3]);  add_161 = None
    clone_161: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_389, memory_format = torch.contiguous_format);  permute_389 = None
    view_518: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_161, [4, 512, 768]);  clone_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_519: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_518, [2048, 768]);  view_518 = None
    mm_71: "f32[2048, 768]" = torch.ops.aten.mm.default(view_519, permute_390);  permute_390 = None
    permute_391: "f32[768, 2048]" = torch.ops.aten.permute.default(view_519, [1, 0])
    mm_72: "f32[768, 768]" = torch.ops.aten.mm.default(permute_391, view_143);  permute_391 = None
    permute_392: "f32[768, 768]" = torch.ops.aten.permute.default(mm_72, [1, 0]);  mm_72 = None
    sum_104: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_519, [0], True);  view_519 = None
    view_520: "f32[768]" = torch.ops.aten.reshape.default(sum_104, [768]);  sum_104 = None
    permute_393: "f32[768, 768]" = torch.ops.aten.permute.default(permute_392, [1, 0]);  permute_392 = None
    view_521: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_71, [4, 512, 768]);  mm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    add_163: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_162, view_521);  add_162 = view_521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_245: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_513, 0.125);  view_513 = None
    view_522: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_245, [2048, 768]);  mul_245 = None
    mm_73: "f32[2048, 768]" = torch.ops.aten.mm.default(view_522, permute_394);  permute_394 = None
    permute_395: "f32[768, 2048]" = torch.ops.aten.permute.default(view_522, [1, 0])
    mm_74: "f32[768, 768]" = torch.ops.aten.mm.default(permute_395, view_217);  permute_395 = view_217 = None
    permute_396: "f32[768, 768]" = torch.ops.aten.permute.default(mm_74, [1, 0]);  mm_74 = None
    sum_105: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_522, [0], True);  view_522 = None
    view_523: "f32[768]" = torch.ops.aten.reshape.default(sum_105, [768]);  sum_105 = None
    permute_397: "f32[768, 768]" = torch.ops.aten.permute.default(permute_396, [1, 0]);  permute_396 = None
    view_524: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_73, [4, 512, 768]);  mm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_164: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_241, view_524);  mul_241 = view_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_247: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_164, primals_165);  primals_165 = None
    mul_248: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_247, 768)
    sum_106: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_247, [2], True)
    mul_249: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_247, mul_77);  mul_247 = None
    sum_107: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_249, [2], True);  mul_249 = None
    mul_250: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_77, sum_107);  sum_107 = None
    sub_91: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_248, sum_106);  mul_248 = sum_106 = None
    sub_92: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_91, mul_250);  sub_91 = mul_250 = None
    mul_251: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_29, sub_92);  div_29 = sub_92 = None
    mul_252: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_164, mul_77);  mul_77 = None
    sum_108: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_252, [0, 1]);  mul_252 = None
    sum_109: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_164, [0, 1]);  add_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_525: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_251, [2048, 768])
    mm_75: "f32[2048, 768]" = torch.ops.aten.mm.default(view_525, permute_398);  permute_398 = None
    permute_399: "f32[768, 2048]" = torch.ops.aten.permute.default(view_525, [1, 0])
    mm_76: "f32[768, 768]" = torch.ops.aten.mm.default(permute_399, view_215);  permute_399 = view_215 = None
    permute_400: "f32[768, 768]" = torch.ops.aten.permute.default(mm_76, [1, 0]);  mm_76 = None
    sum_110: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_525, [0], True);  view_525 = None
    view_526: "f32[768]" = torch.ops.aten.reshape.default(sum_110, [768]);  sum_110 = None
    permute_401: "f32[768, 768]" = torch.ops.aten.permute.default(permute_400, [1, 0]);  permute_400 = None
    view_527: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_75, [4, 512, 768]);  mm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_528: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_527, [4, 512, 12, 64]);  view_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_402: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_528, [0, 2, 1, 3]);  view_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    clone_162: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_402, memory_format = torch.contiguous_format);  permute_402 = None
    view_529: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_162, [48, 512, 64]);  clone_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_64: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_403, view_529);  permute_403 = None
    bmm_65: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_529, permute_404);  view_529 = permute_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_253: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(bmm_65, alias_25);  bmm_65 = None
    sum_111: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_253, [-1], True)
    mul_254: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(alias_25, sum_111);  alias_25 = sum_111 = None
    sub_93: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(mul_253, mul_254);  mul_253 = mul_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_530: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(sub_93, [4, 12, 512, 512]);  sub_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_531: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(view_530, [48, 512, 512]);  view_530 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_66: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_405, view_531);  permute_405 = None
    bmm_67: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_531, permute_406);  view_531 = permute_406 = None
    permute_407: "f32[48, 512, 64]" = torch.ops.aten.permute.default(bmm_66, [0, 2, 1]);  bmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_532: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_64, [4, 12, 512, 64]);  bmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    add_165: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_11, view_532);  tangents_11 = view_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_533: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(permute_407, [4, 12, 512, 64]);  permute_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    add_166: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_10, view_533);  tangents_10 = view_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_534: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_67, [4, 12, 512, 64]);  bmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_408: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_534, [0, 2, 1, 3]);  view_534 = None
    clone_163: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_408, memory_format = torch.contiguous_format);  permute_408 = None
    view_535: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_163, [4, 512, 768]);  clone_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_409: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_165, [0, 2, 1, 3]);  add_165 = None
    clone_164: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_409, memory_format = torch.contiguous_format);  permute_409 = None
    view_536: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_164, [4, 512, 768]);  clone_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_537: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_536, [2048, 768]);  view_536 = None
    mm_77: "f32[2048, 768]" = torch.ops.aten.mm.default(view_537, permute_410);  permute_410 = None
    permute_411: "f32[768, 2048]" = torch.ops.aten.permute.default(view_537, [1, 0])
    mm_78: "f32[768, 768]" = torch.ops.aten.mm.default(permute_411, view_199);  permute_411 = None
    permute_412: "f32[768, 768]" = torch.ops.aten.permute.default(mm_78, [1, 0]);  mm_78 = None
    sum_112: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_537, [0], True);  view_537 = None
    view_538: "f32[768]" = torch.ops.aten.reshape.default(sum_112, [768]);  sum_112 = None
    permute_413: "f32[768, 768]" = torch.ops.aten.permute.default(permute_412, [1, 0]);  permute_412 = None
    view_539: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_77, [4, 512, 768]);  mm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    add_167: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_251, view_539);  mul_251 = view_539 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_414: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_166, [0, 2, 1, 3]);  add_166 = None
    clone_165: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_414, memory_format = torch.contiguous_format);  permute_414 = None
    view_540: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_165, [4, 512, 768]);  clone_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_541: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_540, [2048, 768]);  view_540 = None
    mm_79: "f32[2048, 768]" = torch.ops.aten.mm.default(view_541, permute_415);  permute_415 = None
    permute_416: "f32[768, 2048]" = torch.ops.aten.permute.default(view_541, [1, 0])
    mm_80: "f32[768, 768]" = torch.ops.aten.mm.default(permute_416, view_199);  permute_416 = None
    permute_417: "f32[768, 768]" = torch.ops.aten.permute.default(mm_80, [1, 0]);  mm_80 = None
    sum_113: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_541, [0], True);  view_541 = None
    view_542: "f32[768]" = torch.ops.aten.reshape.default(sum_113, [768]);  sum_113 = None
    permute_418: "f32[768, 768]" = torch.ops.aten.permute.default(permute_417, [1, 0]);  permute_417 = None
    view_543: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_79, [4, 512, 768]);  mm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_168: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_167, view_543);  add_167 = view_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_255: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_535, 0.125);  view_535 = None
    view_544: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_255, [2048, 768]);  mul_255 = None
    mm_81: "f32[2048, 768]" = torch.ops.aten.mm.default(view_544, permute_419);  permute_419 = None
    permute_420: "f32[768, 2048]" = torch.ops.aten.permute.default(view_544, [1, 0])
    mm_82: "f32[768, 768]" = torch.ops.aten.mm.default(permute_420, view_199);  permute_420 = view_199 = None
    permute_421: "f32[768, 768]" = torch.ops.aten.permute.default(mm_82, [1, 0]);  mm_82 = None
    sum_114: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_544, [0], True);  view_544 = None
    view_545: "f32[768]" = torch.ops.aten.reshape.default(sum_114, [768]);  sum_114 = None
    permute_422: "f32[768, 768]" = torch.ops.aten.permute.default(permute_421, [1, 0]);  permute_421 = None
    view_546: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_81, [4, 512, 768]);  mm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_169: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_168, view_546);  add_168 = view_546 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_257: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_169, primals_155);  primals_155 = None
    mul_258: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_257, 768)
    sum_115: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_257, [2], True)
    mul_259: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_257, mul_74);  mul_257 = None
    sum_116: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_259, [2], True);  mul_259 = None
    mul_260: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_74, sum_116);  sum_116 = None
    sub_95: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_258, sum_115);  mul_258 = sum_115 = None
    sub_96: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_95, mul_260);  sub_95 = mul_260 = None
    mul_261: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_30, sub_96);  div_30 = sub_96 = None
    mul_262: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_169, mul_74);  mul_74 = None
    sum_117: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_262, [0, 1]);  mul_262 = None
    sum_118: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_169, [0, 1]);  add_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_547: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_261, [2048, 768])
    mm_83: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_547, permute_423);  permute_423 = None
    permute_424: "f32[768, 2048]" = torch.ops.aten.permute.default(view_547, [1, 0])
    mm_84: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_424, view_197);  permute_424 = view_197 = None
    permute_425: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_84, [1, 0]);  mm_84 = None
    sum_119: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_547, [0], True);  view_547 = None
    view_548: "f32[768]" = torch.ops.aten.reshape.default(sum_119, [768]);  sum_119 = None
    permute_426: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_425, [1, 0]);  permute_425 = None
    view_549: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_83, [4, 512, 3072]);  mm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_264: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_69, 0.5);  add_69 = None
    mul_265: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_196, view_196)
    mul_266: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_265, -0.5);  mul_265 = None
    exp_22: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_266);  mul_266 = None
    mul_267: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_268: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_196, mul_267);  view_196 = mul_267 = None
    add_171: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_264, mul_268);  mul_264 = mul_268 = None
    mul_269: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_549, add_171);  view_549 = add_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_550: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_269, [2048, 3072]);  mul_269 = None
    mm_85: "f32[2048, 768]" = torch.ops.aten.mm.default(view_550, permute_427);  permute_427 = None
    permute_428: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_550, [1, 0])
    mm_86: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_428, view_195);  permute_428 = view_195 = None
    permute_429: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_86, [1, 0]);  mm_86 = None
    sum_120: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_550, [0], True);  view_550 = None
    view_551: "f32[3072]" = torch.ops.aten.reshape.default(sum_120, [3072]);  sum_120 = None
    permute_430: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_429, [1, 0]);  permute_429 = None
    view_552: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_85, [4, 512, 768]);  mm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    add_172: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_261, view_552);  mul_261 = view_552 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:456, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    mul_271: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_172, primals_149);  primals_149 = None
    mul_272: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_271, 768)
    sum_121: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_271, [2], True)
    mul_273: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_271, mul_69);  mul_271 = None
    sum_122: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_273, [2], True);  mul_273 = None
    mul_274: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_69, sum_122);  sum_122 = None
    sub_98: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_272, sum_121);  mul_272 = sum_121 = None
    sub_99: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_98, mul_274);  sub_98 = mul_274 = None
    mul_275: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_31, sub_99);  div_31 = sub_99 = None
    mul_276: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_172, mul_69);  mul_69 = None
    sum_123: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_276, [0, 1]);  mul_276 = None
    sum_124: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_172, [0, 1]);  add_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_553: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_275, [2048, 768])
    mm_87: "f32[2048, 768]" = torch.ops.aten.mm.default(view_553, permute_431);  permute_431 = None
    permute_432: "f32[768, 2048]" = torch.ops.aten.permute.default(view_553, [1, 0])
    mm_88: "f32[768, 768]" = torch.ops.aten.mm.default(permute_432, view_193);  permute_432 = view_193 = None
    permute_433: "f32[768, 768]" = torch.ops.aten.permute.default(mm_88, [1, 0]);  mm_88 = None
    sum_125: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_553, [0], True);  view_553 = None
    view_554: "f32[768]" = torch.ops.aten.reshape.default(sum_125, [768]);  sum_125 = None
    permute_434: "f32[768, 768]" = torch.ops.aten.permute.default(permute_433, [1, 0]);  permute_433 = None
    view_555: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_87, [4, 512, 768]);  mm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_556: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_555, [4, 512, 12, 64]);  view_555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_435: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_556, [0, 2, 1, 3]);  view_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    clone_166: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_435, memory_format = torch.contiguous_format);  permute_435 = None
    view_557: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_166, [48, 512, 64]);  clone_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_68: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_436, view_557);  permute_436 = None
    bmm_69: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_557, permute_437);  view_557 = permute_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_277: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(bmm_69, div_9);  bmm_69 = None
    sum_126: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_277, [-1], True)
    mul_278: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(div_9, sum_126);  div_9 = sum_126 = None
    sub_100: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_70: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_438, sub_100);  permute_438 = None
    bmm_71: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(sub_100, permute_439);  sub_100 = permute_439 = None
    permute_440: "f32[48, 512, 64]" = torch.ops.aten.permute.default(bmm_70, [0, 2, 1]);  bmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_558: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_68, [4, 12, 512, 64]);  bmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    add_173: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_9, view_558);  tangents_9 = view_558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_559: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(permute_440, [4, 12, 512, 64]);  permute_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    add_174: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_8, view_559);  tangents_8 = view_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_560: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_71, [4, 12, 512, 64]);  bmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_441: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_560, [0, 2, 1, 3]);  view_560 = None
    clone_167: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_441, memory_format = torch.contiguous_format);  permute_441 = None
    view_561: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_167, [4, 512, 768]);  clone_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_442: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_173, [0, 2, 1, 3]);  add_173 = None
    clone_168: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_442, memory_format = torch.contiguous_format);  permute_442 = None
    view_562: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_168, [4, 512, 768]);  clone_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_563: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_562, [2048, 768]);  view_562 = None
    mm_89: "f32[2048, 768]" = torch.ops.aten.mm.default(view_563, permute_443);  permute_443 = None
    permute_444: "f32[768, 2048]" = torch.ops.aten.permute.default(view_563, [1, 0])
    mm_90: "f32[768, 768]" = torch.ops.aten.mm.default(permute_444, view_143);  permute_444 = None
    permute_445: "f32[768, 768]" = torch.ops.aten.permute.default(mm_90, [1, 0]);  mm_90 = None
    sum_127: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_563, [0], True);  view_563 = None
    view_564: "f32[768]" = torch.ops.aten.reshape.default(sum_127, [768]);  sum_127 = None
    permute_446: "f32[768, 768]" = torch.ops.aten.permute.default(permute_445, [1, 0]);  permute_445 = None
    view_565: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_89, [4, 512, 768]);  mm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    add_175: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_163, view_565);  add_163 = view_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_447: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_174, [0, 2, 1, 3]);  add_174 = None
    clone_169: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_447, memory_format = torch.contiguous_format);  permute_447 = None
    view_566: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_169, [4, 512, 768]);  clone_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_567: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_566, [2048, 768]);  view_566 = None
    mm_91: "f32[2048, 768]" = torch.ops.aten.mm.default(view_567, permute_448);  permute_448 = None
    permute_449: "f32[768, 2048]" = torch.ops.aten.permute.default(view_567, [1, 0])
    mm_92: "f32[768, 768]" = torch.ops.aten.mm.default(permute_449, view_143);  permute_449 = None
    permute_450: "f32[768, 768]" = torch.ops.aten.permute.default(mm_92, [1, 0]);  mm_92 = None
    sum_128: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_567, [0], True);  view_567 = None
    view_568: "f32[768]" = torch.ops.aten.reshape.default(sum_128, [768]);  sum_128 = None
    permute_451: "f32[768, 768]" = torch.ops.aten.permute.default(permute_450, [1, 0]);  permute_450 = None
    view_569: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_91, [4, 512, 768]);  mm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    add_176: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_175, view_569);  add_175 = view_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_279: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_561, 0.125);  view_561 = None
    view_570: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_279, [2048, 768]);  mul_279 = None
    mm_93: "f32[2048, 768]" = torch.ops.aten.mm.default(view_570, permute_452);  permute_452 = None
    permute_453: "f32[768, 2048]" = torch.ops.aten.permute.default(view_570, [1, 0])
    mm_94: "f32[768, 768]" = torch.ops.aten.mm.default(permute_453, view_179);  permute_453 = view_179 = None
    permute_454: "f32[768, 768]" = torch.ops.aten.permute.default(mm_94, [1, 0]);  mm_94 = None
    sum_129: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_570, [0], True);  view_570 = None
    view_571: "f32[768]" = torch.ops.aten.reshape.default(sum_129, [768]);  sum_129 = None
    permute_455: "f32[768, 768]" = torch.ops.aten.permute.default(permute_454, [1, 0]);  permute_454 = None
    view_572: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_93, [4, 512, 768]);  mm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_177: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_275, view_572);  mul_275 = view_572 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_281: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_177, primals_139);  primals_139 = None
    mul_282: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_281, 768)
    sum_130: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_281, [2], True)
    mul_283: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_281, mul_66);  mul_281 = None
    sum_131: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_283, [2], True);  mul_283 = None
    mul_284: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_66, sum_131);  sum_131 = None
    sub_102: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_282, sum_130);  mul_282 = sum_130 = None
    sub_103: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_102, mul_284);  sub_102 = mul_284 = None
    mul_285: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_32, sub_103);  div_32 = sub_103 = None
    mul_286: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_177, mul_66);  mul_66 = None
    sum_132: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_286, [0, 1]);  mul_286 = None
    sum_133: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_177, [0, 1]);  add_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_573: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_285, [2048, 768])
    mm_95: "f32[2048, 768]" = torch.ops.aten.mm.default(view_573, permute_456);  permute_456 = None
    permute_457: "f32[768, 2048]" = torch.ops.aten.permute.default(view_573, [1, 0])
    mm_96: "f32[768, 768]" = torch.ops.aten.mm.default(permute_457, view_177);  permute_457 = view_177 = None
    permute_458: "f32[768, 768]" = torch.ops.aten.permute.default(mm_96, [1, 0]);  mm_96 = None
    sum_134: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_573, [0], True);  view_573 = None
    view_574: "f32[768]" = torch.ops.aten.reshape.default(sum_134, [768]);  sum_134 = None
    permute_459: "f32[768, 768]" = torch.ops.aten.permute.default(permute_458, [1, 0]);  permute_458 = None
    view_575: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_95, [4, 512, 768]);  mm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_576: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_575, [4, 512, 12, 64]);  view_575 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_460: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_576, [0, 2, 1, 3]);  view_576 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    clone_170: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_460, memory_format = torch.contiguous_format);  permute_460 = None
    view_577: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_170, [48, 512, 64]);  clone_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_72: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_461, view_577);  permute_461 = None
    bmm_73: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_577, permute_462);  view_577 = permute_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_287: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(bmm_73, alias_27);  bmm_73 = None
    sum_135: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_287, [-1], True)
    mul_288: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(alias_27, sum_135);  alias_27 = sum_135 = None
    sub_104: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(mul_287, mul_288);  mul_287 = mul_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_578: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(sub_104, [4, 12, 512, 512]);  sub_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_579: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(view_578, [48, 512, 512]);  view_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_74: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_463, view_579);  permute_463 = None
    bmm_75: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_579, permute_464);  view_579 = permute_464 = None
    permute_465: "f32[48, 512, 64]" = torch.ops.aten.permute.default(bmm_74, [0, 2, 1]);  bmm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_580: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_72, [4, 12, 512, 64]);  bmm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    add_178: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_7, view_580);  tangents_7 = view_580 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_581: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(permute_465, [4, 12, 512, 64]);  permute_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    add_179: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_6, view_581);  tangents_6 = view_581 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_582: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_75, [4, 12, 512, 64]);  bmm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_466: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_582, [0, 2, 1, 3]);  view_582 = None
    clone_171: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_466, memory_format = torch.contiguous_format);  permute_466 = None
    view_583: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_171, [4, 512, 768]);  clone_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_467: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_178, [0, 2, 1, 3]);  add_178 = None
    clone_172: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_467, memory_format = torch.contiguous_format);  permute_467 = None
    view_584: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_172, [4, 512, 768]);  clone_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_585: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_584, [2048, 768]);  view_584 = None
    mm_97: "f32[2048, 768]" = torch.ops.aten.mm.default(view_585, permute_468);  permute_468 = None
    permute_469: "f32[768, 2048]" = torch.ops.aten.permute.default(view_585, [1, 0])
    mm_98: "f32[768, 768]" = torch.ops.aten.mm.default(permute_469, view_161);  permute_469 = None
    permute_470: "f32[768, 768]" = torch.ops.aten.permute.default(mm_98, [1, 0]);  mm_98 = None
    sum_136: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_585, [0], True);  view_585 = None
    view_586: "f32[768]" = torch.ops.aten.reshape.default(sum_136, [768]);  sum_136 = None
    permute_471: "f32[768, 768]" = torch.ops.aten.permute.default(permute_470, [1, 0]);  permute_470 = None
    view_587: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_97, [4, 512, 768]);  mm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    add_180: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_285, view_587);  mul_285 = view_587 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_472: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_179, [0, 2, 1, 3]);  add_179 = None
    clone_173: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_472, memory_format = torch.contiguous_format);  permute_472 = None
    view_588: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_173, [4, 512, 768]);  clone_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_589: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_588, [2048, 768]);  view_588 = None
    mm_99: "f32[2048, 768]" = torch.ops.aten.mm.default(view_589, permute_473);  permute_473 = None
    permute_474: "f32[768, 2048]" = torch.ops.aten.permute.default(view_589, [1, 0])
    mm_100: "f32[768, 768]" = torch.ops.aten.mm.default(permute_474, view_161);  permute_474 = None
    permute_475: "f32[768, 768]" = torch.ops.aten.permute.default(mm_100, [1, 0]);  mm_100 = None
    sum_137: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_589, [0], True);  view_589 = None
    view_590: "f32[768]" = torch.ops.aten.reshape.default(sum_137, [768]);  sum_137 = None
    permute_476: "f32[768, 768]" = torch.ops.aten.permute.default(permute_475, [1, 0]);  permute_475 = None
    view_591: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_99, [4, 512, 768]);  mm_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_181: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_180, view_591);  add_180 = view_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_289: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_583, 0.125);  view_583 = None
    view_592: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_289, [2048, 768]);  mul_289 = None
    mm_101: "f32[2048, 768]" = torch.ops.aten.mm.default(view_592, permute_477);  permute_477 = None
    permute_478: "f32[768, 2048]" = torch.ops.aten.permute.default(view_592, [1, 0])
    mm_102: "f32[768, 768]" = torch.ops.aten.mm.default(permute_478, view_161);  permute_478 = view_161 = None
    permute_479: "f32[768, 768]" = torch.ops.aten.permute.default(mm_102, [1, 0]);  mm_102 = None
    sum_138: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_592, [0], True);  view_592 = None
    view_593: "f32[768]" = torch.ops.aten.reshape.default(sum_138, [768]);  sum_138 = None
    permute_480: "f32[768, 768]" = torch.ops.aten.permute.default(permute_479, [1, 0]);  permute_479 = None
    view_594: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_101, [4, 512, 768]);  mm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_182: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_181, view_594);  add_181 = view_594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_291: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_182, primals_129);  primals_129 = None
    mul_292: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_291, 768)
    sum_139: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_291, [2], True)
    mul_293: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_291, mul_63);  mul_291 = None
    sum_140: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_293, [2], True);  mul_293 = None
    mul_294: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_63, sum_140);  sum_140 = None
    sub_106: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_292, sum_139);  mul_292 = sum_139 = None
    sub_107: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_106, mul_294);  sub_106 = mul_294 = None
    mul_295: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_33, sub_107);  div_33 = sub_107 = None
    mul_296: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_182, mul_63);  mul_63 = None
    sum_141: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_296, [0, 1]);  mul_296 = None
    sum_142: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_182, [0, 1]);  add_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_595: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_295, [2048, 768])
    mm_103: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_595, permute_481);  permute_481 = None
    permute_482: "f32[768, 2048]" = torch.ops.aten.permute.default(view_595, [1, 0])
    mm_104: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_482, view_159);  permute_482 = view_159 = None
    permute_483: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_104, [1, 0]);  mm_104 = None
    sum_143: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_595, [0], True);  view_595 = None
    view_596: "f32[768]" = torch.ops.aten.reshape.default(sum_143, [768]);  sum_143 = None
    permute_484: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_483, [1, 0]);  permute_483 = None
    view_597: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_103, [4, 512, 3072]);  mm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_298: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_58, 0.5);  add_58 = None
    mul_299: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_158, view_158)
    mul_300: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_299, -0.5);  mul_299 = None
    exp_23: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_300);  mul_300 = None
    mul_301: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_302: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_158, mul_301);  view_158 = mul_301 = None
    add_184: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_298, mul_302);  mul_298 = mul_302 = None
    mul_303: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_597, add_184);  view_597 = add_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_598: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_303, [2048, 3072]);  mul_303 = None
    mm_105: "f32[2048, 768]" = torch.ops.aten.mm.default(view_598, permute_485);  permute_485 = None
    permute_486: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_598, [1, 0])
    mm_106: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_486, view_157);  permute_486 = view_157 = None
    permute_487: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_106, [1, 0]);  mm_106 = None
    sum_144: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_598, [0], True);  view_598 = None
    view_599: "f32[3072]" = torch.ops.aten.reshape.default(sum_144, [3072]);  sum_144 = None
    permute_488: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_487, [1, 0]);  permute_487 = None
    view_600: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_105, [4, 512, 768]);  mm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    add_185: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_295, view_600);  mul_295 = view_600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:456, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    mul_305: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_185, primals_123);  primals_123 = None
    mul_306: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_305, 768)
    sum_145: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_305, [2], True)
    mul_307: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_305, mul_58);  mul_305 = None
    sum_146: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_307, [2], True);  mul_307 = None
    mul_308: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_58, sum_146);  sum_146 = None
    sub_109: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_306, sum_145);  mul_306 = sum_145 = None
    sub_110: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_109, mul_308);  sub_109 = mul_308 = None
    mul_309: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_34, sub_110);  div_34 = sub_110 = None
    mul_310: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_185, mul_58);  mul_58 = None
    sum_147: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_310, [0, 1]);  mul_310 = None
    sum_148: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_185, [0, 1]);  add_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_601: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_309, [2048, 768])
    mm_107: "f32[2048, 768]" = torch.ops.aten.mm.default(view_601, permute_489);  permute_489 = None
    permute_490: "f32[768, 2048]" = torch.ops.aten.permute.default(view_601, [1, 0])
    mm_108: "f32[768, 768]" = torch.ops.aten.mm.default(permute_490, view_155);  permute_490 = view_155 = None
    permute_491: "f32[768, 768]" = torch.ops.aten.permute.default(mm_108, [1, 0]);  mm_108 = None
    sum_149: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_601, [0], True);  view_601 = None
    view_602: "f32[768]" = torch.ops.aten.reshape.default(sum_149, [768]);  sum_149 = None
    permute_492: "f32[768, 768]" = torch.ops.aten.permute.default(permute_491, [1, 0]);  permute_491 = None
    view_603: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_107, [4, 512, 768]);  mm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_604: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_603, [4, 512, 12, 64]);  view_603 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_493: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_604, [0, 2, 1, 3]);  view_604 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    clone_174: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_493, memory_format = torch.contiguous_format);  permute_493 = None
    view_605: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_174, [48, 512, 64]);  clone_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_76: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_494, view_605);  permute_494 = None
    bmm_77: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_605, permute_495);  view_605 = permute_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_311: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(bmm_77, div_7);  bmm_77 = None
    sum_150: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_311, [-1], True)
    mul_312: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(div_7, sum_150);  div_7 = sum_150 = None
    sub_111: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(mul_311, mul_312);  mul_311 = mul_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_78: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_496, sub_111);  permute_496 = None
    bmm_79: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(sub_111, permute_497);  sub_111 = permute_497 = None
    permute_498: "f32[48, 512, 64]" = torch.ops.aten.permute.default(bmm_78, [0, 2, 1]);  bmm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_606: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_76, [4, 12, 512, 64]);  bmm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    add_186: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_5, view_606);  tangents_5 = view_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_607: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(permute_498, [4, 12, 512, 64]);  permute_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    add_187: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_4, view_607);  tangents_4 = view_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_608: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_79, [4, 12, 512, 64]);  bmm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_499: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_608, [0, 2, 1, 3]);  view_608 = None
    clone_175: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_499, memory_format = torch.contiguous_format);  permute_499 = None
    view_609: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_175, [4, 512, 768]);  clone_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_500: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_186, [0, 2, 1, 3]);  add_186 = None
    clone_176: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_500, memory_format = torch.contiguous_format);  permute_500 = None
    view_610: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_176, [4, 512, 768]);  clone_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_611: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_610, [2048, 768]);  view_610 = None
    mm_109: "f32[2048, 768]" = torch.ops.aten.mm.default(view_611, permute_501);  permute_501 = None
    permute_502: "f32[768, 2048]" = torch.ops.aten.permute.default(view_611, [1, 0])
    mm_110: "f32[768, 768]" = torch.ops.aten.mm.default(permute_502, view_143);  permute_502 = None
    permute_503: "f32[768, 768]" = torch.ops.aten.permute.default(mm_110, [1, 0]);  mm_110 = None
    sum_151: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_611, [0], True);  view_611 = None
    view_612: "f32[768]" = torch.ops.aten.reshape.default(sum_151, [768]);  sum_151 = None
    permute_504: "f32[768, 768]" = torch.ops.aten.permute.default(permute_503, [1, 0]);  permute_503 = None
    view_613: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_109, [4, 512, 768]);  mm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    add_188: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_176, view_613);  add_176 = view_613 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_505: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_187, [0, 2, 1, 3]);  add_187 = None
    clone_177: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_505, memory_format = torch.contiguous_format);  permute_505 = None
    view_614: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_177, [4, 512, 768]);  clone_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_615: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_614, [2048, 768]);  view_614 = None
    mm_111: "f32[2048, 768]" = torch.ops.aten.mm.default(view_615, permute_506);  permute_506 = None
    permute_507: "f32[768, 2048]" = torch.ops.aten.permute.default(view_615, [1, 0])
    mm_112: "f32[768, 768]" = torch.ops.aten.mm.default(permute_507, view_143);  permute_507 = view_143 = None
    permute_508: "f32[768, 768]" = torch.ops.aten.permute.default(mm_112, [1, 0]);  mm_112 = None
    sum_152: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_615, [0], True);  view_615 = None
    view_616: "f32[768]" = torch.ops.aten.reshape.default(sum_152, [768]);  sum_152 = None
    permute_509: "f32[768, 768]" = torch.ops.aten.permute.default(permute_508, [1, 0]);  permute_508 = None
    view_617: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_111, [4, 512, 768]);  mm_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    add_189: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_188, view_617);  add_188 = view_617 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_313: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_609, 0.125);  view_609 = None
    view_618: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_313, [2048, 768]);  mul_313 = None
    mm_113: "f32[2048, 768]" = torch.ops.aten.mm.default(view_618, permute_510);  permute_510 = None
    permute_511: "f32[768, 2048]" = torch.ops.aten.permute.default(view_618, [1, 0])
    mm_114: "f32[768, 768]" = torch.ops.aten.mm.default(permute_511, view_141);  permute_511 = view_141 = None
    permute_512: "f32[768, 768]" = torch.ops.aten.permute.default(mm_114, [1, 0]);  mm_114 = None
    sum_153: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_618, [0], True);  view_618 = None
    view_619: "f32[768]" = torch.ops.aten.reshape.default(sum_153, [768]);  sum_153 = None
    permute_513: "f32[768, 768]" = torch.ops.aten.permute.default(permute_512, [1, 0]);  permute_512 = None
    view_620: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_113, [4, 512, 768]);  mm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_190: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_309, view_620);  mul_309 = view_620 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_315: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_190, primals_113);  primals_113 = None
    mul_316: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_315, 768)
    sum_154: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_315, [2], True)
    mul_317: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_315, mul_55);  mul_315 = None
    sum_155: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_317, [2], True);  mul_317 = None
    mul_318: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_55, sum_155);  sum_155 = None
    sub_113: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_316, sum_154);  mul_316 = sum_154 = None
    sub_114: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_113, mul_318);  sub_113 = mul_318 = None
    mul_319: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_35, sub_114);  div_35 = sub_114 = None
    mul_320: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_190, mul_55);  mul_55 = None
    sum_156: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_320, [0, 1]);  mul_320 = None
    sum_157: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_190, [0, 1]);  add_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_621: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_319, [2048, 768])
    mm_115: "f32[2048, 768]" = torch.ops.aten.mm.default(view_621, permute_514);  permute_514 = None
    permute_515: "f32[768, 2048]" = torch.ops.aten.permute.default(view_621, [1, 0])
    mm_116: "f32[768, 768]" = torch.ops.aten.mm.default(permute_515, view_139);  permute_515 = view_139 = None
    permute_516: "f32[768, 768]" = torch.ops.aten.permute.default(mm_116, [1, 0]);  mm_116 = None
    sum_158: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_621, [0], True);  view_621 = None
    view_622: "f32[768]" = torch.ops.aten.reshape.default(sum_158, [768]);  sum_158 = None
    permute_517: "f32[768, 768]" = torch.ops.aten.permute.default(permute_516, [1, 0]);  permute_516 = None
    view_623: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_115, [4, 512, 768]);  mm_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_624: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_623, [4, 512, 12, 64]);  view_623 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_518: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_624, [0, 2, 1, 3]);  view_624 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    clone_178: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_518, memory_format = torch.contiguous_format);  permute_518 = None
    view_625: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_178, [48, 512, 64]);  clone_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_80: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_519, view_625);  permute_519 = None
    bmm_81: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_625, permute_520);  view_625 = permute_520 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_321: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(bmm_81, alias_29);  bmm_81 = None
    sum_159: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_321, [-1], True)
    mul_322: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(alias_29, sum_159);  alias_29 = sum_159 = None
    sub_115: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(mul_321, mul_322);  mul_321 = mul_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_626: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(sub_115, [4, 12, 512, 512]);  sub_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_627: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(view_626, [48, 512, 512]);  view_626 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_82: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_521, view_627);  permute_521 = None
    bmm_83: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_627, permute_522);  view_627 = permute_522 = None
    permute_523: "f32[48, 512, 64]" = torch.ops.aten.permute.default(bmm_82, [0, 2, 1]);  bmm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_628: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_80, [4, 12, 512, 64]);  bmm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    add_191: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_3, view_628);  tangents_3 = view_628 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_629: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(permute_523, [4, 12, 512, 64]);  permute_523 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    add_192: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_2, view_629);  tangents_2 = view_629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_630: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_83, [4, 12, 512, 64]);  bmm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_524: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_630, [0, 2, 1, 3]);  view_630 = None
    clone_179: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_524, memory_format = torch.contiguous_format);  permute_524 = None
    view_631: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_179, [4, 512, 768]);  clone_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_525: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_191, [0, 2, 1, 3]);  add_191 = None
    clone_180: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_525, memory_format = torch.contiguous_format);  permute_525 = None
    view_632: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_180, [4, 512, 768]);  clone_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_633: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_632, [2048, 768]);  view_632 = None
    mm_117: "f32[2048, 768]" = torch.ops.aten.mm.default(view_633, permute_526);  permute_526 = None
    permute_527: "f32[768, 2048]" = torch.ops.aten.permute.default(view_633, [1, 0])
    mm_118: "f32[768, 768]" = torch.ops.aten.mm.default(permute_527, view_123);  permute_527 = None
    permute_528: "f32[768, 768]" = torch.ops.aten.permute.default(mm_118, [1, 0]);  mm_118 = None
    sum_160: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_633, [0], True);  view_633 = None
    view_634: "f32[768]" = torch.ops.aten.reshape.default(sum_160, [768]);  sum_160 = None
    permute_529: "f32[768, 768]" = torch.ops.aten.permute.default(permute_528, [1, 0]);  permute_528 = None
    view_635: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_117, [4, 512, 768]);  mm_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    add_193: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_319, view_635);  mul_319 = view_635 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_530: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_192, [0, 2, 1, 3]);  add_192 = None
    clone_181: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_530, memory_format = torch.contiguous_format);  permute_530 = None
    view_636: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_181, [4, 512, 768]);  clone_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_637: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_636, [2048, 768]);  view_636 = None
    mm_119: "f32[2048, 768]" = torch.ops.aten.mm.default(view_637, permute_531);  permute_531 = None
    permute_532: "f32[768, 2048]" = torch.ops.aten.permute.default(view_637, [1, 0])
    mm_120: "f32[768, 768]" = torch.ops.aten.mm.default(permute_532, view_123);  permute_532 = None
    permute_533: "f32[768, 768]" = torch.ops.aten.permute.default(mm_120, [1, 0]);  mm_120 = None
    sum_161: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_637, [0], True);  view_637 = None
    view_638: "f32[768]" = torch.ops.aten.reshape.default(sum_161, [768]);  sum_161 = None
    permute_534: "f32[768, 768]" = torch.ops.aten.permute.default(permute_533, [1, 0]);  permute_533 = None
    view_639: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_119, [4, 512, 768]);  mm_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_194: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_193, view_639);  add_193 = view_639 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_323: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_631, 0.125);  view_631 = None
    view_640: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_323, [2048, 768]);  mul_323 = None
    mm_121: "f32[2048, 768]" = torch.ops.aten.mm.default(view_640, permute_535);  permute_535 = None
    permute_536: "f32[768, 2048]" = torch.ops.aten.permute.default(view_640, [1, 0])
    mm_122: "f32[768, 768]" = torch.ops.aten.mm.default(permute_536, view_123);  permute_536 = view_123 = None
    permute_537: "f32[768, 768]" = torch.ops.aten.permute.default(mm_122, [1, 0]);  mm_122 = None
    sum_162: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_640, [0], True);  view_640 = None
    view_641: "f32[768]" = torch.ops.aten.reshape.default(sum_162, [768]);  sum_162 = None
    permute_538: "f32[768, 768]" = torch.ops.aten.permute.default(permute_537, [1, 0]);  permute_537 = None
    view_642: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_121, [4, 512, 768]);  mm_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_195: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_194, view_642);  add_194 = view_642 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:1075, code: hidden_states = self.layernorm_embedding(hidden_states)
    mul_325: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_195, primals_103);  primals_103 = None
    mul_326: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_325, 768)
    sum_163: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_325, [2], True)
    mul_327: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_325, mul_52);  mul_325 = None
    sum_164: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_327, [2], True);  mul_327 = None
    mul_328: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_52, sum_164);  sum_164 = None
    sub_117: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_326, sum_163);  mul_326 = sum_163 = None
    sub_118: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_117, mul_328);  sub_117 = mul_328 = None
    mul_329: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_36, sub_118);  div_36 = sub_118 = None
    mul_330: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_195, mul_52);  mul_52 = None
    sum_165: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_330, [0, 1]);  mul_330 = None
    sum_166: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_195, [0, 1]);  add_195 = None
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    full_default_2: "b8[4, 512, 1]" = torch.ops.aten.full.default([4, 512, 1], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where_1: "f32[4, 512, 768]" = torch.ops.aten.where.self(full_default_2, full_default_1, mul_329)
    full_default_4: "f32[1026, 768]" = torch.ops.aten.full.default([1026, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put: "f32[1026, 768]" = torch.ops.aten._unsafe_index_put.default(full_default_4, [add], where_1, True);  where_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:1059, code: inputs_embeds = self.embed_tokens(input) * self.embed_scale
    mul_331: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_329, 1.0);  mul_329 = None
    eq_1: "b8[4, 512]" = torch.ops.aten.eq.Scalar(primals_264, 1)
    unsqueeze_5: "b8[4, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
    where_2: "f32[4, 512, 768]" = torch.ops.aten.where.self(unsqueeze_5, full_default_1, mul_331);  unsqueeze_5 = mul_331 = None
    full_default_6: "f32[50265, 768]" = torch.ops.aten.full.default([50265, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_1: "f32[50265, 768]" = torch.ops.aten._unsafe_index_put.default(full_default_6, [primals_264], where_2, True);  primals_264 = where_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_333: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_189, primals_100);  primals_100 = None
    mul_334: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_333, 768)
    sum_167: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_333, [2], True)
    mul_335: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_333, mul_49);  mul_333 = None
    sum_168: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_335, [2], True);  mul_335 = None
    mul_336: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_49, sum_168);  sum_168 = None
    sub_120: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_334, sum_167);  mul_334 = sum_167 = None
    sub_121: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_120, mul_336);  sub_120 = mul_336 = None
    mul_337: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_37, sub_121);  div_37 = sub_121 = None
    mul_338: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_189, mul_49);  mul_49 = None
    sum_169: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_338, [0, 1]);  mul_338 = None
    sum_170: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_189, [0, 1]);  add_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    view_643: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_337, [2048, 768])
    mm_123: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_643, permute_539);  permute_539 = None
    permute_540: "f32[768, 2048]" = torch.ops.aten.permute.default(view_643, [1, 0])
    mm_124: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_540, view_119);  permute_540 = view_119 = None
    permute_541: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_124, [1, 0]);  mm_124 = None
    sum_171: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_643, [0], True);  view_643 = None
    view_644: "f32[768]" = torch.ops.aten.reshape.default(sum_171, [768]);  sum_171 = None
    permute_542: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_541, [1, 0]);  permute_541 = None
    view_645: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_123, [4, 512, 3072]);  mm_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_340: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_42, 0.5);  add_42 = None
    mul_341: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_118, view_118)
    mul_342: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_341, -0.5);  mul_341 = None
    exp_24: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_342);  mul_342 = None
    mul_343: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_344: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_118, mul_343);  view_118 = mul_343 = None
    add_197: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_340, mul_344);  mul_340 = mul_344 = None
    mul_345: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_645, add_197);  view_645 = add_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_646: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_345, [2048, 3072]);  mul_345 = None
    mm_125: "f32[2048, 768]" = torch.ops.aten.mm.default(view_646, permute_543);  permute_543 = None
    permute_544: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_646, [1, 0])
    mm_126: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_544, view_117);  permute_544 = view_117 = None
    permute_545: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_126, [1, 0]);  mm_126 = None
    sum_172: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_646, [0], True);  view_646 = None
    view_647: "f32[3072]" = torch.ops.aten.reshape.default(sum_172, [3072]);  sum_172 = None
    permute_546: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_545, [1, 0]);  permute_545 = None
    view_648: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_125, [4, 512, 768]);  mm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    add_198: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_337, view_648);  mul_337 = view_648 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:340, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_347: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_198, primals_94);  primals_94 = None
    mul_348: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_347, 768)
    sum_173: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_347, [2], True)
    mul_349: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_347, mul_44);  mul_347 = None
    sum_174: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_349, [2], True);  mul_349 = None
    mul_350: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_44, sum_174);  sum_174 = None
    sub_123: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_348, sum_173);  mul_348 = sum_173 = None
    sub_124: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_123, mul_350);  sub_123 = mul_350 = None
    mul_351: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_38, sub_124);  div_38 = sub_124 = None
    mul_352: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_198, mul_44);  mul_44 = None
    sum_175: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_352, [0, 1]);  mul_352 = None
    sum_176: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_198, [0, 1]);  add_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_649: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_351, [2048, 768])
    mm_127: "f32[2048, 768]" = torch.ops.aten.mm.default(view_649, permute_547);  permute_547 = None
    permute_548: "f32[768, 2048]" = torch.ops.aten.permute.default(view_649, [1, 0])
    mm_128: "f32[768, 768]" = torch.ops.aten.mm.default(permute_548, view_115);  permute_548 = view_115 = None
    permute_549: "f32[768, 768]" = torch.ops.aten.permute.default(mm_128, [1, 0]);  mm_128 = None
    sum_177: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_649, [0], True);  view_649 = None
    view_650: "f32[768]" = torch.ops.aten.reshape.default(sum_177, [768]);  sum_177 = None
    permute_550: "f32[768, 768]" = torch.ops.aten.permute.default(permute_549, [1, 0]);  permute_549 = None
    view_651: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_127, [4, 512, 768]);  mm_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_652: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_651, [4, 512, 12, 64]);  view_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_551: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_652, [0, 2, 1, 3]);  view_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    clone_182: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_551, memory_format = torch.contiguous_format);  permute_551 = None
    view_653: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_182, [48, 512, 64]);  clone_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_84: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_552, view_653);  permute_552 = None
    bmm_85: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_653, permute_553);  view_653 = permute_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_353: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(bmm_85, div_5);  bmm_85 = None
    sum_178: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_353, [-1], True)
    mul_354: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(div_5, sum_178);  div_5 = sum_178 = None
    sub_125: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(mul_353, mul_354);  mul_353 = mul_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_86: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_554, sub_125);  permute_554 = None
    bmm_87: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(sub_125, permute_555);  sub_125 = permute_555 = None
    permute_556: "f32[48, 512, 64]" = torch.ops.aten.permute.default(bmm_86, [0, 2, 1]);  bmm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_654: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_84, [4, 12, 512, 64]);  bmm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_655: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(permute_556, [4, 12, 512, 64]);  permute_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_656: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_87, [4, 12, 512, 64]);  bmm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_557: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_656, [0, 2, 1, 3]);  view_656 = None
    clone_183: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_557, memory_format = torch.contiguous_format);  permute_557 = None
    view_657: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_183, [4, 512, 768]);  clone_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_558: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_654, [0, 2, 1, 3]);  view_654 = None
    clone_184: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_558, memory_format = torch.contiguous_format);  permute_558 = None
    view_658: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_184, [4, 512, 768]);  clone_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_659: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_658, [2048, 768]);  view_658 = None
    mm_129: "f32[2048, 768]" = torch.ops.aten.mm.default(view_659, permute_559);  permute_559 = None
    permute_560: "f32[768, 2048]" = torch.ops.aten.permute.default(view_659, [1, 0])
    mm_130: "f32[768, 768]" = torch.ops.aten.mm.default(permute_560, view_101);  permute_560 = None
    permute_561: "f32[768, 768]" = torch.ops.aten.permute.default(mm_130, [1, 0]);  mm_130 = None
    sum_179: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_659, [0], True);  view_659 = None
    view_660: "f32[768]" = torch.ops.aten.reshape.default(sum_179, [768]);  sum_179 = None
    permute_562: "f32[768, 768]" = torch.ops.aten.permute.default(permute_561, [1, 0]);  permute_561 = None
    view_661: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_129, [4, 512, 768]);  mm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    add_199: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_351, view_661);  mul_351 = view_661 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_563: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_655, [0, 2, 1, 3]);  view_655 = None
    view_662: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(permute_563, [4, 512, 768]);  permute_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    clone_185: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_662, memory_format = torch.contiguous_format);  view_662 = None
    view_663: "f32[2048, 768]" = torch.ops.aten.reshape.default(clone_185, [2048, 768]);  clone_185 = None
    mm_131: "f32[2048, 768]" = torch.ops.aten.mm.default(view_663, permute_564);  permute_564 = None
    permute_565: "f32[768, 2048]" = torch.ops.aten.permute.default(view_663, [1, 0])
    mm_132: "f32[768, 768]" = torch.ops.aten.mm.default(permute_565, view_101);  permute_565 = None
    permute_566: "f32[768, 768]" = torch.ops.aten.permute.default(mm_132, [1, 0]);  mm_132 = None
    sum_180: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_663, [0], True);  view_663 = None
    view_664: "f32[768]" = torch.ops.aten.reshape.default(sum_180, [768]);  sum_180 = None
    permute_567: "f32[768, 768]" = torch.ops.aten.permute.default(permute_566, [1, 0]);  permute_566 = None
    view_665: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_131, [4, 512, 768]);  mm_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_200: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_199, view_665);  add_199 = view_665 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_355: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_657, 0.125);  view_657 = None
    view_666: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_355, [2048, 768]);  mul_355 = None
    mm_133: "f32[2048, 768]" = torch.ops.aten.mm.default(view_666, permute_568);  permute_568 = None
    permute_569: "f32[768, 2048]" = torch.ops.aten.permute.default(view_666, [1, 0])
    mm_134: "f32[768, 768]" = torch.ops.aten.mm.default(permute_569, view_101);  permute_569 = view_101 = None
    permute_570: "f32[768, 768]" = torch.ops.aten.permute.default(mm_134, [1, 0]);  mm_134 = None
    sum_181: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_666, [0], True);  view_666 = None
    view_667: "f32[768]" = torch.ops.aten.reshape.default(sum_181, [768]);  sum_181 = None
    permute_571: "f32[768, 768]" = torch.ops.aten.permute.default(permute_570, [1, 0]);  permute_570 = None
    view_668: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_133, [4, 512, 768]);  mm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_201: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_200, view_668);  add_200 = view_668 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_357: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_201, primals_84);  primals_84 = None
    mul_358: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_357, 768)
    sum_182: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_357, [2], True)
    mul_359: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_357, mul_41);  mul_357 = None
    sum_183: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_359, [2], True);  mul_359 = None
    mul_360: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_41, sum_183);  sum_183 = None
    sub_127: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_358, sum_182);  mul_358 = sum_182 = None
    sub_128: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_127, mul_360);  sub_127 = mul_360 = None
    mul_361: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_39, sub_128);  div_39 = sub_128 = None
    mul_362: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_201, mul_41);  mul_41 = None
    sum_184: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_362, [0, 1]);  mul_362 = None
    sum_185: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_201, [0, 1]);  add_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    view_669: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_361, [2048, 768])
    mm_135: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_669, permute_572);  permute_572 = None
    permute_573: "f32[768, 2048]" = torch.ops.aten.permute.default(view_669, [1, 0])
    mm_136: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_573, view_99);  permute_573 = view_99 = None
    permute_574: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_136, [1, 0]);  mm_136 = None
    sum_186: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_669, [0], True);  view_669 = None
    view_670: "f32[768]" = torch.ops.aten.reshape.default(sum_186, [768]);  sum_186 = None
    permute_575: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_574, [1, 0]);  permute_574 = None
    view_671: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_135, [4, 512, 3072]);  mm_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_364: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_35, 0.5);  add_35 = None
    mul_365: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_98, view_98)
    mul_366: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_365, -0.5);  mul_365 = None
    exp_25: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_366);  mul_366 = None
    mul_367: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
    mul_368: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_98, mul_367);  view_98 = mul_367 = None
    add_203: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_364, mul_368);  mul_364 = mul_368 = None
    mul_369: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_671, add_203);  view_671 = add_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_672: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_369, [2048, 3072]);  mul_369 = None
    mm_137: "f32[2048, 768]" = torch.ops.aten.mm.default(view_672, permute_576);  permute_576 = None
    permute_577: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_672, [1, 0])
    mm_138: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_577, view_97);  permute_577 = view_97 = None
    permute_578: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_138, [1, 0]);  mm_138 = None
    sum_187: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_672, [0], True);  view_672 = None
    view_673: "f32[3072]" = torch.ops.aten.reshape.default(sum_187, [3072]);  sum_187 = None
    permute_579: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_578, [1, 0]);  permute_578 = None
    view_674: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_137, [4, 512, 768]);  mm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    add_204: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_361, view_674);  mul_361 = view_674 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:340, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_371: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_204, primals_78);  primals_78 = None
    mul_372: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_371, 768)
    sum_188: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_371, [2], True)
    mul_373: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_371, mul_36);  mul_371 = None
    sum_189: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_373, [2], True);  mul_373 = None
    mul_374: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_36, sum_189);  sum_189 = None
    sub_130: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_372, sum_188);  mul_372 = sum_188 = None
    sub_131: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_130, mul_374);  sub_130 = mul_374 = None
    mul_375: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_40, sub_131);  div_40 = sub_131 = None
    mul_376: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_204, mul_36);  mul_36 = None
    sum_190: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_376, [0, 1]);  mul_376 = None
    sum_191: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_204, [0, 1]);  add_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_675: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_375, [2048, 768])
    mm_139: "f32[2048, 768]" = torch.ops.aten.mm.default(view_675, permute_580);  permute_580 = None
    permute_581: "f32[768, 2048]" = torch.ops.aten.permute.default(view_675, [1, 0])
    mm_140: "f32[768, 768]" = torch.ops.aten.mm.default(permute_581, view_95);  permute_581 = view_95 = None
    permute_582: "f32[768, 768]" = torch.ops.aten.permute.default(mm_140, [1, 0]);  mm_140 = None
    sum_192: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_675, [0], True);  view_675 = None
    view_676: "f32[768]" = torch.ops.aten.reshape.default(sum_192, [768]);  sum_192 = None
    permute_583: "f32[768, 768]" = torch.ops.aten.permute.default(permute_582, [1, 0]);  permute_582 = None
    view_677: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_139, [4, 512, 768]);  mm_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_678: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_677, [4, 512, 12, 64]);  view_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_584: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_678, [0, 2, 1, 3]);  view_678 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    clone_186: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_584, memory_format = torch.contiguous_format);  permute_584 = None
    view_679: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_186, [48, 512, 64]);  clone_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_88: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_585, view_679);  permute_585 = None
    bmm_89: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_679, permute_586);  view_679 = permute_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_377: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(bmm_89, div_4);  bmm_89 = None
    sum_193: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_377, [-1], True)
    mul_378: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(div_4, sum_193);  div_4 = sum_193 = None
    sub_132: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(mul_377, mul_378);  mul_377 = mul_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_90: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_587, sub_132);  permute_587 = None
    bmm_91: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(sub_132, permute_588);  sub_132 = permute_588 = None
    permute_589: "f32[48, 512, 64]" = torch.ops.aten.permute.default(bmm_90, [0, 2, 1]);  bmm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_680: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_88, [4, 12, 512, 64]);  bmm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_681: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(permute_589, [4, 12, 512, 64]);  permute_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_682: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_91, [4, 12, 512, 64]);  bmm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_590: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_682, [0, 2, 1, 3]);  view_682 = None
    clone_187: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_590, memory_format = torch.contiguous_format);  permute_590 = None
    view_683: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_187, [4, 512, 768]);  clone_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_591: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_680, [0, 2, 1, 3]);  view_680 = None
    clone_188: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_591, memory_format = torch.contiguous_format);  permute_591 = None
    view_684: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_188, [4, 512, 768]);  clone_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_685: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_684, [2048, 768]);  view_684 = None
    mm_141: "f32[2048, 768]" = torch.ops.aten.mm.default(view_685, permute_592);  permute_592 = None
    permute_593: "f32[768, 2048]" = torch.ops.aten.permute.default(view_685, [1, 0])
    mm_142: "f32[768, 768]" = torch.ops.aten.mm.default(permute_593, view_81);  permute_593 = None
    permute_594: "f32[768, 768]" = torch.ops.aten.permute.default(mm_142, [1, 0]);  mm_142 = None
    sum_194: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_685, [0], True);  view_685 = None
    view_686: "f32[768]" = torch.ops.aten.reshape.default(sum_194, [768]);  sum_194 = None
    permute_595: "f32[768, 768]" = torch.ops.aten.permute.default(permute_594, [1, 0]);  permute_594 = None
    view_687: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_141, [4, 512, 768]);  mm_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    add_205: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_375, view_687);  mul_375 = view_687 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_596: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_681, [0, 2, 1, 3]);  view_681 = None
    view_688: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(permute_596, [4, 512, 768]);  permute_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    clone_189: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_688, memory_format = torch.contiguous_format);  view_688 = None
    view_689: "f32[2048, 768]" = torch.ops.aten.reshape.default(clone_189, [2048, 768]);  clone_189 = None
    mm_143: "f32[2048, 768]" = torch.ops.aten.mm.default(view_689, permute_597);  permute_597 = None
    permute_598: "f32[768, 2048]" = torch.ops.aten.permute.default(view_689, [1, 0])
    mm_144: "f32[768, 768]" = torch.ops.aten.mm.default(permute_598, view_81);  permute_598 = None
    permute_599: "f32[768, 768]" = torch.ops.aten.permute.default(mm_144, [1, 0]);  mm_144 = None
    sum_195: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_689, [0], True);  view_689 = None
    view_690: "f32[768]" = torch.ops.aten.reshape.default(sum_195, [768]);  sum_195 = None
    permute_600: "f32[768, 768]" = torch.ops.aten.permute.default(permute_599, [1, 0]);  permute_599 = None
    view_691: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_143, [4, 512, 768]);  mm_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_206: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_205, view_691);  add_205 = view_691 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_379: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_683, 0.125);  view_683 = None
    view_692: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_379, [2048, 768]);  mul_379 = None
    mm_145: "f32[2048, 768]" = torch.ops.aten.mm.default(view_692, permute_601);  permute_601 = None
    permute_602: "f32[768, 2048]" = torch.ops.aten.permute.default(view_692, [1, 0])
    mm_146: "f32[768, 768]" = torch.ops.aten.mm.default(permute_602, view_81);  permute_602 = view_81 = None
    permute_603: "f32[768, 768]" = torch.ops.aten.permute.default(mm_146, [1, 0]);  mm_146 = None
    sum_196: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_692, [0], True);  view_692 = None
    view_693: "f32[768]" = torch.ops.aten.reshape.default(sum_196, [768]);  sum_196 = None
    permute_604: "f32[768, 768]" = torch.ops.aten.permute.default(permute_603, [1, 0]);  permute_603 = None
    view_694: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_145, [4, 512, 768]);  mm_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_207: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_206, view_694);  add_206 = view_694 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_381: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_207, primals_68);  primals_68 = None
    mul_382: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_381, 768)
    sum_197: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_381, [2], True)
    mul_383: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_381, mul_33);  mul_381 = None
    sum_198: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_383, [2], True);  mul_383 = None
    mul_384: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_33, sum_198);  sum_198 = None
    sub_134: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_382, sum_197);  mul_382 = sum_197 = None
    sub_135: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_134, mul_384);  sub_134 = mul_384 = None
    mul_385: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_41, sub_135);  div_41 = sub_135 = None
    mul_386: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_207, mul_33);  mul_33 = None
    sum_199: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_386, [0, 1]);  mul_386 = None
    sum_200: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_207, [0, 1]);  add_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    view_695: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_385, [2048, 768])
    mm_147: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_695, permute_605);  permute_605 = None
    permute_606: "f32[768, 2048]" = torch.ops.aten.permute.default(view_695, [1, 0])
    mm_148: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_606, view_79);  permute_606 = view_79 = None
    permute_607: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_148, [1, 0]);  mm_148 = None
    sum_201: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_695, [0], True);  view_695 = None
    view_696: "f32[768]" = torch.ops.aten.reshape.default(sum_201, [768]);  sum_201 = None
    permute_608: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_607, [1, 0]);  permute_607 = None
    view_697: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_147, [4, 512, 3072]);  mm_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_388: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_28, 0.5);  add_28 = None
    mul_389: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_78, view_78)
    mul_390: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_389, -0.5);  mul_389 = None
    exp_26: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_390);  mul_390 = None
    mul_391: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_392: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_78, mul_391);  view_78 = mul_391 = None
    add_209: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_388, mul_392);  mul_388 = mul_392 = None
    mul_393: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_697, add_209);  view_697 = add_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_698: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_393, [2048, 3072]);  mul_393 = None
    mm_149: "f32[2048, 768]" = torch.ops.aten.mm.default(view_698, permute_609);  permute_609 = None
    permute_610: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_698, [1, 0])
    mm_150: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_610, view_77);  permute_610 = view_77 = None
    permute_611: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_150, [1, 0]);  mm_150 = None
    sum_202: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_698, [0], True);  view_698 = None
    view_699: "f32[3072]" = torch.ops.aten.reshape.default(sum_202, [3072]);  sum_202 = None
    permute_612: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_611, [1, 0]);  permute_611 = None
    view_700: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_149, [4, 512, 768]);  mm_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    add_210: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_385, view_700);  mul_385 = view_700 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:340, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_395: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_210, primals_62);  primals_62 = None
    mul_396: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_395, 768)
    sum_203: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_395, [2], True)
    mul_397: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_395, mul_28);  mul_395 = None
    sum_204: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_397, [2], True);  mul_397 = None
    mul_398: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_28, sum_204);  sum_204 = None
    sub_137: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_396, sum_203);  mul_396 = sum_203 = None
    sub_138: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_137, mul_398);  sub_137 = mul_398 = None
    mul_399: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_42, sub_138);  div_42 = sub_138 = None
    mul_400: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_210, mul_28);  mul_28 = None
    sum_205: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_400, [0, 1]);  mul_400 = None
    sum_206: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_210, [0, 1]);  add_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_701: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_399, [2048, 768])
    mm_151: "f32[2048, 768]" = torch.ops.aten.mm.default(view_701, permute_613);  permute_613 = None
    permute_614: "f32[768, 2048]" = torch.ops.aten.permute.default(view_701, [1, 0])
    mm_152: "f32[768, 768]" = torch.ops.aten.mm.default(permute_614, view_75);  permute_614 = view_75 = None
    permute_615: "f32[768, 768]" = torch.ops.aten.permute.default(mm_152, [1, 0]);  mm_152 = None
    sum_207: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_701, [0], True);  view_701 = None
    view_702: "f32[768]" = torch.ops.aten.reshape.default(sum_207, [768]);  sum_207 = None
    permute_616: "f32[768, 768]" = torch.ops.aten.permute.default(permute_615, [1, 0]);  permute_615 = None
    view_703: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_151, [4, 512, 768]);  mm_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_704: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_703, [4, 512, 12, 64]);  view_703 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_617: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_704, [0, 2, 1, 3]);  view_704 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    clone_190: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_617, memory_format = torch.contiguous_format);  permute_617 = None
    view_705: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_190, [48, 512, 64]);  clone_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_92: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_618, view_705);  permute_618 = None
    bmm_93: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_705, permute_619);  view_705 = permute_619 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_401: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(bmm_93, div_3);  bmm_93 = None
    sum_208: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_401, [-1], True)
    mul_402: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(div_3, sum_208);  div_3 = sum_208 = None
    sub_139: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(mul_401, mul_402);  mul_401 = mul_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_94: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_620, sub_139);  permute_620 = None
    bmm_95: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(sub_139, permute_621);  sub_139 = permute_621 = None
    permute_622: "f32[48, 512, 64]" = torch.ops.aten.permute.default(bmm_94, [0, 2, 1]);  bmm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_706: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_92, [4, 12, 512, 64]);  bmm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_707: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(permute_622, [4, 12, 512, 64]);  permute_622 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_708: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_95, [4, 12, 512, 64]);  bmm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_623: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_708, [0, 2, 1, 3]);  view_708 = None
    clone_191: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_623, memory_format = torch.contiguous_format);  permute_623 = None
    view_709: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_191, [4, 512, 768]);  clone_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_624: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_706, [0, 2, 1, 3]);  view_706 = None
    clone_192: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_624, memory_format = torch.contiguous_format);  permute_624 = None
    view_710: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_192, [4, 512, 768]);  clone_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_711: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_710, [2048, 768]);  view_710 = None
    mm_153: "f32[2048, 768]" = torch.ops.aten.mm.default(view_711, permute_625);  permute_625 = None
    permute_626: "f32[768, 2048]" = torch.ops.aten.permute.default(view_711, [1, 0])
    mm_154: "f32[768, 768]" = torch.ops.aten.mm.default(permute_626, view_61);  permute_626 = None
    permute_627: "f32[768, 768]" = torch.ops.aten.permute.default(mm_154, [1, 0]);  mm_154 = None
    sum_209: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_711, [0], True);  view_711 = None
    view_712: "f32[768]" = torch.ops.aten.reshape.default(sum_209, [768]);  sum_209 = None
    permute_628: "f32[768, 768]" = torch.ops.aten.permute.default(permute_627, [1, 0]);  permute_627 = None
    view_713: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_153, [4, 512, 768]);  mm_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    add_211: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_399, view_713);  mul_399 = view_713 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_629: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_707, [0, 2, 1, 3]);  view_707 = None
    view_714: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(permute_629, [4, 512, 768]);  permute_629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    clone_193: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_714, memory_format = torch.contiguous_format);  view_714 = None
    view_715: "f32[2048, 768]" = torch.ops.aten.reshape.default(clone_193, [2048, 768]);  clone_193 = None
    mm_155: "f32[2048, 768]" = torch.ops.aten.mm.default(view_715, permute_630);  permute_630 = None
    permute_631: "f32[768, 2048]" = torch.ops.aten.permute.default(view_715, [1, 0])
    mm_156: "f32[768, 768]" = torch.ops.aten.mm.default(permute_631, view_61);  permute_631 = None
    permute_632: "f32[768, 768]" = torch.ops.aten.permute.default(mm_156, [1, 0]);  mm_156 = None
    sum_210: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_715, [0], True);  view_715 = None
    view_716: "f32[768]" = torch.ops.aten.reshape.default(sum_210, [768]);  sum_210 = None
    permute_633: "f32[768, 768]" = torch.ops.aten.permute.default(permute_632, [1, 0]);  permute_632 = None
    view_717: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_155, [4, 512, 768]);  mm_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_212: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_211, view_717);  add_211 = view_717 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_403: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_709, 0.125);  view_709 = None
    view_718: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_403, [2048, 768]);  mul_403 = None
    mm_157: "f32[2048, 768]" = torch.ops.aten.mm.default(view_718, permute_634);  permute_634 = None
    permute_635: "f32[768, 2048]" = torch.ops.aten.permute.default(view_718, [1, 0])
    mm_158: "f32[768, 768]" = torch.ops.aten.mm.default(permute_635, view_61);  permute_635 = view_61 = None
    permute_636: "f32[768, 768]" = torch.ops.aten.permute.default(mm_158, [1, 0]);  mm_158 = None
    sum_211: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_718, [0], True);  view_718 = None
    view_719: "f32[768]" = torch.ops.aten.reshape.default(sum_211, [768]);  sum_211 = None
    permute_637: "f32[768, 768]" = torch.ops.aten.permute.default(permute_636, [1, 0]);  permute_636 = None
    view_720: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_157, [4, 512, 768]);  mm_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_213: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_212, view_720);  add_212 = view_720 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_405: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_213, primals_52);  primals_52 = None
    mul_406: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_405, 768)
    sum_212: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_405, [2], True)
    mul_407: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_405, mul_25);  mul_405 = None
    sum_213: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_407, [2], True);  mul_407 = None
    mul_408: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_25, sum_213);  sum_213 = None
    sub_141: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_406, sum_212);  mul_406 = sum_212 = None
    sub_142: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_141, mul_408);  sub_141 = mul_408 = None
    mul_409: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_43, sub_142);  div_43 = sub_142 = None
    mul_410: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_213, mul_25);  mul_25 = None
    sum_214: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_410, [0, 1]);  mul_410 = None
    sum_215: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_213, [0, 1]);  add_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    view_721: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_409, [2048, 768])
    mm_159: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_721, permute_638);  permute_638 = None
    permute_639: "f32[768, 2048]" = torch.ops.aten.permute.default(view_721, [1, 0])
    mm_160: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_639, view_59);  permute_639 = view_59 = None
    permute_640: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_160, [1, 0]);  mm_160 = None
    sum_216: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_721, [0], True);  view_721 = None
    view_722: "f32[768]" = torch.ops.aten.reshape.default(sum_216, [768]);  sum_216 = None
    permute_641: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_640, [1, 0]);  permute_640 = None
    view_723: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_159, [4, 512, 3072]);  mm_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_412: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_21, 0.5);  add_21 = None
    mul_413: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_58, view_58)
    mul_414: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_413, -0.5);  mul_413 = None
    exp_27: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_414);  mul_414 = None
    mul_415: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
    mul_416: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_58, mul_415);  view_58 = mul_415 = None
    add_215: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_412, mul_416);  mul_412 = mul_416 = None
    mul_417: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_723, add_215);  view_723 = add_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_724: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_417, [2048, 3072]);  mul_417 = None
    mm_161: "f32[2048, 768]" = torch.ops.aten.mm.default(view_724, permute_642);  permute_642 = None
    permute_643: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_724, [1, 0])
    mm_162: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_643, view_57);  permute_643 = view_57 = None
    permute_644: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_162, [1, 0]);  mm_162 = None
    sum_217: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_724, [0], True);  view_724 = None
    view_725: "f32[3072]" = torch.ops.aten.reshape.default(sum_217, [3072]);  sum_217 = None
    permute_645: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_644, [1, 0]);  permute_644 = None
    view_726: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_161, [4, 512, 768]);  mm_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    add_216: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_409, view_726);  mul_409 = view_726 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:340, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_419: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_216, primals_46);  primals_46 = None
    mul_420: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_419, 768)
    sum_218: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_419, [2], True)
    mul_421: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_419, mul_20);  mul_419 = None
    sum_219: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_421, [2], True);  mul_421 = None
    mul_422: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_20, sum_219);  sum_219 = None
    sub_144: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_420, sum_218);  mul_420 = sum_218 = None
    sub_145: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_144, mul_422);  sub_144 = mul_422 = None
    mul_423: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_44, sub_145);  div_44 = sub_145 = None
    mul_424: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_216, mul_20);  mul_20 = None
    sum_220: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_424, [0, 1]);  mul_424 = None
    sum_221: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_216, [0, 1]);  add_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_727: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_423, [2048, 768])
    mm_163: "f32[2048, 768]" = torch.ops.aten.mm.default(view_727, permute_646);  permute_646 = None
    permute_647: "f32[768, 2048]" = torch.ops.aten.permute.default(view_727, [1, 0])
    mm_164: "f32[768, 768]" = torch.ops.aten.mm.default(permute_647, view_55);  permute_647 = view_55 = None
    permute_648: "f32[768, 768]" = torch.ops.aten.permute.default(mm_164, [1, 0]);  mm_164 = None
    sum_222: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_727, [0], True);  view_727 = None
    view_728: "f32[768]" = torch.ops.aten.reshape.default(sum_222, [768]);  sum_222 = None
    permute_649: "f32[768, 768]" = torch.ops.aten.permute.default(permute_648, [1, 0]);  permute_648 = None
    view_729: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_163, [4, 512, 768]);  mm_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_730: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_729, [4, 512, 12, 64]);  view_729 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_650: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_730, [0, 2, 1, 3]);  view_730 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    clone_194: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_650, memory_format = torch.contiguous_format);  permute_650 = None
    view_731: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_194, [48, 512, 64]);  clone_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_96: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_651, view_731);  permute_651 = None
    bmm_97: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_731, permute_652);  view_731 = permute_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_425: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(bmm_97, div_2);  bmm_97 = None
    sum_223: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_425, [-1], True)
    mul_426: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(div_2, sum_223);  div_2 = sum_223 = None
    sub_146: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(mul_425, mul_426);  mul_425 = mul_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_98: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_653, sub_146);  permute_653 = None
    bmm_99: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(sub_146, permute_654);  sub_146 = permute_654 = None
    permute_655: "f32[48, 512, 64]" = torch.ops.aten.permute.default(bmm_98, [0, 2, 1]);  bmm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_732: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_96, [4, 12, 512, 64]);  bmm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_733: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(permute_655, [4, 12, 512, 64]);  permute_655 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_734: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_99, [4, 12, 512, 64]);  bmm_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_656: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_734, [0, 2, 1, 3]);  view_734 = None
    clone_195: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_656, memory_format = torch.contiguous_format);  permute_656 = None
    view_735: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_195, [4, 512, 768]);  clone_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_657: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_732, [0, 2, 1, 3]);  view_732 = None
    clone_196: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_657, memory_format = torch.contiguous_format);  permute_657 = None
    view_736: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_196, [4, 512, 768]);  clone_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_737: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_736, [2048, 768]);  view_736 = None
    mm_165: "f32[2048, 768]" = torch.ops.aten.mm.default(view_737, permute_658);  permute_658 = None
    permute_659: "f32[768, 2048]" = torch.ops.aten.permute.default(view_737, [1, 0])
    mm_166: "f32[768, 768]" = torch.ops.aten.mm.default(permute_659, view_41);  permute_659 = None
    permute_660: "f32[768, 768]" = torch.ops.aten.permute.default(mm_166, [1, 0]);  mm_166 = None
    sum_224: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_737, [0], True);  view_737 = None
    view_738: "f32[768]" = torch.ops.aten.reshape.default(sum_224, [768]);  sum_224 = None
    permute_661: "f32[768, 768]" = torch.ops.aten.permute.default(permute_660, [1, 0]);  permute_660 = None
    view_739: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_165, [4, 512, 768]);  mm_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    add_217: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_423, view_739);  mul_423 = view_739 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_662: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_733, [0, 2, 1, 3]);  view_733 = None
    view_740: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(permute_662, [4, 512, 768]);  permute_662 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    clone_197: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_740, memory_format = torch.contiguous_format);  view_740 = None
    view_741: "f32[2048, 768]" = torch.ops.aten.reshape.default(clone_197, [2048, 768]);  clone_197 = None
    mm_167: "f32[2048, 768]" = torch.ops.aten.mm.default(view_741, permute_663);  permute_663 = None
    permute_664: "f32[768, 2048]" = torch.ops.aten.permute.default(view_741, [1, 0])
    mm_168: "f32[768, 768]" = torch.ops.aten.mm.default(permute_664, view_41);  permute_664 = None
    permute_665: "f32[768, 768]" = torch.ops.aten.permute.default(mm_168, [1, 0]);  mm_168 = None
    sum_225: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_741, [0], True);  view_741 = None
    view_742: "f32[768]" = torch.ops.aten.reshape.default(sum_225, [768]);  sum_225 = None
    permute_666: "f32[768, 768]" = torch.ops.aten.permute.default(permute_665, [1, 0]);  permute_665 = None
    view_743: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_167, [4, 512, 768]);  mm_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_218: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_217, view_743);  add_217 = view_743 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_427: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_735, 0.125);  view_735 = None
    view_744: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_427, [2048, 768]);  mul_427 = None
    mm_169: "f32[2048, 768]" = torch.ops.aten.mm.default(view_744, permute_667);  permute_667 = None
    permute_668: "f32[768, 2048]" = torch.ops.aten.permute.default(view_744, [1, 0])
    mm_170: "f32[768, 768]" = torch.ops.aten.mm.default(permute_668, view_41);  permute_668 = view_41 = None
    permute_669: "f32[768, 768]" = torch.ops.aten.permute.default(mm_170, [1, 0]);  mm_170 = None
    sum_226: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_744, [0], True);  view_744 = None
    view_745: "f32[768]" = torch.ops.aten.reshape.default(sum_226, [768]);  sum_226 = None
    permute_670: "f32[768, 768]" = torch.ops.aten.permute.default(permute_669, [1, 0]);  permute_669 = None
    view_746: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_169, [4, 512, 768]);  mm_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_219: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_218, view_746);  add_218 = view_746 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_429: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_219, primals_36);  primals_36 = None
    mul_430: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_429, 768)
    sum_227: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_429, [2], True)
    mul_431: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_429, mul_17);  mul_429 = None
    sum_228: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_431, [2], True);  mul_431 = None
    mul_432: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_17, sum_228);  sum_228 = None
    sub_148: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_430, sum_227);  mul_430 = sum_227 = None
    sub_149: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_148, mul_432);  sub_148 = mul_432 = None
    mul_433: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_45, sub_149);  div_45 = sub_149 = None
    mul_434: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_219, mul_17);  mul_17 = None
    sum_229: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_434, [0, 1]);  mul_434 = None
    sum_230: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_219, [0, 1]);  add_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    view_747: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_433, [2048, 768])
    mm_171: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_747, permute_671);  permute_671 = None
    permute_672: "f32[768, 2048]" = torch.ops.aten.permute.default(view_747, [1, 0])
    mm_172: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_672, view_39);  permute_672 = view_39 = None
    permute_673: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_172, [1, 0]);  mm_172 = None
    sum_231: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_747, [0], True);  view_747 = None
    view_748: "f32[768]" = torch.ops.aten.reshape.default(sum_231, [768]);  sum_231 = None
    permute_674: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_673, [1, 0]);  permute_673 = None
    view_749: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_171, [4, 512, 3072]);  mm_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_436: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_14, 0.5);  add_14 = None
    mul_437: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_38, view_38)
    mul_438: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_437, -0.5);  mul_437 = None
    exp_28: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_438);  mul_438 = None
    mul_439: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
    mul_440: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_38, mul_439);  view_38 = mul_439 = None
    add_221: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_436, mul_440);  mul_436 = mul_440 = None
    mul_441: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_749, add_221);  view_749 = add_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_750: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_441, [2048, 3072]);  mul_441 = None
    mm_173: "f32[2048, 768]" = torch.ops.aten.mm.default(view_750, permute_675);  permute_675 = None
    permute_676: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_750, [1, 0])
    mm_174: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_676, view_37);  permute_676 = view_37 = None
    permute_677: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_174, [1, 0]);  mm_174 = None
    sum_232: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_750, [0], True);  view_750 = None
    view_751: "f32[3072]" = torch.ops.aten.reshape.default(sum_232, [3072]);  sum_232 = None
    permute_678: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_677, [1, 0]);  permute_677 = None
    view_752: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_173, [4, 512, 768]);  mm_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    add_222: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_433, view_752);  mul_433 = view_752 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:340, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_443: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_222, primals_30);  primals_30 = None
    mul_444: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_443, 768)
    sum_233: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_443, [2], True)
    mul_445: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_443, mul_12);  mul_443 = None
    sum_234: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_445, [2], True);  mul_445 = None
    mul_446: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_12, sum_234);  sum_234 = None
    sub_151: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_444, sum_233);  mul_444 = sum_233 = None
    sub_152: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_151, mul_446);  sub_151 = mul_446 = None
    mul_447: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_46, sub_152);  div_46 = sub_152 = None
    mul_448: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_222, mul_12);  mul_12 = None
    sum_235: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_448, [0, 1]);  mul_448 = None
    sum_236: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_222, [0, 1]);  add_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_753: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_447, [2048, 768])
    mm_175: "f32[2048, 768]" = torch.ops.aten.mm.default(view_753, permute_679);  permute_679 = None
    permute_680: "f32[768, 2048]" = torch.ops.aten.permute.default(view_753, [1, 0])
    mm_176: "f32[768, 768]" = torch.ops.aten.mm.default(permute_680, view_35);  permute_680 = view_35 = None
    permute_681: "f32[768, 768]" = torch.ops.aten.permute.default(mm_176, [1, 0]);  mm_176 = None
    sum_237: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_753, [0], True);  view_753 = None
    view_754: "f32[768]" = torch.ops.aten.reshape.default(sum_237, [768]);  sum_237 = None
    permute_682: "f32[768, 768]" = torch.ops.aten.permute.default(permute_681, [1, 0]);  permute_681 = None
    view_755: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_175, [4, 512, 768]);  mm_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_756: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_755, [4, 512, 12, 64]);  view_755 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_683: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_756, [0, 2, 1, 3]);  view_756 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    clone_198: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_683, memory_format = torch.contiguous_format);  permute_683 = None
    view_757: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_198, [48, 512, 64]);  clone_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_100: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_684, view_757);  permute_684 = None
    bmm_101: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_757, permute_685);  view_757 = permute_685 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_449: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(bmm_101, div_1);  bmm_101 = None
    sum_238: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_449, [-1], True)
    mul_450: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(div_1, sum_238);  div_1 = sum_238 = None
    sub_153: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(mul_449, mul_450);  mul_449 = mul_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_102: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_686, sub_153);  permute_686 = None
    bmm_103: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(sub_153, permute_687);  sub_153 = permute_687 = None
    permute_688: "f32[48, 512, 64]" = torch.ops.aten.permute.default(bmm_102, [0, 2, 1]);  bmm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_758: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_100, [4, 12, 512, 64]);  bmm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_759: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(permute_688, [4, 12, 512, 64]);  permute_688 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_760: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_103, [4, 12, 512, 64]);  bmm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_689: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_760, [0, 2, 1, 3]);  view_760 = None
    clone_199: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_689, memory_format = torch.contiguous_format);  permute_689 = None
    view_761: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_199, [4, 512, 768]);  clone_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_690: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_758, [0, 2, 1, 3]);  view_758 = None
    clone_200: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_690, memory_format = torch.contiguous_format);  permute_690 = None
    view_762: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_200, [4, 512, 768]);  clone_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_763: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_762, [2048, 768]);  view_762 = None
    mm_177: "f32[2048, 768]" = torch.ops.aten.mm.default(view_763, permute_691);  permute_691 = None
    permute_692: "f32[768, 2048]" = torch.ops.aten.permute.default(view_763, [1, 0])
    mm_178: "f32[768, 768]" = torch.ops.aten.mm.default(permute_692, view_21);  permute_692 = None
    permute_693: "f32[768, 768]" = torch.ops.aten.permute.default(mm_178, [1, 0]);  mm_178 = None
    sum_239: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_763, [0], True);  view_763 = None
    view_764: "f32[768]" = torch.ops.aten.reshape.default(sum_239, [768]);  sum_239 = None
    permute_694: "f32[768, 768]" = torch.ops.aten.permute.default(permute_693, [1, 0]);  permute_693 = None
    view_765: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_177, [4, 512, 768]);  mm_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    add_223: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_447, view_765);  mul_447 = view_765 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_695: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_759, [0, 2, 1, 3]);  view_759 = None
    view_766: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(permute_695, [4, 512, 768]);  permute_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    clone_201: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_766, memory_format = torch.contiguous_format);  view_766 = None
    view_767: "f32[2048, 768]" = torch.ops.aten.reshape.default(clone_201, [2048, 768]);  clone_201 = None
    mm_179: "f32[2048, 768]" = torch.ops.aten.mm.default(view_767, permute_696);  permute_696 = None
    permute_697: "f32[768, 2048]" = torch.ops.aten.permute.default(view_767, [1, 0])
    mm_180: "f32[768, 768]" = torch.ops.aten.mm.default(permute_697, view_21);  permute_697 = None
    permute_698: "f32[768, 768]" = torch.ops.aten.permute.default(mm_180, [1, 0]);  mm_180 = None
    sum_240: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_767, [0], True);  view_767 = None
    view_768: "f32[768]" = torch.ops.aten.reshape.default(sum_240, [768]);  sum_240 = None
    permute_699: "f32[768, 768]" = torch.ops.aten.permute.default(permute_698, [1, 0]);  permute_698 = None
    view_769: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_179, [4, 512, 768]);  mm_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_224: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_223, view_769);  add_223 = view_769 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_451: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_761, 0.125);  view_761 = None
    view_770: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_451, [2048, 768]);  mul_451 = None
    mm_181: "f32[2048, 768]" = torch.ops.aten.mm.default(view_770, permute_700);  permute_700 = None
    permute_701: "f32[768, 2048]" = torch.ops.aten.permute.default(view_770, [1, 0])
    mm_182: "f32[768, 768]" = torch.ops.aten.mm.default(permute_701, view_21);  permute_701 = view_21 = None
    permute_702: "f32[768, 768]" = torch.ops.aten.permute.default(mm_182, [1, 0]);  mm_182 = None
    sum_241: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_770, [0], True);  view_770 = None
    view_771: "f32[768]" = torch.ops.aten.reshape.default(sum_241, [768]);  sum_241 = None
    permute_703: "f32[768, 768]" = torch.ops.aten.permute.default(permute_702, [1, 0]);  permute_702 = None
    view_772: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_181, [4, 512, 768]);  mm_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_225: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_224, view_772);  add_224 = view_772 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_453: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_225, primals_20);  primals_20 = None
    mul_454: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_453, 768)
    sum_242: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_453, [2], True)
    mul_455: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_453, mul_9);  mul_453 = None
    sum_243: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_455, [2], True);  mul_455 = None
    mul_456: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_9, sum_243);  sum_243 = None
    sub_155: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_454, sum_242);  mul_454 = sum_242 = None
    sub_156: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_155, mul_456);  sub_155 = mul_456 = None
    mul_457: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_47, sub_156);  div_47 = sub_156 = None
    mul_458: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_225, mul_9);  mul_9 = None
    sum_244: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_458, [0, 1]);  mul_458 = None
    sum_245: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_225, [0, 1]);  add_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    view_773: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_457, [2048, 768])
    mm_183: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_773, permute_704);  permute_704 = None
    permute_705: "f32[768, 2048]" = torch.ops.aten.permute.default(view_773, [1, 0])
    mm_184: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_705, view_19);  permute_705 = view_19 = None
    permute_706: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_184, [1, 0]);  mm_184 = None
    sum_246: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_773, [0], True);  view_773 = None
    view_774: "f32[768]" = torch.ops.aten.reshape.default(sum_246, [768]);  sum_246 = None
    permute_707: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_706, [1, 0]);  permute_706 = None
    view_775: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_183, [4, 512, 3072]);  mm_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_460: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_7, 0.5);  add_7 = None
    mul_461: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_18, view_18)
    mul_462: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_461, -0.5);  mul_461 = None
    exp_29: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_462);  mul_462 = None
    mul_463: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
    mul_464: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_18, mul_463);  view_18 = mul_463 = None
    add_227: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_460, mul_464);  mul_460 = mul_464 = None
    mul_465: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_775, add_227);  view_775 = add_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_776: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_465, [2048, 3072]);  mul_465 = None
    mm_185: "f32[2048, 768]" = torch.ops.aten.mm.default(view_776, permute_708);  permute_708 = None
    permute_709: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_776, [1, 0])
    mm_186: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_709, view_17);  permute_709 = view_17 = None
    permute_710: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_186, [1, 0]);  mm_186 = None
    sum_247: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_776, [0], True);  view_776 = None
    view_777: "f32[3072]" = torch.ops.aten.reshape.default(sum_247, [3072]);  sum_247 = None
    permute_711: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_710, [1, 0]);  permute_710 = None
    view_778: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_185, [4, 512, 768]);  mm_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    add_228: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_457, view_778);  mul_457 = view_778 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:340, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_467: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_228, primals_14);  primals_14 = None
    mul_468: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_467, 768)
    sum_248: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_467, [2], True)
    mul_469: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_467, mul_4);  mul_467 = None
    sum_249: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_469, [2], True);  mul_469 = None
    mul_470: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_4, sum_249);  sum_249 = None
    sub_158: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_468, sum_248);  mul_468 = sum_248 = None
    sub_159: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_158, mul_470);  sub_158 = mul_470 = None
    mul_471: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_48, sub_159);  div_48 = sub_159 = None
    mul_472: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_228, mul_4);  mul_4 = None
    sum_250: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_472, [0, 1]);  mul_472 = None
    sum_251: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_228, [0, 1]);  add_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_779: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_471, [2048, 768])
    mm_187: "f32[2048, 768]" = torch.ops.aten.mm.default(view_779, permute_712);  permute_712 = None
    permute_713: "f32[768, 2048]" = torch.ops.aten.permute.default(view_779, [1, 0])
    mm_188: "f32[768, 768]" = torch.ops.aten.mm.default(permute_713, view_15);  permute_713 = view_15 = None
    permute_714: "f32[768, 768]" = torch.ops.aten.permute.default(mm_188, [1, 0]);  mm_188 = None
    sum_252: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_779, [0], True);  view_779 = None
    view_780: "f32[768]" = torch.ops.aten.reshape.default(sum_252, [768]);  sum_252 = None
    permute_715: "f32[768, 768]" = torch.ops.aten.permute.default(permute_714, [1, 0]);  permute_714 = None
    view_781: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_187, [4, 512, 768]);  mm_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_782: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_781, [4, 512, 12, 64]);  view_781 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_716: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_782, [0, 2, 1, 3]);  view_782 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    clone_202: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_716, memory_format = torch.contiguous_format);  permute_716 = None
    view_783: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_202, [48, 512, 64]);  clone_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_104: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_717, view_783);  permute_717 = None
    bmm_105: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_783, permute_718);  view_783 = permute_718 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_473: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(bmm_105, div);  bmm_105 = None
    sum_253: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_473, [-1], True)
    mul_474: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(div, sum_253);  div = sum_253 = None
    sub_160: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(mul_473, mul_474);  mul_473 = mul_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_106: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_719, sub_160);  permute_719 = None
    bmm_107: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(sub_160, permute_720);  sub_160 = permute_720 = None
    permute_721: "f32[48, 512, 64]" = torch.ops.aten.permute.default(bmm_106, [0, 2, 1]);  bmm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_784: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_104, [4, 12, 512, 64]);  bmm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_785: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(permute_721, [4, 12, 512, 64]);  permute_721 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_786: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_107, [4, 12, 512, 64]);  bmm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_722: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_786, [0, 2, 1, 3]);  view_786 = None
    clone_203: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_722, memory_format = torch.contiguous_format);  permute_722 = None
    view_787: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_203, [4, 512, 768]);  clone_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_723: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_784, [0, 2, 1, 3]);  view_784 = None
    clone_204: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_723, memory_format = torch.contiguous_format);  permute_723 = None
    view_788: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_204, [4, 512, 768]);  clone_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_789: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_788, [2048, 768]);  view_788 = None
    mm_189: "f32[2048, 768]" = torch.ops.aten.mm.default(view_789, permute_724);  permute_724 = None
    permute_725: "f32[768, 2048]" = torch.ops.aten.permute.default(view_789, [1, 0])
    mm_190: "f32[768, 768]" = torch.ops.aten.mm.default(permute_725, view_1);  permute_725 = None
    permute_726: "f32[768, 768]" = torch.ops.aten.permute.default(mm_190, [1, 0]);  mm_190 = None
    sum_254: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_789, [0], True);  view_789 = None
    view_790: "f32[768]" = torch.ops.aten.reshape.default(sum_254, [768]);  sum_254 = None
    permute_727: "f32[768, 768]" = torch.ops.aten.permute.default(permute_726, [1, 0]);  permute_726 = None
    view_791: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_189, [4, 512, 768]);  mm_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    add_229: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_471, view_791);  mul_471 = view_791 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_728: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_785, [0, 2, 1, 3]);  view_785 = None
    view_792: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(permute_728, [4, 512, 768]);  permute_728 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    clone_205: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_792, memory_format = torch.contiguous_format);  view_792 = None
    view_793: "f32[2048, 768]" = torch.ops.aten.reshape.default(clone_205, [2048, 768]);  clone_205 = None
    mm_191: "f32[2048, 768]" = torch.ops.aten.mm.default(view_793, permute_729);  permute_729 = None
    permute_730: "f32[768, 2048]" = torch.ops.aten.permute.default(view_793, [1, 0])
    mm_192: "f32[768, 768]" = torch.ops.aten.mm.default(permute_730, view_1);  permute_730 = None
    permute_731: "f32[768, 768]" = torch.ops.aten.permute.default(mm_192, [1, 0]);  mm_192 = None
    sum_255: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_793, [0], True);  view_793 = None
    view_794: "f32[768]" = torch.ops.aten.reshape.default(sum_255, [768]);  sum_255 = None
    permute_732: "f32[768, 768]" = torch.ops.aten.permute.default(permute_731, [1, 0]);  permute_731 = None
    view_795: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_191, [4, 512, 768]);  mm_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_230: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_229, view_795);  add_229 = view_795 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_475: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_787, 0.125);  view_787 = None
    view_796: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_475, [2048, 768]);  mul_475 = None
    mm_193: "f32[2048, 768]" = torch.ops.aten.mm.default(view_796, permute_733);  permute_733 = None
    permute_734: "f32[768, 2048]" = torch.ops.aten.permute.default(view_796, [1, 0])
    mm_194: "f32[768, 768]" = torch.ops.aten.mm.default(permute_734, view_1);  permute_734 = view_1 = None
    permute_735: "f32[768, 768]" = torch.ops.aten.permute.default(mm_194, [1, 0]);  mm_194 = None
    sum_256: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_796, [0], True);  view_796 = None
    view_797: "f32[768]" = torch.ops.aten.reshape.default(sum_256, [768]);  sum_256 = None
    permute_736: "f32[768, 768]" = torch.ops.aten.permute.default(permute_735, [1, 0]);  permute_735 = None
    view_798: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_193, [4, 512, 768]);  mm_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_231: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_230, view_798);  add_230 = view_798 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:824, code: hidden_states = self.layernorm_embedding(hidden_states)
    mul_477: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_231, primals_4);  primals_4 = None
    mul_478: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_477, 768)
    sum_257: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_477, [2], True)
    mul_479: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_477, mul_1);  mul_477 = None
    sum_258: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_479, [2], True);  mul_479 = None
    mul_480: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_1, sum_258);  sum_258 = None
    sub_162: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_478, sum_257);  mul_478 = sum_257 = None
    sub_163: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_162, mul_480);  sub_162 = mul_480 = None
    mul_481: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_49, sub_163);  div_49 = sub_163 = None
    mul_482: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_231, mul_1);  mul_1 = None
    sum_259: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_482, [0, 1]);  mul_482 = None
    sum_260: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_231, [0, 1]);  add_231 = None
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    where_3: "f32[4, 512, 768]" = torch.ops.aten.where.self(full_default_2, full_default_1, mul_481);  full_default_2 = None
    _unsafe_index_put_2: "f32[1026, 768]" = torch.ops.prims._unsafe_index_put_.default(full_default_4, [add], where_3, True);  full_default_4 = add = where_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:818, code: inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    mul_483: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_481, 1.0);  mul_481 = None
    eq_3: "b8[4, 512]" = torch.ops.aten.eq.Scalar(view, 1)
    unsqueeze_7: "b8[4, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_3, -1);  eq_3 = None
    where_4: "f32[4, 512, 768]" = torch.ops.aten.where.self(unsqueeze_7, full_default_1, mul_483);  unsqueeze_7 = full_default_1 = mul_483 = None
    _unsafe_index_put_3: "f32[50265, 768]" = torch.ops.prims._unsafe_index_put_.default(full_default_6, [view], where_4, True);  full_default_6 = view = where_4 = None
    return [_unsafe_index_put_2, _unsafe_index_put, _unsafe_index_put_3, sum_259, sum_260, permute_736, view_797, permute_732, view_794, permute_727, view_790, permute_715, view_780, sum_250, sum_251, permute_711, view_777, permute_707, view_774, sum_244, sum_245, permute_703, view_771, permute_699, view_768, permute_694, view_764, permute_682, view_754, sum_235, sum_236, permute_678, view_751, permute_674, view_748, sum_229, sum_230, permute_670, view_745, permute_666, view_742, permute_661, view_738, permute_649, view_728, sum_220, sum_221, permute_645, view_725, permute_641, view_722, sum_214, sum_215, permute_637, view_719, permute_633, view_716, permute_628, view_712, permute_616, view_702, sum_205, sum_206, permute_612, view_699, permute_608, view_696, sum_199, sum_200, permute_604, view_693, permute_600, view_690, permute_595, view_686, permute_583, view_676, sum_190, sum_191, permute_579, view_673, permute_575, view_670, sum_184, sum_185, permute_571, view_667, permute_567, view_664, permute_562, view_660, permute_550, view_650, sum_175, sum_176, permute_546, view_647, permute_542, view_644, sum_169, sum_170, _unsafe_index_put_1, sum_165, sum_166, permute_538, view_641, permute_534, view_638, permute_529, view_634, permute_517, view_622, sum_156, sum_157, permute_513, view_619, permute_509, view_616, permute_504, view_612, permute_492, view_602, sum_147, sum_148, permute_488, view_599, permute_484, view_596, sum_141, sum_142, permute_480, view_593, permute_476, view_590, permute_471, view_586, permute_459, view_574, sum_132, sum_133, permute_455, view_571, permute_451, view_568, permute_446, view_564, permute_434, view_554, sum_123, sum_124, permute_430, view_551, permute_426, view_548, sum_117, sum_118, permute_422, view_545, permute_418, view_542, permute_413, view_538, permute_401, view_526, sum_108, sum_109, permute_397, view_523, permute_393, view_520, permute_388, view_516, permute_376, view_506, sum_99, sum_100, permute_372, view_503, permute_368, view_500, sum_93, sum_94, permute_364, view_497, permute_360, view_494, permute_355, view_490, permute_343, view_478, sum_84, sum_85, permute_339, view_475, permute_335, view_472, permute_330, view_468, permute_318, view_458, sum_75, sum_76, permute_314, view_455, permute_310, view_452, sum_69, sum_70, permute_306, view_449, permute_302, view_446, permute_297, view_442, permute_285, view_430, sum_60, sum_61, permute_281, view_427, permute_277, view_424, permute_272, view_420, permute_260, view_410, sum_51, sum_52, permute_256, view_407, permute_252, view_404, sum_45, sum_46, permute_248, view_401, permute_244, view_398, permute_239, view_394, permute_227, view_382, sum_36, sum_37, permute_223, view_379, permute_219, view_376, permute_214, view_372, permute_202, view_362, sum_27, sum_28, permute_198, view_359, permute_194, view_356, sum_21, sum_22, permute_190, None, None, None]
    